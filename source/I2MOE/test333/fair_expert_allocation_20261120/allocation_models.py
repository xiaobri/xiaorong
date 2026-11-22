from dataclasses import replace
import torch
from torch import nn

import bootstrap
from alignment_models import matched_rank
from block_features import SEGMENTS,base_weights,register_blocks
from cohort_models import focal_per_row
from i2moe_eeg.losses import task_loss
from peak_models import PeakShapeMoE,method_contract as peak_contract,method_loss_terms as peak_loss

METHODS=('independent','independent_aux','transport','transport_aux')


def method_contract(method):
    r=peak_contract('peaks')
    r.update(method=method,independent_peak_shape_anchor=True,
        original_regional_architecture_extension='Learned re-pooling of eight pre-LayerNorm temporal blocks through original five frozen regional classifiers.',
        regional_time_assignment='Six row/column normalization iterations with half native and half learned shared time capacity' if method.startswith('transport') else 'Separate distribution of time weights for each regional expert',
        base_temporal_mass='Eight consecutive segments of 8,8,8,8,8,8,8,7 final CNN time bins',
        effective_expert_scores='First five original classifiers evaluated on temporally re-pooled embeddings; global original redundancy plus bounded residual from all five pooling changes',
        final_fusion='Original six routing weights on changes in six expert margins plus fresh, separately trained peak-shape anchor',
        original_core_receives_only_original_loss_gradients=True,auxiliary_regional_weight=.1 if method.endswith('aux') else 0.,
        additional_head_loss='Cap64-weighted focal and .03 matched ranking; optional .1 regional focal loss mean across five',
        added_source_backbone_forwards=0,regional_residual_is_bounded=False,global_margin_delta_bound=1.5,
        bounded_margin_correction=None,
        inspiration='Lewis et al., ICML 2021, balanced expert assignment. Here finite capacities are per-window times and allocation is soft; all six experts are evaluated.')
    return r


def allocate(scores,prior,coordinated):
    kernel=scores.clamp(-3,3).exp()*prior
    if not coordinated:
        return kernel/kernel.sum(-1,keepdim=True)
    consensus=(scores.mean(1).clamp(-3,3).exp()*prior)
    capacity=.5*prior+.5*consensus/consensus.sum(-1,keepdim=True)
    for _ in range(6):
        kernel=kernel/kernel.sum(-1,keepdim=True).clamp_min(1e-8)
        kernel=kernel*(5*capacity[:,None,:]/kernel.sum(1,keepdim=True).clamp_min(1e-8))
    return kernel/kernel.sum(-1,keepdim=True).clamp_min(1e-8)


class AllocationLayer(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.scorers=nn.ModuleList([nn.Sequential(nn.LayerNorm(width+3),nn.Linear(width+3,32),nn.GELU(),
                                                   nn.Linear(32,1)) for _ in range(5)])
        for scorer in self.scorers:
            nn.init.zeros_(scorer[-1].weight)
            nn.init.zeros_(scorer[-1].bias)
        self.redundancy=nn.Sequential(nn.Linear(5*(width+SEGMENTS),64,bias=False),nn.Tanh(),
                                      nn.Linear(64,1,bias=False))

    def forward(self,blocks,prior,features,original_logits,sources,coordinated):
        token=torch.stack([s.encoder.projection[1](blocks[:,i]) for i,s in enumerate(sources)],1)
        token_logits=torch.stack([s.classifier(token[:,i]) for i,s in enumerate(sources)],1)
        local=token_logits[...,1]-token_logits[...,0]
        margin=original_logits[...,1]-original_logits[...,0]
        inputs=torch.cat((token-features[:,:,None],local[...,None],
                          (local-margin[:,:,None])[...,None],local.mean(1,keepdim=True).expand_as(local)[...,None]),-1)
        scores=torch.stack([layer(inputs[:,i]).squeeze(-1) for i,layer in enumerate(self.scorers)],1)
        scores=3*torch.tanh(scores/3)
        weights=allocate(scores,prior,coordinated)
        reference=allocate(torch.zeros_like(scores),prior,coordinated)
        delta_mass=weights-reference
        uniform=(prior[None,None,:,None]*blocks).sum(2)
        shift=(delta_mass[...,None]*blocks).sum(2)
        change=torch.stack([s.encoder.projection[1](uniform[:,i]+shift[:,i])
                            -s.encoder.projection[1](uniform[:,i]) for i,s in enumerate(sources)],1)
        adjusted=features+change
        regional=original_logits+torch.stack([s.classifier(adjusted[:,i])-s.classifier(features[:,i])
                                               for i,s in enumerate(sources)],1)
        global_delta=1.5*self.redundancy(torch.cat((change,delta_mass),-1).flatten(1)).squeeze(-1).tanh()
        return regional,global_delta,weights


class ExpertAllocationMoE(PeakShapeMoE):
    def __init__(self,method,**kwargs):
        if method not in METHODS:
            raise ValueError(method)
        super().__init__('peaks',**kwargs)
        self.allocation_method=method
        self.block_capture,handles=register_blocks(self.frozen_experts)
        self._source_hooks.extend(handles)
        with torch.random.fork_rng(devices=[]):
            self.correction.allocation=AllocationLayer(self.hidden_dim)

    def forward(self,eeg,compute_interactions=None):
        peak=super().forward(eeg,compute_interactions)
        if eeg.ndim==4 and tuple(eeg.shape[1:])==(5,4,1280):
            blocks=eeg[:,:,3,256:1280].reshape(len(eeg),5,SEGMENTS,128)
            length=63
        else:
            blocks=torch.stack([self.block_capture[i][0] for i in range(5)],1)
            length=self.block_capture[0][1]
        prior=base_weights(length,blocks)
        sources=[getattr(e,'module',e) for e in self.frozen_experts.values()]
        original=peak.expert_logits.detach()
        regions,global_change,time=self.correction.allocation(blocks,prior,peak.frozen_features.detach(),original[:,:5],
            sources,self.allocation_method.startswith('transport'))
        global_logits=original[:,5]+torch.stack((-global_change/2,global_change/2),1)
        final_experts=torch.cat((regions,global_logits[:,None]),1)
        shift=((final_experts-original)*peak.interaction_weights.detach()[...,None]).sum(1)
        out=replace(peak,logits=peak.logits.detach()+shift,expert_logits=final_experts)
        out.core_output,out.anchor_output,out.peak_output=peak.core_output,peak.anchor_output,peak
        out.training_groups,out.cohort_state=peak.training_groups,self.correction
        out.correction_margin=peak.correction_margin.detach()+shift[:,1]-shift[:,0]
        out.pooled_risk=(original[:,:5,1]-original[:,:5,0]).mean(1)
        out.time_allocation=time
        return out

    def architecture_summary(self):
        r=super().architecture_summary()
        r.update(method=self.allocation_method,method_contract=method_contract(self.allocation_method))
        return r


def method_loss_terms(method,output,labels,args,focal):
    parts=peak_loss('peaks',output.peak_output,labels,args,focal)
    ref=parts['primary']
    weights=output.cohort_state.group_base_weights[labels,output.training_groups]
    extra=(weights*focal_per_row(output.logits,labels,focal)).mean()+.03*matched_rank(output.logits,labels,weights,output.pooled_risk)
    if method.endswith('_aux'):
        regional=focal_per_row(output.expert_logits[:,:5].reshape(-1,2),labels.repeat_interleave(5),focal)
        extra=extra+.1*(weights*regional.reshape(-1,5).mean(1)).mean()
    fused=task_loss(output.logits,labels,args.task_loss,focal)
    parts['total']=parts['total']+extra
    parts['primary']=fused
    parts['extra']=parts['extra']+ref+extra-fused
    return parts
