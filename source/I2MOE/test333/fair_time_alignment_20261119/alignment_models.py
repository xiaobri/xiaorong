"""Learn from coincidence and lead/lag of five pretrained regional score traces."""
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

import bootstrap
from cohort_models import focal_per_row
from complement_models import ComplementMoE, method_contract as anchor_contract, method_loss_terms as anchor_loss
from i2moe_eeg.losses import task_loss
from risk_models import hard_rank

METHODS = ('shifted','aligned','shifted_match','aligned_match')


def method_contract(method):
    result = anchor_contract('risk_power')
    result.update(method=method, independent_risk_power_anchor=True,
        descriptor='Five within-window regional classifier score traces, 63 bins. Node shape, pairwise simultaneous correlation, soft peak overlap, symmetric and directed two-bin lag.',
        time_alignment='Intact' if method.startswith('aligned') else 'Fixed circular shift of regions by 0,11,23,37,49 bins for edge summaries only',
        ranking='Top16 difficult negatives ranked by final score' if not method.endswith('_match') else '16 nearest opposite-label negatives by mean complete-window frozen expert margin for each positive',
        extra_backbone_forwards=0, extra_source_classifier_calls_per_expert='One batched call on projected local features',
        final_fusion='Fresh independently trained risk_power plus detached bounded per-window temporal correction',
        bounded_margin_correction=3.,extra_refinement_bound=1.,
        additional_head_loss='Population-normalized static cap64 weighted focal and 0.03 pairwise ranking on refined logits',
        related_paper='Tang et al., ICLR 2022, EEG dynamic connectivity. Here observations are learned full-input expert scores, not electrode potentials or physiological connectivity.')
    return result


def local_descriptors(traces,pooled,shifted):
    center = traces-traces.mean(-1,keepdim=True)
    scale = center.square().mean(-1,keepdim=True).clamp_min(1e-12).sqrt()
    z = center/scale
    weights = z.softmax(-1)
    node = torch.stack((pooled,center.amax(-1),traces.topk(min(4,traces.shape[-1])).values.mean(-1)-pooled,
                        scale.squeeze(-1),z.diff(dim=-1).abs().mean(-1),
                        -(weights*weights.clamp_min(1e-10).log()).sum(-1)),-1)
    if shifted:
        offsets = (0,11,23,37,49)
        z = torch.stack([column.roll(offset,-1) for column,offset in zip(z.unbind(1),offsets)],1)
        weights = z.softmax(-1)
    length = z.shape[-1]
    corr = z@z.transpose(1,2)/length
    overlap = (weights@weights.transpose(1,2))*length
    lag = z[:,:,2:]@z[:,:,:-2].transpose(1,2)/(length-2)
    axes = torch.triu_indices(5,5,1,device=z.device)
    i,j = axes.unbind(0)
    edge = torch.stack((corr[:,i,j],overlap[:,i,j],.5*(lag[:,i,j]+lag[:,j,i]),
                        .5*(lag[:,i,j]-lag[:,j,i])),-1)
    return node,edge


class TemporalRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.nodes = nn.Sequential(nn.Linear(6,16),nn.GELU())
        self.edges = nn.Sequential(nn.Linear(4,16),nn.GELU())
        self.readout = nn.Sequential(nn.LayerNorm(15*16+1),nn.Linear(15*16+1,64),nn.GELU(),nn.Linear(64,1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self,traces,pooled,anchor_margin,shifted):
        node,edge = local_descriptors(traces,pooled,shifted)
        inputs = torch.cat((self.nodes(node).flatten(1),self.edges(edge).flatten(1),anchor_margin[:,None]),1)
        return self.readout(inputs).squeeze(-1).tanh()


class TimeAlignmentMoE(ComplementMoE):
    def __init__(self,method,**kwargs):
        if method not in METHODS:
            raise ValueError(method)
        super().__init__('risk_power',**kwargs)
        self.alignment_method = method
        with torch.random.fork_rng(devices=[]):
            self.correction.refiner = TemporalRefiner()
        self.traces = {}
        for i,expert in enumerate(self.frozen_experts.values()):
            source = getattr(expert,'module',expert)
            def collect(module,inputs,output,i=i,source=source):
                token = source.encoder.projection(output.mean(-2).transpose(1,2))
                logits = source.classifier(token)
                self.traces[i] = (logits[...,1]-logits[...,0]).detach()
            self._source_hooks.append(source.encoder.features.register_forward_hook(collect))

    def forward(self,eeg,compute_interactions=None):
        self.traces.clear()
        anchor = super().forward(eeg,compute_interactions)
        traces = (eeg[:,:,3,256:319] if eeg.ndim==4 and tuple(eeg.shape[1:])==(5,4,1280)
                  else torch.stack([self.traces[i] for i in range(5)],1))
        logits = anchor.expert_logits[:,:5].detach()
        pooled = logits[...,1]-logits[...,0]
        reference = anchor.logits.detach()
        delta = self.correction.refiner(traces,pooled,reference[:,1]-reference[:,0],
                                        self.alignment_method.startswith('shifted'))
        final = reference + torch.stack((-delta/2,delta/2),-1)
        output = replace(anchor,logits=final)
        output.core_output,output.anchor_output = anchor.core_output,anchor
        output.correction_margin = anchor.correction_margin.detach()+delta
        output.training_groups,output.cohort_state = anchor.training_groups,self.correction
        output.pooled_risk = pooled.mean(1)
        return output

    def architecture_summary(self):
        result = super().architecture_summary()
        result.update(method=self.alignment_method,method_contract=method_contract(self.alignment_method))
        return result


def matched_rank(logits,labels,weights,pooled):
    margin = logits[:,1]-logits[:,0]
    pos,neg = torch.where(labels==1)[0],torch.where(labels==0)[0]
    if not len(pos) or not len(neg):
        return margin.sum()*0
    distance = (pooled[pos,None]-pooled[None,neg]).abs()
    chosen = neg[distance.topk(min(16,len(neg)),largest=False).indices]
    weight = weights[pos,None]*weights[chosen]
    loss = F.softplus(.5+margin[chosen]-margin[pos,None])
    return (loss*weight).sum()/weight.sum().clamp_min(1e-8)


def method_loss_terms(method,output,labels,args,focal):
    parts = anchor_loss('risk_power',output.anchor_output,labels,args,focal)
    reference_primary = parts['primary']
    weight = output.cohort_state.group_base_weights[labels,output.training_groups]
    losses = focal_per_row(output.logits,labels,focal)
    ranking = (matched_rank(output.logits,labels,weight,output.pooled_risk) if method.endswith('_match')
               else hard_rank(output.logits,labels,weight,16))
    extra = (weight*losses).mean()+.03*ranking
    fused = task_loss(output.logits,labels,args.task_loss,focal)
    parts['total'] = parts['total']+extra
    parts['primary'] = fused
    parts['extra'] = parts['extra']+reference_primary+extra-fused
    return parts
