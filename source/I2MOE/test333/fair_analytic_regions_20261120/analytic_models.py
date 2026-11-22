"""Accumulate forward-only regional likelihood equations alongside the six-expert EEG model."""
from dataclasses import replace
import math

import torch
from torch import nn

from allocation_models import ExpertAllocationMoE, method_contract as parent_contract
from allocation_models import method_loss_terms as parent_loss
from i2moe_eeg.losses import task_loss

METHODS=('pooled_residual','temporal_residual','pooled_direct','temporal_direct')


def method_contract(method: str) -> dict:
    result=parent_contract('independent_aux')
    result.update(method=method,independent_allocation_parent=True,
        analytic_features='First 32 original pooled hidden coordinates for each of five experts'
            + (' plus regional margin and six summaries of eight frozen time-block margins' if method.startswith('temporal') else ''),
        analytic_target='First logistic Newton residual about the prequentially calibrated parent'
            if method.endswith('residual') else 'Independent online logistic likelihood with recursive IRLS',
        normal_equations='After each training prediction, accumulate sufficient statistics on the original batch, with 0.025 diagonal ridge after centering. Refresh every eight batches and before evaluation. No past EEG is revisited.',
        fusion='Add calibrated residual, or equally mix independent and parent margins in calibrated coordinates. Return to the original parent margin scale. Limit additional margin magnitude to six.',
        analytic_updates_are_additional_supervised_state=True,extra_adamw_steps=0,
        extra_validation_training=False,analytic_source_weights=False,
        parent_uses_original_acquisition_weights=True, original_source_networks_unchanged=True,
        readout_inspiration='McDonnell et al., RanPAC, NeurIPS 2023: incremental supervised Gram statistics on frozen features. Here we use logistic curvature and original ordered EEG batches.')
    return result


class NormalEquations(nn.Module):
    def __init__(self,width: int):
        super().__init__()
        self.register_buffer('gram',torch.zeros(width,width,dtype=torch.float64))
        self.register_buffer('target',torch.zeros(width,dtype=torch.float64))
        self.register_buffer('coef',torch.zeros(width,dtype=torch.float64))

    def add(self, x, curvature, rhs):
        self.gram.add_(x.T@(curvature[:,None]*x))
        self.target.add_(x.T@rhs)

    def solve(self):
        gram=(self.gram+self.gram.T)*.5
        mass=gram[-1,-1].clamp_min(1e-12)
        var=(gram.diag()[:-1]-gram[:-1,-1].square()/mass).clamp_min(1e-5*mass)
        penalty=torch.cat((.025*var,(1e-8*mass).reshape(1)))
        self.coef.copy_(torch.linalg.solve(gram+penalty.diag(),self.target))


class RegionalLikelihood(nn.Module):
    def __init__(self,width: int,direct: bool):
        super().__init__()
        self.direct=direct
        self.calibration=NormalEquations(2)
        self.calibration.coef[0]=1
        self.score=NormalEquations(width+1)
        if direct:
            self.score.coef[-1]=-math.log(30.)
        self.register_buffer('updates',torch.zeros((),dtype=torch.long))
        self.register_buffer('refresh_step',torch.zeros((),dtype=torch.long))
        self.register_buffer('counts',torch.zeros(2,dtype=torch.long))
        self.register_buffer('ready',torch.zeros((),dtype=torch.bool))

    def refresh(self):
        if self.counts[1].item()<32 or self.counts[0].item()<128:
            return
        self.calibration.solve()
        self.calibration.coef[0].clamp_(.15,5.)
        self.calibration.coef[1].clamp_(-12,8.)
        self.score.solve()
        self.refresh_step.copy_(self.updates)
        self.ready.fill_(True)

    @staticmethod
    def design(features):
        return torch.cat((features.double().clamp(-32,32),torch.ones_like(features[:,:1],dtype=torch.float64)),-1)

    @torch.no_grad()
    def update(self,features,margin,labels):
        x=self.design(features)
        z=self.design(margin[:,None])
        y=labels.double()
        offset=z@self.calibration.coef
        p=offset.clamp(-12,12).sigmoid()
        curvature=(p*(1-p)).clamp_min(1e-5)
        self.calibration.add(z,curvature,curvature*offset+y-p)
        if self.direct:
            score=(x@self.score.coef).clamp(-12,12)
            prob=score.sigmoid()
            h=(prob*(1-prob)).clamp_min(1e-5)
            self.score.add(x,h,h*score+y-prob)
        else:
            self.score.add(x,curvature,y-p)
        self.counts.add_(torch.bincount(labels,minlength=2))
        self.updates.add_(1)
        if self.updates.item()%8==0:
            self.refresh()

    @torch.no_grad()
    def forward(self,features,margin):
        if not self.training and self.refresh_step.item()!=self.updates.item():
            self.refresh()
        if not self.ready.item():
            return torch.zeros_like(margin)
        a,b=self.calibration.coef
        score=self.design(features)@self.score.coef
        if self.direct:
            delta=.5*((score-b)/a-margin)
        else:
            delta=2*(score/2).tanh()/a
        return delta.clamp(-6,6).to(margin)


class AnalyticRegionMoE(ExpertAllocationMoE):
    def __init__(self,method: str,**kwargs):
        if method not in METHODS:
            raise ValueError(method)
        super().__init__('independent_aux',**kwargs)
        self.analytic_method=method
        self.temporal=method.startswith('temporal')
        self.width=min(32,self.hidden_dim)
        self.correction.analytic=RegionalLikelihood(5*(self.width+7*int(self.temporal)),method.endswith('direct'))

    @torch.no_grad()
    def observations(self,eeg,parent):
        features=parent.frozen_features[:,:,:self.width]
        if self.temporal:
            if eeg.ndim==4 and tuple(eeg.shape[1:])==(5,4,1280):
                blocks=eeg[:,:,3,256:].reshape(len(eeg),5,8,128)
            else:
                blocks=torch.stack([self.block_capture[i][0] for i in range(5)],1)
            traces=[]
            for i,expert in enumerate(self.frozen_experts.values()):
                source=getattr(expert,'module',expert)
                logits=source.classifier(source.encoder.projection[1](blocks[:,i]))
                traces.append(logits[...,1]-logits[...,0])
            t=torch.stack(traces,1)
            original=parent.expert_logits[:,:5]
            stats=torch.stack((original[...,1]-original[...,0],t.mean(2),t.std(2,unbiased=False),
                               t.amax(2),t.amin(2),t.topk(2,dim=2).values.mean(2),t.diff(dim=2).abs().mean(2)),-1)
            features=torch.cat((features,stats),-1)
        return features.flatten(1)

    def forward(self,eeg,compute_interactions=None):
        parent=super().forward(eeg,compute_interactions)
        features=self.observations(eeg,parent)
        margin=(parent.logits[:,1]-parent.logits[:,0]).detach()
        delta=self.correction.analytic(features,margin)
        out=replace(parent,logits=parent.logits+torch.stack((-delta/2,delta/2),1))
        out.core_output,out.anchor_output,out.peak_output=parent.core_output,parent.anchor_output,parent.peak_output
        out.allocation_output=parent
        out.training_groups,out.cohort_state=parent.training_groups,self.correction
        out.correction_margin=parent.correction_margin.detach()+delta
        out.analytic_features,out.parent_margin=features,margin
        return out

    def architecture_summary(self):
        result=super().architecture_summary()
        result.update(method=self.analytic_method,method_contract=method_contract(self.analytic_method))
        return result


def method_loss_terms(method,output,labels,args,focal):
    parts=parent_loss('independent_aux',output.allocation_output,labels,args,focal)
    solver=output.cohort_state.analytic
    if solver.training and torch.is_grad_enabled():
        solver.update(output.analytic_features,output.parent_margin,labels)
    primary=task_loss(output.logits,labels,args.task_loss,focal)
    parts['extra']=parts['extra']+parts['primary']-primary
    parts['primary']=primary
    return parts
