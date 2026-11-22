"""Refine an independently trained risk_power anchor using regional peak shape."""
from dataclasses import replace
import bootstrap
import torch
from torch import nn
from cohort_models import focal_per_row
from complement_models import ComplementMoE, method_contract as anchor_contract, method_loss_terms as anchor_loss
from i2moe_eeg.losses import task_loss
from risk_models import hard_rank

METHODS = ('peaks', 'peaks_matched', 'styled_peaks', 'styled_peaks_matched')


def method_contract(method):
    r = anchor_contract('risk_power')
    r.update(method=method, independent_risk_power_anchor=True,
        original_six_expert_and_anchor_receive_only_reference_objective=True,
        shape_features='Per-region standardized activation maximum, signed log mean and log population SD, calculated from frozen source features',
        original_anchor_margin_bound=2., additional_refinement_bound=1., bounded_margin_correction=3.,
        final_fusion='Detached original risk_power logits plus learned bounded per-window shape correction',
        refinement_weighting='Static 50% shared acquisition mass' if method.endswith('_matched') else 'Static cap64 anchor source weights',
        auxiliary_style_view=method.startswith('styled'), extra_source_cnn_forwards=0,
        added_module='Five peak-shape encoders and small score refinement',
        additional_head_loss='Focal and 0.03 top16 ranking on refined score. Styled arms average real and synthetic moment focal losses from same batch; ranking on real window.',
        inspiration='MixStyle, ICLR 2021. Same-label different-source moment interpolation preserving regional standardized maxima. No test-time transformation.')
    return r


def descriptors(moments):
    mean, maximum, deviation = moments.unbind(2)
    peak = ((maximum-mean).clamp_min(0)/deviation.clamp_min(1e-6)).clamp_max(32).log1p()
    return peak, mean, deviation


def cohort_donors(labels,groups):
    batch = len(labels)
    where = torch.arange(batch,device=labels.device)
    indices = (where[:,None]+where[None,:]+1) % batch
    available = (labels[indices]==labels[:,None]) & (groups[indices]!=groups[:,None])
    first = available.long().argmax(1)
    return torch.where(available.any(1),indices[where,first],where)


class PeakRefinement(nn.Module):
    def __init__(self,source_width):
        super().__init__()
        self.encoders = nn.ModuleList([nn.Sequential(nn.LayerNorm(3*source_width),nn.Linear(3*source_width,48),
                                                     nn.GELU()) for _ in range(5)])
        self.score = nn.Sequential(nn.LayerNorm(5*48+7),nn.Linear(5*48+7,96),nn.GELU(),nn.Linear(96,1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, shapes, context, donor=None):
        peak, mean, deviation = shapes
        if donor is not None:
            mean = (mean+mean[donor])*.5
            deviation = (deviation+deviation[donor])*.5
        style = torch.cat((peak,mean.sign()*mean.abs().log1p(), deviation.clamp_min(1e-6).log()),-1)
        values = [encoder(style[:,i]) for i,encoder in enumerate(self.encoders)]
        return self.score(torch.cat((context,*values),-1)).squeeze(-1).tanh()


def corrected(anchor_logits,delta):
    return anchor_logits.detach()+torch.stack((-delta/2,delta/2),-1)


class PeakShapeMoE(ComplementMoE):
    def __init__(self,method,**kwargs):
        if method not in METHODS:
            raise ValueError(method)
        super().__init__('risk_power',**kwargs)
        self.peak_method = method
        first = next(iter(self.frozen_experts.values()))
        source = getattr(first,'module',first)
        with torch.random.fork_rng(devices=[]):
            self.correction.refiner = PeakRefinement(source.encoder.projection[0].in_features)
        self.correction.register_buffer('refinement_weights',torch.ones_like(self.correction.group_base_weights))

    def forward(self,eeg,compute_interactions=None):
        anchor = super().forward(eeg,compute_interactions)
        expert = anchor.expert_logits.detach()
        asc = anchor.logits.detach()
        context = torch.cat((expert[...,1]-expert[...,0],(asc[:,1]-asc[:,0])[:,None]),1)
        moments = torch.stack([self.moments[i].detach() for i in range(5)],1)
        shapes = descriptors(moments)
        delta = self.correction.refiner(shapes,context)
        output = replace(anchor,logits=corrected(anchor.logits,delta))
        output.core_output, output.anchor_output = anchor.core_output, anchor
        output.correction_margin = anchor.correction_margin.detach()+delta
        output.training_groups, output.cohort_state = anchor.training_groups,self.correction
        output.shape_context,output.shapes = context,shapes
        return output

    def architecture_summary(self):
        result = super().architecture_summary()
        result.update(method=self.peak_method,method_contract=method_contract(self.peak_method),
                      added_module='Detached regional peak-shape correction')
        return result


def method_loss_terms(method,output,labels,args,focal):
    parts = anchor_loss('risk_power',output.anchor_output,labels,args,focal)
    reference_primary = parts['primary']
    group = output.training_groups
    state = output.cohort_state
    weights = state.refinement_weights[labels,group]
    per_row = focal_per_row(output.logits,labels,focal)
    if method.startswith('styled'):
        donor = cohort_donors(labels,group)
        delta = state.refiner(output.shapes,output.shape_context,donor)
        swapped = corrected(output.anchor_output.logits,delta)
        per_row = .5*(per_row+focal_per_row(swapped,labels,focal))
    extra = (weights*per_row).mean()+.03*hard_rank(output.logits,labels,weights,16)
    fused = task_loss(output.logits,labels,args.task_loss,focal)
    parts['total'] = parts['total']+extra
    parts['primary'] = fused
    parts['extra'] = parts['extra']+reference_primary+extra-fused
    return parts
