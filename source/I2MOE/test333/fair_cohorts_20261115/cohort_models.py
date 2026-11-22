"""Train-only conditional cohort robustness on a detached correction path."""
from dataclasses import replace
import math

import bootstrap
import torch
from torch import nn
from torch.nn import functional as F

from protected_models import ProtectedCorrectionMoE
from i2moe_eeg.f1_variants import loss_terms as baseline_loss_terms
from i2moe_eeg.losses import task_loss

METHODS = ('rank_control', 'cohort_balanced', 'cohort_dro', 'cohort_cvar')
MARGIN_BOUND = 2.
RANK_WEIGHT = .03
MAX_GROUPS = 64


def method_contract(method):
    return dict(original_six_expert_objective_retained=True, unchanged_total_loss=False,
        baseline_path_receives_only_original_loss_gradients=True,
        correction_inputs_and_base_logits_detached=True, historical_baseline_checkpoint_reused=False,
        same_batches_and_adamw_updates=True, extra_training_draws=0, extra_optimizer_steps=0,
        bounded_margin_correction=MARGIN_BOUND, extra_fused_task_loss_weight=1.,
        extra_rank_weight=RANK_WEIGHT, rank_top_negative_count=16,
        cohort_weighting=method, cohort_metadata_used_only_in_training_loss=True,
        validation_cohort_metadata_used=False, inference_uses_eeg_only=True,
        static_weights='Inverse square-root Train cohort size within each class; capped at4 with exact class-conditional population mean1',
        dro='EMA per-cohort per-class focal risk; bounded square-root risk ratio; re-normalized using Train class-conditional counts' if method == 'cohort_dro' else None,
        cvar='Half weighted mean plus half class-prior-preserving worst20-percent focal loss' if method == 'cohort_cvar' else None,
        final_f1_noninferiority_guaranteed=False)


class CohortCorrectionMoE(ProtectedCorrectionMoE):
    def __init__(self, method, **kwargs):
        super().__init__('protected_moments', **kwargs)
        if method not in METHODS:
            raise ValueError(method)
        self.cohort_method = method
        self.correction.register_buffer('group_risk', torch.ones(2, MAX_GROUPS))
        self.correction.register_buffer('group_seen', torch.zeros(2, MAX_GROUPS, dtype=torch.bool))
        self.correction.register_buffer('group_probabilities', torch.zeros(2, MAX_GROUPS))
        self.correction.register_buffer('group_base_weights', torch.zeros(2, MAX_GROUPS))

    def forward(self, eeg, compute_interactions=None):
        output = super().forward(eeg, compute_interactions)
        margin = MARGIN_BOUND*output.correction_margin
        result = replace(output, logits=output.core_output.logits.detach()+torch.stack((-margin/2, margin/2), -1))
        result.core_output, result.correction_margin = output.core_output, margin
        result.cohort_state = self.correction
        if eeg.ndim == 4 and tuple(eeg.shape[1:]) == (5, 4, 1280):
            result.training_groups = eeg[:, 0, 3, 130].long()
        else:
            result.training_groups = torch.full((len(eeg),), -1, device=eeg.device, dtype=torch.long)
        return result

    def architecture_summary(self):
        result = super().architecture_summary()
        result.update(method=self.cohort_method, method_contract=method_contract(self.cohort_method))
        return result


def focal_per_row(logits, labels, focal):
    log_p = F.log_softmax(logits, -1).gather(1, labels[:, None]).squeeze(1)
    alpha = torch.where(labels == 1, focal.positive_alpha, 1-focal.positive_alpha)
    return -alpha*(1-log_p.exp()).pow(focal.gamma)*log_p


def group_weights(method, output, labels, losses):
    if method == 'rank_control':
        return torch.ones_like(losses)
    groups, state = output.training_groups, output.cohort_state
    if bool(((groups < 0) | (groups >= MAX_GROUPS)).any()):
        raise ValueError('Training requires valid Train-only acquisition group IDs')
    if bool((state.group_probabilities[labels, groups] <= 0).any()):
        raise ValueError('Training group/class absent from Train metadata')
    table = state.group_base_weights.clone()
    if method == 'cohort_dro':
        with torch.no_grad():
            cells = labels*MAX_GROUPS+groups
            sums = torch.zeros(2*MAX_GROUPS, device=losses.device).scatter_add_(0, cells, losses.detach())
            counts = torch.zeros_like(sums).scatter_add_(0, cells, torch.ones_like(losses))
            present = counts.view(2, MAX_GROUPS) > 0
            observed = (sums/counts.clamp_min(1)).view(2, MAX_GROUPS)
            previous = torch.where(state.group_seen, state.group_risk, observed)
            state.group_risk.copy_(torch.where(present, .95*previous+.05*observed, state.group_risk))
            state.group_seen.logical_or_(present)
            probabilities = state.group_probabilities
            seen_probability = probabilities*state.group_seen
            class_risk = (seen_probability*state.group_risk).sum(-1, keepdim=True)/seen_probability.sum(-1, keepdim=True).clamp_min(1e-8)
            effective_risk = torch.where(state.group_seen, state.group_risk, class_risk)
            factor = (effective_risk/class_risk.clamp_min(1e-8)).sqrt().clamp(.5, 2.)
            table = table*factor
            table /= (table*probabilities).sum(-1, keepdim=True).clamp_min(1e-8)
    return table[labels, groups].detach()


def weighted_rank(logits, labels, weights):
    margin = logits[:, 1]-logits[:, 0]
    pos, neg = torch.where(labels == 1)[0], torch.where(labels == 0)[0]
    if not len(pos) or not len(neg):
        return margin.sum()*0
    neg = neg[margin[neg].detach().topk(min(16, len(neg))).indices]
    pair = F.softplus(margin[neg][None, :]-margin[pos][:, None]+.5)
    pair_weights = weights[pos][:, None]*weights[neg][None, :]
    return (pair*pair_weights).sum()/pair_weights.sum().clamp_min(1e-8)


def method_loss_terms(method, output, labels, args, focal):
    parts = baseline_loss_terms(output.core_output, labels, args, focal)
    original_primary = parts['primary']
    losses = focal_per_row(output.logits, labels, focal)
    weights = group_weights(method, output, labels, losses)
    correction = (losses*weights).mean()
    if method == 'cohort_cvar':
        tail = losses.sum()*0
        for label in (0, 1):
            selected = (losses*weights)[labels == label]
            if len(selected):
                tail = tail+len(selected)/len(labels)*selected.topk(max(1, math.ceil(.2*len(selected)))).values.mean()
        correction = .5*correction+.5*tail
    rank = RANK_WEIGHT*weighted_rank(output.logits, labels, weights)
    # Keep the original primary task in the logs; all added terms explicit.
    fused = task_loss(output.logits, labels, args.task_loss, focal)
    parts['total'] = parts['total']+correction+rank
    parts['primary'] = fused
    parts['extra'] = parts['extra']+original_primary+correction+rank-fused
    return parts
