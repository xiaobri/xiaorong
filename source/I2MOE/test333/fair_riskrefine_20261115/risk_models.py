"""Refine cohort robustness and top-negative ranking under the fixed budget."""
import bootstrap
import torch
from torch.nn import functional as F

from cohort_models import CohortCorrectionMoE, focal_per_row, group_weights
from cohort_models import method_contract as original_contract
from i2moe_eeg.f1_variants import loss_terms as baseline_loss_terms
from i2moe_eeg.losses import task_loss

METHODS = ('risk_cap16', 'risk_hard', 'risk_core', 'risk_core_hard')


def method_contract(method):
    result = original_contract('cohort_dro')
    result.update(method=method, cohort_weighting='Train-only capped16 inverse-square-root and EMA risk',
        static_weights='Inverse square-root Train cohort size within each class; cap16; population mean1',
        rank_top_negative_count=4 if method.endswith('hard') else 16,
        baseline_path_receives_only_original_loss_gradients='core' not in method,
        extra_core_cohort_focal_weight=.5 if 'core' in method else 0.,
        original_core_loss_terms_retained=True,
        core_trajectory_expected_equal_to_baseline='core' not in method,
        correction_gradient_still_detached_from_core=True)
    return result


class RiskRefinementMoE(CohortCorrectionMoE):
    def __init__(self, method, **kwargs):
        if method not in METHODS:
            raise ValueError(method)
        super().__init__('cohort_dro', **kwargs)
        self.risk_method = method

    def architecture_summary(self):
        result = super().architecture_summary()
        result.update(method=self.risk_method, method_contract=method_contract(self.risk_method))
        return result


def hard_rank(logits, labels, weights, count):
    margin = logits[:, 1]-logits[:, 0]
    positive, negative = torch.where(labels == 1)[0], torch.where(labels == 0)[0]
    if not len(positive) or not len(negative):
        return margin.sum()*0
    selected = negative[margin[negative].detach().topk(min(count, len(negative))).indices]
    losses = F.softplus(margin[selected][None]-margin[positive][:, None]+.5)
    pair_weights = weights[positive][:, None]*weights[selected][None]
    return (pair_weights*losses).sum()/pair_weights.sum().clamp_min(1e-8)


def method_loss_terms(method, output, labels, args, focal):
    parts = baseline_loss_terms(output.core_output, labels, args, focal)
    core_primary = parts['primary']
    per_row = focal_per_row(output.logits, labels, focal)
    # Update risk exactly once per original training batch.
    weights = group_weights('cohort_dro', output, labels, per_row)
    corrected = (per_row*weights).mean()
    ranking = .03*hard_rank(output.logits, labels, weights, 4 if method.endswith('hard') else 16)
    core_aux = .5*(focal_per_row(output.core_output.logits, labels, focal)*weights).mean() if 'core' in method else corrected.new_zeros(())
    fused = task_loss(output.logits, labels, args.task_loss, focal)
    parts['total'] = parts['total']+corrected+ranking+core_aux
    parts['primary'] = fused
    parts['extra'] = parts['extra']+core_primary+corrected+ranking+core_aux-fused
    return parts
