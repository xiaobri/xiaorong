"""Confidence-aware six-expert routing and a separate pair-ranking ablation.

All source experts and the redundancy Transformer are unchanged. The original
focal/auxiliary/consistency losses are retained. Two explicitly named methods
add a within-batch ranking term; no extra data, draws or optimizer steps occur.
"""
import bootstrap
import torch
from torch import nn
from torch.nn import functional as F

from i2moe_eeg.f1_variants import SixExpertF1, loss_terms as baseline_loss_terms

METHODS = ('evidence_logit', 'evidence_probability', 'baseline_rank', 'evidence_rank')
RANK_WEIGHT = .01
RANK_MARGIN = .5
HARD_NEGATIVES = 16


def method_contract(method: str) -> dict:
    ranked = method in ('baseline_rank', 'evidence_rank')
    return dict(
        original_focal_auxiliary_and_consistency_terms_retained=True,
        unchanged_total_loss=not ranked,
        ranking_weight=RANK_WEIGHT if ranked else 0.,
        ranking_margin=RANK_MARGIN if ranked else None,
        maximum_hard_negatives_per_current_batch=HARD_NEGATIVES if ranked else None,
        extra_training_draws=0,
        extra_optimizer_steps=0,
        validation_labels_used_by_model=False,
        fusion='arithmetic probability mixture' if method == 'evidence_probability' else 'original weighted logits',
        evidence_router=method != 'baseline_rank',
        loss_only_control=method == 'baseline_rank',
    )


def within_batch_rank_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    margins = logits[:, 1] - logits[:, 0]
    positives, negatives = margins[labels == 1], margins[labels == 0]
    if positives.numel() == 0 or negatives.numel() == 0:
        return margins.sum() * 0
    negative_scores = negatives.topk(min(HARD_NEGATIVES, negatives.numel())).values
    return F.softplus(negative_scores[None] - positives[:, None] + RANK_MARGIN).mean()


def method_loss_terms(method, output, labels, args, focal):
    parts = baseline_loss_terms(output, labels, args, focal)
    if method in ('baseline_rank', 'evidence_rank'):
        extra = RANK_WEIGHT * within_batch_rank_loss(output.logits, labels)
        parts['total'] = parts['total'] + extra
        parts['extra'] = parts['extra'] + extra
    return parts


class EvidenceRoutingMoE(SixExpertF1):
    def __init__(self, method: str, **kwargs):
        super().__init__(**kwargs)
        if method not in METHODS:
            raise ValueError(method)
        self.method = method
        self.routing_adapter = None
        if method != 'baseline_rank':
            with torch.random.fork_rng(devices=[]):
                self.routing_adapter = nn.Sequential(
                    nn.LayerNorm(self.hidden_dim * 6 + 48),
                    nn.Linear(self.hidden_dim * 6 + 48, 128), nn.GELU(),
                    nn.Dropout(kwargs.get('router_dropout', .5)), nn.Linear(128, 6))
                nn.init.zeros_(self.routing_adapter[-1].weight)
                nn.init.zeros_(self.routing_adapter[-1].bias)

    def forward(self, eeg, compute_interactions=None):
        result = super().forward(eeg, compute_interactions)
        if self.routing_adapter is not None:
            # Match the original rule: the routing path cannot train global
            # expert representations/logits through its evidence inputs.
            logits = result.expert_logits.detach()
            margins = logits[..., 1] - logits[..., 0]
            probabilities = logits.softmax(-1)
            entropy = -(probabilities * logits.log_softmax(-1)).sum(-1)
            disagreement = (margins[:, :, None] - margins[:, None, :]).flatten(1)
            evidence = torch.cat((result.router_features.flatten(1), margins, entropy, disagreement), -1)
            devices = [eeg.device.index] if eeg.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                adjustment = self.routing_adapter(evidence) / self.router.temperature
            result.interaction_weights = (result.interaction_weights.clamp_min(1e-30).log() + adjustment).softmax(-1)
        if self.method == 'evidence_probability':
            # Stable arithmetic mixture, normalized over the two labels. This
            # is an architecture change from the original geometric mixture.
            result.logits = torch.logsumexp(
                result.expert_logits.log_softmax(-1)
                + result.interaction_weights.clamp_min(1e-30).log()[..., None], dim=1)
        else:
            result.logits = (result.expert_logits * result.interaction_weights[..., None]).sum(1)
        return result

    def architecture_summary(self):
        result = super().architecture_summary()
        result.update(method=self.method, retained_experts=list(self.expert_names),
                      method_contract=method_contract(self.method),
                      original_redundancy_transformer_preserved=True,
                      routing_inputs='original six representations plus detached expert margins, entropy and pairwise margin differences',
                      original_parameter_initialization_preserved=True,
                      new_router_initialization='zero final layer; original weights up to FP32 normalization rounding',
                      full_system_passes=1, outer_tta=False)
        return result
