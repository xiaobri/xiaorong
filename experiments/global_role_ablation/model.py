"""Controlled interventions on the existing global expert's two output paths.

All arms retain the baseline modules, initialization, corrupted global views,
and auxiliary/consistency training. The gap used by E is a label-conditioned
empirical capability proxy on training examples, not a PID or Bayes quantity.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.nn import functional as F

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "source" / "I2MOE6_F1"
for import_root in (SOURCE_ROOT.parent, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from i2moe_eeg.f1_variants import SixExpertF1, loss_terms as baseline_loss_terms
from i2moe_eeg.losses import interaction_feature_losses, task_loss
from i2moe_eeg.model import I2MoEOutput


ARMS = ("A", "B", "C", "D", "E", "E_mean")
GAP_AUXILIARY_WEIGHT = 0.05


class GlobalRoleAblation(SixExpertF1):
    """A: neither path; B: router only; C: prediction only; D/E: both.

    The global token is detached for routing in every arm, as in the baseline.
    A/B disable its prediction by masking gate logits before softmax. A/C keep
    the router's six-token shape but replace its last input token with zeros.
    E changes only the global auxiliary loss (see :func:`loss_terms`).
    """

    def __init__(self, arm: str = "D", **kwargs):
        if arm not in ARMS:
            raise ValueError(f"Unknown global-role arm {arm!r}; expected one of {ARMS}.")
        variant = kwargs.pop("variant", "baseline")
        if variant != "baseline":
            raise ValueError("Global-role ablations require variant='baseline'.")
        super().__init__(variant="baseline", **kwargs)
        self.arm = arm

    def forward(self, eeg: torch.Tensor, compute_interactions: Optional[bool] = None):
        if self.arm in {"D", "E", "E_mean"}:
            # A single, unmodified baseline call preserves all dropout/noise RNG.
            output = super().forward(eeg, compute_interactions)
        else:
            output = self._intervened_forward(eeg, compute_interactions)
        output.global_role_arm = self.arm
        return output

    def _intervened_forward(self, eeg, compute_interactions):
        if compute_interactions is None:
            compute_interactions = self.training
        frozen_features, frozen_logits = self._encode_frozen(eeg)
        # Match the baseline's order, shapes, and number of stochastic calls.
        if compute_interactions:
            views = self._feature_views(frozen_features)
            features, logits = self.redundancy_expert(views)
            interaction_view_features = features.unsqueeze(1)
            interaction_view_logits = logits.unsqueeze(1)
            new_features = interaction_view_features[:, :, 0, :]
            new_logits = interaction_view_logits[:, :, 0, :]
        else:
            interaction_view_features = None
            interaction_view_logits = None
            features, logits = self.redundancy_expert(frozen_features)
            new_features = features.unsqueeze(1)
            new_logits = logits.unsqueeze(1)

        global_router_token = new_features.detach()
        if self.arm in {"A", "C"}:
            global_router_token = torch.zeros_like(global_router_token)
        router_features = torch.cat((frozen_features, global_router_token), dim=1)
        if self.arm in {"A", "B"}:
            scores = self.router.network(router_features.flatten(start_dim=1))
            scores = scores / self.router.temperature
            scores = torch.cat((scores[:, :5], torch.full_like(scores[:, 5:], -torch.inf)), dim=1)
            weights = scores.softmax(dim=1)
        else:
            weights = self.router(router_features)

        expert_logits = torch.cat((frozen_logits, new_logits), dim=1)
        if self.arm in {"A", "B"}:
            # Exclude the disabled branch entirely, also avoiding 0 * nonfinite
            # global logits from contaminating the five-expert prediction.
            combined = torch.sum(expert_logits[:, :5] * weights[:, :5, None], dim=1)
        else:
            combined = torch.sum(expert_logits * weights.unsqueeze(-1), dim=1)
        return I2MoEOutput(
            logits=combined,
            expert_logits=expert_logits,
            interaction_weights=weights,
            frozen_features=frozen_features,
            router_features=router_features,
            interaction_view_features=interaction_view_features,
            interaction_view_logits=interaction_view_logits,
        )

    def architecture_summary(self):
        summary = super().architecture_summary()
        summary.update(
            model="global_role_ablation",
            global_role_arm=self.arm,
            global_prediction_enabled=self.arm not in {"A", "B"},
            global_router_token_enabled=self.arm not in {"A", "C"},
            global_router_token_detached=True,
            global_gate_mask_before_softmax=self.arm in {"A", "B"},
            global_auxiliary_loss="normalized_empirical_gap_weighted_focal"
            if self.arm in {"E", "E_mean"} else "baseline",
            empirical_gap_aggregation=("mean" if self.arm == "E_mean" else "max")
            if self.arm in {"E", "E_mean"} else None,
        )
        summary["outer_router_includes_interaction_features"] = self.arm not in {"A", "C"}
        if self.arm in {"A", "C"}:
            summary["outer_router_input"] = "five_frozen_features_plus_zero_global_token"
        return summary


def empirical_gap_weights(regional_logits, labels, aggregation="max"):
    """Return detached (normalized weights, empirical gap) for training labels.

    For E, g = 1 - max_r p_r(y); E_mean uses the mean as a matched control.
    Normalization keeps the minibatch mean weight at one and never uses a
    router prediction, a global-expert prediction, or a validation label.
    """
    if regional_logits.ndim != 3 or tuple(regional_logits.shape[1:]) != (5, 2):
        raise ValueError("Expected regional_logits with shape [B,5,2].")
    if aggregation not in {"max", "mean"}:
        raise ValueError("aggregation must be 'max' or 'mean'.")
    if labels.ndim != 1 or labels.shape[0] != regional_logits.shape[0] or not labels.numel():
        raise ValueError("Expected a nonempty training-label vector with shape [B].")
    with torch.no_grad():
        target_indices = labels.long()[:, None, None].expand(-1, 5, 1)
        true_class_probabilities = regional_logits.detach().softmax(-1).gather(-1, target_indices).squeeze(-1)
        capability = true_class_probabilities.amax(1) if aggregation == "max" else true_class_probabilities.mean(1)
        gap = 1.0 - capability
        unnormalized = 1.0 + gap
        weights = unnormalized / unnormalized.mean()
    return weights, gap


def _per_sample_focal(logits, labels, focal):
    # Same formula and dtype conventions as the baseline MulticlassFocalLoss,
    # with the reduction deferred until after example weights are applied.
    labels = labels.long()
    log_probabilities = F.log_softmax(logits, dim=1)
    log_p_t = log_probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
    p_t = log_p_t.exp()
    alpha_t = torch.where(
        labels == 1,
        torch.as_tensor(focal.positive_alpha, device=logits.device, dtype=logits.dtype),
        torch.as_tensor(1.0 - focal.positive_alpha, device=logits.device, dtype=logits.dtype),
    )
    return -alpha_t * (1.0 - p_t).pow(focal.gamma) * log_p_t


def loss_terms(output, labels, args, focal):
    """Baseline-compatible loss API; E replaces, rather than adds, global aux.

    A-D delegate unchanged to the baseline loss. E/E_mean require the fixed
    focal / 0.05 auxiliary protocol so a caller cannot silently run a different
    experiment under the same arm name.
    """
    arm = getattr(output, "global_role_arm", None)
    if arm not in ARMS:
        raise ValueError("Output must come from GlobalRoleAblation.")
    configured_arm = getattr(args, "arm", arm)
    if configured_arm != arm:
        raise ValueError(f"Loss arm {configured_arm!r} does not match model arm {arm!r}.")
    if getattr(args, "variant", "baseline") != "baseline":
        raise ValueError("Global-role losses require variant='baseline'.")
    if arm not in {"E", "E_mean"}:
        return baseline_loss_terms(output, labels, args, focal)
    if args.task_loss != "focal" or not math.isclose(
        args.auxiliary_task_loss_weight, GAP_AUXILIARY_WEIGHT, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("E/E_mean require focal loss and auxiliary_task_loss_weight=0.05.")

    primary = task_loss(output.logits, labels, args.task_loss, focal)
    zero = primary.new_zeros(())
    weights, gap = empirical_gap_weights(
        output.expert_logits[:, :5], labels, "mean" if arm == "E_mean" else "max"
    )
    auxiliary = (weights * _per_sample_focal(output.expert_logits[:, 5], labels, focal)).mean()
    if output.interaction_view_features is not None:
        interaction, interaction_components = interaction_feature_losses(output.interaction_view_features)
    else:
        interaction, interaction_components = zero, {}
    total = primary + GAP_AUXILIARY_WEIGHT * auxiliary + args.interaction_loss_weight * interaction + zero
    return dict(
        total=total, primary=primary, auxiliary=auxiliary, interaction=interaction,
        extra=zero, routing=zero, regional_mass_penalty=zero,
        auxiliary_components={"redundancy": auxiliary},
        interaction_components=interaction_components,
        empirical_gap=gap, empirical_gap_weights=weights,
    )
