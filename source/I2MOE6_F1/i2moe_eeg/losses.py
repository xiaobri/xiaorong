"""Task, auxiliary-task and feature-space interaction losses for I2MOE9E."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .regions import NUM_INTERACTION_EXPERTS, NUM_REGIONS


class MulticlassFocalLoss(nn.Module):
    def __init__(self, positive_alpha: float = 0.85, gamma: float = 2.0) -> None:
        super().__init__()
        if not 0 <= positive_alpha <= 1:
            raise ValueError("positive_alpha must be in [0,1].")
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        self.positive_alpha = float(positive_alpha)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError(f"Expected two-class logits [B,2], got {tuple(logits.shape)}")
        targets = targets.long()
        log_probabilities = F.log_softmax(logits, dim=1)
        log_p_t = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = log_p_t.exp()
        alpha_t = torch.where(
            targets == 1,
            torch.as_tensor(self.positive_alpha, device=logits.device, dtype=logits.dtype),
            torch.as_tensor(1.0 - self.positive_alpha, device=logits.device, dtype=logits.dtype),
        )
        return (-alpha_t * (1.0 - p_t).pow(self.gamma) * log_p_t).mean()


def task_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    kind: str,
    focal: MulticlassFocalLoss,
) -> torch.Tensor:
    if kind == "focal":
        return focal(logits, targets)
    if kind == "cross_entropy":
        return F.cross_entropy(logits, targets.long())
    raise ValueError(f"Unknown task loss: {kind!r}")


def auxiliary_interaction_task_loss(
    expert_logits: torch.Tensor,
    targets: torch.Tensor,
    kind: str,
    focal: MulticlassFocalLoss,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """Supervise the redundancy expert; frozen experts are excluded."""

    if expert_logits.ndim != 3 or tuple(expert_logits.shape[1:]) != (
        NUM_INTERACTION_EXPERTS,
        2,
    ):
        raise ValueError(
            f"Expected expert logits [B,{NUM_INTERACTION_EXPERTS},2], "
            f"got {tuple(expert_logits.shape)}."
        )
    redundancy = task_loss(expert_logits[:, NUM_REGIONS], targets, kind, focal)
    return redundancy, {"redundancy": redundancy}


def interaction_feature_losses(
    view_features: torch.Tensor,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """Average cosine distance between the intact and five perturbed features."""
    if view_features.ndim != 4 or tuple(view_features.shape[1:3]) != (1, NUM_REGIONS + 1):
        raise ValueError(f"Expected [B,1,{NUM_REGIONS + 1},D], got {tuple(view_features.shape)}.")
    views = view_features[:, 0]
    redundancy = (1 - F.cosine_similarity(views[:, :1], views[:, 1:], dim=-1, eps=1e-8)).mean()
    return redundancy, {"redundancy": redundancy}


def router_statistics(weights: torch.Tensor) -> dict[str, torch.Tensor]:
    if weights.ndim != 2 or weights.shape[1] != NUM_INTERACTION_EXPERTS:
        raise ValueError(
            f"Expected router weights [B,{NUM_INTERACTION_EXPERTS}], got {tuple(weights.shape)}"
        )
    safe = weights.clamp_min(torch.finfo(weights.dtype).tiny)
    entropy = -(safe * safe.log()).sum(dim=1).mean()
    return {
        "entropy": entropy,
        "maximum_weight": weights.max(dim=1).values.mean(),
        "minimum_weight": weights.min(dim=1).values.mean(),
    }
