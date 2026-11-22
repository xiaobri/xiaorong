"""Six-expert ablations; immutable source models retain their audited contract."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .model import EEGFrozenRegionalI2MoE, I2MoEOutput
from .losses import task_loss, auxiliary_interaction_task_loss, interaction_feature_losses


class RegionalResidualAdapter(nn.Module):
    """Port of RICAI/T2MOE A's zero-initialized feature and logit residuals."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, 32)
        self.up = nn.Linear(32, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier_delta = nn.Linear(hidden_dim, 2)
        for layer in (self.up, self.classifier_delta):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features: torch.Tensor, logits: torch.Tensor):
        adapted = features + self.up(self.dropout(F.gelu(self.down(self.norm(features)))))
        return adapted, logits + self.classifier_delta(adapted)


class TemporalMILExpert(nn.Module):
    """Raw 19-channel CNN followed by gated attention over 0.5/1 s instances.

    At the fixed 500 Hz input rate, total convolution stride 50 yields 0.1 s
    tokens. Local mean/max pools of 5/10 tokens produce two instance scales.
    Only the final bag receives a label; no crop-level labels are fabricated.
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(19, 32, 15, stride=5, padding=7),
            nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv1d(32, 64, 9, stride=5, padding=4),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv1d(64, hidden_dim, 5, stride=2, padding=2),
            nn.GroupNorm(8, hidden_dim), nn.GELU(),
        )
        self.instance_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.scale_embedding = nn.Parameter(torch.zeros(2, hidden_dim))
        self.attention_v = nn.Linear(hidden_dim, 64)
        self.attention_u = nn.Linear(hidden_dim, 64)
        self.attention_w = nn.Linear(64, 1, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    def forward(self, eeg: torch.Tensor):
        encoded = self.encoder(eeg)
        instances = []
        for index, (width, stride) in enumerate(((5, 2), (10, 5))):
            windows = encoded.unfold(-1, width, stride)
            pooled = torch.cat((windows.mean(-1), windows.amax(-1)), dim=1).transpose(1, 2)
            instances.append(self.instance_projection(pooled) + self.scale_embedding[index])
        tokens = torch.cat(instances, dim=1)
        scores = self.attention_w(torch.tanh(self.attention_v(tokens)) *
                                  torch.sigmoid(self.attention_u(tokens)))
        attention = scores.softmax(dim=1)
        features = self.norm((tokens * attention).sum(dim=1))
        return features, self.head(features)


class SixExpertF1(EEGFrozenRegionalI2MoE):
    def __init__(self, variant: str = "baseline", frozen_chunk_size: int = 0, **kwargs):
        if variant not in {"baseline", "sampling", "competence", "temporal_mil"}:
            raise ValueError(f"Unknown variant {variant}")
        super().__init__(**kwargs)
        self.variant = variant
        self.frozen_chunk_size = frozen_chunk_size
        if variant == "competence":
            self.adapters = nn.ModuleList([
                RegionalResidualAdapter(self.hidden_dim, kwargs.get("dropout", 0.3))
                for _ in range(5)
            ])
            self.log_scale = nn.Parameter(torch.zeros(6))
            self.margin_bias = nn.Parameter(torch.zeros(6))
        elif variant == "temporal_mil":
            self.redundancy_expert = TemporalMILExpert(self.hidden_dim, kwargs.get("dropout", 0.3))

    @property
    def expert_names(self):
        names = super().expert_names
        return names[:-1] + ("temporal_mil",) if self.variant == "temporal_mil" else names

    def _encode_frozen(self, eeg: torch.Tensor):
        if not self.frozen_chunk_size or eeg.shape[0] <= self.frozen_chunk_size:
            return super()._encode_frozen(eeg)
        # Sources use eval BatchNorm/Dropout and no gradients. The trainable
        # heads still receive the entire B128 batch in one optimizer update.
        chunks = [super(SixExpertF1, self)._encode_frozen(part)
                  for part in eeg.split(self.frozen_chunk_size)]
        return tuple(torch.cat([part[i] for part in chunks], dim=0) for i in (0, 1))

    def forward(self, eeg: torch.Tensor, compute_interactions: Optional[bool] = None):
        if self.variant in {"baseline", "sampling"}:
            return super().forward(eeg, compute_interactions)
        features, source_logits = self._encode_frozen(eeg)
        source_features = features
        if self.variant == "competence":
            adapted = [adapter(features[:, i], source_logits[:, i])
                       for i, adapter in enumerate(self.adapters)]
            features = torch.stack([pair[0] for pair in adapted], dim=1)
            regional_logits = torch.stack([pair[1] for pair in adapted], dim=1)
            new_features, new_logits = self.redundancy_expert(features)
        else:
            regional_logits = source_logits
            new_features, new_logits = self.redundancy_expert(eeg)
        router_features = torch.cat((features, new_features.detach().unsqueeze(1)), dim=1)
        weights = self.router(router_features)
        expert_logits = torch.cat((regional_logits, new_logits.unsqueeze(1)), dim=1)
        if self.variant == "competence":
            margin = (expert_logits[..., 1] - expert_logits[..., 0]) * self.log_scale.exp() + self.margin_bias
            expert_logits = torch.stack((-0.5 * margin, 0.5 * margin), dim=-1)
        output = I2MoEOutput(
            logits=(expert_logits * weights.unsqueeze(-1)).sum(1),
            expert_logits=expert_logits, interaction_weights=weights,
            frozen_features=source_features, router_features=router_features,
            interaction_view_features=None, interaction_view_logits=None,
        )
        output.source_logits = source_logits
        return output

    def architecture_summary(self):
        summary = super().architecture_summary()
        summary.update(model="six_expert_f1_screen", variant=self.variant,
                       frozen_forward_chunk_size=self.frozen_chunk_size)
        if self.variant in {"competence", "temporal_mil"}:
            summary.update(corruption=None, interaction_loss_space=None)
        if self.variant == "competence":
            summary.update(regional_adapters="zero-initialized residuals", margin_calibration=True,
                           competence_target="detached source five-expert CE", regional_mass_floor=0.55)
        if self.variant == "temporal_mil":
            names = list(self.expert_names)
            names[-1] = "temporal_mil"
            summary.update(experts=names, interaction_experts=["temporal_mil"],
                           interaction_architecture="raw_cnn_gated_attention_mil",
                           interaction_tokens="0.5s and 1s pooled raw waveform features",
                           instance_labels=False, encoder_stride_samples=50)
        return summary


def loss_terms(output, labels, args, focal):
    primary = task_loss(output.logits, labels, args.task_loss, focal)
    zero = primary.new_zeros(())
    if args.variant == "competence":
        per_expert = [task_loss(output.expert_logits[:, i], labels, args.task_loss, focal)
                      for i in range(6)]
        auxiliary = torch.stack(per_expert).mean()
        expanded = labels[:, None].expand(-1, 5).reshape(-1)
        source_ce = F.cross_entropy(output.source_logits.reshape(-1, 2), expanded,
                                    reduction="none").reshape(-1, 5)
        target = torch.softmax(-source_ce.detach() / args.routing_target_temperature, dim=1)
        regional = output.interaction_weights[:, :5]
        mass = regional.sum(1)
        normalized = regional / mass.unsqueeze(1).clamp_min(1e-8)
        routing = F.kl_div(normalized.clamp_min(1e-8).log(), target, reduction="batchmean")
        mass_penalty = F.relu(args.regional_mass_floor - mass).mean()
        extra = args.routing_loss_weight * routing + args.regional_mass_loss_weight * mass_penalty
        auxiliary_components = {f"expert_{i}": value for i, value in enumerate(per_expert)}
    else:
        auxiliary, auxiliary_components = auxiliary_interaction_task_loss(
            output.expert_logits, labels, args.task_loss, focal)
        if args.variant == "temporal_mil":
            auxiliary_components = {"temporal_mil": auxiliary}
        extra, routing, mass_penalty = zero, zero, zero
    if output.interaction_view_features is not None:
        interaction, interaction_components = interaction_feature_losses(output.interaction_view_features)
    else:
        interaction, interaction_components = zero, {}
    total = primary + args.auxiliary_task_loss_weight * auxiliary + args.interaction_loss_weight * interaction + extra
    return dict(total=total, primary=primary, auxiliary=auxiliary, interaction=interaction,
                extra=extra, routing=routing, regional_mass_penalty=mass_penalty,
                auxiliary_components=auxiliary_components, interaction_components=interaction_components)
