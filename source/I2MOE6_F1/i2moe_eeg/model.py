"""Five frozen distilled regional experts plus a trainable redundancy Transformer.

The router mixes six logits from the full view. Its input uses all six
representations, with the redundancy feature detached as in I2MOE9E.
"""

from __future__ import annotations
import czb_paths as _paths

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .regions import (
    EXPERT_NAMES,
    NUM_EEG_CHANNELS,
    NUM_INTERACTION_EXPERTS,
    NUM_REGIONS,
    REGION_NAMES,
)


FROZEN_EXPERT_ENCODER_NAME = "efficientnet_v2_s"
FROZEN_EXPERT_ENCODER_WEIGHTS = (
    "New01-distilled per-region selected validation-AUPRC best_auprc.pt"
)
FROZEN_EXPERT_FEATURE_POOLING = "global_adaptive_average_pool"
FROZEN_EXPERT_INPUT_LAYOUT = "eeg_[B,19,T]_as_single_channel_image_[B,1,19,T]"
FROZEN_EXPERT_STEM_TRANSFER = "rgb_weight_mean_to_single_channel"
FROZEN_EXPERT_CHECKPOINT_ROOTS: dict[str, str] = {
    "unique_frontal": (
        f"{_paths.STORAGE}/I2MOE7E/I2MOE7E2/"
        "new01_distill/frontal"
    ),
    "unique_left_temporal": (
        f"{_paths.STORAGE}/I2MOE7E/I2MOE7E2/"
        "new01_distill/left_temporal"
    ),
    "unique_right_temporal": (
        f"{_paths.STORAGE}/I2MOE7E/I2MOE7E2/"
        "new01_distill/right_temporal"
    ),
    "unique_centro_parietal": f"{_paths.STORAGE}/I2MOE7_new01_distill",
    "unique_occipital": (
        f"{_paths.STORAGE}/I2MOE7E/I2MOE7E2/"
        "new01_distill/occipital"
    ),
}
if tuple(FROZEN_EXPERT_CHECKPOINT_ROOTS) != EXPERT_NAMES[:NUM_REGIONS]:
    raise RuntimeError("Frozen checkpoint mapping must follow the canonical expert order.")


FROZEN_EXPERT_BEST_SEEDS: dict[str, int] = {
    "unique_frontal": 2,
    "unique_left_temporal": 2,
    "unique_right_temporal": 42,
    "unique_centro_parietal": 0,
    "unique_occipital": 3,
}
FROZEN_EXPERT_BEST_AUPRC: dict[str, float] = {
    "unique_frontal": 0.7055532839435087,
    "unique_left_temporal": 0.7289177632448219,
    "unique_right_temporal": 0.7194483686079627,
    "unique_centro_parietal": 0.7050140668503487,
    "unique_occipital": 0.730433608344525,
}
FROZEN_EXPERT_BEST_EPOCHS: dict[str, int] = {
    "unique_frontal": 18,
    "unique_left_temporal": 18,
    "unique_right_temporal": 20,
    "unique_centro_parietal": 15,
    "unique_occipital": 18,
}
DISTILLATION_TEACHER_TABLE_SHA256 = (
    "f045d2cf6419a8920f88bd74f2775a9bf184026e84042c171cb290c031b84857"
)
DISTILLATION_TEMPERATURE = 4.0
DISTILLATION_HARD_WEIGHT = 1.0
DISTILLATION_KD_WEIGHT = 1.0
DISTILLATION_TEACHER_AGGREGATION = "logit_mean"
for mapping in (
    FROZEN_EXPERT_BEST_SEEDS,
    FROZEN_EXPERT_BEST_AUPRC,
    FROZEN_EXPERT_BEST_EPOCHS,
):
    if tuple(mapping) != EXPERT_NAMES[:NUM_REGIONS]:
        raise RuntimeError("Frozen best-checkpoint metadata has wrong expert order.")


def default_frozen_checkpoint_paths() -> dict[str, str]:
    """Return the fixed user-confirmed distilled source checkpoints."""

    return {
        name: str(
            Path(root)
            / f"seed{FROZEN_EXPERT_BEST_SEEDS[name]}"
            / "best_auprc.pt"
        )
        for name, root in FROZEN_EXPERT_CHECKPOINT_ROOTS.items()
    }


def _replace_first_conv_to_1ch(model: nn.Module) -> nn.Module:
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            replacement = nn.Conv2d(
                1,
                module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
                padding_mode=getattr(module, "padding_mode", "zeros"),
            )
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], replacement)
            return model
    raise RuntimeError("EfficientNetV2-S has no three-channel input convolution.")


def _make_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    dropout: float,
) -> nn.Sequential:
    if min(input_dim, hidden_dim, output_dim, num_layers) <= 0:
        raise ValueError("MLP dimensions and layer count must be positive.")
    if not 0 <= dropout < 1:
        raise ValueError("MLP dropout must be in [0,1).")
    if num_layers == 1:
        return nn.Sequential(nn.Linear(input_dim, output_dim))
    layers: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
    ]
    for _ in range(num_layers - 2):
        layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)))
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class FrozenFullChannelExpert(nn.Module):
    """I2MOE7-compatible full-channel EfficientNet feature and classifier."""

    def __init__(
        self,
        hidden_dim: int = 128,
        classifier_hidden_dim: int = 128,
        classifier_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        backbone = torchvision.models.efficientnet_v2_s(weights=None)
        backbone = _replace_first_conv_to_1ch(backbone)
        feature_dim = int(backbone.classifier[-1].in_features)
        self.encoder = nn.Module()
        self.encoder.features = backbone.features
        self.encoder.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.classifier = _make_mlp(
            hidden_dim,
            classifier_hidden_dim,
            2,
            classifier_layers,
            dropout,
        )
        self.hidden_dim = int(hidden_dim)

    def forward(self, eeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if eeg.ndim != 3 or eeg.shape[1] != NUM_EEG_CHANNELS:
            raise ValueError(f"Expected EEG [B,19,T], got {tuple(eeg.shape)}.")
        image = eeg.float().unsqueeze(1)
        feature_map = self.encoder.features(image)
        pooled = F.adaptive_avg_pool2d(feature_map, (1, 1)).flatten(1)
        features = self.encoder.projection(pooled)
        return features, self.classifier(features)


class FrozenExpertStub(nn.Module):
    """Small injected expert contract used by unit tests."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, eeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.module(eeg)
        if not isinstance(output, tuple) or len(output) != 2:
            raise TypeError("Injected frozen experts must return (features, logits).")
        return output


def _source_state_for_expert(
    payload: Mapping[str, Any], expert_name: str
) -> dict[str, torch.Tensor]:
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"Source checkpoint for {expert_name} has no model state.")
    source_prefix = f"expert_encoders.{expert_name}."
    translated: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith(source_prefix):
            translated["encoder." + key[len(source_prefix) :]] = value
        elif key.startswith("classifier."):
            translated[key] = value
    if not translated:
        raise ValueError(
            f"Source checkpoint does not contain expected prefix {source_prefix!r}."
        )
    return translated


class TransformerInteractionExpert(nn.Module):
    """Fuse five frozen 128-D tokens and expose both representation and logits."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        feedforward_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * feedforward_multiplier,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, NUM_REGIONS, hidden_dim)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.hidden_dim = int(hidden_dim)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim < 3 or tuple(tokens.shape[-2:]) != (
            NUM_REGIONS,
            self.hidden_dim,
        ):
            raise ValueError(
                f"Expected tokens [...,{NUM_REGIONS},{self.hidden_dim}], "
                f"got {tuple(tokens.shape)}."
            )
        leading = tokens.shape[:-2]
        fused = tokens.reshape(-1, NUM_REGIONS, self.hidden_dim)
        fused = fused + self.position_embedding
        for layer in self.layers:
            fused = layer(fused)
        representation = self.norm(fused).mean(dim=1)
        logits = self.head(representation)
        return (
            representation.reshape(*leading, self.hidden_dim),
            logits.reshape(*leading, 2),
        )


class InteractionRouter(nn.Module):
    """Dense router from all six expert representations to six weights."""

    def __init__(
        self,
        hidden_dim: int,
        router_hidden_dim: int,
        router_layers: int,
        router_dropout: float,
        temperature: float,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("Router temperature must be positive.")
        self.network = _make_mlp(
            NUM_INTERACTION_EXPERTS * hidden_dim,
            router_hidden_dim,
            NUM_INTERACTION_EXPERTS,
            router_layers,
            router_dropout,
        )
        self.hidden_dim = int(hidden_dim)
        self.temperature = float(temperature)

    def forward(self, expert_features: torch.Tensor) -> torch.Tensor:
        expected = (NUM_INTERACTION_EXPERTS, self.hidden_dim)
        if expert_features.ndim != 3 or tuple(expert_features.shape[1:]) != expected:
            raise ValueError(
                f"Expected all-expert features [B,{expected[0]},{expected[1]}], "
                f"got {tuple(expert_features.shape)}."
            )
        flattened = expert_features.flatten(start_dim=1)
        return torch.softmax(self.network(flattened) / self.temperature, dim=1)


@dataclass
class I2MoEOutput:
    logits: torch.Tensor
    expert_logits: torch.Tensor
    interaction_weights: torch.Tensor
    frozen_features: torch.Tensor
    router_features: torch.Tensor
    interaction_view_features: Optional[torch.Tensor] = None
    interaction_view_logits: Optional[torch.Tensor] = None


class EEGFrozenRegionalI2MoE(nn.Module):
    """Five frozen regional experts plus one trainable redundancy Transformer."""

    def __init__(
        self,
        checkpoint_paths: Optional[Mapping[str, str]] = None,
        expected_source_seeds: Optional[Mapping[str, int]] = None,
        hidden_dim: int = 128,
        num_heads: int = 4,
        fusion_layers: int = 2,
        feedforward_multiplier: int = 2,
        dropout: float = 0.3,
        router_hidden_dim: int = 128,
        router_layers: int = 2,
        router_dropout: float = 0.5,
        router_temperature: float = 2.0,
        replacement_std: float = 1.0,
        frozen_expert_factory: Optional[Callable[[str], nn.Module]] = None,
    ) -> None:
        super().__init__()
        if replacement_std <= 0:
            raise ValueError("replacement_std must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        frozen: dict[str, nn.Module] = {}
        source_metadata: dict[str, dict[str, Any]] = {}
        if frozen_expert_factory is not None:
            for name in EXPERT_NAMES[:NUM_REGIONS]:
                frozen[name] = FrozenExpertStub(frozen_expert_factory(name))
                source_metadata[name] = {"injected_for_test": True}
        else:
            if checkpoint_paths is None or expected_source_seeds is None:
                raise ValueError(
                    "checkpoint_paths and expected_source_seeds are required for "
                    "formal models."
                )
            if tuple(checkpoint_paths) != EXPERT_NAMES[:NUM_REGIONS]:
                raise ValueError("Frozen checkpoint paths have wrong expert order.")
            if tuple(expected_source_seeds) != EXPERT_NAMES[:NUM_REGIONS]:
                raise ValueError("Expected frozen source seeds have wrong expert order.")
            for name in EXPERT_NAMES[:NUM_REGIONS]:
                path = Path(checkpoint_paths[name])
                if not path.is_file() or path.stat().st_size == 0:
                    raise FileNotFoundError(f"Missing frozen checkpoint: {path}")
                payload = torch.load(path, map_location="cpu", weights_only=False)
                expected_source_seed = int(expected_source_seeds[name])
                if int(payload.get("seed", -1)) != expected_source_seed:
                    raise ValueError(
                        f"{name} source seed={payload.get('seed')} does not match "
                        f"its selected best seed={expected_source_seed}."
                    )
                expected_region = name.removeprefix("unique_")
                expected_model = f"eeg19_single_{expected_region}_effnetv2s_zero_mask"
                config = payload.get("config", {})
                source_region = config.get("region_contract", {}).get("target_region")
                if (
                    payload.get("model_name") != expected_model
                    or config.get("model") != expected_model
                    or source_region != expected_region
                ):
                    raise ValueError(
                        f"{name} checkpoint identity mismatch: "
                        f"model={payload.get('model_name')!r}, region={source_region!r}."
                    )
                if (
                    int(payload.get("epoch", -1)) <= 0
                    or int(payload.get("epoch", -1))
                    != int(payload.get("best_epoch", -2))
                    or int(config.get("hidden_dim", -1)) != hidden_dim
                    or int(config.get("classifier_hidden_dim", -1)) != 128
                    or int(config.get("classifier_layers", -1)) != 2
                ):
                    raise ValueError(f"{name} checkpoint architecture/selection mismatch.")
                if (
                    config.get("distillation_enabled") is not True
                    or not math.isclose(
                        float(config.get("hard_weight", float("nan"))),
                        DISTILLATION_HARD_WEIGHT,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or not math.isclose(
                        float(config.get("kd_weight", float("nan"))),
                        DISTILLATION_KD_WEIGHT,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or not math.isclose(
                        float(config.get("temperature", float("nan"))),
                        DISTILLATION_TEMPERATURE,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or config.get("teacher_aggregation")
                    != DISTILLATION_TEACHER_AGGREGATION
                    or config.get("teacher_table_sha256")
                    != DISTILLATION_TEACHER_TABLE_SHA256
                ):
                    raise ValueError(
                        f"{name} source checkpoint does not satisfy the recorded "
                        "New01 distillation contract."
                    )
                if (
                    int(payload["best_epoch"]) != FROZEN_EXPERT_BEST_EPOCHS[name]
                    or not math.isclose(
                        float(payload["best_auprc"]),
                        FROZEN_EXPERT_BEST_AUPRC[name],
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ):
                    raise ValueError(
                        f"{name} is not the recorded selected distilled "
                        "checkpoint."
                    )
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(0)
                    expert = FrozenFullChannelExpert(hidden_dim=hidden_dim)
                expert.load_state_dict(
                    _source_state_for_expert(payload, name), strict=True
                )
                frozen[name] = expert
                source_metadata[name] = {
                    "path": str(path.resolve()),
                    "source_model": payload.get("model_name"),
                    "seed": int(payload["seed"]),
                    "epoch": int(payload["epoch"]),
                    "best_epoch": int(payload["best_epoch"]),
                    "best_auprc": float(payload["best_auprc"]),
                    "distillation_enabled": True,
                    "distillation_temperature": float(config["temperature"]),
                    "distillation_hard_weight": float(config["hard_weight"]),
                    "distillation_kd_weight": float(config["kd_weight"]),
                    "teacher_aggregation": config["teacher_aggregation"],
                    "teacher_table_sha256": config["teacher_table_sha256"],
                }
        self.frozen_experts = nn.ModuleDict(frozen)
        for parameter in self.frozen_experts.parameters():
            parameter.requires_grad_(False)
        self.frozen_experts.eval()

        self.redundancy_expert = TransformerInteractionExpert(
            hidden_dim,
            num_heads,
            fusion_layers,
            feedforward_multiplier,
            dropout,
        )
        self.router = InteractionRouter(
            hidden_dim,
            router_hidden_dim,
            router_layers,
            router_dropout,
            router_temperature,
        )
        self.source_metadata = source_metadata
        self.hidden_dim = int(hidden_dim)
        self.replacement_std = float(replacement_std)

    @property
    def expert_names(self) -> tuple[str, ...]:
        return EXPERT_NAMES

    def train(self, mode: bool = True) -> "EEGFrozenRegionalI2MoE":
        super().train(mode)
        # Frozen BatchNorm and Dropout must retain source-checkpoint inference
        # semantics even while the redundancy expert is training.
        self.frozen_experts.eval()
        return self

    def _encode_frozen(
        self, eeg: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []
        with torch.no_grad():
            for expert in self.frozen_experts.values():
                current_features, current_logits = expert(eeg)
                features.append(current_features)
                logits.append(current_logits)
        return torch.stack(features, dim=1), torch.stack(logits, dim=1)

    def _feature_views(self, features: torch.Tensor) -> torch.Tensor:
        views = [features]
        for index in range(NUM_REGIONS):
            noise = torch.randn_like(features[:, index : index + 1])
            noise = noise * self.replacement_std
            views.append(
                torch.cat(
                    (features[:, :index], noise, features[:, index + 1 :]),
                    dim=1,
                )
            )
        return torch.stack(views, dim=1)

    def forward(
        self, eeg: torch.Tensor, compute_interactions: Optional[bool] = None
    ) -> I2MoEOutput:
        if compute_interactions is None:
            compute_interactions = self.training
        frozen_features, frozen_logits = self._encode_frozen(eeg)

        if compute_interactions:
            redundancy_views = self._feature_views(frozen_features)
            redundancy_features, redundancy_logits = self.redundancy_expert(
                redundancy_views
            )
            interaction_view_features = redundancy_features.unsqueeze(1)
            interaction_view_logits = redundancy_logits.unsqueeze(1)
            new_logits = interaction_view_logits[:, :, 0, :]
            new_features = interaction_view_features[:, :, 0, :]
        else:
            interaction_view_features = None
            interaction_view_logits = None
            redundancy_features, redundancy_logits = self.redundancy_expert(
                frozen_features
            )
            new_logits = redundancy_logits.unsqueeze(1)
            new_features = redundancy_features.unsqueeze(1)

        # Prevent the routing path from changing the redundancy representation.
        router_features = torch.cat(
            (frozen_features, new_features.detach()), dim=1
        )
        weights = self.router(router_features)
        expert_logits = torch.cat((frozen_logits, new_logits), dim=1)
        logits = torch.sum(expert_logits * weights.unsqueeze(-1), dim=1)
        return I2MoEOutput(
            logits=logits,
            expert_logits=expert_logits,
            interaction_weights=weights,
            frozen_features=frozen_features,
            router_features=router_features,
            interaction_view_features=interaction_view_features,
            interaction_view_logits=interaction_view_logits,
        )

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: value
            for name, value in self.state_dict().items()
            if not name.startswith("frozen_experts.")
        }

    def load_trainable_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        expected = set(self.trainable_state_dict())
        if set(state) != expected:
            raise ValueError(
                "Trainable checkpoint schema mismatch; "
                f"missing={sorted(expected-set(state))[:5]}, "
                f"extra={sorted(set(state)-expected)[:5]}."
            )
        incompatible = self.load_state_dict(dict(state), strict=False)
        unexpected = list(incompatible.unexpected_keys)
        invalid_missing = [
            name
            for name in incompatible.missing_keys
            if not name.startswith("frozen_experts.")
        ]
        if unexpected or invalid_missing:
            raise ValueError(
                f"Failed to load trainable state: missing={invalid_missing}, "
                f"unexpected={unexpected}."
            )

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def architecture_summary(self) -> dict[str, object]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        frozen_by_expert = {
            name: sum(parameter.numel() for parameter in expert.parameters())
            for name, expert in self.frozen_experts.items()
        }
        return {
            "model": "eeg19_distilled_frozen5_redundancy_router6_detach",
            "regions": list(REGION_NAMES),
            "experts": list(EXPERT_NAMES),
            "num_regions": NUM_REGIONS,
            "num_experts": NUM_INTERACTION_EXPERTS,
            "hidden_dim": self.hidden_dim,
            "frozen_expert_encoder": FROZEN_EXPERT_ENCODER_NAME,
            "frozen_expert_source": FROZEN_EXPERT_ENCODER_WEIGHTS,
            "frozen_experts_distilled": True,
            "distillation_teacher_table_sha256": DISTILLATION_TEACHER_TABLE_SHA256,
            "frozen_expert_selection": (
                "user-confirmed fixed per-region distilled checkpoints"
            ),
            "frozen_expert_best_seeds": dict(FROZEN_EXPERT_BEST_SEEDS),
            "frozen_sources_shared_across_training_seeds": True,
            "frozen_expert_input_layout": FROZEN_EXPERT_INPUT_LAYOUT,
            "frozen_expert_feature_pooling": FROZEN_EXPERT_FEATURE_POOLING,
            "frozen_experts_receive_all_19_channels": True,
            "frozen_experts_eval_mode": not self.frozen_experts.training,
            "frozen_experts_fully_frozen": not any(
                parameter.requires_grad
                for parameter in self.frozen_experts.parameters()
            ),
            "frozen_source_checkpoints": self.source_metadata,
            "interaction_experts": ["redundancy"],
            "interaction_architecture": "redundancy_transformer_encoder",
            "interaction_tokens": "five_frozen_128d_regional_expert_features",
            "interaction_loss_space": "128d_transformer_fused_features",
            "corruption": "independent_gaussian_replacement_per_feature_token",
            "replacement_std": self.replacement_std,
            "outer_router": "dense_temperature_softmax",
            "outer_router_input": "all_six_expert_128d_representations",
            "outer_router_input_experts": list(EXPERT_NAMES),
            "outer_router_input_shape": "[B,6,128]",
            "outer_router_input_dim": NUM_INTERACTION_EXPERTS * self.hidden_dim,
            "outer_router_includes_interaction_features": True,
            "interaction_features_detached_before_router": True,
            "top_k_routing": False,
            "parameters_total_with_frozen_sources": total,
            "parameters_trainable": trainable,
            "parameters_frozen_by_expert": frozen_by_expert,
            "checkpoint_contains_frozen_source_weights": False,
        }


# Compatibility names used by the existing trainer/evaluator scaffolding.
EEGRegionI2MoE = EEGFrozenRegionalI2MoE
EEGFullChannelI2MoE = EEGFrozenRegionalI2MoE
