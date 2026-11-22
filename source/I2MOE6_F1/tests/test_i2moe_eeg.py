from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from i2moe_eeg.losses import (  # noqa: E402
    MulticlassFocalLoss,
    auxiliary_interaction_task_loss,
    interaction_feature_losses,
    router_statistics,
)
from i2moe_eeg.data import sampled_indices, sampling_plan  # noqa: E402
from i2moe_eeg.model import (  # noqa: E402
    EEGRegionI2MoE,
    FROZEN_EXPERT_BEST_SEEDS,
    default_frozen_checkpoint_paths,
)
from i2moe_eeg.regions import (  # noqa: E402
    EEG_CHANNEL_NAMES,
    EXPERT_NAMES,
    NUM_EEG_CHANNELS,
    REGION_CHANNELS,
    REGION_NAMES,
    region_metadata,
)


class TinyFrozenExpert(nn.Module):
    def __init__(self, hidden_dim: int, offset: float) -> None:
        super().__init__()
        self.projection = nn.Linear(NUM_EEG_CHANNELS, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)
        with torch.no_grad():
            self.projection.bias.add_(offset)

    def forward(self, eeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = eeg.mean(dim=2)
        features = self.projection(pooled)
        return features, self.classifier(features)


def tiny_model(hidden_dim: int = 8) -> EEGRegionI2MoE:
    offsets = {name: index / 10 for index, name in enumerate(EXPERT_NAMES[:5])}
    return EEGRegionI2MoE(
        hidden_dim=hidden_dim,
        num_heads=2,
        fusion_layers=2,
        feedforward_multiplier=2,
        dropout=0.0,
        router_hidden_dim=12,
        router_layers=2,
        router_dropout=0.0,
        router_temperature=2.0,
        replacement_std=1.0,
        frozen_expert_factory=lambda name: TinyFrozenExpert(
            hidden_dim, offsets[name]
        ),
    )


class RegionContractTests(unittest.TestCase):
    def test_regions_cover_raw_channels_exactly_once(self) -> None:
        flattened = [index for values in REGION_CHANNELS.values() for index in values]
        self.assertEqual(sorted(flattened), list(range(19)))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(tuple(REGION_CHANNELS), REGION_NAMES)
        self.assertEqual(len(EXPERT_NAMES), 6)

    def test_metadata_proves_raw_19_channel_contract(self) -> None:
        metadata = region_metadata()
        self.assertEqual(metadata["selected_raw_channel_count"], 19)
        self.assertEqual(metadata["raw_eeg_channel_names"], list(EEG_CHANNEL_NAMES))
        self.assertFalse(metadata["ecg_or_emg_constructed"])

    def test_default_checkpoints_use_confirmed_distilled_seeds(self) -> None:
        paths = default_frozen_checkpoint_paths()
        self.assertEqual(tuple(paths), EXPERT_NAMES[:5])
        self.assertEqual(
            {
                name: int(Path(path).parent.name.removeprefix("seed"))
                for name, path in paths.items()
            },
            FROZEN_EXPERT_BEST_SEEDS,
        )
        self.assertEqual(
            FROZEN_EXPERT_BEST_SEEDS,
            {
                "unique_frontal": 2,
                "unique_left_temporal": 2,
                "unique_right_temporal": 42,
                "unique_centro_parietal": 0,
                "unique_occipital": 3,
            },
        )


class SamplingContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = np.concatenate(
            (np.zeros(25_000, dtype=np.int64), np.ones(5_000, dtype=np.int64))
        )

    def test_batch128_plan_keeps_all_twenty_thousand_samples(self) -> None:
        plan = sampling_plan(self.labels, 20_000, 30, "match_cam", 128)
        self.assertEqual(plan["samples_before_batch_truncation"], 20_000)
        self.assertEqual(plan["samples_after_batch_truncation"], 20_000)
        self.assertEqual(plan["full_batches_per_epoch"], 156)
        self.assertEqual(plan["final_partial_batch_size"], 32)
        self.assertEqual(plan["steps_per_epoch"], 157)

    def test_batch128_sampler_keeps_final_partial_batch(self) -> None:
        indices, stats = sampled_indices(
            self.labels, 20_000, 30, "match_cam", 128, seed=0, epoch=1
        )
        self.assertEqual(len(indices), 20_000)
        self.assertEqual(stats["samples"], 20_000)
        self.assertEqual(stats["class_0"], 19_355)
        self.assertEqual(stats["class_1"], 645)
        self.assertEqual(stats["full_batches"], 156)
        self.assertEqual(stats["final_partial_batch_size"], 32)
        self.assertEqual(stats["steps"], 157)


class ModelContractTests(unittest.TestCase):
    def test_training_shapes_router_and_independent_corruptions(self) -> None:
        torch.manual_seed(7)
        model = tiny_model()
        model.train()
        output = model(torch.randn(3, 19, 64), compute_interactions=True)
        self.assertEqual(tuple(output.logits.shape), (3, 2))
        self.assertEqual(tuple(output.expert_logits.shape), (3, 6, 2))
        self.assertEqual(tuple(output.frozen_features.shape), (3, 5, 8))
        self.assertEqual(tuple(output.router_features.shape), (3, 6, 8))
        self.assertEqual(tuple(output.interaction_weights.shape), (3, 6))
        self.assertEqual(tuple(output.interaction_view_features.shape), (3, 1, 6, 8))
        self.assertEqual(tuple(output.interaction_view_logits.shape), (3, 1, 6, 2))
        self.assertTrue(
            torch.allclose(output.interaction_weights.sum(dim=1), torch.ones(3))
        )
        self.assertTrue(
            torch.equal(output.router_features[:, :5], output.frozen_features)
        )
        self.assertTrue(
            torch.equal(
                output.router_features[:, 5:],
                output.interaction_view_features[:, :, 0, :],
            )
        )
        self.assertEqual(model.router.network[0].in_features, 6 * 8)
        self.assertFalse(output.router_features.requires_grad)

    def test_router_only_loss_does_not_update_interaction_experts(self) -> None:
        torch.manual_seed(12)
        model = tiny_model().train()
        output = model(torch.randn(4, 19, 48), compute_interactions=True)
        router_only_loss = -torch.log(output.interaction_weights[:, 0]).mean()
        router_only_loss.backward()

        router_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.router.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(router_gradient, 0.0)
        self.assertFalse(hasattr(model, "synergy_expert"))
        self.assertTrue(all(p.grad is None for p in model.redundancy_expert.parameters()))


    def test_router_rejects_five_feature_input(self) -> None:
        model = tiny_model()
        with self.assertRaisesRegex(ValueError, "all-expert features"):
            model.router(torch.randn(2, 5, 8))

    def test_frozen_experts_stay_eval_and_receive_no_gradients(self) -> None:
        torch.manual_seed(8)
        model = tiny_model()
        model.train()
        self.assertFalse(model.frozen_experts.training)
        output = model(torch.randn(4, 19, 48), compute_interactions=True)
        labels = torch.tensor([0, 1, 0, 1])
        focal = MulticlassFocalLoss()
        final = focal(output.logits, labels)
        auxiliary, _ = auxiliary_interaction_task_loss(
            output.expert_logits, labels, "focal", focal
        )
        interaction, _ = interaction_feature_losses(
            output.interaction_view_features
        )
        (final + 0.05 * auxiliary + 0.005 * interaction).backward()
        for parameter in model.frozen_experts.parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
        missing = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        self.assertEqual(missing, [])

    def test_eval_forward_does_not_consume_rng(self) -> None:
        model = tiny_model().eval()
        eeg = torch.randn(2, 19, 32)
        before = torch.get_rng_state().clone()
        with torch.no_grad():
            first = model(eeg, compute_interactions=False)
        after = torch.get_rng_state()
        self.assertTrue(torch.equal(before, after))
        self.assertIsNone(first.interaction_view_features)
        self.assertIsNone(first.interaction_view_logits)

    def test_trainable_checkpoint_excludes_frozen_weights_and_round_trips(self) -> None:
        torch.manual_seed(10)
        source = tiny_model().eval()
        state = source.trainable_state_dict()
        self.assertTrue(state)
        self.assertFalse(any(name.startswith("frozen_experts.") for name in state))
        torch.manual_seed(11)
        target = tiny_model().eval()
        target.load_trainable_state_dict(state)
        for name, value in source.trainable_state_dict().items():
            self.assertTrue(torch.equal(value, target.trainable_state_dict()[name]))

    def test_architecture_summary_is_explicit_about_freezing(self) -> None:
        summary = tiny_model().architecture_summary()
        self.assertEqual(summary["num_experts"], 6)
        self.assertTrue(summary["frozen_experts_fully_frozen"])
        self.assertEqual(
            summary["interaction_loss_space"], "128d_transformer_fused_features"
        )
        self.assertEqual(
            summary["outer_router_input"],
            "all_six_expert_128d_representations",
        )
        self.assertEqual(summary["outer_router_input_dim"], 6 * 8)
        self.assertTrue(summary["outer_router_includes_interaction_features"])
        self.assertTrue(summary["interaction_features_detached_before_router"])
        self.assertTrue(summary["frozen_experts_distilled"])
        self.assertFalse(summary["checkpoint_contains_frozen_source_weights"])


class LossTests(unittest.TestCase):
    def test_feature_role_losses_have_expected_values(self) -> None:
        views = torch.zeros(2, 1, 6, 4)
        views[..., 0] = 1
        views[:, 0, 1, 0] = -1
        total, components = interaction_feature_losses(views)
        self.assertEqual(set(components), {"redundancy"})
        self.assertAlmostEqual(float(total), 2 / 5, places=6)

    def test_auxiliary_loss_supervises_only_redundancy(self) -> None:
        logits = torch.randn(3, 6, 2, requires_grad=True)
        labels = torch.tensor([0, 1, 0])
        total, components = auxiliary_interaction_task_loss(
            logits, labels, "focal", MulticlassFocalLoss()
        )
        total.backward()
        self.assertEqual(set(components), {"redundancy"})
        self.assertEqual(float(logits.grad[:, :5].abs().sum()), 0.0)
        self.assertGreater(float(logits.grad[:, 5:].abs().sum()), 0.0)

    def test_router_statistics_are_finite(self) -> None:
        stats = router_statistics(torch.full((4, 6), 1 / 6))
        self.assertTrue(all(torch.isfinite(value) for value in stats.values()))


if __name__ == "__main__":
    unittest.main()
