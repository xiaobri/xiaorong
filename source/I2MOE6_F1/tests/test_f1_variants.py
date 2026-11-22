import unittest
from types import SimpleNamespace

import numpy as np
import torch

from test_i2moe_eeg import TinyFrozenExpert
from i2moe_eeg.f1_variants import SixExpertF1, RegionalResidualAdapter, loss_terms
from i2moe_eeg.losses import MulticlassFocalLoss
from i2moe_eeg.data import sampled_indices
from i2moe_eeg.metrics import _threshold_metrics
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def make_model(variant="baseline", chunk=0):
    return SixExpertF1(
        variant=variant, frozen_chunk_size=chunk, hidden_dim=8, num_heads=2,
        fusion_layers=1, dropout=0.0, router_hidden_dim=12, router_dropout=0.0,
        frozen_expert_factory=lambda _: TinyFrozenExpert(8, 0.0),
    )


def loss_args(variant):
    return SimpleNamespace(variant=variant, task_loss="focal", auxiliary_task_loss_weight=0.1,
                           interaction_loss_weight=0.005 if variant == "baseline" else 0.0,
                           routing_loss_weight=0.02, regional_mass_loss_weight=0.02,
                           regional_mass_floor=0.55, routing_target_temperature=0.5)


class F1VariantTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def test_adapter_initialization_preserves_source(self):
        adapter = RegionalResidualAdapter(8, 0.0)
        features, logits = torch.randn(4, 8), torch.randn(4, 2)
        adapted, result = adapter(features, logits)
        torch.testing.assert_close(adapted, features, rtol=0, atol=0)
        torch.testing.assert_close(result, logits, rtol=0, atol=0)

    def test_chunked_frozen_inference_matches(self):
        model = make_model().eval()
        eeg = torch.randn(5, 19, 500)
        expected = model(eeg).logits
        model.frozen_chunk_size = 2
        torch.testing.assert_close(model(eeg).logits, expected)

    def test_variants_backpropagate_without_updating_frozen_sources(self):
        for variant in ("baseline", "sampling", "competence", "temporal_mil"):
            with self.subTest(variant=variant):
                model = make_model(variant).train()
                eeg = torch.randn(4, 19, 2000)
                output = model(eeg)
                self.assertEqual(tuple(output.expert_logits.shape), (4, 6, 2))
                torch.testing.assert_close(output.interaction_weights.sum(1), torch.ones(4))
                loss = loss_terms(output, torch.tensor([0, 1, 0, 1]), loss_args(variant), MulticlassFocalLoss())
                self.assertTrue(torch.isfinite(loss["total"]))
                loss["total"].backward()
                self.assertFalse(any(p.grad is not None for p in model.frozen_experts.parameters()))
                self.assertTrue(all(p.grad is not None for p in model.trainable_parameters()))
                if variant == "temporal_mil":
                    self.assertGreater(float(model.redundancy_expert.encoder[0].weight.grad.abs().sum()), 0)

    def test_new_variant_checkpoints_round_trip(self):
        for variant in ("competence", "temporal_mil"):
            model = make_model(variant).eval()
            state = model.trainable_state_dict()
            self.assertFalse(any(k.startswith("frozen_experts.") for k in state))
            before = {k: v.clone() for k, v in state.items()}
            with torch.no_grad():
                for parameter in model.trainable_parameters():
                    parameter.add_(1)
            model.load_trainable_state_dict(before)
            for name, value in model.trainable_state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_sampling_changes_quota_but_preserves_budget(self):
        labels = np.r_[np.zeros(30000, dtype=np.int64), np.ones(4000, dtype=np.int64)]
        indices, stats = sampled_indices(labels, 20000, 10, "match_cam", 128, 3, 1)
        self.assertEqual(len(indices), 20000)
        self.assertEqual(stats["class_1"], 1818)
        self.assertEqual(stats["steps"], 157)

    def test_fast_threshold_metrics_match_reference_including_ties(self):
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 2, 1001)
        probabilities = np.round(rng.random(1001), 2)
        for threshold in (0.0, 0.01, 0.5, 0.8, 0.99, 1.0):
            result = _threshold_metrics(labels, probabilities, threshold)
            predictions = probabilities > threshold
            counts = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
            self.assertEqual([result[k] for k in ("tn", "fp", "fn", "tp")], list(counts))
            self.assertAlmostEqual(result["f1"], f1_score(labels, predictions, zero_division=0))
            self.assertAlmostEqual(result["precision"], precision_score(labels, predictions, zero_division=0))
            self.assertAlmostEqual(result["sensitivity"], recall_score(labels, predictions, zero_division=0))


if __name__ == "__main__":
    unittest.main()
