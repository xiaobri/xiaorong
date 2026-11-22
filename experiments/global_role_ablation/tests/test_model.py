"""CPU contract checks; tiny injected experts require no data/checkpoints."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.global_role_ablation.model import (
    ARMS, GlobalRoleAblation, empirical_gap_weights, loss_terms,
)
from i2moe_eeg.f1_variants import SixExpertF1, loss_terms as baseline_loss_terms
from i2moe_eeg.losses import MulticlassFocalLoss


class TinyFrozenExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(19, 8)
        self.classifier = nn.Linear(8, 2)

    def forward(self, eeg):
        features = self.projection(eeg.mean(dim=2))
        return features, self.classifier(features)


def make_model(arm="D", baseline=False):
    kwargs = dict(
        hidden_dim=8, num_heads=2, fusion_layers=1, dropout=0.2,
        router_hidden_dim=12, router_dropout=0.3, router_temperature=2.0,
        frozen_expert_factory=lambda _: TinyFrozenExpert(),
    )
    return SixExpertF1(variant="baseline", **kwargs) if baseline else GlobalRoleAblation(arm=arm, **kwargs)


def loss_args(arm="D"):
    return SimpleNamespace(
        arm=arm, variant="baseline", task_loss="focal",
        auxiliary_task_loss_weight=0.05, interaction_loss_weight=0.005,
    )


class GlobalRoleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.eeg = torch.randn(4, 19, 40)
        self.labels = torch.tensor([0, 1, 0, 1])
        self.focal = MulticlassFocalLoss()

    def test_all_arms_preserve_initialization_parameters_and_rng_budget(self):
        torch.manual_seed(23)
        baseline = make_model(baseline=True)
        expected_rng = torch.random.get_rng_state()
        state = baseline.state_dict()
        for arm in ARMS:
            with self.subTest(arm=arm):
                torch.manual_seed(23)
                model = make_model(arm)
                self.assertTrue(torch.equal(expected_rng, torch.random.get_rng_state()))
                self.assertEqual(list(state), list(model.state_dict()))
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, state[key], rtol=0, atol=0)
                for training in (True, False):
                    baseline.train(training)
                    model.train(training)
                    torch.manual_seed(31)
                    reference = baseline(self.eeg)
                    forward_rng = torch.random.get_rng_state()
                    torch.manual_seed(31)
                    result = model(self.eeg)
                    self.assertTrue(torch.equal(forward_rng, torch.random.get_rng_state()))
                    torch.testing.assert_close(result.expert_logits, reference.expert_logits, rtol=0, atol=0)
                    if training:
                        torch.testing.assert_close(result.interaction_view_features, reference.interaction_view_features, rtol=0, atol=0)

    def test_d_is_bitwise_baseline_for_forward_loss_and_gradients(self):
        torch.manual_seed(23)
        baseline = make_model(baseline=True)
        torch.manual_seed(23)
        model = make_model("D")
        for training in (True, False):
            for interactions in (None, True, False):
                with self.subTest(training=training, interactions=interactions):
                    baseline.train(training).zero_grad(set_to_none=True)
                    model.train(training).zero_grad(set_to_none=True)
                    torch.manual_seed(31)
                    expected = baseline(self.eeg, compute_interactions=interactions)
                    expected_rng = torch.random.get_rng_state()
                    torch.manual_seed(31)
                    actual = model(self.eeg, compute_interactions=interactions)
                    self.assertTrue(torch.equal(expected_rng, torch.random.get_rng_state()))
                    for name in ("logits", "expert_logits", "interaction_weights", "frozen_features", "router_features", "interaction_view_features", "interaction_view_logits"):
                        wanted, got = getattr(expected, name), getattr(actual, name)
                        if wanted is None:
                            self.assertIsNone(got)
                        else:
                            torch.testing.assert_close(got, wanted, rtol=0, atol=0)
                    expected_loss = baseline_loss_terms(expected, self.labels, loss_args(), self.focal)
                    actual_loss = loss_terms(actual, self.labels, loss_args(), self.focal)
                    for name in ("total", "primary", "auxiliary", "interaction"):
                        torch.testing.assert_close(actual_loss[name], expected_loss[name], rtol=0, atol=0)
                    expected_loss["total"].backward()
                    actual_loss["total"].backward()
                    for (name, a), (_, b) in zip(model.named_parameters(), baseline.named_parameters()):
                        if b.grad is None:
                            self.assertIsNone(a.grad, name)
                        else:
                            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def _intervene_global(self, model, feature_delta=0.0, logit_delta=0.0):
        def hook(_module, _inputs, result):
            features, logits = result
            return features + feature_delta, logits + logits.new_tensor([0.0, logit_delta])
        return model.redundancy_expert.register_forward_hook(hook)

    def test_disabled_paths_are_invariant_to_global_interventions(self):
        for arm in ("A", "B", "C"):
            with self.subTest(arm=arm):
                model = make_model(arm).eval()
                before = model(self.eeg)
                if arm in {"A", "C"}:
                    hook = self._intervene_global(model, feature_delta=100.0)
                    after = model(self.eeg)
                    hook.remove()
                    torch.testing.assert_close(after.interaction_weights, before.interaction_weights, rtol=0, atol=0)
                    self.assertTrue(torch.equal(after.router_features[:, 5], torch.zeros_like(after.router_features[:, 5])))
                if arm in {"A", "B"}:
                    hook = self._intervene_global(model, logit_delta=100.0)
                    after = model(self.eeg)
                    hook.remove()
                    torch.testing.assert_close(after.logits, before.logits, rtol=0, atol=0)
                    torch.testing.assert_close(after.interaction_weights, before.interaction_weights, rtol=0, atol=0)

    def test_enabled_paths_respond_to_separate_global_interventions(self):
        for arm in ("B", "D"):
            model = make_model(arm).eval()
            # Make global-token dependence deterministic instead of relying
            # on randomly initialized router connections being large enough.
            model.router.network = nn.Linear(6 * 8, 6, bias=False)
            with torch.no_grad():
                model.router.network.weight.zero_()
                model.router.network.weight[0, 5 * 8] = 1.0
            before = model(self.eeg)
            hook = self._intervene_global(model, feature_delta=5.0)
            after = model(self.eeg)
            hook.remove()
            self.assertFalse(torch.equal(after.interaction_weights, before.interaction_weights))
        for arm in ("C", "D"):
            model = make_model(arm).eval()
            before = model(self.eeg)
            hook = self._intervene_global(model, logit_delta=100.0)
            after = model(self.eeg)
            hook.remove()
            self.assertFalse(torch.equal(after.logits, before.logits))
            torch.testing.assert_close(after.interaction_weights, before.interaction_weights, rtol=0, atol=0)

    def test_gate_is_masked_before_softmax_with_extreme_sixth_score(self):
        for arm in ("A", "B"):
            model = make_model(arm).eval()
            def enormous_sixth_score(_module, _inputs, scores):
                return torch.cat((scores[:, :5], torch.full_like(scores[:, 5:], 1e30)), dim=1)
            hook = model.router.network.register_forward_hook(enormous_sixth_score)
            output = model(self.eeg)
            hook.remove()
            self.assertTrue(torch.isfinite(output.logits).all())
            self.assertTrue(torch.isfinite(output.interaction_weights).all())
            self.assertTrue(torch.equal(output.interaction_weights[:, 5], torch.zeros(4)))
            torch.testing.assert_close(output.interaction_weights[:, :5].sum(1), torch.ones(4))
            output.logits.square().mean().backward()
            self.assertFalse(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.redundancy_expert.parameters()))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_b_auxiliary_can_train_global_without_router_or_frozen_gradients(self):
        model = make_model("B").train()
        output = model(self.eeg)
        auxiliary = loss_terms(output, self.labels, loss_args("B"), self.focal)["auxiliary"]
        auxiliary.backward()
        self.assertFalse(any(p.grad is not None for p in model.router.parameters()))
        self.assertFalse(any(p.grad is not None for p in model.frozen_experts.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.redundancy_expert.parameters()))
        parameter = next(p for p in model.redundancy_expert.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        before = parameter.detach().clone()
        torch.optim.SGD(model.trainable_parameters(), lr=0.01).step()
        self.assertFalse(torch.equal(before, parameter))

    def test_empirical_gap_uses_correct_class_best_expert_and_detaches(self):
        logits = torch.tensor([
            [[4., -4.], [1., -1.], [0., 0.], [0., 0.], [0., 0.]],
            [[4., -4.], [1., -1.], [0., 0.], [0., 0.], [0., 0.]],
        ], requires_grad=True)
        labels = torch.tensor([0, 1])
        weights, gap = empirical_gap_weights(logits, labels)
        probabilities = logits.detach().softmax(-1)
        expected = 1 - torch.stack((probabilities[0, :, 0].max(), probabilities[1, :, 1].max()))
        torch.testing.assert_close(gap, expected)
        torch.testing.assert_close(weights, (1 + expected) / (1 + expected).mean())
        self.assertFalse(weights.requires_grad)
        self.assertFalse(gap.requires_grad)
        self.assertGreater(float(weights[1]), float(weights[0]))
        self.assertAlmostEqual(float(weights.mean()), 1.0)
        _, mean_gap = empirical_gap_weights(logits, labels, "mean")
        self.assertFalse(torch.equal(mean_gap, gap))

    def test_e_replaces_auxiliary_and_preserves_primary_consistency(self):
        for arm in ("E", "E_mean"):
            with self.subTest(arm=arm):
                model = make_model(arm).train()
                output = model(self.eeg)
                actual = loss_terms(output, self.labels, loss_args(arm), self.focal)
                original = baseline_loss_terms(output, self.labels, loss_args(arm), self.focal)
                for key in ("primary", "interaction"):
                    torch.testing.assert_close(actual[key], original[key], rtol=0, atol=0)
                per_example = torch.stack([self.focal(output.expert_logits[i:i+1, 5], self.labels[i:i+1]) for i in range(4)])
                wanted_aux = (actual["empirical_gap_weights"] * per_example).mean()
                torch.testing.assert_close(actual["auxiliary"], wanted_aux, rtol=0, atol=0)
                wanted_total = actual["primary"] + 0.05 * wanted_aux + 0.005 * actual["interaction"]
                torch.testing.assert_close(actual["total"], wanted_total, rtol=0, atol=0)
                actual["auxiliary"].backward()
                self.assertFalse(any(p.grad is not None for p in model.router.parameters()))
                self.assertFalse(any(p.grad is not None for p in model.frozen_experts.parameters()))
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.redundancy_expert.parameters()))

    def test_invalid_arm_and_loss_protocol_fail_explicitly(self):
        with self.assertRaises(ValueError):
            make_model("unknown")
        with self.assertRaises(ValueError):
            GlobalRoleAblation(variant="competence")
        model = make_model("E").eval()
        output = model(self.eeg)
        args = loss_args("E")
        args.auxiliary_task_loss_weight = 0.1
        with self.assertRaises(ValueError):
            loss_terms(output, self.labels, args, self.focal)
        with self.assertRaises(ValueError):
            loss_terms(output, self.labels, loss_args("D"), self.focal)


if __name__ == "__main__":
    unittest.main()
