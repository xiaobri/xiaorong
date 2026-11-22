from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap
import torch
from torch import nn
from models import StrictCorrectionMoE
from i2moe_eeg.f1_variants import SixExpertF1, loss_terms
from i2moe_eeg.losses import MulticlassFocalLoss


class TinyFrozen(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(19, 8)
        self.logits = nn.Linear(8, 2)

    def forward(self, eeg):
        features = self.features(eeg.mean(-1))
        return features, self.logits(features)


def make_model(method):
    kwargs = dict(variant='baseline', hidden_dim=8, num_heads=2,
                  fusion_layers=1, dropout=.3, router_hidden_dim=12,
                  router_dropout=.5, frozen_expert_factory=lambda _: TinyFrozen())
    return SixExpertF1(**kwargs) if method == 'baseline' else StrictCorrectionMoE(method, **kwargs)


class StrictTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_initialization_preserves_baseline_weights_and_rng(self):
        for method in ('channel', 'orbit', 'joint'):
            torch.manual_seed(3)
            baseline = make_model('baseline')
            expected_rng = torch.get_rng_state().clone()
            torch.manual_seed(3)
            added = make_model(method)
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
            for key, value in baseline.state_dict().items():
                torch.testing.assert_close(value, added.state_dict()[key], rtol=0, atol=0)

    def test_zero_correction_preserves_train_outputs_and_random_stream(self):
        eeg = torch.randn(2, 19, 2000)
        for method in ('channel', 'orbit', 'joint'):
            torch.manual_seed(42)
            baseline = make_model('baseline').train()
            torch.manual_seed(42)
            added = make_model(method).train()
            torch.manual_seed(99)
            expected = baseline(eeg)
            expected_rng = torch.get_rng_state().clone()
            torch.manual_seed(99)
            observed = added(eeg)
            torch.testing.assert_close(expected.logits, observed.logits, rtol=0, atol=0)
            torch.testing.assert_close(expected.interaction_view_features,
                                       observed.interaction_view_features, rtol=0, atol=0)
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))

    def test_original_loss_updates_new_encoders_but_not_frozen_sources(self):
        args = SimpleNamespace(variant='baseline', task_loss='focal',
                               auxiliary_task_loss_weight=.05, interaction_loss_weight=.005)
        eeg, labels = torch.randn(2, 19, 2000), torch.tensor([0, 1])
        for method in ('channel', 'orbit', 'joint'):
            torch.manual_seed(3)
            model = make_model(method).train()
            optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=5e-4, weight_decay=.01)
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                output = model(eeg)
                losses = loss_terms(output, labels, args, MulticlassFocalLoss(.85, 2))
                self.assertTrue(torch.isfinite(losses['total']))
                losses['total'].backward()
                self.assertTrue(all(p.grad is None for p in model.frozen_experts.parameters()))
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad and parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                if step == 1:
                    for branch in model.corrections.values():
                        total = sum(float(p.grad.abs().sum()) for p in branch.encoder.parameters()
                                    if p.grad is not None)
                        self.assertGreater(total, 0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                before = model(eeg).logits.clone()
                state = {k: v.clone() for k, v in model.trainable_state_dict().items()}
                for p in model.trainable_parameters():
                    p.add_(1)
                model.load_trainable_state_dict(state)
                torch.testing.assert_close(before, model(eeg).logits, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()

