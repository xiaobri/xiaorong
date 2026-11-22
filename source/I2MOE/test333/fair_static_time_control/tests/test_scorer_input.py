import copy
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from time_control import configure, parameter_digest


class ScorerInputTests(unittest.TestCase):
    def test_fixed_scores_depend_on_position_but_are_identical_across_windows(self):
        torch.manual_seed(3)
        model = nn.Module()
        model.correction = nn.Module()
        model.correction.allocation = nn.Module()
        model.correction.allocation.scorers = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(131), nn.Linear(131, 32), nn.GELU(), nn.Linear(32, 1))
             for _ in range(5)])
        dynamic = configure(model, "dynamic")
        fixed = configure(copy.deepcopy(model), "fixed")
        self.assertEqual(parameter_digest(dynamic), parameter_digest(fixed))
        self.assertEqual(list(dynamic.state_dict()), list(fixed.state_dict()))
        x = torch.randn(4, 8, 131)
        for a, b in zip(dynamic.correction.allocation.scorers, fixed.correction.allocation.scorers):
            self.assertGreater(a(x).std(dim=0).max().item(), 0.001)
            static = b(x)
            torch.testing.assert_close(static, static[:1].expand_as(static), rtol=0, atol=0)
            self.assertGreater(static.std(dim=1).max().item(), 0.001)
            static[:, 0].sum().backward()
            self.assertGreater(b[1].weight.grad.abs().sum().item(), 0)
