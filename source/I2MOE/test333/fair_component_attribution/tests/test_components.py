import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "fair_refreshed_pairs_20261120"))
import bootstrap
from components import configure, ARMS
from protected_models import CorrectiveHead


class FeatureControlTests(unittest.TestCase):
    def test_masks_change_only_the_prespecified_inputs(self):
        torch.manual_seed(42)
        model = nn.Module()
        model.hidden_dim = 8
        model.correction = CorrectiveHead("protected_moments", 8, 6, 0., ()).eval()
        # The actual trainer starts at zero residual; exercise informative weights here.
        nn.init.normal_(model.correction.predictor[-1].weight)
        core = SimpleNamespace(expert_logits=torch.randn(4, 6, 2), logits=torch.randn(4, 2),
                               router_features=torch.randn(4, 6, 8))
        base = {i:torch.randn(4,3,6) for i in range(5)}
        for arm in ARMS:
            fitted = configure(copy.deepcopy(model),arm)
            self.assertEqual(fitted.state_dict().keys(),model.state_dict().keys())
            with torch.no_grad():
                result = fitted.correction(core,{},base)
                if arm in ("scores_only","no_moments","mean_only"):
                    changed = {i:x.clone() for i,x in base.items()}
                    for x in changed.values():
                        x[:,1:] += 5
                    torch.testing.assert_close(result,fitted.correction(core,{},changed))
                    if arm != "mean_only":
                        for x in changed.values():
                            x[:,0] = torch.randn_like(x[:,0])
                        torch.testing.assert_close(result,fitted.correction(core,{},changed))
                    other = copy.copy(core)
                    other.router_features = torch.randn_like(core.router_features)
                    if arm == "scores_only":
                        torch.testing.assert_close(result,fitted.correction(other,{},base))
                    else:
                        self.assertGreater((result-fitted.correction(other,{},base)).abs().max().item(),1e-6)
