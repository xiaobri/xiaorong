"""CPU regressions for isolation, checkpoint continuity and threshold semantics."""
import copy
from fractions import Fraction
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from metrics import scan
from study import Learner, corrected, original


class ControlTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.x = torch.randn(8,195,requires_grad=True)
        self.logits = torch.randn(8,2,requires_grad=True)
        self.y = torch.arange(8) % 2
        self.mask = torch.tensor([True,False]*4)
        self.focal = original.MulticlassFocalLoss(.85,2.)

    def test_initialization_and_both_gradient_paths(self):
        rng = torch.get_rng_state().clone()
        learner = Learner('linear_focal_all',.0005,'cpu')
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in learner.head.parameters()),196)
        raw = corrected(self.logits,self.x,learner.head)
        torch.testing.assert_close(raw,self.logits, rtol=0, atol=0)
        experts = torch.randn(8,6,2,requires_grad=True)
        out = SimpleNamespace(logits=self.logits, expert_logits=experts,interaction_view_features=None)
        args = SimpleNamespace(variant='baseline', task_loss='focal', auxiliary_task_loss_weight=.05,
                               interaction_loss_weight=.005)
        original.loss_terms(out,self.y,args,self.focal)['total'].backward()
        self.assertTrue(all(p.grad is None for p in learner.head.parameters()))
        self.logits.grad = None
        learner.step(self.logits,self.x,self.y,self.mask,self.focal)
        self.assertIsNone(self.x.grad)
        self.assertIsNone(self.logits.grad)
        self.assertGreater(learner.head.weight.grad.abs().max().item(),0.)
        self.assertGreater(learner.head.weight.norm().item(),0.)

    def test_empty_mask_and_resume_preserve_moments(self):
        for arm in ('linear_focal_eligible','linear_bce_eligible'):
            a,b = [Learner(arm,.002,'cpu') for _ in range(2)]
            a.step(self.logits,self.x,self.y,self.mask,self.focal)
            saved = copy.deepcopy(a.save())
            self.assertIsNone(a.step(self.logits,self.x,self.y,~torch.ones_like(self.mask),self.focal))
            b.restore(saved)
            for learner in (a,b):
                learner.step(self.logits,self.x,self.y,~self.mask,self.focal)
            torch.testing.assert_close(a.head.state_dict(),b.head.state_dict(),rtol=0,atol=0)
            torch.testing.assert_close(a.opt.state_dict()['state'],b.opt.state_dict()['state'],rtol=0,atol=0)
            self.assertEqual([g['weight_decay'] for g in b.opt.param_groups],[.01,0.])

    def test_grouped_and_grid_scans(self):
        y = np.array([0,1,0,1,1,0])
        grid = original.threshold_grid(SimpleNamespace(threshold_start=.01,threshold_step=.01,threshold_end=.99))
        for p in [np.ones(6)*.4,np.array([.1,.1,.7,.8,.9,.9]),np.array([.1,.55,.2,.65,.29,.8])]:
            candidates = original.binary_metrics(y,p,grid)['best_f1_on_validation_grid']
            got = scan(y,p,grid)
            self.assertEqual((got['tp'],got['fp'],got['threshold']),
                             (candidates['tp'],candidates['fp'],candidates['threshold']))
            ts = [*sorted(set(p),reverse=True),np.nextafter(min(p),-np.inf)]
            def score(t):
                pred = p > t
                tp = int(y[pred].sum())
                return Fraction(2*tp,int(pred.sum()+y.sum()))
            self.assertEqual(scan(y,p)['threshold'],max(ts,key=score))


if __name__ == '__main__':
    unittest.main()
