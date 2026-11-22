from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
from pair_models import PairState


class PairStateTests(unittest.TestCase):
    def test_no_use_of_current_batch_negatives_and_exact_domain_matching(self):
        state=PairState(3,within=True,nearest=True,capacity=8)
        state.support[:2]=True
        x=torch.arange(24).reshape(8,3).float()/20
        y=torch.tensor([1,0,0,0,0,0,1,0])
        g=torch.tensor([0,0,0,1,1,1,2,2])
        state.update(x,x[:,0],y,g,x[:,0])
        self.assertEqual(state.pair_count.item(),0)
        state.update(x,x[:,0],y,g,x[:,0])
        self.assertEqual(state.paired_positives.item(),1)
        self.assertEqual(state.pair_count.item(),2)
        self.assertEqual(state.eligible_counts.tolist(),[10,2])
        self.assertEqual(state.counts.tolist(),[12,4])

    def test_checkpoint_and_inference_ignore_negative_bank(self):
        torch.manual_seed(42)
        for within in (False,True):
            for nearest in (False,True):
                state=PairState(4,within,nearest,capacity=24)
                state.support[:2]=True
                x=torch.randn(129,4)
                y=(x[:,1]>.35).long()
                g=torch.arange(len(y))%2
                for part in torch.arange(len(y)).split(16):
                    state.update(x[part],x[part,0],y[part],g[part],x[part,2])
                state.eval()
                pred=state(x,x[:,0])
                self.assertTrue(state.ready.item())
                self.assertEqual(state.updates.item(),state.refresh_step.item())
                self.assertGreater(pred.std().item(),.01)
                second=PairState(4,within,nearest,capacity=24).eval()
                second.load_state_dict(state.state_dict())
                second.negative_x.fill_(1e3)
                second.negative_group.fill_(-1)
                torch.testing.assert_close(pred,second(x,x[:,0]),atol=0,rtol=0)
                torch.testing.assert_close(pred,torch.cat([second(v,v[:,0]) for v in x.split(9)]))
