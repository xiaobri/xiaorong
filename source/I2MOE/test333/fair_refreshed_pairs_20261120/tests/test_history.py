from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap
from refresh_models import HistoryState


class HistoryStateTests(unittest.TestCase):
    def test_positions_track_only_eligible_negatives_through_wrap_and_reload(self):
        state = HistoryState(2, capacity=5)
        state.support[0] = True
        for epoch, offset in ((1,0), (2,8)):
            y = torch.tensor([0,0,1,0,0,1,0,0])
            g = torch.tensor([0,1,0,0,0,0,1,0])
            ids = offset+torch.arange(8)
            x = torch.stack((ids.float(),-ids.float()),1)
            state.remember_positions(y,g,ids,torch.full_like(ids,epoch))
            state.update(x,x[:,0],y,g,x[:,0])
        torch.testing.assert_close(state.row_ids, state.negative_x[:,0].long())
        torch.testing.assert_close(state.epochs, 1+(state.row_ids>=8).long())
        loaded = HistoryState(2, capacity=5)
        loaded.load_state_dict(state.state_dict())
        torch.testing.assert_close(loaded.row_ids, state.row_ids)
        torch.testing.assert_close(loaded.select(torch.tensor(0), torch.tensor(0.)),
                                   state.select(torch.tensor(0), torch.tensor(0.)))
