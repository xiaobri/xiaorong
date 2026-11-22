"""Check the pair implementation and the readout on a native raw EEG batch."""
import argparse
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from study import OUT, Observations, RawLinear, corrected, fit, load, new_head, original, state_hash
from cohort_execution import make_metadata


def main():
    seed=3
    ck = load(fit.OUT/'formal/pairs/seed3/best_f1.pt')
    flags = argparse.Namespace(**ck['args'])
    frame = pd.read_csv(flags.csv_path,usecols=['file_path','mode','target'])
    before,_ = make_metadata(frame)
    shifted = frame.copy()
    v = shifted['mode'].eq('Val')
    shifted.loc[v,'target'] = 1-shifted.loc[v,'target']
    shifted.loc[v,'file_path'] = '/ignored/unknown/subject/window.npy'
    after,_ = make_metadata(shifted)
    assert before == after
    data = original.EEG19NpyDataset(frame,'Val',flags.stored_prefix,flags.data_root,
                               target_samples=flags.target_samples,eeg_scale_divisor=flags.eeg_scale_divisor)
    indices = np.r_[np.flatnonzero(data.labels==0)[:64],np.flatnonzero(data.labels==1)[:64]]
    eeg = torch.stack([data[int(i)][0] for i in indices]).cuda()
    with patch.object(original,'SixExpertF1',lambda **kw:fit.build('pairs',**kw)):
        pair = original.build_model(flags)
    pair.load_trainable_state_dict(ck['model'])
    pair.cuda().eval()
    core = original.build_model(flags)
    core.load_trainable_state_dict({k:t for k,t in ck['model'].items() if not k.startswith('correction.')})
    model = RawLinear(core,new_head('cpu')).cuda().eval()
    h = state_hash(model.core.trainable_state_dict()) + state_hash(model.head.state_dict())
    with torch.inference_mode():
        p = pair(eeg,False)
        out = model.core(eeg,False)
        x = model.features(eeg,out)
        torch.testing.assert_close(x,p.analytic_features.detach().clamp(-32,32),rtol=0,atol=0)
        torch.testing.assert_close(model(eeg).logits,p.core_output.logits,rtol=0,atol=0)
        packet = torch.zeros(len(eeg),5,4,1280,device=eeg.device)
        for i in range(5):
            packet[:,i,3,256:] = model.features.block_capture[i][0].flatten(1)
        for position in (130,240,241):
            packet[:,0,3,position] = torch.arange(len(packet),device=eeg.device)
        torch.testing.assert_close(model.features(packet,out),x,rtol=0,atol=0)
        assert not x.requires_grad and x.shape == (128,195)
    assert h == state_hash(model.core.trainable_state_dict()) + state_hash(model.head.state_dict())
    result = dict(seed=seed, windows=128, positive=64, features=195, pair_feature_max_abs=0.,
                  zero_head_logit_max_abs=0., metadata_feature_max_abs=0.,
                  train_support_unchanged_after_val_mutation=True,eval_state_unchanged=True)
    original.atomic_write_json(OUT/'probe.json',result)
    print(result)


if __name__ == '__main__':
    main()
