"""Lock each selection once, then replay source EEG at its own operating point."""
import argparse
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from single import HERE, OUT, SEEDS, build
from queue_fits import jobs
from fit import reference_dir
from additional import digest, original, read
from i2moe_eeg.utils import atomic_write_json, set_seed


def run(study, seed, shard, shards):
    sys.path.insert(0, str(study / 'tools'))
    import external_eval
    assert 0 <= shard < shards and torch.cuda.device_count() == 1
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(12*1024**3 / torch.cuda.get_device_properties(0).total_memory)
    models, locks, cached, frozen = {}, {}, {}, {}
    for arm, current in jobs():
        if current != seed:
            continue
        folder = OUT / 'formal' / arm / f'seed{seed}'
        path = folder / 'best_f1.pt'
        complete = read(folder/'completed.json')
        assert complete['epochs'] == 20 and complete['seed'] == seed
        ck = torch.load(path, map_location='cpu', weights_only=False)
        config = ck['config']
        assert config['single_component']['code'] == {f:digest(HERE/f) for f in ('single.py','fit.py')}
        assert config['single_component']['arm'] == arm and ck['seed'] == config['seed'] == seed
        records = [json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()]
        assert [r['epoch'] for r in records] == list(range(1,21))
        epoch = max(records,key=lambda r:(r['validation']['best_f1_on_validation_grid']['f1'],r['epoch']))['epoch']
        metric = ck['validation']['best_f1_on_validation_grid']
        assert ck['epoch'] == epoch and metric == records[epoch-1]['validation']['best_f1_on_validation_grid']
        for ep in range(1,21):
            name = f'sampling_epoch{ep:02d}.json'
            assert read(folder/name)['actual_batch_order_sha256'] == read(reference_dir(seed)/name)['actual_batch_order_sha256']
        flags = argparse.Namespace(**ck['args'])
        for name,p in original.frozen_checkpoint_paths(flags).items():
            if p not in frozen:
                frozen[p] = digest(Path(p))
            assert frozen[p] == config['frozen_checkpoint_identity'][name]['sha256']
        set_seed(seed, flags.deterministic)
        with patch.object(original,'SixExpertF1',lambda **kw:build(arm,**kw)):
            model = original.build_model(flags)
        model.load_trainable_state_dict(ck['model'])
        models[arm] = model.cuda().eval()
        locks[arm] = dict(epoch=epoch, threshold=metric['threshold'], cached_f1=metric['f1'], sha256=digest(path))
        with np.load(folder/f'predictions_epoch{epoch:02d}.npz') as p:
            cached[arm] = p['labels'], p['probabilities']
    assert len(frozen) == 5 and digest(Path(flags.csv_path)) == config['csv_sha256']
    frame = pd.read_csv(flags.csv_path, usecols=['target','mode','file_path'])
    data = original.EEG19NpyDataset(frame,'Val',flags.stored_prefix,flags.data_root,
                             target_samples=flags.target_samples,eeg_scale_divisor=flags.eeg_scale_divisor)
    n, batch = len(data), flags.eval_batch_size
    assert n == 166079 and batch == 128
    steps = math.ceil(n/batch)
    lo, hi = (min(n,(steps*i//shards)*batch) for i in (shard,shard+1))
    dest = OUT/'raw'/f'seed{seed}_shard{shard}.json'
    contract = dict(seed=seed,locks=locks,manifest=config['csv_sha256'], start=lo,stop=hi,
                    shards=shards, replay=digest(Path(__file__)), evaluator=digest(study/'tools/external_eval.py'))
    if dest.exists():
        assert read(dest)['contract'] == contract
        return
    loader = DataLoader(Subset(data,range(lo,hi)),batch_size=batch,num_workers=2,pin_memory=True)
    counts={a:np.zeros(4,dtype=np.int64) for a in models}
    changes=dict.fromkeys(models,0)
    offset=lo
    with patch.object(external_eval,'KEYS',tuple(models)):
        for eeg,y in loader:
            scores,_ = external_eval.predict(models,eeg.cuda(non_blocking=True),offset==lo or len(y)!=batch)
            end=offset+len(y)
            for i,a in enumerate(models):
                lab,p=cached[a]
                np.testing.assert_array_equal(y.numpy(),lab[offset:end])
                threshold=locks[a]['threshold']
                pred=scores[:,i,0].astype(np.float64)>threshold
                counts[a]+=np.bincount(2*y.numpy()+pred,minlength=4)
                changes[a]+=int(np.sum(pred!=(p[offset:end].astype(np.float64)>threshold)))
            offset=end
    assert offset==hi
    atomic_write_json(dest,dict(contract=contract,changes=changes,
                      counts={a:dict(zip(('tn','fp','fn','tp'),map(int,c))) for a,c in counts.items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--shard',type=int,required=True)
    p.add_argument('--shards',type=int,default=8)
    a=p.parse_args()
    run(a.study,a.seed,a.shard,a.shards)
