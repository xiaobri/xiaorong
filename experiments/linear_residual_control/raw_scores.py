"""Fresh raw-EEG exports for the three references and four locked linear arms."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from study import (ARMS, REFS, ROOT, OUT, HERE, RawLinear, digest, fit, folder,
                   load, name, new_head, original, read, state_hash)
from selection import winners
sys.path.insert(0, str(ROOT/'tools'))
import external_eval

STAGES = {'references': ((0,1,2,3,42), REFS), 'development': ((3,42), ARMS),
          'additional': ((0,1,2), ARMS)}


def load_models(stage, seeds=None):
    expected = load(fit.REFERENCE / 'best_f1.pt')['config']
    old = {(r['arm'],int(r['seed'])):r for r in csv.DictReader((ROOT/'results/single_component_seeds.csv').open())}
    models, locks, frozen = {}, {}, {}
    stage_seeds, arms = STAGES[stage]
    for seed in seeds or stage_seeds:
        ref = fit.reference_dir(seed)
        init = read(ref/'initialization.json')['original_trainable_state_sha256']
        expected['seed'] = seed
        if arms == ARMS:
            rates, selected = read(OUT/'selection.json')['rates'], winners(seed)
        for arm in arms:
            if arm in REFS:
                d = ref if arm == 'baseline' else fit.OUT/'formal'/arm/f'seed{seed}'
                arch = old[arm,seed]
                epoch, threshold = int(arch['epoch']), float(arch['threshold'])
                path = d/'best_f1.pt'
                ck = load(path)
                metric = ck['validation']['best_f1_on_validation_grid']
                assert epoch == ck['epoch'] and metric['threshold'] == threshold
                ledger = [json.loads(s) for s in (d/'metrics.jsonl').read_text().splitlines()]
                assert [r['epoch'] for r in ledger] == list(range(1,21))
                assert epoch == max(ledger, key=lambda r: (r['validation']['best_f1_on_validation_grid']['f1'],r['epoch']))['epoch']
                if arm != 'baseline':
                    ext = ck['config']['single_component']
                    assert ext['arm'] == arm and ext['code'] == {f:digest(fit.HERE/f) for f in ('fit.py','single.py')}
                    assert read(d/'initialization.json')['core'] == init
                    for ep in range(1,21):
                        f = f'sampling_epoch{ep:02d}.json'
                        assert read(d/f)['actual_batch_order_sha256'] == read(ref/f)['actual_batch_order_sha256']
                ck_hash = digest(path)
                lr = None
            else:
                lr = rates[arm]
                k = name(arm,lr)
                row = selected[k]
                epoch, threshold = row['epoch'], row['threshold']
                d = folder(seed)
                path = d/f'epoch{epoch:02d}.pt'
                ck = load(path)
                heads = load(d/f'heads{epoch:02d}.pt')
                assert ck['epoch'] == heads['epoch'] == epoch
                assert heads['design'] == ck['config']['linear_control']
                assert heads['design']['code'] == {f:digest(HERE/f) for f in ('train.py','metrics.py','study.py')}
                assert read(d/'init.json')['sha256'] == init
                assert state_hash(ck['model']) == row['core_state_sha256'] == row['matched_core_sha256']
                head = new_head('cpu')
                learned = heads['learners'][k]
                assert (learned['lr'],learned['arm']) == (lr,arm)
                head.load_state_dict(learned['head'])
                ck_hash = hashlib.sha256((digest(path) + state_hash(head.state_dict())).encode()).hexdigest()
            assert ck['seed'] == seed and all(ck['config'].get(k) == v for k,v in expected.items())
            flags = argparse.Namespace(**ck['args'])
            for n,p in original.frozen_checkpoint_paths(flags).items():
                if p not in frozen:
                    frozen[p] = digest(Path(p))
                assert frozen[p] == expected['frozen_checkpoint_identity'][n]['sha256']
            original.set_seed(seed, flags.deterministic)
            constructor = (lambda **kw:fit.build(arm, **kw)) if arm in ('pairs','pairs_refresh') else original.SixExpertF1
            with patch.object(original,'SixExpertF1',constructor):
                model = original.build_model(flags)
            model.load_trainable_state_dict(ck['model'])
            if arm in ARMS:
                model = RawLinear(model,head)
            ident = f'{arm}.{seed}'
            models[ident] = model.cuda().eval()
            locks[ident] = dict(arm=arm, seed=seed, epoch=epoch, threshold=threshold,
                                head_lr=lr, checkpoint_sha256=ck_hash)
    assert len(frozen) == 5 and digest(Path(flags.csv_path)) == expected['csv_sha256']
    return models, locks, flags, expected['csv_sha256']


def run(stage, shard, shards):
    assert 0 <= shard < shards and torch.cuda.device_count() == 1
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(20 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    models, locks, flags, manifest = load_models(stage)
    frame = pd.read_csv(flags.csv_path, usecols=['target','mode','file_path'])
    data = original.EEG19NpyDataset(frame, 'Val', flags.stored_prefix, flags.data_root,
                         target_samples=flags.target_samples, eeg_scale_divisor=flags.eeg_scale_divisor)
    n, batch = len(data), flags.eval_batch_size
    assert n == 166079 and batch == 128 and data.labels.sum() == 3471
    steps = math.ceil(n/batch)
    lo,hi = (min(n,(steps*i//shards)*batch) for i in (shard,shard+1))
    dest = OUT/'raw'/stage
    dest.mkdir(parents=True, exist_ok=True)
    ck = dest/f'shard{shard}.json'
    file = dest/f'shard{shard}.npz'
    ids = list(models)
    contract = dict(start=lo,stop=hi,shards=shards,keys=ids,locks=locks,
                    manifest_sha256=manifest,code=digest(Path(__file__)),evaluation=digest(ROOT/'tools/external_eval.py'))
    if ck.exists():
        assert read(ck)['contract'] == contract and read(ck)['score_sha256'] == digest(file)
        return
    loader = DataLoader(Subset(data,range(lo,hi)),batch_size=batch,num_workers=2,pin_memory=True)
    chunks, ys = [], []
    with patch.object(external_eval, 'KEYS',tuple(ids)):
        for i,(eeg,y) in enumerate(loader):
            scores,_ = external_eval.predict(models,eeg.cuda(non_blocking=True), i in (0,len(loader)-1))
            chunks.append(scores[:,:,0].copy())
            ys.append(y.numpy())
            if i % 50 == 0:
                print(stage, shard, i, len(loader), flush=True)
    labels = np.concatenate(ys)
    np.testing.assert_array_equal(labels,data.labels[lo:hi])
    with file.with_suffix('.tmp').open('wb') as handle:
        np.savez(handle, scores=np.concatenate(chunks), labels=labels)
    file.with_suffix('.tmp').replace(file)
    original.atomic_write_json(ck,dict(contract=contract,score_sha256=digest(file)))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=STAGES,required=True)
    p.add_argument('--shard',type=int,required=True)
    p.add_argument('--shards',type=int,default=6)
    a=p.parse_args()
    run(a.stage,a.shard,a.shards)
