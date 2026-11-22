"""Replay the locked source operating points on native raw EEG batches."""
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

from additional import METHOD, OUT, SEEDS, digest, original, read
from i2moe_eeg.utils import atomic_write_json, set_seed
from refresh_models import RefreshedPairMoE


def run(study_root, shard, nshards):
    sys.path.insert(0, str(study_root / 'tools'))
    import external_eval as replay
    if not (0 <= shard < nshards and torch.cuda.device_count() == 1):
        raise ValueError('Expected a valid partition and one visible device')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(8 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    models, locks, cached = {}, {}, {}
    frozen = {}
    for method in (METHOD, 'baseline'):
        for seed in SEEDS:
            key = f'{method}_{seed}'
            folder = OUT / 'formal' / method / f'seed{seed}'
            path = folder / 'best_f1.pt'
            complete = read(folder / 'completed.json')
            assert complete['epochs'] == 20 and complete['seed'] == seed
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            epochs = [json.loads(s) for s in (folder / 'metrics.jsonl').read_text().splitlines()]
            assert [e['epoch'] for e in epochs] == list(range(1, 21))
            selected = max(epochs, key=lambda e: (e['validation']['best_f1_on_validation_grid']['f1'], e['epoch']))
            epoch = checkpoint['epoch']
            config = checkpoint['config']
            assert epoch == selected['epoch'] and checkpoint['seed'] == config['seed'] == seed
            assert config['strict_extension']['method'] == method
            metric = checkpoint['validation']['best_f1_on_validation_grid']
            assert metric == selected['validation']['best_f1_on_validation_grid']
            flags = argparse.Namespace(**checkpoint['args'])
            for name, source in original.frozen_checkpoint_paths(flags).items():
                if source not in frozen:
                    frozen[source] = digest(Path(source))
                assert frozen[source] == config['frozen_checkpoint_identity'][name]['sha256']
            set_seed(seed, flags.deterministic)
            constructor = original.SixExpertF1 if method == 'baseline' else lambda **kw: RefreshedPairMoE(METHOD, **kw)
            with patch.object(original, 'SixExpertF1', constructor):
                model = original.build_model(flags)
            model.load_trainable_state_dict(checkpoint['model'])
            models[key] = model.cuda().eval()
            locks[key] = dict(epoch=epoch, threshold=metric['threshold'], checkpoint_sha256=digest(path))
            with np.load(folder / f'predictions_epoch{epoch:02d}.npz') as a:
                cached[key] = a['labels'], a['probabilities']
    assert len(frozen) == 5
    frame = pd.read_csv(flags.csv_path, usecols=['target', 'mode', 'file_path'])
    assert digest(Path(flags.csv_path)) == config['csv_sha256']
    data = original.EEG19NpyDataset(frame, 'Val', flags.stored_prefix, flags.data_root,
                                    target_samples=flags.target_samples, eeg_scale_divisor=flags.eeg_scale_divisor)
    batch = flags.eval_batch_size
    assert len(data) == 166079 and batch == 128
    steps = math.ceil(len(data) / batch)
    lo, hi = (min(len(data), (steps * i // nshards) * batch) for i in (shard, shard + 1))
    path = OUT / 'raw_replay' / f'shard{shard}.json'
    contract = dict(locks=locks, start=lo, stop=hi, manifest=config['csv_sha256'], script=digest(Path(__file__)),
                    prediction_code=digest(study_root / 'tools/external_eval.py'))
    if path.exists():
        if read(path)['contract'] != contract:
            raise ValueError('Existing replay uses a different protocol')
        return
    loader = DataLoader(Subset(data, range(lo, hi)), batch_size=batch, num_workers=2, pin_memory=True)
    counts = {key: np.zeros(4, dtype=np.int64) for key in models}
    changes = dict.fromkeys(models, 0)
    errors = dict.fromkeys(models, 0.)
    offset = lo
    with patch.object(replay, 'KEYS', tuple(models)):
        for eeg, y in loader:
            scores, _ = replay.predict(models, eeg.cuda(non_blocking=True), offset == lo or len(y) != batch)
            end = offset + len(y)
            for idx, key in enumerate(models):
                labels, ref = cached[key]
                np.testing.assert_array_equal(y.numpy(), labels[offset:end])
                p = scores[:, idx, 0].astype(np.float64)
                p_ref = ref[offset:end].astype(np.float64)
                cut = locks[key]['threshold']
                pred = p > cut
                counts[key] += np.bincount(2 * y.numpy() + pred, minlength=4)
                changes[key] += int(np.sum(pred != (p_ref > cut)))
                errors[key] = max(errors[key], float(np.abs(p - p_ref).max()))
            offset = end
            if (offset - lo) % (batch * 100) == 0:
                print(offset - lo, '/', hi - lo, flush=True)
    assert offset == hi
    atomic_write_json(path, dict(contract=contract, changed_decisions=changes, max_probability_error=errors,
                      counts={k: dict(zip(('tn', 'fp', 'fn', 'tp'), map(int, c))) for k, c in counts.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--shards', type=int, default=8)
    args = parser.parse_args()
    run(args.study_root, args.shard, args.shards)
