"""Cache exact original preprocessed FP32 EEG, without fitting or resampling.

Coverage is exactly the two fixed20epoch draw unions plus the full original
Val. This cache enables new raw-waveform branches alongside cached frozen
baseline features. Source data.preprocessing is called unchanged.
"""
import czb_paths as _paths
from pathlib import Path
import sys

BASELINE = Path(f'{_paths.BASELINE}')
ROOT = Path(__file__).resolve().parent
OUTPUT = Path(f'{_paths.STORAGE}/I2MOE_test333_fair_raw_cache')
REFERENCE = Path(f'{_paths.STORAGE}/I2MOE_test333_fair_temporal_cache')
PYTHON = f'{_paths.PYTHON}'
sys.path.insert(0, str(BASELINE))

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from i2moe_eeg.data import EEG19NpyDataset, load_manifest, DEFAULT_CSV_PATH, DEFAULT_DATA_ROOT, DEFAULT_STORED_PREFIX
from i2moe_eeg.utils import atomic_write_json


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def dataset():
    frame, audit = load_manifest(DEFAULT_CSV_PATH)
    parts = [EEG19NpyDataset(frame, mode, DEFAULT_STORED_PREFIX, DEFAULT_DATA_ROOT)
             for mode in ('Train', 'Val')]
    if [len(part) for part in parts] != [731610, 166079] or audit['train_val_exact_path_overlap_rows']:
        raise RuntimeError('Original partition changed')
    return ConcatDataset(parts)


def make_plan():
    reference = json.loads((REFERENCE/'ready.json').read_text())
    rows = np.load(REFERENCE/'rows.npy', mmap_mode='r')
    plan = dict(shape=[len(rows), 19, 2000], dtype='float32', reference=str(REFERENCE),
        original_preprocessing_unchanged=True, no_fitted_representation=True,
        no_new_training_samples=True, no_downsampling_or_precision_change=True,
        row_mapping_sha256=reference['row_mapping_sha256'],
        sources={str(path): digest(path) for path in (Path(__file__), Path(DEFAULT_CSV_PATH), BASELINE/'i2moe_eeg/data.py')})
    for path in (str(DEFAULT_CSV_PATH), str(BASELINE/'i2moe_eeg/data.py')):
        assert plan['sources'][path] == reference['plan']['sources'][path]
    if (OUTPUT/'plan.json').exists():
        assert json.loads((OUTPUT/'plan.json').read_text()) == plan
        return plan
    for name in ('rows.npy', 'row_to_cache.npy', 'labels.npy'):
        shutil.copyfile(REFERENCE/name, OUTPUT/name)
    array = np.lib.format.open_memmap(OUTPUT/'eeg.npy', mode='w+', dtype=np.float32, shape=tuple(plan['shape']))
    array.flush()
    atomic_write_json(OUTPUT/'plan.json', plan)
    return plan


def extract(shard):
    torch.set_num_threads(2)
    plan = json.loads((OUTPUT/'plan.json').read_text())
    done = OUTPUT/f'shard{shard}_complete.json'
    if done.exists():
        assert json.loads(done.read_text())['sources'] == plan['sources']
        return
    data = dataset()
    rows = np.load(OUTPUT/'rows.npy')
    lo, hi = len(rows)*shard//8, len(rows)*(shard+1)//8
    array = np.load(OUTPUT/'eeg.npy', mmap_mode='r+')
    labels = np.load(OUTPUT/'labels.npy', mmap_mode='r')
    loader = DataLoader(Subset(data, rows[lo:hi].tolist()), batch_size=128,
        shuffle=False, num_workers=2, pin_memory=False)
    offset, started = lo, time.time()
    for step, (eeg, target) in enumerate(loader):
        values = eeg.numpy()
        if values.dtype != np.float32 or values.shape[1:] != (19, 2000) or not np.isfinite(values).all():
            raise ValueError('Original EEG contract differs')
        np.testing.assert_array_equal(target.numpy(), labels[rows[offset:offset+len(values)]])
        array[offset:offset+len(values)] = values
        offset += len(values)
        if step % 128 == 0:
            status = dict(shard=shard, rows=offset-lo, total=hi-lo, seconds=time.time()-started)
            atomic_write_json(OUTPUT/f'shard{shard}_progress.json', status)
            print(json.dumps(status), flush=True)
    array.flush()
    assert offset == hi
    # Re-open original data to prove row order and original dtype scaling.
    for slot in np.linspace(lo, hi-1, 16, dtype=int):
        original, target = data[int(rows[slot])]
        np.testing.assert_array_equal(array[slot], original.numpy())
    atomic_write_json(done, dict(lo=lo, hi=hi, rows=hi-lo, seconds=time.time()-started,
        sampled_raw_rows_bit_exact=16, sources=plan['sources']))


def worker(shard):
    environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    with (OUTPUT/f'shard{shard}.log').open('a') as log:
        result = subprocess.run([PYTHON, '-u', str(ROOT/'prepare_cache.py'), '--shard', str(shard)],
            env=environment, stdout=log, stderr=subprocess.STDOUT)
    return dict(shard=shard, return_code=result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, choices=range(8))
    args = parser.parse_args()
    if args.shard is not None:
        extract(args.shard)
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT/'extraction.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = make_plan()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(worker, range(8)))
    atomic_write_json(OUTPUT/'jobs.json', dict(jobs=results))
    if any(row['return_code'] for row in results):
        raise RuntimeError('Raw EEG extraction failed')
    shards = [json.loads((OUTPUT/f'shard{i}_complete.json').read_text()) for i in range(8)]
    assert sum(row['rows'] for row in shards) == plan['shape'][0]
    assert all(row['sources'] == plan['sources'] for row in shards)
    atomic_write_json(OUTPUT/'ready.json', dict(plan=plan, shards=shards,
        eeg_sha256=digest(OUTPUT/'eeg.npy'), row_mapping_sha256=digest(OUTPUT/'row_to_cache.npy')))
    print('Exact original FP32 EEG cache complete', flush=True)


if __name__ == '__main__':
    main()
