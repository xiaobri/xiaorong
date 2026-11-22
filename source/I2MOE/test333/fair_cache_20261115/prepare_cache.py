#!/usr/bin/env python3
"""Cache only deterministic frozen-source computations in FP32.

No training or labels are used to fit a representation. Coverage is the union
of the original baseline's 20 sampled epochs for seeds 3/42, plus full Val.
Original 32-row training tails and 63-row validation tail have separate entries
so subsequent experiments can preserve native frozen-source batch shapes.
"""
import czb_paths as _paths
from pathlib import Path
import sys

BASELINE = Path(f'{_paths.BASELINE}')
ROOT = Path(__file__).resolve().parent
OUTPUT = Path(f'{_paths.STORAGE}/I2MOE_test333_fair_temporal_cache')
sys.path.insert(0, str(BASELINE))

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, SubsetRandomSampler

from i2moe_eeg.data import (EEG19NpyDataset, load_manifest, sampled_indices,
                            DEFAULT_CSV_PATH, DEFAULT_DATA_ROOT, DEFAULT_STORED_PREFIX)
from i2moe_eeg.f1_variants import SixExpertF1
from i2moe_eeg.model import default_frozen_checkpoint_paths, FROZEN_EXPERT_BEST_SEEDS
from i2moe_eeg.utils import atomic_write_json, set_seed

PYTHON = f'{_paths.PYTHON}'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def datasets():
    frame, audit = load_manifest(DEFAULT_CSV_PATH)
    data = [EEG19NpyDataset(frame, mode, DEFAULT_STORED_PREFIX, DEFAULT_DATA_ROOT)
            for mode in ('Train', 'Val')]
    if [len(part) for part in data] != [731610, 166079] or audit['train_val_exact_path_overlap_rows']:
        raise RuntimeError('Original data partition changed')
    return data


def make_plan():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = datasets()
    selected, tails = [], []
    for seed in (3, 42):
        for epoch in range(1, 21):
            indices, stats = sampled_indices(data[0].labels, 20000, 30, 'match_cam', 128, seed, epoch)
            order = np.asarray(list(SubsetRandomSampler(indices.tolist(),
                generator=torch.Generator().manual_seed(seed+epoch*10000))), dtype=np.int64)
            selected.append(indices)
            tails.append(dict(seed=seed, epoch=epoch, rows=order[-32:].tolist()))
    train_rows = np.unique(np.concatenate(selected))
    rows = np.concatenate((train_rows, np.arange(166079, dtype=np.int64)+731610))
    sources = {str(path): digest(path) for path in [Path(DEFAULT_CSV_PATH), Path(__file__),
               BASELINE/'i2moe_eeg/data.py', BASELINE/'i2moe_eeg/model.py',
               *map(Path, default_frozen_checkpoint_paths().values())]}
    plan = dict(sources=sources, seed_order=[3, 42], train_rows=731610, val_rows=166079,
                cached_train_rows=len(train_rows), shape=[len(rows), 5, 65, 128],
                dtype='float32', time_tokens=63, feature_token=63, logits_token=64,
                original_frozen_models_unmodified=True, no_fitted_representation=True,
                original_preprocessing=True, training_draw_budget_unchanged=True,
                tails=tails, validation_tail_rows=list(range(897689-63, 897689)),
                execution_note='Cached FP32 frozen features; downstream new models still require a raw-input full validation replay before a target claim.')
    plan_path = OUTPUT/'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise RuntimeError('Cache plan/source differs')
        return plan
    mapping = np.full(897689, -1, dtype=np.int64)
    mapping[rows] = np.arange(len(rows))
    np.save(OUTPUT/'rows.npy', rows)
    np.save(OUTPUT/'row_to_cache.npy', mapping)
    np.save(OUTPUT/'labels.npy', np.concatenate([part.labels for part in data]))
    array = np.lib.format.open_memmap(OUTPUT/'packets.npy', mode='w+', dtype=np.float32,
                                    shape=tuple(plan['shape']))
    array.flush()
    atomic_write_json(plan_path, plan)
    return plan


def load_sources():
    set_seed(3, deterministic=False)
    model = SixExpertF1(variant='baseline', checkpoint_paths=default_frozen_checkpoint_paths(),
                       expected_source_seeds=FROZEN_EXPERT_BEST_SEEDS).cuda().eval().requires_grad_(False)
    tokens = {}
    for index, expert in enumerate(model.frozen_experts.values()):
        def capture(module, inputs, output, index=index, expert=expert):
            tokens[index] = expert.encoder.projection(output.mean(-2).transpose(1, 2))
        expert.encoder.features.register_forward_hook(capture)
    return model, tokens


@torch.no_grad()
def encode(model, tokens, eeg):
    features, logits = model._encode_frozen(eeg)
    packet = torch.zeros(len(eeg), 5, 65, 128, device=eeg.device)
    packet[:, :, :63] = torch.stack([tokens[index] for index in range(5)], 1)
    packet[:, :, 63] = features
    packet[:, :, 64, :2] = logits
    if not torch.isfinite(packet).all():
        raise FloatingPointError('Nonfinite frozen cache')
    return packet.cpu().numpy()


def extract(shard):
    plan = json.loads((OUTPUT/'plan.json').read_text())
    if (OUTPUT/f'shard{shard}_complete.json').exists():
        return
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.09)
    data = ConcatDataset(datasets())
    rows = np.load(OUTPUT/'rows.npy')
    blocks = (len(rows)+127)//128
    lo, hi = blocks*shard//8*128, min(len(rows), blocks*(shard+1)//8*128)
    model, tokens = load_sources()
    output = np.load(OUTPUT/'packets.npy', mmap_mode='r+')
    loader = DataLoader(Subset(data, rows[lo:hi].tolist()), batch_size=128,
                        num_workers=2, pin_memory=True, shuffle=False)
    offset, started = lo, time.time()
    for step, (eeg, _) in enumerate(loader):
        count = len(eeg)
        # Ordinary cached rows use the standard training/evaluation B128 shape.
        if count != 128:
            eeg = torch.cat((eeg, eeg[:1].expand(128-count, -1, -1)), 0)
        packet = encode(model, tokens, eeg.cuda(non_blocking=True))[:count]
        output[offset:offset+count] = packet
        offset += count
        if step % 64 == 0:
            output.flush()
            status = dict(shard=shard, rows=offset-lo, total=hi-lo, seconds=time.time()-started)
            atomic_write_json(OUTPUT/f'shard{shard}_progress.json', status)
            print(json.dumps(status), flush=True)
    output.flush()
    if offset != hi:
        raise RuntimeError('Incomplete feature cache shard')
    # A separate native-shape cache for each original final training batch.
    tail_rows = [entry for index, entry in enumerate(plan['tails']) if index % 8 == shard]
    for entry in tail_rows:
        eeg = torch.stack([data[index][0] for index in entry['rows']]).cuda()
        path = OUTPUT/f"tail_seed{entry['seed']}_epoch{entry['epoch']:02d}.npy"
        np.save(path, encode(model, tokens, eeg))
    if shard == 0:
        eeg = torch.stack([data[index][0] for index in plan['validation_tail_rows']]).cuda()
        np.save(OUTPUT/'validation_tail.npy', encode(model, tokens, eeg))
    atomic_write_json(OUTPUT/f'shard{shard}_complete.json', dict(
        lo=lo, hi=hi, rows=hi-lo, seconds=time.time()-started,
        peak_gib=torch.cuda.max_memory_allocated()/2**30, sources=plan['sources']))


def run_worker(gpu):
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    with (OUTPUT/f'shard{gpu}.log').open('a') as log:
        result = subprocess.run([PYTHON, '-u', str(Path(__file__).resolve()), '--shard', str(gpu)],
                                env=environment, stdout=log, stderr=subprocess.STDOUT)
    return dict(gpu=gpu, return_code=result.returncode)


@torch.no_grad()
def smoke_cache():
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.09)
    data = datasets()[0]
    indices = np.r_[np.flatnonzero(data.labels == 0)[:124], np.flatnonzero(data.labels == 1)[:4]]
    eeg = torch.stack([data[int(index)][0] for index in indices]).cuda()
    model, tokens = load_sources()
    packet = encode(model, tokens, eeg)
    features, logits = model._encode_frozen(eeg)
    torch.testing.assert_close(torch.from_numpy(packet[:, :, 63]).cuda(), features, rtol=0, atol=0)
    torch.testing.assert_close(torch.from_numpy(packet[:, :, 64, :2]).cuda(), logits, rtol=0, atol=0)
    torch.testing.assert_close(torch.from_numpy(packet[:, :, :63]).cuda(),
                               torch.stack([tokens[index] for index in range(5)], 1), rtol=0, atol=0)
    atomic_write_json(ROOT/'smoke_passed.json', dict(
        source_sha256=digest(__file__), original_input_shape=list(eeg.shape),
        packet_shape=list(packet.shape), dtype=str(packet.dtype),
        frozen_features_logits_and_temporal_tokens_repeat_exact=True,
        peak_gib=torch.cuda.max_memory_allocated()/2**30))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, choices=range(8))
    args = parser.parse_args()
    if args.shard is not None:
        extract(args.shard)
        return
    smoke_cache()
    torch.cuda.empty_cache()
    plan = make_plan()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run_worker, range(8)))
    atomic_write_json(OUTPUT/'jobs.json', dict(jobs=results))
    if any(row['return_code'] != 0 for row in results):
        raise RuntimeError('Cache extraction failed')
    shards = [json.loads((OUTPUT/f'shard{index}_complete.json').read_text()) for index in range(8)]
    if sum(item['rows'] for item in shards) != plan['shape'][0]:
        raise RuntimeError('Cache coverage incomplete')
    for item in shards:
        if item['sources'] != plan['sources']:
            raise RuntimeError('Cache source identity differs')
    atomic_write_json(OUTPUT/'ready.json', dict(plan=plan, shards=shards,
                      packets_sha256=digest(OUTPUT/'packets.npy'),
                      row_mapping_sha256=digest(OUTPUT/'row_to_cache.npy')))
    print('Frozen source cache complete', flush=True)


if __name__ == '__main__':
    main()
