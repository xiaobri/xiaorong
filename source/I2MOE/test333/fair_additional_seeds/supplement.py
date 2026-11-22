"""Compute only missing FP32 frozen inputs and each new seed's native B32 tails."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from additional import SEEDS, REFERENCE, SUPPLEMENT, bootstrap, digest, ordered, original, read
from block_features import register_blocks
from i2moe_eeg.utils import atomic_write_json, set_seed

SHARDS = 4


def build_plan():
    SUPPLEMENT.mkdir(parents=True, exist_ok=True)
    args = read(REFERENCE / 'args.json')
    base = read(bootstrap.CACHE / 'ready.json')
    for name in [args['csv_path'], *original.frozen_checkpoint_paths(argparse.Namespace(**args)).values()]:
        if digest(Path(name)) != base['plan']['sources'][name]:
            raise ValueError('Frozen sources or manifest changed')
    labels = np.load(bootstrap.CACHE / 'labels.npy')[:base['plan']['train_rows']]
    mapping = np.load(bootstrap.CACHE / 'row_to_cache.npy')
    all_rows, tails = [], []
    for seed in SEEDS:
        for epoch in range(1, args['epochs'] + 1):
            indices, stats = original.sampled_indices(labels, args['samples_per_epoch'], args['ratio'],
                                                      args['sampling'], args['batch_size'], seed, epoch)
            order = ordered(indices, seed, epoch)
            assert stats['samples'] == 20000 and stats['final_partial_batch_size'] == 32
            tails.append(dict(seed=seed, epoch=epoch, rows=order[-32:].tolist(),
                              order_sha256=hashlib.sha256(order.tobytes()).hexdigest()))
            all_rows.append(indices)
    rows = np.unique(np.concatenate(all_rows))
    missing = rows[mapping[rows] < 0]
    plan = dict(seeds=list(SEEDS), shards=SHARDS, unique_drawn_rows=len(rows), missing_rows=len(missing),
                missing_sha256=hashlib.sha256(missing.tobytes()).hexdigest(), tails=tails,
                base_mapping=base['row_mapping_sha256'], prep_sha256=digest(Path(__file__)))
    path = SUPPLEMENT / 'plan.json'
    if path.exists():
        previous = read(path)
        if previous == plan:
            return
        if (list(SUPPLEMENT.glob('shard*.json')) or
                dict(previous, prep_sha256=plan['prep_sha256']) != plan):
            raise RuntimeError('Existing supplement has a different design')
        atomic_write_json(path, plan)
        return
    np.save(SUPPLEMENT / 'rows.npy', missing)
    extra_map = np.full(len(labels), -1, dtype=np.int64)
    extra_map[missing] = np.arange(len(missing))
    np.save(SUPPLEMENT / 'mapping.npy', extra_map)
    np.lib.format.open_memmap(SUPPLEMENT / 'packets.npy', mode='w+', dtype=np.float32,
                             shape=(len(missing), 5, 4, 1280)).flush()
    atomic_write_json(path, plan)
    print('Union', len(rows), 'new regular rows', len(missing), 'native tails', len(tails), flush=True)


def extract(shard):
    plan = read(SUPPLEMENT / 'plan.json')
    assert plan['prep_sha256'] == digest(Path(__file__))
    marker = SUPPLEMENT / f'shard{shard}.json'
    if marker.exists():
        assert read(marker)['plan_sha256'] == digest(SUPPLEMENT / 'plan.json')
        return
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    set_seed(3, False)
    args = argparse.Namespace(**read(REFERENCE / 'args.json'))
    base = original.build_model(args).cuda().eval().requires_grad_(False)
    moments = {}
    for i, expert in enumerate(base.frozen_experts.values()):
        def capture(module, inputs, value, index=i):
            moments[index] = torch.stack((F.adaptive_avg_pool2d(value, (1, 1)).flatten(1),
                                          value.amax((-2, -1)), value.flatten(2).std(-1, unbiased=False)), 1)
        expert.encoder.features.register_forward_hook(capture)
    blocks, handles = register_blocks(base.frozen_experts)

    @torch.no_grad()
    def pack(eeg):
        features, logits = base._encode_frozen(eeg.cuda())
        value = eeg.new_zeros((len(eeg), 5, 4, 1280), device=features.device)
        value[:, :, :3] = torch.stack([moments[i] for i in range(5)], 1)
        value[:, :, 3, :128] = features
        value[:, :, 3, 128:130] = logits
        value[:, :, 3, 256:] = torch.stack([blocks[i][0].flatten(1) for i in range(5)], 1)
        if not value.isfinite().all():
            raise FloatingPointError('Nonfinite frozen inputs')
        return value.cpu().numpy()

    frame, _ = original.load_manifest(args.csv_path)
    data = original.EEG19NpyDataset(frame, 'Train', args.stored_prefix, args.data_root,
                                    target_samples=args.target_samples, eeg_scale_divisor=args.eeg_scale_divisor)
    del frame
    source_labels = np.load(bootstrap.CACHE / 'labels.npy')[:len(data)]
    np.testing.assert_array_equal(source_labels, data.labels)
    mapping = np.load(bootstrap.CACHE / 'row_to_cache.npy')
    old = np.flatnonzero(mapping[:len(data)] >= 0)[shard::2500][:128]
    assert len(old) == 128
    raw = pack(torch.stack([data[int(i)][0] for i in old]))
    slots = mapping[old]
    temporal = np.load(bootstrap.CACHE / 'packets.npy', mmap_mode='r')[slots, :, 63:65]
    moments_ref = np.load(bootstrap.MOMENT_CACHE / 'packets.npy', mmap_mode='r')[slots]
    blocks_ref = np.load(bootstrap.BLOCK_CACHE / 'packets.npy', mmap_mode='r')[slots]
    errors = {}
    for name, first, second in [('moments', raw[:, :, :3], moments_ref),
                                ('embedding', raw[:, :, 3, :128], temporal[:, :, 0]),
                                ('logits', raw[:, :, 3, 128:130], temporal[:, :, 1, :2]),
                                ('blocks', raw[:, :, 3, 256:], blocks_ref.reshape(128, 5, -1))]:
        errors[name] = float(np.max(np.abs(first - second)))
        # The original caches were produced in separate cuDNN benchmark runs.
        # Compare each window and expert, including the post-normalization
        # embedding and class logits consumed by the original router.
        a, b = first.reshape(128, 5, -1), second.reshape(128, 5, -1)
        relative = np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1).clip(min=1)
        errors[name + '_max_relative_l2'] = float(relative.max())
        if relative.max() > .001 or (name in ('embedding', 'logits') and errors[name] > .002):
            raise ValueError(f'Frozen source parity failure: {errors}')

    rows = np.load(SUPPLEMENT / 'rows.npy')
    dest = np.load(SUPPLEMENT / 'packets.npy', mmap_mode='r+')
    nblocks = (len(rows) + 127) // 128
    lo = min(len(rows), (nblocks * shard // SHARDS) * 128)
    hi = min(len(rows), (nblocks * (shard + 1) // SHARDS) * 128)
    loader = DataLoader(Subset(data, rows[lo:hi].tolist()), batch_size=128, num_workers=2)
    offset = lo
    for eeg, _ in loader:
        n = len(eeg)
        if n < 128:
            eeg = torch.cat((eeg, eeg[:1].expand(128 - n, -1, -1)))
        dest[offset:offset + n] = pack(eeg)[:n]
        offset += n
    assert offset == hi
    dest.flush()
    for entry in plan['tails'][shard::SHARDS]:
        eeg = torch.stack([data[i][0] for i in entry['rows']])
        np.save(SUPPLEMENT / 'tail_{}_{}.npy'.format(entry['seed'], entry['epoch']), pack(eeg))
    atomic_write_json(marker, dict(plan_sha256=digest(SUPPLEMENT / 'plan.json'), rows=hi-lo, parity_max_abs=errors))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, choices=range(SHARDS))
    options = parser.parse_args()
    if options.shard is None:
        build_plan()
    else:
        extract(options.shard)
