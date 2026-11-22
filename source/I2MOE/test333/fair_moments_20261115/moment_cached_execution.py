"""Execution-only adapter; keep the original optimizer, sampler and loss code."""
import bootstrap
import json
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from i2moe_eeg.model import EEGFrozenRegionalI2MoE


def install(original, seed: int, method: str) -> dict:
    ready_path = bootstrap.CACHE/'ready.json'
    if not ready_path.exists():
        raise RuntimeError('Frozen cache is not complete')
    ready = json.loads(ready_path.read_text())
    plan = ready['plan']
    moments_ready = json.loads((bootstrap.MOMENT_CACHE/'ready.json').read_text())
    assert moments_ready['row_mapping_sha256'] == ready['row_mapping_sha256']
    assert moments_ready['plan']['tails'] == plan['tails']
    assert moments_ready['plan']['validation_tail_rows'] == plan['validation_tail_rows']
    for path in set(plan['sources']) & set(moments_ready['plan']['sources']):
        assert plan['sources'][path] == moments_ready['plan']['sources'][path], path
    source_dataset = original.EEG19NpyDataset
    source_preflight = original.preflight_dataset
    live = {}

    class CachedDataset(Dataset):
        def __init__(self, frame, mode, stored_prefix, data_root, **kwargs):
            self.source = source_dataset(frame, mode, stored_prefix, data_root, **kwargs)
            self.mode, self.labels = mode, self.source.labels
            self.offset = 0 if mode == 'Train' else 731610
            self.packets = np.load(bootstrap.CACHE/'packets.npy', mmap_mode='r')
            self.moments = np.load(bootstrap.MOMENT_CACHE/'packets.npy', mmap_mode='r')
            self.moment_tail = None
            self.mapping = np.load(bootstrap.CACHE/'row_to_cache.npy', mmap_mode='r')
            all_labels = np.load(bootstrap.CACHE/'labels.npy', mmap_mode='r')
            if not np.array_equal(self.labels, all_labels[self.offset:self.offset+len(self)]):
                raise RuntimeError('Cached labels/order differ from original dataset')
            self.tail_lookup, self.tail_packet = {}, None
            if mode == 'Val':
                self.tail_lookup = {row: index for index, row in enumerate(plan['validation_tail_rows'])}
                self.tail_packet = np.load(bootstrap.CACHE/'validation_tail.npy', mmap_mode='r')
                self.moment_tail = np.load(bootstrap.MOMENT_CACHE/'validation_tail.npy', mmap_mode='r')
            live[mode] = self

        def __len__(self):
            return len(self.labels)

        def set_epoch(self, epoch):
            entry = next(row for row in plan['tails'] if row['seed'] == seed and row['epoch'] == epoch)
            self.tail_lookup = {row: index for index, row in enumerate(entry['rows'])}
            self.tail_packet = np.load(bootstrap.CACHE/f'tail_seed{seed}_epoch{epoch:02d}.npy', mmap_mode='r')
            self.moment_tail = np.load(bootstrap.MOMENT_CACHE/f'tail_seed{seed}_epoch{epoch:02d}.npy', mmap_mode='r')

        def __getitem__(self, index):
            row = self.offset + int(index)
            if row in self.tail_lookup:
                packet = self.tail_packet[self.tail_lookup[row]]
                moments = self.moment_tail[self.tail_lookup[row]]
            else:
                slot = int(self.mapping[row])
                if slot < 0:
                    raise RuntimeError('An unscheduled training row is absent from the frozen cache')
                packet = self.packets[slot]
                moments = self.moments[slot]
            combined = np.zeros((5, 4, 1280), dtype=np.float32)
            combined[:, :3] = moments
            combined[:, 3, :128] = packet[:, 63]
            combined[:, 3, 128:130] = packet[:, 64, :2]
            return torch.from_numpy(combined), int(self.labels[index])

    def cached_frozen(model, packet):
        if packet.ndim != 4 or tuple(packet.shape[1:]) != (5, 4, 1280):
            raise ValueError('Invalid frozen-source moment packet')
        for index in range(5):
            model.moments[index] = packet[:, index, :3].contiguous()
        return packet[:, :, 3, :128].contiguous(), packet[:, :, 3, 128:130].contiguous()

    original.EEG19NpyDataset = CachedDataset
    original.preflight_dataset = lambda dataset, requested: dict(
        source_preflight(dataset.source, requested), execution='FP32 frozen-source cache; original raw source inspected')
    sampler = original.sampled_indices

    def epoch_sampler(*values, **kwargs):
        live['Train'].set_epoch(int(values[-1]))
        return sampler(*values, **kwargs)

    original.sampled_indices = epoch_sampler
    EEGFrozenRegionalI2MoE._encode_frozen = cached_frozen
    if hasattr(original, 'smoke_test'):
        smoke = original.smoke_test

        def cached_smoke(model, dataset, focal, args, device):
            # The cache intentionally omits rows never sampled in either
            # formal seed. Restrict only the disposable smoke-test pool to
            # cached training rows; formal sampling remains untouched.
            available = np.flatnonzero(dataset.mapping[:len(dataset)] >= 0)
            view = Subset(dataset, available.tolist())
            view.labels = dataset.labels[available]
            result = smoke(model, view, focal, args, device)
            result['smoke_pool'] = 'Original Train rows present in the fixed formal sampling union'
            return result

        original.smoke_test = cached_smoke
    return dict(cache=str(bootstrap.CACHE), packets_sha256=ready['packets_sha256'],
                moment_cache=str(bootstrap.MOMENT_CACHE), moment_packets_sha256=moments_ready['packets_sha256'],
                raw_source_manifest_sha256=plan['sources'][str(original.DEFAULT_CSV_PATH)],
                dtype='float32', native_batch_tails_preserved=True,
                scientific_protocol_unchanged=True,
                target_requires_raw_input_replay=True,
                initial_raw_and_cached_frozen_tensor_parity='see fair_cache_20261115/smoke_passed.json and fair_moment_cache_20261115/smoke_passed.json')
