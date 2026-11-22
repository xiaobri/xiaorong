"""Apply the original regional classifiers to cached local EEG features."""
import czb_paths as _paths
import fcntl
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASELINE = Path(f'{_paths.BASELINE}')
sys.path.insert(0, str(BASELINE))
from i2moe_eeg.f1_variants import SixExpertF1
from i2moe_eeg.model import default_frozen_checkpoint_paths, FROZEN_EXPERT_BEST_SEEDS
from i2moe_eeg.utils import atomic_write_json

CACHE = Path(f'{_paths.STORAGE}/I2MOE_test333_fair_temporal_cache')
OUTPUT = Path(f'{_paths.RUNS}/I2MOE_test333_fair_time_scores_20261119')


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            checksum.update(block)
    return checksum.hexdigest()


@torch.no_grad()
def describe(classifiers, array):
    packet = torch.from_numpy(np.array(array, copy=True))
    scores = []
    for region, classifier in enumerate(classifiers):
        logits = classifier(packet[:, region, :64])
        pooled = packet[:, region, 64, :2]
        if (logits[:, 63] - pooled).abs().max() > .002:
            raise ValueError('Original frozen classifier or pooled feature parity changed')
        scores.append(logits[..., 1] - logits[..., 0])
    result = torch.stack(scores, 1)
    if not result.isfinite().all():
        raise ValueError('Nonfinite regional trace')
    return result.numpy()


def main():
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT / 'extract.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(6)
        info = json.loads((CACHE / 'ready.json').read_text())
        plan = info['plan']
        for name, expected in plan['sources'].items():
            if digest(name) != expected:
                raise ValueError(f'Source cache input changed at {name}')
        if digest(CACHE/'row_to_cache.npy') != info['row_mapping_sha256']:
            raise ValueError('Unexpected row map')
        array = np.load(CACHE / 'packets.npy', mmap_mode='r')
        shape = (len(array), 5, 64)
        contract = dict(shape=list(shape), temporal_packet_hash=info['packets_sha256'],
                        row_mapping_sha256=info['row_mapping_sha256'], sources=plan['sources'],
                        tails=plan['tails'], val_tail=plan['validation_tail_rows'], extractor=digest(__file__),
                        dtype='float32', description='63 time-local logit margins and 1 pooled margin per original classifier; CPU, no model fitting')
        contract_path = OUTPUT / 'plan.json'
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError('Existing extraction contract changed')
        atomic_write_json(contract_path, contract)
        if (OUTPUT / 'ready.json').exists():
            return
        model = SixExpertF1(variant='baseline', checkpoint_paths=default_frozen_checkpoint_paths(),
                            expected_source_seeds=FROZEN_EXPERT_BEST_SEEDS).eval().requires_grad_(False)
        classifiers = [source.classifier for source in model.frozen_experts.values()]
        progress = OUTPUT/'progress.json'
        start = json.loads(progress.read_text())['rows'] if progress.exists() else 0
        dest = np.lib.format.open_memmap(OUTPUT/'packets.npy', mode='r+' if start else 'w+',
                                         dtype='float32', shape=shape)
        tick = time.monotonic()
        for index in range(start, len(array), 256):
            end = min(index+256,len(array))
            dest[index:end] = describe(classifiers,array[index:end])
            if end % 8192 == 0 or end == len(array):
                dest.flush()
                atomic_write_json(progress, dict(rows=end, total=len(array), seconds_this_invocation=time.monotonic()-tick))
        tail_files = ['validation_tail.npy'] + [f"tail_seed{e['seed']}_epoch{e['epoch']:02d}.npy" for e in plan['tails']]
        for file in tail_files:
            np.save(OUTPUT/file, describe(classifiers,np.load(CACHE/file)))
        atomic_write_json(OUTPUT/'ready.json', dict(plan=contract, packets_sha256=digest(OUTPUT/'packets.npy'),
                          row_mapping_sha256=info['row_mapping_sha256'], native_tail_count=len(tail_files)))


if __name__ == '__main__':
    main()
