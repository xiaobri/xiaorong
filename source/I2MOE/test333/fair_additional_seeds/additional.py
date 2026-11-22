"""Paths and the preregistered seed-replication design."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'fair_refreshed_pairs_20261120'))
import bootstrap
import train_eeg19_i2moe as original

SEEDS = (0, 1, 2)
METHOD = 'fresh_both'
REFERENCE = bootstrap.CACHE.parent / 'I2MOE6_F1/screen_20261112/baseline/seed3'
OUT = bootstrap.CACHE.parent / 'I2MOE_test333_fresh_both_additional_20261122'
SUPPLEMENT = OUT / 'frozen_train_supplement'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        result = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def ordered(indices, seed, epoch):
    import numpy as np
    import torch
    from torch.utils.data import SubsetRandomSampler
    generator = torch.Generator().manual_seed(seed + epoch * 10000)
    return np.asarray(list(SubsetRandomSampler(indices.tolist(), generator=generator)), dtype=np.int64)
