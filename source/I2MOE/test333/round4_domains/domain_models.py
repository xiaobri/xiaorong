"""Cohort-robust raw detail branch with unchanged class-prior expectation."""
import czb_paths as _paths
from pathlib import Path

import numpy as np
import torch

from raw_data import RawDataset
from raw_models import RawCorrection, supervised_contrastive
from models4 import focal_each

OUTPUT = Path(f'{_paths.RUNS}/I2MOE_test333_round4_domains_20261114')


class DomainDataset(RawDataset):
    def __init__(self, mode: str, seed: int, variant: str):
        super().__init__(mode, seed, variant)
        self.train_weights = np.load(OUTPUT / 'train_weights.npy', mmap_mode='r') if mode == 'Train' else None
        if self.train_weights is not None and self.train_weights.shape != (731610,):
            raise RuntimeError('Cohort weights have the wrong training row count')

    def __getitem__(self, index: int) -> dict:
        result = super().__getitem__(index)
        result['row_weight'] = torch.tensor(float(self.train_weights[index]) if self.train_weights is not None else 1.)
        return result


class DomainCorrection(RawCorrection):
    def __init__(self, variant: str):
        if variant not in ('cohort_contrast', 'cohort_fsam'):
            raise ValueError(variant)
        super().__init__('raw_contrast')
        self.variant = variant

    def forward(self, batch: dict) -> dict:
        result = super().forward(batch)
        result['features'] = result.pop('embedding')
        return result


def objective(output: dict, batch: dict, variant: str) -> torch.Tensor:
    weights = batch['row_weight'].float()
    labels = batch['label']
    return ((weights * focal_each(output['fused'], labels)).mean()
            + .02 * (output['fused'] - batch['anchor']).square().mean()
            + .1 * (weights * focal_each(output['standalone'], labels)).mean()
            + .003 * supervised_contrastive(output['features'], labels))
