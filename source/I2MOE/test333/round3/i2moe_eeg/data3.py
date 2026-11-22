"""Read-only shared input and six-expert prediction caches."""
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from .protocol3 import BASE_CACHE, OUTPUT


class CachedEEG(Dataset):
    def __init__(self, mode: str, seed: int, require_ready: bool = True):
        if require_ready:
            ready = json.loads((OUTPUT / 'inputs_ready.json').read_text())
            if ready['n_train'] != 731610 or ready['n_val'] != 166079:
                raise RuntimeError('Input cache row contract changed')
        self.offset = 0 if mode == 'Train' else 731610
        self.length = 731610 if mode == 'Train' else 166079
        self.seed_index = (3, 42).index(seed)
        self.fm = np.load(OUTPUT / 'fm_inputs.npy', mmap_mode='r')
        self.base = np.load(BASE_CACHE / 'baseline_margins.npy', mmap_mode='r')
        self.labels = np.load(OUTPUT / 'labels.npy', mmap_mode='r')[self.offset:self.offset+self.length]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        row = self.offset + int(index)
        return {'fm': torch.from_numpy(np.array(self.fm[row], copy=True)),
                'base': torch.from_numpy(np.array(self.base[row, self.seed_index], copy=True)),
                'label': int(self.labels[index]), 'index': int(index)}
