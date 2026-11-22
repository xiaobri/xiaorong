"""Read-only caches with unchanged original row and seed ordering."""
import json
import numpy as np
import torch
from i2moe_eeg.data3 import CachedEEG
from protocol4 import OUTPUT


class AnchorDataset(CachedEEG):
    def __init__(self, mode: str, seed: int, variant: str):
        super().__init__(mode,seed)
        self.variant=variant
        root=OUTPUT/'anchor_cache'/f'seed{seed}'
        ready=json.loads((root/'ready.json').read_text())
        if ready['identity']['rows']!=897689 or ready['identity']['seed']!=seed:
            raise RuntimeError('Anchor cache contract changed')
        self.summary=np.load(root/'summary.npy',mmap_mode='r')
        self.tokens=np.load(root/'tokens.npy',mmap_mode='r') if variant=='local_mil' else None

    def __getitem__(self,index: int) -> dict:
        row=self.offset+int(index)
        summary=np.array(self.summary[row],copy=True)
        result=dict(features=torch.from_numpy(summary[:128]),anchor=torch.tensor(summary[129]),
                    base=torch.from_numpy(np.array(self.base[row,self.seed_index],copy=True)),
                    label=int(self.labels[index]),index=int(index))
        if self.tokens is not None:result['tokens']=torch.from_numpy(np.array(self.tokens[row],copy=True))
        if self.variant=='wave_detail':result['fm']=torch.from_numpy(np.array(self.fm[row],copy=True))
        return result

