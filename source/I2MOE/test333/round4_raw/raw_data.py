import json
import numpy as np
import torch
from data4 import AnchorDataset
from protocol4 import OUTPUT


class RawDataset(AnchorDataset):
    def __init__(self,mode,seed,variant):
        super().__init__(mode,seed,'hard_rank')
        self.variant=variant
        ready=json.loads((OUTPUT/'raw_cache/ready.json').read_text())
        if ready['identity']['shape']!=[897689,19,2000] or ready['identity']['hz']!=500:
            raise RuntimeError('Raw input contract changed')
        self.raw=np.load(OUTPUT/'raw_cache/raw_inputs.npy',mmap_mode='r')

    def __getitem__(self,index):
        result=super().__getitem__(index)
        result['raw']=torch.from_numpy(np.array(self.raw[self.offset+int(index)],copy=True))
        return result

