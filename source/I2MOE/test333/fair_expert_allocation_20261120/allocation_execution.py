import json
import numpy as np
import torch

import bootstrap
from complement_execution import install as install_anchor
from block_features import SEGMENTS


def install(original,seed,method):
    contract=install_anchor(original,seed,'risk_power')
    base=json.loads((bootstrap.CACHE/'ready.json').read_text())
    path=bootstrap.BLOCK_CACHE
    info=json.loads((path/'ready.json').read_text())
    plan=info['plan']
    if (base['row_mapping_sha256']!=info['row_mapping_sha256']
            or base['plan']['tails']!=plan['tails'] or base['plan']['validation_tail_rows']!=plan['val_tail']):
        raise ValueError('Segment features differs from the original row order')
    source=original.EEG19NpyDataset

    class BlockDataset(source):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.blocks=np.load(path/'packets.npy',mmap_mode='r')
            self.block_tail=np.load(path/'validation_tail.npy',mmap_mode='r') if self.mode=='Val' else None

        def set_epoch(self,epoch):
            super().set_epoch(epoch)
            self.block_tail=np.load(path/f'tail_seed{seed}_epoch{epoch:02d}.npy',mmap_mode='r')

        def __getitem__(self,index):
            packet,label=super().__getitem__(index)
            row=self.offset+int(index)
            blocks=(self.block_tail[self.tail_lookup[row]] if row in self.tail_lookup else self.blocks[self.mapping[row]])
            packet[:,3,256:256+SEGMENTS*128]=torch.from_numpy(np.array(blocks,copy=True)).flatten(1)
            return packet,label

    original.EEG19NpyDataset=BlockDataset
    contract.update(pre_layernorm_block_cache=str(path),block_sha256=info['packets_sha256'],block_definition=plan['descriptor'],
                    block_sources=plan['sources'])
    return contract
