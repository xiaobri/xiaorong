"""Read previous in-stream control embeddings using their original row and batch-tail epoch."""
from copy import copy
from functools import lru_cache

import torch

from allocation_execution import install as install_blocks

ROW_FIELD = 240
EPOCH_FIELD = 241
_history_batch = None


def history_batch(rows, epochs):
    if _history_batch is None:
        raise RuntimeError('Original Train execution adapter is required to refresh negative controls')
    return _history_batch(rows, epochs)


def install(original, seed: int, method: str):
    global _history_batch
    contract = install_blocks(original, seed, 'independent_aux')
    dataset_class = original.EEG19NpyDataset
    live = {}

    class IndexedDataset(dataset_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.original_epoch = 1
            if self.mode == 'Train':
                live['Train'] = self

        def set_epoch(self, epoch):
            super().set_epoch(epoch)
            self.original_epoch = int(epoch)

        def __getitem__(self, index):
            packet, label = super().__getitem__(index)
            if self.mode == 'Train':
                packet[0, 3, ROW_FIELD] = int(index)
                packet[0, 3, EPOCH_FIELD] = self.original_epoch
            return packet, label

    @lru_cache(maxsize=2)
    def snapshot(epoch):
        # Copy the object so historical native tail selection cannot alter the running sampler.
        prior = copy(live['Train'])
        prior.set_epoch(int(epoch))
        return prior

    def load(rows, epochs):
        if not len(rows) or len(rows) != len(epochs):
            raise ValueError('Invalid historical negative record request')
        packets = []
        for row, epoch in zip(rows, epochs):
            if int(row) < 0 or not 1 <= int(epoch) <= 20:
                raise ValueError('Historical negative has no valid original position')
            packet, label = snapshot(int(epoch))[int(row)]
            if int(label) != 0:
                raise RuntimeError('Negative observation provenance did not match the labels')
            packets.append(packet)
        return torch.stack(packets)

    _history_batch = load
    original.EEG19NpyDataset = IndexedDataset
    contract.update(history_row_epoch_stored_for_train_only=True, history_slots=[ROW_FIELD, EPOCH_FIELD],
        original_negative_draws_can_be_re_evaluated=True, training_row_ids_used_in_predictions=False)
    return contract
