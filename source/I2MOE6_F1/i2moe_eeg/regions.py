"""Canonical 19-channel ordering and the five requested anatomical regions.

Indices in :data:`REGION_CHANNELS` always refer to the *raw NPY* channel axis.
The loader deliberately preserves channels 0..18 in their raw order so these
indices must never be applied to an already region-reordered tensor.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, Sequence

import torch


EEG_CHANNEL_NAMES: tuple[str, ...] = (
    "Fp1",  # 0
    "Fp2",  # 1
    "F3",   # 2
    "F4",   # 3
    "C3",   # 4
    "C4",   # 5
    "P3",   # 6
    "P4",   # 7
    "O1",   # 8
    "O2",   # 9
    "F7",   # 10
    "F8",   # 11
    "T3",   # 12
    "T4",   # 13
    "T5",   # 14
    "T6",   # 15
    "Fz",   # 16
    "Cz",   # 17
    "Pz",   # 18
)

REGION_CHANNELS: Mapping[str, tuple[int, ...]] = OrderedDict(
    (
        ("frontal", (0, 1, 2, 3, 16)),
        ("left_temporal", (10, 12, 14)),
        ("right_temporal", (11, 13, 15)),
        ("centro_parietal", (4, 5, 17, 6, 7, 18)),
        ("occipital", (8, 9)),
    )
)
REGION_NAMES: tuple[str, ...] = tuple(REGION_CHANNELS)
NUM_EEG_CHANNELS = len(EEG_CHANNEL_NAMES)
NUM_REGIONS = len(REGION_NAMES)
NUM_INTERACTION_EXPERTS = NUM_REGIONS + 1
EXPERT_NAMES: tuple[str, ...] = tuple(
    [f"unique_{name}" for name in REGION_NAMES] + ["redundancy"]
)


def _validate_region_contract() -> None:
    flattened = [channel for channels in REGION_CHANNELS.values() for channel in channels]
    if len(flattened) != NUM_EEG_CHANNELS:
        raise RuntimeError(
            f"Brain-region map must contain {NUM_EEG_CHANNELS} entries, got {len(flattened)}."
        )
    if len(set(flattened)) != NUM_EEG_CHANNELS:
        raise RuntimeError("Brain-region map contains duplicated channel indices.")
    expected = set(range(NUM_EEG_CHANNELS))
    observed = set(flattened)
    if observed != expected:
        raise RuntimeError(
            "Brain-region map must cover raw EEG indices 0..18 exactly; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}."
        )


_validate_region_contract()


def channel_names(indices: Sequence[int]) -> tuple[str, ...]:
    return tuple(EEG_CHANNEL_NAMES[index] for index in indices)


def region_metadata() -> dict[str, object]:
    return {
        "selected_raw_channel_count": NUM_EEG_CHANNELS,
        "raw_eeg_channel_names": list(EEG_CHANNEL_NAMES),
        "raw_eeg_channel_indices": list(range(NUM_EEG_CHANNELS)),
        "raw_channels_19_and_above_selected": False,
        "ecg_or_emg_constructed": False,
        "channel_name_sidecar_verified": False,
        "raw_channel_mapping_confirmed_by_data_owner": True,
        "channel_name_provenance": (
            "The data owner confirmed the raw NPY 0..18 channel-name and brain-region "
            "mapping; it is also consistent with the existing CAM-MoE region config. "
            "The NPY/CSV corpus itself has no separate channel-name sidecar."
        ),
        "regions": {
            name: {
                "raw_indices": list(indices),
                "channel_names": list(channel_names(indices)),
            }
            for name, indices in REGION_CHANNELS.items()
        },
        "expert_names": list(EXPERT_NAMES),
        "index_semantics": "All region indices address raw NPY channels before reordering.",
    }


def split_regions(eeg: torch.Tensor) -> list[torch.Tensor]:
    """Split ``[B, 19, T]`` (or ``[19, T]``) in canonical region order."""

    if eeg.ndim not in (2, 3):
        raise ValueError(f"Expected EEG tensor [19,T] or [B,19,T], got {tuple(eeg.shape)}")
    channel_axis = 0 if eeg.ndim == 2 else 1
    if eeg.shape[channel_axis] != NUM_EEG_CHANNELS:
        raise ValueError(
            f"Expected {NUM_EEG_CHANNELS} raw-order EEG channels, got {eeg.shape[channel_axis]}."
        )
    outputs: list[torch.Tensor] = []
    for indices in REGION_CHANNELS.values():
        index = torch.as_tensor(indices, device=eeg.device, dtype=torch.long)
        outputs.append(torch.index_select(eeg, channel_axis, index))
    return outputs
