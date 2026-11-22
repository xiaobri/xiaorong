"""Dataset and class-sampling utilities for the Xiehe EEG NPY corpus."""

from __future__ import annotations
import czb_paths as _paths

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .regions import NUM_EEG_CHANNELS, region_metadata


DEFAULT_DATA_ROOT = f"{_paths.STORAGE}/finetune-xiehe-dataset"
DEFAULT_CSV_NAME = (
    "train_xiehe_all_add_test_2025_0430_normal_4s_and_inpatient_6s_with_mode.csv"
)
DEFAULT_CSV_PATH = str(Path(DEFAULT_DATA_ROOT) / DEFAULT_CSV_NAME)
DEFAULT_STORED_PREFIX = (
    f"{_paths.STORED_PREFIX}"
)


def load_manifest(csv_path: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = ["file_path", "target", "mode"]
    frame = pd.read_csv(csv_path, usecols=required)
    if frame[required].isna().any().any():
        raise ValueError("Manifest contains null file_path, target, or mode values.")
    frame["target"] = pd.to_numeric(frame["target"], errors="raise").astype(np.int64)
    invalid_targets = sorted(set(frame["target"].unique()) - {0, 1})
    if invalid_targets:
        raise ValueError(f"Expected binary targets 0/1, found {invalid_targets}.")
    invalid_modes = sorted(set(frame["mode"].unique()) - {"Train", "Val"})
    if invalid_modes:
        raise ValueError(f"Unexpected mode values: {invalid_modes}.")
    for mode in ("Train", "Val"):
        if not bool((frame["mode"] == mode).any()):
            raise ValueError(f"Manifest has no {mode} rows.")

    train_paths = set(frame.loc[frame["mode"] == "Train", "file_path"].astype(str))
    val_paths = frame.loc[frame["mode"] == "Val", "file_path"].astype(str)
    # The CSV has no patient column. The immediate parent directory is the
    # recording/patient identifier in the current corpus, so report this as a
    # heuristic audit without silently changing the published split.
    inferred_subject = frame["file_path"].astype(str).str.rsplit("/", n=2).str[-2]
    train_subjects = set(inferred_subject[frame["mode"] == "Train"])
    val_subjects = set(inferred_subject[frame["mode"] == "Val"])
    inferred_subject_overlap = sorted(train_subjects & val_subjects)
    label_counts = frame.groupby(["mode", "target"], observed=True).size().to_dict()
    summary: dict[str, Any] = {
        "csv_path": os.path.realpath(csv_path),
        "rows": int(len(frame)),
        "mode_counts": {
            str(key): int(value) for key, value in frame["mode"].value_counts().to_dict().items()
        },
        "label_counts": {
            f"{mode}/target{target}": int(count)
            for (mode, target), count in label_counts.items()
        },
        "duplicate_file_path_rows": int(
            frame.duplicated(subset=["file_path"], keep="first").sum()
        ),
        "train_val_exact_path_overlap_rows": int(val_paths.isin(train_paths).sum()),
        "patient_level_split_verified": False,
        "patient_level_split_note": (
            "The CSV has no explicit patient/group column. Exact path overlap is checked, "
            "but patient-level independence cannot be proven from the selected columns."
        ),
        "inferred_parent_subject_audit": {
            "heuristic_only": True,
            "train_subjects": len(train_subjects),
            "val_subjects": len(val_subjects),
            "overlap_count": len(inferred_subject_overlap),
            "overlap_ids": inferred_subject_overlap,
            "note": (
                "Subject IDs are inferred from each NPY parent directory. This audit is "
                "not a substitute for an explicit patient_id column."
            ),
        },
        "input_contract": region_metadata(),
    }
    return frame, summary


class EEG19NpyDataset(Dataset[tuple[torch.Tensor, int]]):
    """Select raw NPY channels 0..18; ECG/EMG are never derived or returned."""

    def __init__(
        self,
        frame: pd.DataFrame,
        mode: str,
        stored_prefix: str,
        data_root: str,
        target_samples: int = 2000,
        eeg_scale_divisor: float = 1e-3,
    ) -> None:
        rows = frame.loc[frame["mode"] == mode, ["file_path", "target"]]
        if rows.empty:
            raise ValueError(f"No rows found for mode={mode!r}.")
        if target_samples <= 0:
            raise ValueError("target_samples must be positive.")
        if eeg_scale_divisor <= 0:
            raise ValueError("eeg_scale_divisor must be positive.")
        self.paths = rows["file_path"].astype(str).to_numpy(copy=True)
        self.labels = rows["target"].to_numpy(dtype=np.int64, copy=True)
        self.mode = mode
        self.stored_prefix = stored_prefix.rstrip("/")
        self.data_root = data_root.rstrip("/")
        self.target_samples = int(target_samples)
        self.eeg_scale_divisor = float(eeg_scale_divisor)

    def __len__(self) -> int:
        return len(self.labels)

    def map_path(self, stored_path: str) -> str:
        if stored_path == self.stored_prefix:
            return self.data_root
        expected = self.stored_prefix + "/"
        if stored_path.startswith(expected):
            return self.data_root + stored_path[len(self.stored_prefix) :]
        if os.path.isfile(stored_path):
            return stored_path
        raise FileNotFoundError(
            f"Cannot map {stored_path!r}; expected stored prefix {self.stored_prefix!r}."
        )

    def preprocess(self, raw: np.ndarray) -> np.ndarray:
        if raw.ndim != 2 or raw.shape[0] < NUM_EEG_CHANNELS:
            raise ValueError(
                f"Expected raw NPY [>=19,T], got shape={getattr(raw, 'shape', None)}."
            )
        # Preserve the existing CAM-MoE/New33 numeric operation order: scale in
        # the source dtype first, then copy into the final float32 tensor. This
        # avoids a subtle input mismatch for the many float16 NPY files.
        eeg = np.array(raw[:NUM_EEG_CHANNELS], copy=True)
        eeg /= self.eeg_scale_divisor
        output = np.zeros((NUM_EEG_CHANNELS, self.target_samples), dtype=np.float32)
        width = min(self.target_samples, eeg.shape[1])
        output[:, :width] = eeg[:, :width]
        if not np.isfinite(output).all():
            raise FloatingPointError("Non-finite values found in the 19-channel EEG tensor.")
        return output

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        mapped = self.map_path(str(self.paths[index]))
        try:
            # Memory mapping avoids materializing the unused trailing channels;
            # preprocess() copies only raw channels 0..18 into model input.
            raw = np.load(mapped, allow_pickle=False, mmap_mode="r")
        except Exception as exc:
            raise RuntimeError(f"Failed to load NPY: {mapped}") from exc
        return torch.from_numpy(self.preprocess(raw)), int(self.labels[index])


def preflight_dataset(dataset: EEG19NpyDataset, count: int) -> dict[str, Any]:
    if count <= 0:
        return {"checked": 0}
    requested = min(int(count), len(dataset))
    candidates: list[int] = []
    # With a small preflight budget, cover both classes before adding boundary
    # and evenly spaced examples.
    for label in (0, 1):
        indices = np.flatnonzero(dataset.labels == label)
        if len(indices):
            candidates.append(int(indices[0]))
    for label in (0, 1):
        indices = np.flatnonzero(dataset.labels == label)
        if len(indices):
            candidates.append(int(indices[-1]))
    candidates.extend((0, len(dataset) - 1))
    candidates.extend(
        np.linspace(0, len(dataset) - 1, max(requested, 2), dtype=int).tolist()
    )
    selected = list(dict.fromkeys(candidates))
    if len(selected) < requested:
        selected_set = set(selected)
        selected.extend(
            index
            for index in range(len(dataset))
            if index not in selected_set
        )
    selected = selected[:requested]
    observations = []
    for index in selected:
        tensor, label = dataset[index]
        observations.append(
            {
                "index": index,
                "label": int(label),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "finite": bool(torch.isfinite(tensor).all()),
                "minimum": float(tensor.min()),
                "maximum": float(tensor.max()),
            }
        )
    return {"checked": len(observations), "observations": observations}


def sampling_plan(
    labels: np.ndarray,
    samples_per_epoch: int,
    ratio: int,
    sampling: str,
    batch_size: int,
) -> dict[str, int]:
    count_0 = int((labels == 0).sum())
    count_1 = int((labels == 1).sum())
    if sampling == "natural" or samples_per_epoch <= 0:
        sampled_0, sampled_1 = count_0, count_1
    else:
        requested_1 = max(1, samples_per_epoch // (ratio + 1))
        requested_0 = max(1, samples_per_epoch - requested_1)
        if sampling == "match_cam":
            sampled_0 = min(requested_0, count_0)
            sampled_1 = min(requested_1, count_1)
        elif sampling == "exact_ratio":
            sampled_0, sampled_1 = requested_0, requested_1
        else:
            raise ValueError(f"Unknown sampling mode: {sampling!r}")
    before = sampled_0 + sampled_1
    full_batches = before // batch_size
    final_partial_batch_size = before % batch_size
    return {
        "available_class_0": count_0,
        "available_class_1": count_1,
        "sampled_class_0_before_batch_truncation": sampled_0,
        "sampled_class_1_before_batch_truncation": sampled_1,
        "samples_before_batch_truncation": before,
        # Kept for ledger compatibility. Sampling no longer truncates the
        # final partial batch, so all requested samples are retained.
        "samples_after_batch_truncation": before,
        "batch_size": int(batch_size),
        "full_batches_per_epoch": full_batches,
        "final_partial_batch_size": final_partial_batch_size,
        "steps_per_epoch": full_batches + int(final_partial_batch_size > 0),
    }


def sampled_indices(
    labels: np.ndarray,
    samples_per_epoch: int,
    ratio: int,
    sampling: str,
    batch_size: int,
    seed: int,
    epoch: int,
) -> tuple[np.ndarray, dict[str, int]]:
    indices_0 = np.flatnonzero(labels == 0)
    indices_1 = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(seed + epoch)
    if sampling == "natural" or samples_per_epoch <= 0:
        sampled = np.arange(len(labels), dtype=np.int64)
        rng.shuffle(sampled)
    else:
        requested_1 = max(1, samples_per_epoch // (ratio + 1))
        requested_0 = max(1, samples_per_epoch - requested_1)
        if sampling == "match_cam":
            sample_0 = min(requested_0, len(indices_0))
            sample_1 = min(requested_1, len(indices_1))
            replace_0 = replace_1 = False
        elif sampling == "exact_ratio":
            sample_0, sample_1 = requested_0, requested_1
            replace_0 = sample_0 > len(indices_0)
            replace_1 = sample_1 > len(indices_1)
        else:
            raise ValueError(f"Unknown sampling mode: {sampling!r}")
        sampled = np.concatenate(
            (
                rng.choice(indices_0, size=sample_0, replace=replace_0),
                rng.choice(indices_1, size=sample_1, replace=replace_1),
            )
        ).astype(np.int64, copy=False)
        rng.shuffle(sampled)
    if len(sampled) <= 0:
        raise ValueError("Sampling plan does not contain any samples.")
    full_batches = len(sampled) // batch_size
    final_partial_batch_size = len(sampled) % batch_size
    stats = {
        "samples": int(len(sampled)),
        "class_0": int((labels[sampled] == 0).sum()),
        "class_1": int((labels[sampled] == 1).sum()),
        "full_batches": int(full_batches),
        "final_partial_batch_size": int(final_partial_batch_size),
        "steps": int(full_batches + (final_partial_batch_size > 0)),
    }
    return sampled, stats
