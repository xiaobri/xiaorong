"""Binary classification and expert-weight metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .regions import EXPERT_NAMES, NUM_INTERACTION_EXPERTS


def _threshold_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    # Match the existing New33 EEG evaluation contract exactly. In particular,
    # a probability equal to the threshold is classified as negative.
    predictions = probabilities > threshold
    positive = labels == 1
    tp = int(np.count_nonzero(predictions & positive))
    fp = int(np.count_nonzero(predictions & ~positive))
    fn = int(np.count_nonzero(~predictions & positive))
    tn = int(labels.size - tp - fp - fn)
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    return {
        "threshold": float(threshold),
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
        "accuracy": float((tp + tn) / labels.size),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Iterable[float],
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1 or labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must be equal-length 1-D arrays.")
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("Non-finite validation probabilities.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Validation metrics require both binary classes.")
    fixed = _threshold_metrics(labels, probabilities, 0.5)
    candidates = [_threshold_metrics(labels, probabilities, value) for value in thresholds]
    # Existing EEG scripts keep the first threshold in an F1 tie.
    best = max(candidates, key=lambda record: float(record["f1"]))
    return {
        "samples": int(labels.size),
        "positives": int(labels.sum()),
        "negatives": int((labels == 0).sum()),
        "positive_rate": float(labels.mean()),
        "auprc": float(average_precision_score(labels, probabilities)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "fixed_0_5": fixed,
        "best_f1_on_validation_grid": best,
    }


def interaction_weight_metrics(
    weights: np.ndarray, labels: np.ndarray, expert_names: tuple[str, ...] = EXPERT_NAMES
) -> dict[str, object]:
    weights = np.asarray(weights, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if weights.ndim != 2 or weights.shape[1] != NUM_INTERACTION_EXPERTS:
        raise ValueError(
            f"Expected weights [N,{NUM_INTERACTION_EXPERTS}], got {weights.shape}."
        )
    if weights.shape[0] != labels.shape[0]:
        raise ValueError("Weight/label sample counts differ.")
    if not np.isfinite(weights).all():
        raise FloatingPointError("Non-finite interaction weights.")
    row_sum_error = np.abs(weights.sum(axis=1) - 1.0)
    safe = np.clip(weights, np.finfo(np.float64).tiny, 1.0)
    entropy = -(safe * np.log(safe)).sum(axis=1)
    top = weights.argmax(axis=1)

    def named(values: np.ndarray) -> dict[str, float]:
        return {name: float(values[index]) for index, name in enumerate(expert_names)}

    by_target = {}
    for target in (0, 1):
        selected = weights[labels == target]
        by_target[str(target)] = {
            "samples": int(selected.shape[0]),
            "mean": named(selected.mean(axis=0)),
            "std": named(selected.std(axis=0)),
        }
    counts = np.bincount(top, minlength=NUM_INTERACTION_EXPERTS)
    return {
        "mean": named(weights.mean(axis=0)),
        "std": named(weights.std(axis=0)),
        "by_target": by_target,
        "top_expert_counts": {
            name: int(counts[index]) for index, name in enumerate(expert_names)
        },
        "entropy_mean": float(entropy.mean()),
        "entropy_std": float(entropy.std()),
        "maximum_row_sum_error": float(row_sum_error.max(initial=0.0)),
    }
