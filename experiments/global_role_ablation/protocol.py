"""Shared, dependency-free contract for the global-role experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

VERSION = "global-role-v1"
ARMS = ("A", "B", "C", "D", "E", "E_mean")
MAIN_ARMS = ARMS[:5]
SEEDS = (3, 42, 0, 1, 2)
# Resource settings/paths can vary; scientific settings cannot in formal mode.
FORMAL = dict(
    variant="baseline", epochs=20, batch_size=128, eval_batch_size=128,
    samples_per_epoch=20000, ratio=30, sampling="match_cam", lr=5e-4,
    weight_decay=0.01, task_loss="focal", focal_alpha=0.85, focal_gamma=2.0,
    auxiliary_task_loss_weight=0.05, interaction_loss_weight=0.005,
    hidden_dim=128, num_heads=4, fusion_layers=2, feedforward_multiplier=2,
    dropout=0.3, router_hidden_dim=128, router_layers=2, router_dropout=0.5,
    router_temperature=2.0, replacement_std=1.0, target_samples=2000,
    eeg_scale_divisor=1e-3, amp=False, max_steps_per_epoch=0, max_val_samples=0,
    threshold_start=0.01, threshold_end=0.99, threshold_step=0.01,
    frozen_chunk_size=0,
)


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def validate_formal(args):
    differences = {key: {"expected": expected, "actual": getattr(args, key, None)}
                   for key, expected in FORMAL.items() if getattr(args, key, None) != expected}
    if differences:
        raise ValueError("Formal protocol mismatch: " + json.dumps(differences, sort_keys=True))
    if args.seed not in SEEDS:
        raise ValueError(f"Formal seeds are {SEEDS}.")
    if args.smoke_test:
        raise ValueError("Use --mode smoke for a smoke test.")


def run_directory(root, arm, seed):
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    return Path(root).resolve() / arm / f"seed{seed}"
