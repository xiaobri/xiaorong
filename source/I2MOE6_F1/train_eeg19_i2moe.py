#!/usr/bin/env python3
"""Train I2MOE9E from five selected frozen distilled regional checkpoints."""

from __future__ import annotations
import czb_paths as _paths

import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler

from i2moe_eeg.data import (
    DEFAULT_CSV_PATH,
    DEFAULT_DATA_ROOT,
    DEFAULT_STORED_PREFIX,
    EEG19NpyDataset,
    load_manifest,
    preflight_dataset,
    sampled_indices,
    sampling_plan,
)
from i2moe_eeg.losses import (
    MulticlassFocalLoss,
    auxiliary_interaction_task_loss,
    interaction_feature_losses,
    router_statistics,
    task_loss,
)
from i2moe_eeg.metrics import (
    binary_metrics,
    interaction_weight_metrics,
)
from i2moe_eeg.model import (
    DISTILLATION_HARD_WEIGHT,
    DISTILLATION_KD_WEIGHT,
    DISTILLATION_TEACHER_AGGREGATION,
    DISTILLATION_TEACHER_TABLE_SHA256,
    DISTILLATION_TEMPERATURE,
    EEGRegionI2MoE,
    FROZEN_EXPERT_BEST_AUPRC,
    FROZEN_EXPERT_BEST_EPOCHS,
    FROZEN_EXPERT_BEST_SEEDS,
    FROZEN_EXPERT_ENCODER_NAME,
    FROZEN_EXPERT_ENCODER_WEIGHTS,
    FROZEN_EXPERT_FEATURE_POOLING,
    FROZEN_EXPERT_INPUT_LAYOUT,
    default_frozen_checkpoint_paths,
)
from i2moe_eeg.regions import EXPERT_NAMES, NUM_INTERACTION_EXPERTS, region_metadata
from i2moe_eeg.utils import (
    atomic_torch_save,
    atomic_write_json,
    atomic_write_jsonl,
    capture_rng_state,
    json_ready,
    load_json,
    restore_rng_state,
    set_seed,
)


from i2moe_eeg.f1_variants import SixExpertF1, loss_terms

MODEL_NAME = "six_expert_f1_screen"
DEFAULT_OUTPUT_DIR = (
    f"{_paths.STORAGE}/I2MOE9/test/"
    "eeg19_distilled_frozen5_redundancy_router6_detach_train20k_bs128_lr0.0005_e20/seed42"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("baseline", "sampling", "competence", "temporal_mil"), default="baseline")
    parser.add_argument("--frozen-chunk-size", type=int, default=0)
    parser.add_argument("--routing-loss-weight", type=float, default=0.02)
    parser.add_argument("--regional-mass-loss-weight", type=float, default=0.02)
    parser.add_argument("--regional-mass-floor", type=float, default=0.55)
    parser.add_argument("--routing-target-temperature", type=float, default=0.5)
    parser.add_argument("--memory-cap-gib", type=float, default=4.0)
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--stored-prefix", default=DEFAULT_STORED_PREFIX)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-samples", type=int, default=2000)
    parser.add_argument("--eeg-scale-divisor", type=float, default=1e-3)
    parser.add_argument(
        "--strict-patient-disjoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reject the current split if parent-directory subject IDs overlap. "
            "Disabled by default to preserve the existing baseline split."
        ),
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--samples-per-epoch", type=int, default=20_000)
    parser.add_argument("--ratio", type=int, default=30)
    parser.add_argument(
        "--sampling",
        choices=("match_cam", "exact_ratio", "natural"),
        default="match_cam",
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--task-loss", choices=("focal", "cross_entropy"), default="focal")
    parser.add_argument("--focal-alpha", type=float, default=0.85)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--auxiliary-task-loss-weight", type=float, default=0.05)
    parser.add_argument("--interaction-loss-weight", type=float, default=0.005)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--feedforward-multiplier", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--router-hidden-dim", type=int, default=128)
    parser.add_argument("--router-layers", type=int, default=2)
    parser.add_argument("--router-dropout", type=float, default=0.5)
    parser.add_argument("--router-temperature", type=float, default=2.0)
    parser.add_argument("--replacement-std", type=float, default=1.0)
    for expert_name in EXPERT_NAMES[:5]:
        option = "--" + expert_name.removeprefix("unique_").replace("_", "-") + "-checkpoint"
        parser.add_argument(
            option,
            default="",
            help=(
                f"Override the fixed selected distilled checkpoint for {expert_name}; "
                "formal launchers leave this empty."
            ),
        )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="FP32 is the default comparison protocol.",
    )
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--allow-multiple-visible-gpus",
        action="store_true",
        help="Still uses cuda:0, but bypasses the one-visible-GPU launch guard.",
    )
    parser.add_argument("--resume", default="")
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--val-log-interval", type=int, default=200)
    parser.add_argument("--preflight-samples", type=int, default=8)
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-end", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a real-data forward/backward check and exit without writing a run.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_fields = (
        "epochs",
        "batch_size",
        "eval_batch_size",
        "target_samples",
        "hidden_dim",
        "num_heads",
        "fusion_layers",
        "feedforward_multiplier",
        "router_hidden_dim",
        "router_layers",
    )
    invalid = [name for name in positive_integer_fields if int(getattr(args, name)) <= 0]
    if invalid:
        raise ValueError(f"These values must be positive: {invalid}.")
    if args.num_workers < 0 or args.samples_per_epoch < 0 or args.ratio < 0:
        raise ValueError("num_workers, samples_per_epoch and ratio must be non-negative.")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("lr must be positive and weight_decay non-negative.")
    if args.interaction_loss_weight < 0 or args.auxiliary_task_loss_weight < 0:
        raise ValueError("Auxiliary-task and interaction weights must be non-negative.")
    if args.threshold_step <= 0 or args.threshold_end < args.threshold_start:
        raise ValueError("Invalid validation threshold grid.")
    if args.max_steps_per_epoch < 0 or args.max_val_samples < 0:
        raise ValueError("Debug limits must be non-negative.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This trainer intentionally supports only one process/GPU per run.")


def initialize_device(args: argparse.Namespace) -> torch.device:
    if args.device == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable.")
    visible = torch.cuda.device_count()
    if visible != 1 and not args.allow_multiple_visible_gpus:
        raise RuntimeError(
            "Single-GPU contract requires exactly one visible CUDA device; "
            f"torch sees {visible}. Launch with CUDA_VISIBLE_DEVICES=<one GPU>."
        )
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    return device


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_checkpoint_paths(args: argparse.Namespace) -> dict[str, str]:
    defaults = default_frozen_checkpoint_paths()
    resolved: dict[str, str] = {}
    for expert_name in EXPERT_NAMES[:5]:
        attribute = expert_name.removeprefix("unique_") + "_checkpoint"
        chosen = getattr(args, attribute) or defaults[expert_name]
        resolved[expert_name] = str(Path(chosen).resolve())
    return resolved


def frozen_checkpoint_identity(
    paths: dict[str, str], source_metadata: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    identity: dict[str, dict[str, Any]] = {}
    for name, value in paths.items():
        path = Path(value)
        identity[name] = {
            **source_metadata[name],
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(str(path)),
        }
    return identity


def semantic_config(
    args: argparse.Namespace,
    manifest_sha256: str,
    source_identity: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    keys = (
        "variant", "frozen_chunk_size", "routing_loss_weight", "regional_mass_loss_weight",
        "regional_mass_floor", "routing_target_temperature",
        "stored_prefix",
        "data_root",
        "target_samples",
        "eeg_scale_divisor",
        "strict_patient_disjoint",
        "batch_size",
        "eval_batch_size",
        "num_workers",
        "samples_per_epoch",
        "ratio",
        "sampling",
        "lr",
        "weight_decay",
        "task_loss",
        "focal_alpha",
        "focal_gamma",
        "auxiliary_task_loss_weight",
        "interaction_loss_weight",
        "hidden_dim",
        "num_heads",
        "fusion_layers",
        "feedforward_multiplier",
        "dropout",
        "router_hidden_dim",
        "router_layers",
        "router_dropout",
        "router_temperature",
        "replacement_std",
        "seed",
        "device",
        "amp",
        "deterministic",
        "max_steps_per_epoch",
        "max_val_samples",
        "threshold_start",
        "threshold_end",
        "threshold_step",
    )
    config = {key: getattr(args, key) for key in keys}
    config.update(
        {
            "model": MODEL_NAME,
            "csv_path": os.path.realpath(args.csv_path),
            "csv_sha256": manifest_sha256,
            "data_root": os.path.realpath(args.data_root),
            "world_size": 1,
            "epochs": int(args.epochs),
            "region_contract": region_metadata(),
            "frozen_expert_encoder": FROZEN_EXPERT_ENCODER_NAME,
            "frozen_expert_source": FROZEN_EXPERT_ENCODER_WEIGHTS,
            "frozen_expert_feature_pooling": FROZEN_EXPERT_FEATURE_POOLING,
            "frozen_expert_input_layout": FROZEN_EXPERT_INPUT_LAYOUT,
            "frozen_expert_selection": (
                "user-confirmed fixed per-region distilled checkpoints"
            ),
            "frozen_experts_distilled": True,
            "frozen_source_distillation": {
                "hard_weight": DISTILLATION_HARD_WEIGHT,
                "kd_weight": DISTILLATION_KD_WEIGHT,
                "temperature": DISTILLATION_TEMPERATURE,
                "teacher_aggregation": DISTILLATION_TEACHER_AGGREGATION,
                "teacher_table_sha256": DISTILLATION_TEACHER_TABLE_SHA256,
            },
            "outer_router_input": "all_six_expert_128d_representations",
            "outer_router_input_dim": NUM_INTERACTION_EXPERTS * int(args.hidden_dim),
            "interaction_features_detached_before_router": True,
            "frozen_expert_best_source_seeds": dict(FROZEN_EXPERT_BEST_SEEDS),
            "frozen_expert_best_source_epochs": dict(FROZEN_EXPERT_BEST_EPOCHS),
            "frozen_expert_best_source_auprc": dict(FROZEN_EXPERT_BEST_AUPRC),
            "frozen_sources_shared_across_training_seeds": True,
            "frozen_checkpoint_identity": source_identity,
            "torchvision_version": torchvision.__version__,
        }
    )
    return config


def build_model(args: argparse.Namespace) -> EEGRegionI2MoE:
    return SixExpertF1(
        variant=args.variant, frozen_chunk_size=args.frozen_chunk_size,
        checkpoint_paths=frozen_checkpoint_paths(args),
        expected_source_seeds=FROZEN_EXPERT_BEST_SEEDS,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        fusion_layers=args.fusion_layers,
        feedforward_multiplier=args.feedforward_multiplier,
        dropout=args.dropout,
        router_hidden_dim=args.router_hidden_dim,
        router_layers=args.router_layers,
        router_dropout=args.router_dropout,
        router_temperature=args.router_temperature,
        replacement_std=args.replacement_std,
    )


def make_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    sampler: Optional[SubsetRandomSampler] = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        drop_last=False,
    )


def threshold_grid(args: argparse.Namespace) -> np.ndarray:
    count = int(
        math.floor((args.threshold_end - args.threshold_start) / args.threshold_step + 1e-8)
    )
    return np.asarray(
        [args.threshold_start + index * args.threshold_step for index in range(count + 1)],
        dtype=np.float64,
    )


def train_one_epoch(
    model: EEGRegionI2MoE,
    dataset: EEG19NpyDataset,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    focal: MulticlassFocalLoss,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
) -> dict[str, Any]:
    indices, sample_stats = sampled_indices(
        dataset.labels,
        args.samples_per_epoch,
        args.ratio,
        args.sampling,
        args.batch_size,
        args.seed,
        epoch,
    )
    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(args.seed + epoch * 10_000)
    sampler = SubsetRandomSampler(indices.tolist(), generator=sampler_generator)
    loader = make_loader(dataset, args.batch_size, args.num_workers, device, sampler)

    model.train()
    sums: dict[str, float] = {
        "total_loss": 0.0,
        "final_task_loss": 0.0,
        "auxiliary_task_loss": 0.0,
        "interaction_loss": 0.0,
        "correct": 0.0,
    }
    auxiliary_component_sums = {}
    interaction_component_sums = {}
    extra_sums = {"extra": 0.0, "routing": 0.0, "regional_mass_penalty": 0.0}
    router_sums = {"entropy": 0.0, "maximum_weight": 0.0, "minimum_weight": 0.0}
    seen = 0
    step_count = 0
    started = time.monotonic()
    total_steps = len(loader)
    if args.max_steps_per_epoch:
        total_steps = min(total_steps, args.max_steps_per_epoch)

    for step, (eeg, labels) in enumerate(loader, start=1):
        if args.max_steps_per_epoch and step > args.max_steps_per_epoch:
            break
        eeg = eeg.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=args.amp and device.type == "cuda",
        ):
            output = model(eeg, compute_interactions=True)
            parts = loss_terms(output, labels, args, focal)
            current_final_task_loss = parts["primary"]
            current_auxiliary_task_loss = parts["auxiliary"]
            current_interaction_loss = parts["interaction"]
            auxiliary_components = parts["auxiliary_components"]
            interaction_components = parts["interaction_components"]
            total_loss = parts["total"]
        if not bool(torch.isfinite(total_loss)):
            raise FloatingPointError(
                f"Non-finite loss at seed={args.seed}, epoch={epoch}, step={step}: "
                f"final_task={current_final_task_loss.item()}, "
                f"auxiliary_task={current_auxiliary_task_loss.item()}, "
                f"interaction={current_interaction_loss.item()}."
            )
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch = int(labels.numel())
        seen += batch
        step_count += 1
        sums["total_loss"] += float(total_loss.detach()) * batch
        sums["final_task_loss"] += float(current_final_task_loss.detach()) * batch
        sums["auxiliary_task_loss"] += (
            float(current_auxiliary_task_loss.detach()) * batch
        )
        sums["interaction_loss"] += float(current_interaction_loss.detach()) * batch
        sums["correct"] += float((output.logits.argmax(dim=1) == labels).sum())
        for name, value in auxiliary_components.items():
            auxiliary_component_sums[name] = auxiliary_component_sums.get(name, 0.0) + float(value.detach()) * batch
        for name, value in interaction_components.items():
            interaction_component_sums[name] = interaction_component_sums.get(name, 0.0) + float(value.detach()) * batch
        for name in extra_sums:
            extra_sums[name] += float(parts[name].detach()) * batch
        current_router = router_statistics(output.interaction_weights.detach())
        for name in router_sums:
            router_sums[name] += float(current_router[name]) * batch

        if step == 1 or step % args.log_interval == 0 or step == total_steps:
            elapsed = max(time.monotonic() - started, 1e-9)
            rate = seen / elapsed
            remaining = max(total_steps - step, 0) * (elapsed / step)
            print(
                f"TRAIN model={MODEL_NAME} seed={args.seed} epoch={epoch}/{args.epochs} "
                f"step={step}/{total_steps} samples={seen} rate={rate:.1f}/s "
                f"eta={remaining:.1f}s total_loss={sums['total_loss']/seen:.6f} "
                f"final_task_loss={sums['final_task_loss']/seen:.6f} "
                f"aux_task_loss={sums['auxiliary_task_loss']/seen:.6f} "
                f"interaction_loss={sums['interaction_loss']/seen:.6f}",
                flush=True,
            )

    if seen == 0:
        raise RuntimeError("Training loader produced no samples.")
    return {
        "samples": seen,
        "extra_loss_components": {name: value / seen for name, value in extra_sums.items()},
        "steps": step_count,
        "sampling": sample_stats,
        "loss": sums["total_loss"] / seen,
        "final_task_loss": sums["final_task_loss"] / seen,
        "auxiliary_task_loss": sums["auxiliary_task_loss"] / seen,
        "interaction_loss": sums["interaction_loss"] / seen,
        "auxiliary_task_components": {
            name: value / seen for name, value in auxiliary_component_sums.items()
        },
        "interaction_components": {
            name: value / seen for name, value in interaction_component_sums.items()
        },
        "accuracy": sums["correct"] / seen,
        "router": {name: value / seen for name, value in router_sums.items()},
        "seconds": time.monotonic() - started,
    }


@torch.no_grad()
def evaluate(
    model: EEGRegionI2MoE,
    dataset: EEG19NpyDataset,
    focal: MulticlassFocalLoss,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
) -> dict[str, Any]:
    evaluation_dataset: torch.utils.data.Dataset = dataset
    if args.max_val_samples:
        # Debug-only validation must include both classes; the audited manifest
        # is ordered and its first 512 rows contain only negatives.
        half = args.max_val_samples // 2
        selected = np.concatenate((np.flatnonzero(dataset.labels == 0)[:half],
                                   np.flatnonzero(dataset.labels == 1)[:args.max_val_samples - half]))
        evaluation_dataset = Subset(dataset, selected.tolist())
    loader = make_loader(
        evaluation_dataset, args.eval_batch_size, args.num_workers, device, sampler=None
    )
    model.eval()
    labels_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    expert_probability_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    loss_sum = 0.0
    seen = 0
    started = time.monotonic()

    for step, (eeg, labels) in enumerate(loader, start=1):
        eeg = eeg.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        with torch.amp.autocast(
            device_type=device.type,
            enabled=args.amp and device.type == "cuda",
        ):
            output = model(eeg, compute_interactions=False)
            current_loss = task_loss(output.logits, labels, args.task_loss, focal)
        batch = int(labels.numel())
        seen += batch
        loss_sum += float(current_loss) * batch
        labels_parts.append(labels.cpu().numpy())
        probability_parts.append(torch.softmax(output.logits, dim=1)[:, 1].cpu().numpy())
        expert_probability_parts.append(
            torch.softmax(output.expert_logits, dim=2)[:, :, 1].cpu().numpy()
        )
        weight_parts.append(output.interaction_weights.cpu().numpy())
        if (
            step == 1
            or step % args.val_log_interval == 0
            or step == len(loader)
        ):
            elapsed = max(time.monotonic() - started, 1e-9)
            remaining = max(len(loader) - step, 0) * (elapsed / step)
            print(
                f"VAL model={MODEL_NAME} seed={args.seed} epoch={epoch}/{args.epochs} "
                f"step={step}/{len(loader)} samples={seen} eta={remaining:.1f}s",
                flush=True,
            )

    labels_array = np.concatenate(labels_parts)
    probabilities = np.concatenate(probability_parts)
    expert_probabilities = np.concatenate(expert_probability_parts)
    weights = np.concatenate(weight_parts)
    thresholds = threshold_grid(args)
    metrics = binary_metrics(labels_array, probabilities, thresholds)
    metrics["coarse_grid"] = binary_metrics(labels_array, probabilities, np.arange(0.05, 0.951, 0.05))
    if not args.max_val_samples:
        prediction_path = Path(args.output_dir) / f"predictions_epoch{epoch:02d}.npz"
        with prediction_path.with_suffix(".tmp").open("wb") as stream:
            np.savez_compressed(stream, labels=labels_array, probabilities=probabilities,
                                expert_probabilities=expert_probabilities, weights=weights)
        prediction_path.with_suffix(".tmp").replace(prediction_path)
    metrics.update(
        {
            "loss": loss_sum / seen,
            "seconds": time.monotonic() - started,
            "interaction_weights": interaction_weight_metrics(weights, labels_array, model.expert_names),
            "expert_metrics": {
                name: binary_metrics(
                    labels_array, expert_probabilities[:, index], thresholds
                )
                for index, name in enumerate(model.expert_names)
            },
            "interaction_forward_used": False,
            "selection_metric": "auprc",
            "split": "Val",
        }
    )
    return metrics


def smoke_test(
    model: EEGRegionI2MoE,
    dataset: EEG19NpyDataset,
    focal: MulticlassFocalLoss,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    positive_indices = np.flatnonzero(dataset.labels == 1)
    negative_indices = np.flatnonzero(dataset.labels == 0)
    positive_count = max(1, args.batch_size // (args.ratio + 1))
    negative_count = args.batch_size - positive_count
    if len(positive_indices) < positive_count or len(negative_indices) < negative_count:
        raise RuntimeError(
            "Smoke test cannot construct the requested real-data batch: "
            f"need class0={negative_count}, class1={positive_count}."
        )
    selected = np.concatenate(
        (negative_indices[:negative_count], positive_indices[:positive_count])
    ).tolist()
    eeg = torch.stack([dataset[index][0] for index in selected]).to(device)
    labels = torch.tensor(dataset.labels[selected], dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    output = model(eeg, compute_interactions=True)
    parts = loss_terms(output, labels, args, focal)
    primary, auxiliary, interaction, total = (parts[name] for name in ("primary", "auxiliary", "interaction", "total"))
    auxiliary_components = parts["auxiliary_components"]
    interaction_components = parts["interaction_components"]
    total.backward()
    missing_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
        peak_reserved_gib = torch.cuda.max_memory_reserved(device) / (1024**3)
    else:
        peak_allocated_gib = 0.0
        peak_reserved_gib = 0.0
    training_result = {
        "input_shape": list(eeg.shape),
        "label_counts": {
            "class_0": int((labels == 0).sum()),
            "class_1": int((labels == 1).sum()),
        },
        "logits_shape": list(output.logits.shape),
        "expert_logits_shape": list(output.expert_logits.shape),
        "interaction_view_features_shape": list(output.interaction_view_features.shape) if output.interaction_view_features is not None else None,
        "interaction_view_logits_shape": list(output.interaction_view_logits.shape) if output.interaction_view_logits is not None else None,
        "router_features_shape": list(output.router_features.shape),
        "weights_shape": list(output.interaction_weights.shape),
        "weights_row_sums": output.interaction_weights.sum(dim=1).detach().cpu(),
        "final_task_loss": primary.detach().cpu(),
        "auxiliary_task_loss": auxiliary.detach().cpu(),
        "auxiliary_task_components": {
            key: value.detach().cpu() for key, value in auxiliary_components.items()
        },
        "interaction_loss": interaction.detach().cpu(),
        "interaction_components": {
            key: value.detach().cpu() for key, value in interaction_components.items()
        },
        "total_loss": total.detach().cpu(),
        "all_finite": bool(
            torch.isfinite(output.logits).all()
            and torch.isfinite(output.router_features).all()
            and torch.isfinite(output.interaction_weights).all()
            and (output.interaction_view_features is None or torch.isfinite(output.interaction_view_features).all())
            and (output.interaction_view_logits is None or torch.isfinite(output.interaction_view_logits).all())
            and torch.isfinite(total)
        ),
        "missing_gradient_parameters": missing_gradients,
        "optimizer_step_completed": True,
        "peak_allocated_gib": peak_allocated_gib,
        "peak_reserved_gib": peak_reserved_gib,
    }
    optimizer.zero_grad(set_to_none=True)
    del output, total
    inference_before = torch.get_rng_state().clone()
    model.eval()
    with torch.no_grad():
        inference = model(eeg, compute_interactions=False)
    inference_after = torch.get_rng_state()
    torch_rng_unchanged = torch.equal(inference_before, inference_after)
    return {
        **training_result,
        "inference_shape": list(inference.logits.shape),
        "inference_finite": bool(torch.isfinite(inference.logits).all()),
        "inference_did_not_consume_torch_rng": torch_rng_unchanged,
        "architecture": model.architecture_summary(),
    }


def validate_resume(
    checkpoint: dict[str, Any], current_config: dict[str, Any], args: argparse.Namespace
) -> int:
    required = ("model", "optimizer", "scaler", "rng_state", "config", "epoch")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"Resume checkpoint is missing: {missing}.")
    saved_config = checkpoint["config"]
    if not isinstance(saved_config, dict):
        raise ValueError("Resume checkpoint config is not a mapping.")
    saved_epochs = int(saved_config.get("epochs", -1))
    saved_without_epochs = {key: value for key, value in saved_config.items() if key != "epochs"}
    current_without_epochs = {
        key: value for key, value in current_config.items() if key != "epochs"
    }
    if saved_without_epochs != current_without_epochs:
        differing = sorted(
            key
            for key in set(saved_without_epochs) | set(current_without_epochs)
            if saved_without_epochs.get(key) != current_without_epochs.get(key)
        )
        raise ValueError(f"Resume configuration mismatch in fields: {differing}.")
    epoch = int(checkpoint["epoch"])
    if saved_epochs <= 0 or epoch <= 0 or epoch > saved_epochs:
        raise ValueError(
            f"Invalid resume epoch contract: epoch={epoch}, saved target={saved_epochs}."
        )
    if saved_epochs > args.epochs or epoch > args.epochs:
        raise ValueError(
            f"Cannot resume into a smaller target: checkpoint target/epoch="
            f"{saved_epochs}/{epoch}, requested={args.epochs}."
        )
    return epoch


def _json_equivalent(left: Any, right: Any) -> bool:
    return json_ready(left) == json_ready(right)


def _metrics_entries(path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed metrics.jsonl line {line_number}.") from exc
            if record.get("model") != MODEL_NAME or int(record.get("seed", -1)) != args.seed:
                raise RuntimeError(f"Foreign run record in metrics.jsonl line {line_number}.")
            epoch = int(record.get("epoch", -1))
            if epoch < 1 or epoch > args.epochs:
                raise RuntimeError(f"Out-of-range metrics epoch {epoch}.")
            entries.append(record)
    return entries


def _metrics_by_epoch(path: Path, args: argparse.Namespace) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for record in _metrics_entries(path, args):
        epoch = int(record["epoch"])
        if epoch in records:
            raise RuntimeError(f"Duplicate metrics epoch {epoch}.")
        records[epoch] = record
    return records


def _validate_record_config(
    record: dict[str, Any], current_config: dict[str, Any], epoch: int
) -> None:
    observed = record.get("config")
    if not isinstance(observed, dict):
        raise RuntimeError(f"Metrics epoch {epoch} has no semantic config mapping.")
    observed_without_epochs = {key: value for key, value in observed.items() if key != "epochs"}
    current_without_epochs = {
        key: value for key, value in current_config.items() if key != "epochs"
    }
    if observed_without_epochs != current_without_epochs:
        raise RuntimeError(f"Metrics epoch {epoch} semantic configuration mismatch.")
    target = int(observed.get("epochs", -1))
    if target < epoch or target > int(current_config["epochs"]):
        raise RuntimeError(
            f"Metrics epoch {epoch} has invalid target epochs={target}."
        )


def _state_dict_schema(state: Any, label: str) -> dict[str, tuple[int, ...]]:
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"{label} has no non-empty model state.")
    invalid = [name for name, tensor in state.items() if not isinstance(tensor, torch.Tensor)]
    if invalid:
        raise RuntimeError(f"{label} has non-tensor model entries: {invalid[:5]}.")
    return {name: tuple(tensor.shape) for name, tensor in state.items()}


def _unequal_state_keys(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> list[str]:
    if set(left) != set(right):
        return sorted(set(left) ^ set(right))
    return [name for name in left if not torch.equal(left[name], right[name])]


def _validate_best_payload(
    payload: dict[str, Any],
    expected_epoch: int,
    expected_score: float,
    expected_record: dict[str, Any],
    resume_model_schema: dict[str, tuple[int, ...]],
    args: argparse.Namespace,
) -> None:
    if (
        payload.get("model_name") != MODEL_NAME
        or int(payload.get("seed", -1)) != args.seed
        or int(payload.get("epoch", -1)) != expected_epoch
    ):
        raise RuntimeError("best_auprc.pt identity/epoch mismatch.")
    if not math.isclose(
        float(payload.get("best_auprc", float("nan"))),
        expected_score,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("best_auprc.pt value does not match metrics ledger.")
    if not _json_equivalent(payload.get("epoch_metrics"), expected_record):
        raise RuntimeError("best_auprc.pt epoch_metrics differs from the metrics ledger.")
    if not _json_equivalent(payload.get("config"), expected_record.get("config")):
        raise RuntimeError("best_auprc.pt config differs from the metrics ledger.")
    if _state_dict_schema(payload.get("model"), "best_auprc.pt") != resume_model_schema:
        raise RuntimeError("best_auprc.pt model schema differs from resume.pt.")


def _best_checkpoint_payload(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Keep best-model inference metadata without duplicating Adam/RNG state."""

    keys = (
        "model_name",
        "seed",
        "epoch",
        "model",
        "config",
        "args",
        "best_auprc",
        "best_epoch",
        "validation",
        "epoch_metrics",
    )
    missing = [key for key in keys if key not in checkpoint]
    if missing:
        raise RuntimeError(f"Cannot construct best checkpoint; missing: {missing}.")
    return {key: checkpoint[key] for key in keys}


def repair_or_validate_best(
    output_dir: Path,
    checkpoint: dict[str, Any],
    resumed_epoch: int,
    records: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    selected_epoch = max(
        records,
        key=lambda epoch: (float(records[epoch]["validation"]["auprc"]), epoch),
    )
    selected_score = float(records[selected_epoch]["validation"]["auprc"])
    if (
        int(checkpoint.get("best_epoch", -1)) != selected_epoch
        or not math.isclose(
            float(checkpoint.get("best_auprc", float("nan"))),
            selected_score,
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("resume.pt best metadata disagrees with the committed ledger.")

    resume_schema = _state_dict_schema(checkpoint.get("model"), "resume.pt")
    best_path = output_dir / "best_auprc.pt"
    valid = False
    if best_path.is_file() and best_path.stat().st_size:
        try:
            payload = torch.load(best_path, map_location="cpu", weights_only=False)
            _validate_best_payload(
                payload,
                selected_epoch,
                selected_score,
                records[selected_epoch],
                resume_schema,
                args,
            )
            if selected_epoch == resumed_epoch:
                unequal = _unequal_state_keys(payload["model"], checkpoint["model"])
                if unequal:
                    raise RuntimeError(
                        "Current-epoch best model differs from resume.pt: "
                        f"{unequal[:5]}."
                    )
            valid = True
        except Exception:
            valid = False

    if valid:
        return
    if selected_epoch == resumed_epoch:
        # resume.pt is the commit marker and contains the exact state needed to
        # repair a crash between committing resume and replacing the best file.
        atomic_torch_save(_best_checkpoint_payload(checkpoint), best_path)
        print(f"RESUME_REPAIRED_BEST epoch={selected_epoch}", flush=True)
        return
    raise RuntimeError(
        "The committed best epoch predates resume.pt, but best_auprc.pt is missing "
        "or invalid and cannot be reconstructed from the current model."
    )


def reconcile_committed_ledger(
    output_dir: Path,
    args: argparse.Namespace,
    current_config: dict[str, Any],
    checkpoint: dict[str, Any],
    committed_epoch: int,
) -> dict[int, dict[str, Any]]:
    """Treat resume.pt as commit marker and discard only an E+1 crash tail."""

    ledger_path = output_dir / "metrics.jsonl"
    if not ledger_path.is_file() or not ledger_path.stat().st_size:
        raise RuntimeError("resume.pt exists but metrics.jsonl is missing or empty.")
    entries = _metrics_entries(ledger_path, args)
    committed_records: dict[int, dict[str, Any]] = {}
    tail_started = False
    tail_count = 0
    expected_next = 1
    for record in entries:
        epoch = int(record["epoch"])
        _validate_record_config(record, current_config, epoch)
        if epoch <= committed_epoch:
            if tail_started:
                raise RuntimeError("Committed metrics appear after an uncommitted ledger tail.")
            if epoch != expected_next:
                raise RuntimeError(
                    f"Committed ledger is duplicated/out of order: expected epoch "
                    f"{expected_next}, found {epoch}."
                )
            committed_records[epoch] = record
            expected_next += 1
        else:
            tail_started = True
            if epoch != committed_epoch + 1:
                raise RuntimeError(
                    f"Only an epoch {committed_epoch + 1} crash tail may follow "
                    f"resume epoch {committed_epoch}; found {epoch}."
                )
            tail_count += 1
    if set(committed_records) != set(range(1, committed_epoch + 1)):
        raise RuntimeError("Committed metrics are not a complete 1..resume_epoch ledger.")
    if not _json_equivalent(
        checkpoint.get("epoch_metrics"), committed_records[committed_epoch]
    ):
        raise RuntimeError("resume.pt epoch_metrics differs from its committed ledger row.")
    if not _json_equivalent(
        checkpoint.get("config"), committed_records[committed_epoch].get("config")
    ):
        raise RuntimeError("resume.pt config differs from its committed ledger row.")

    repair_or_validate_best(
        output_dir, checkpoint, committed_epoch, committed_records, args
    )
    ordered = [committed_records[epoch] for epoch in sorted(committed_records)]
    atomic_write_jsonl(ledger_path, ordered)
    atomic_write_json(
        output_dir / "latest_metrics.json", committed_records[committed_epoch]
    )
    atomic_torch_save(checkpoint["model"], output_dir / "last_model.pt")
    if tail_count:
        print(
            f"RESUME_TRIMMED_UNCOMMITTED_METRICS rows={tail_count} "
            f"epoch={committed_epoch + 1}",
            flush=True,
        )
    return committed_records


def validate_completion(
    output_dir: Path,
    args: argparse.Namespace,
    current_config: dict[str, Any],
    committed_epoch: int,
) -> dict[str, Any]:
    if args.max_steps_per_epoch or args.max_val_samples:
        raise RuntimeError("Debug-truncated runs cannot be marked completed.")
    required_paths = {
        name: output_dir / name
        for name in (
            "metrics.jsonl",
            "latest_metrics.json",
            "resume.pt",
            "last_model.pt",
            "best_auprc.pt",
        )
    }
    missing = [name for name, path in required_paths.items() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Cannot complete run; missing/empty files: {missing}.")
    entries = _metrics_entries(required_paths["metrics.jsonl"], args)
    records = _metrics_by_epoch(required_paths["metrics.jsonl"], args)
    expected_epochs = list(range(1, args.epochs + 1))
    if [int(record["epoch"]) for record in entries] != expected_epochs:
        raise RuntimeError("Metrics ledger must contain exactly one ordered row per epoch.")
    for epoch, record in records.items():
        _validate_record_config(record, current_config, epoch)
    latest = load_json(required_paths["latest_metrics.json"])
    if not _json_equivalent(latest, records[args.epochs]):
        raise RuntimeError("latest_metrics.json differs from the final ledger row.")
    if committed_epoch != args.epochs:
        raise RuntimeError(
            f"Resume checkpoint committed epoch={committed_epoch}, target={args.epochs}."
        )
    resume = torch.load(required_paths["resume.pt"], map_location="cpu", weights_only=False)
    if (
        int(resume.get("epoch", -1)) != args.epochs
        or resume.get("model_name") != MODEL_NAME
        or int(resume.get("seed", -1)) != args.seed
        or not _json_equivalent(resume.get("config"), current_config)
    ):
        raise RuntimeError("Final resume checkpoint identity/epoch/config mismatch.")
    if not _json_equivalent(resume.get("epoch_metrics"), records[args.epochs]):
        raise RuntimeError("resume.pt epoch_metrics differs from the final ledger row.")
    if not all(key in resume for key in ("model", "optimizer", "scaler", "rng_state")):
        raise RuntimeError("resume.pt lacks full continuation state.")
    resume_schema = _state_dict_schema(resume.get("model"), "resume.pt")
    last_state = torch.load(
        required_paths["last_model.pt"], map_location="cpu", weights_only=True
    )
    _state_dict_schema(last_state, "last_model.pt")
    unequal = _unequal_state_keys(last_state, resume["model"])
    if unequal:
        raise RuntimeError(f"last_model.pt differs from resume.pt: {unequal[:5]}.")
    best_epoch = max(
        records,
        key=lambda epoch: (float(records[epoch]["validation"]["auprc"]), epoch),
    )
    best = torch.load(required_paths["best_auprc.pt"], map_location="cpu", weights_only=False)
    expected_best = float(records[best_epoch]["validation"]["auprc"])
    _validate_best_payload(
        best,
        best_epoch,
        expected_best,
        records[best_epoch],
        resume_schema,
        args,
    )
    if (
        int(resume.get("best_epoch", -1)) != best_epoch
        or not math.isclose(
            float(resume.get("best_auprc", float("nan"))),
            expected_best,
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("resume.pt best metadata differs from the metrics ledger.")
    return {
        "metrics_epochs": sorted(records),
        "metrics_line_count": len(entries),
        "latest_epoch": int(latest["epoch"]),
        "committed_epoch": committed_epoch,
        "best_auprc_epoch": best_epoch,
        "best_auprc": expected_best,
        "selection_rule": "maximum validation AUPRC; ties choose latest epoch",
        "commit_marker": "resume.pt",
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = initialize_device(args)
    if args.variant in {"competence", "temporal_mil"} and args.interaction_loss_weight != 0:
        raise ValueError("These variants do not use redundancy consistency; set --interaction-loss-weight 0.")
    if args.frozen_chunk_size < 0 or args.memory_cap_gib <= 0 or args.routing_target_temperature <= 0:
        raise ValueError("Invalid chunk size, memory cap, or routing target temperature.")
    if device.type == "cuda":
        total_memory = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(args.memory_cap_gib * 1024**3 / total_memory)
    set_seed(args.seed, args.deterministic)

    manifest_sha256 = sha256_file(args.csv_path)
    manifest, manifest_summary = load_manifest(args.csv_path)
    overlap_audit = manifest_summary["inferred_parent_subject_audit"]
    if args.strict_patient_disjoint and int(overlap_audit["overlap_count"]) > 0:
        raise ValueError(
            "Parent-directory subject audit found Train/Val overlap: "
            f"{overlap_audit['overlap_ids']}."
        )
    train_set = EEG19NpyDataset(
        manifest,
        "Train",
        args.stored_prefix,
        args.data_root,
        target_samples=args.target_samples,
        eeg_scale_divisor=args.eeg_scale_divisor,
    )
    val_set = EEG19NpyDataset(
        manifest,
        "Val",
        args.stored_prefix,
        args.data_root,
        target_samples=args.target_samples,
        eeg_scale_divisor=args.eeg_scale_divisor,
    )
    del manifest
    manifest_summary["csv_sha256"] = manifest_sha256
    manifest_summary["train_preflight"] = preflight_dataset(
        train_set, args.preflight_samples
    )
    manifest_summary["val_preflight"] = preflight_dataset(val_set, args.preflight_samples)
    if int(manifest_summary["train_val_exact_path_overlap_rows"]) != 0:
        raise ValueError("Exact NPY paths overlap between Train and Val.")
    if int(overlap_audit["overlap_count"]) > 0:
        print(
            "WARNING inferred parent-directory subject overlap is present: "
            + ",".join(overlap_audit["overlap_ids"]),
            flush=True,
        )
    print("MANIFEST_JSON=" + json.dumps(json_ready(manifest_summary), sort_keys=True), flush=True)

    model = build_model(args).to(device)
    focal = MulticlassFocalLoss(args.focal_alpha, args.focal_gamma).to(device)
    if args.smoke_test:
        result = smoke_test(model, train_set, focal, args, device)
        if (not result["all_finite"] or result["missing_gradient_parameters"]
                or not result["optimizer_step_completed"]
                or result["weights_shape"] != [args.batch_size, 6]
                or not result["inference_finite"]
                or not torch.allclose(result["weights_row_sums"], torch.ones(args.batch_size))):
            raise RuntimeError("Six-expert smoke test failed.")
        print("SMOKE_TEST_JSON=" + json.dumps(json_ready(result), sort_keys=True), flush=True)
        return

    output_dir = Path(args.output_dir)
    artifacts = (
        "resume.pt",
        "last_model.pt",
        "best_auprc.pt",
        "metrics.jsonl",
        "completed.json",
    )
    if not args.resume and any((output_dir / name).exists() for name in artifacts):
        raise FileExistsError(
            f"Output directory already contains a run: {output_dir}. Use --resume or a new path."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = frozen_checkpoint_paths(args)
    source_identity = frozen_checkpoint_identity(
        checkpoint_paths, model.source_metadata
    )
    current_config = semantic_config(args, manifest_sha256, source_identity)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp and device.type == "cuda"
    )

    resumed_epoch: Optional[int] = None
    start_epoch = 1
    best_auprc = -math.inf
    best_epoch = -1
    metrics_records: dict[int, dict[str, Any]] = {}
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.resolve() != (output_dir / "resume.pt").resolve():
            raise ValueError("--resume must be this output directory's resume.pt.")
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        resumed_epoch = validate_resume(checkpoint, current_config, args)
        model.load_trainable_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint["rng_state"], device)
        start_epoch = resumed_epoch + 1
        best_auprc = float(checkpoint.get("best_auprc", -math.inf))
        best_epoch = int(checkpoint.get("best_epoch", -1))
        metrics_records = reconcile_committed_ledger(
            output_dir,
            args,
            current_config,
            checkpoint,
            resumed_epoch,
        )
        repair_best_f1(output_dir, metrics_records)
        if (output_dir / "completed.json").exists() and resumed_epoch < args.epochs:
            (output_dir / "completed.json").unlink()

    atomic_write_json(output_dir / "args.json", vars(args))
    atomic_write_json(output_dir / "manifest_summary.json", manifest_summary)
    run_summary = {
        "model": MODEL_NAME,
        "seed": args.seed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "world_size": 1,
        "batch_size_per_process": args.batch_size,
        "global_batch_size": args.batch_size,
        "epochs": args.epochs,
        "architecture": model.architecture_summary(),
        "sampling_plan": sampling_plan(
            train_set.labels,
            args.samples_per_epoch,
            args.ratio,
            args.sampling,
            args.batch_size,
        ),
        "loss": {
            "task": args.task_loss,
            "focal_positive_alpha": args.focal_alpha,
            "focal_gamma": args.focal_gamma,
            "final_router_task_weight": 1.0,
            "auxiliary_task": "focal loss on the redundancy expert",
            "auxiliary_task_weight": args.auxiliary_task_loss_weight,
            "frozen_expert_auxiliary_task_losses": False,
            "interaction": "mean redundancy cosine distance to five perturbed views",
            "interaction_space": "128d Transformer fused features",
            "interaction_weight": args.interaction_loss_weight,
            "trainable_expert_task_losses": ["redundancy"],
            "outer_router_balance_or_entropy_loss": False,
        },
        "comparison_contract": {
            "input": "raw NPY EEG channels 0..18 only",
            "ecg": False,
            "emg": False,
            "teachers": False,
            "distillation": False,
            "final_moe_distillation": False,
            "frozen_brain_region_experts_distilled": True,
            "num_brain_region_modalities": 5,
            "brain_region_encoder": FROZEN_EXPERT_ENCODER_NAME,
            "brain_region_encoder_weights": FROZEN_EXPERT_ENCODER_WEIGHTS,
            "brain_region_encoders": 5,
            "brain_region_encoder_parameters_shared": False,
            "brain_region_experts_frozen": True,
            "same_seed_source_checkpoint_pairing": False,
            "per_region_selected_source_checkpoint_pairing": True,
            "frozen_sources_shared_across_training_seeds": True,
            "num_interaction_experts": 1,
            "outer_router_input_experts": 6,
            "outer_router_input_shape": "[B,6,128]",
            "outer_router_includes_interaction_features": True,
            "interaction_features_detached_before_router": True,
            "single_gpu": True,
        },
        "config": current_config,
    }
    run_summary["loss"].update(
        variant=args.variant,
        auxiliary_task="all six calibrated experts" if args.variant == "competence" else "sixth expert only",
        interaction="redundancy consistency" if args.variant in {"baseline", "sampling"} else "disabled",
        trainable_expert_task_losses=list(EXPERT_NAMES) if args.variant == "competence" else [args.variant if args.variant == "temporal_mil" else "redundancy"],
        routing_weight=args.routing_loss_weight if args.variant == "competence" else 0.0,
        regional_mass_weight=args.regional_mass_loss_weight if args.variant == "competence" else 0.0,
        regional_mass_floor=args.regional_mass_floor if args.variant == "competence" else None,
    )
    run_summary["comparison_contract"].update(
        variant=args.variant, sixth_expert=args.variant if args.variant == "temporal_mil" else "redundancy",
        trainable_regional_adapters=args.variant == "competence",
        allocator_cap_gib=args.memory_cap_gib,
    )
    atomic_write_json(output_dir / "run_summary.json", run_summary)
    print("RUN_JSON=" + json.dumps(json_ready(run_summary), sort_keys=True), flush=True)

    committed_epoch = resumed_epoch or 0
    for epoch in range(start_epoch, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.monotonic()
        training = train_one_epoch(
            model, train_set, optimizer, scaler, focal, args, device, epoch
        )
        validation = evaluate(model, val_set, focal, args, device, epoch)
        epoch_metrics = {
            "model": MODEL_NAME,
            "seed": args.seed,
            "epoch": epoch,
            "config": current_config,
            "train": training,
            "validation": validation,
            "epoch_seconds": time.monotonic() - epoch_started,
            "peak_gpu_gib": (
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda"
                else 0.0
            ),
        }
        current_auprc = float(validation["auprc"])
        is_best = current_auprc >= best_auprc
        if is_best:
            best_auprc = current_auprc
            best_epoch = epoch
        checkpoint = {
            "model_name": MODEL_NAME,
            "seed": args.seed,
            "epoch": epoch,
            "model": model.trainable_state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "rng_state": capture_rng_state(),
            "config": current_config,
            "args": vars(args),
            "best_auprc": best_auprc,
            "best_epoch": best_epoch,
            "validation": validation,
            "epoch_metrics": epoch_metrics,
        }
        # resume.pt is the sole commit marker. The ledger is atomically
        # de-duplicated on every epoch. If a crash happens before resume.pt is
        # replaced, resume reconciliation discards only the physical E+1 tail;
        # if it happens just after resume, that payload can repair the best file.
        atomic_torch_save(_best_checkpoint_payload(checkpoint), output_dir / f"epoch{epoch:02d}.pt")
        metrics_records[epoch] = epoch_metrics
        atomic_write_jsonl(
            output_dir / "metrics.jsonl",
            [metrics_records[index] for index in sorted(metrics_records)],
        )
        atomic_write_json(output_dir / "latest_metrics.json", epoch_metrics)
        atomic_torch_save(model.trainable_state_dict(), output_dir / "last_model.pt")
        atomic_torch_save(checkpoint, output_dir / "resume.pt")
        committed_epoch = epoch
        if is_best:
            atomic_torch_save(
                _best_checkpoint_payload(checkpoint),
                output_dir / "best_auprc.pt",
            )
        selected_f1_epoch = max(metrics_records, key=lambda e: (float(metrics_records[e]["validation"]["best_f1_on_validation_grid"]["f1"]), e))
        if selected_f1_epoch == epoch:
            f1_payload = _best_checkpoint_payload(checkpoint)
            f1_payload.update(selection_metric="best_f1", selected_f1=float(validation["best_f1_on_validation_grid"]["f1"]), selected_f1_epoch=epoch)
            atomic_torch_save(f1_payload, output_dir / "best_f1.pt")
        print("RESULT_JSON=" + json.dumps(json_ready(epoch_metrics), sort_keys=True), flush=True)

    repair_best_f1(output_dir, metrics_records)
    if args.max_steps_per_epoch or args.max_val_samples:
        atomic_write_json(output_dir / "debug_finished.json", {"status": "debug_only", "epoch": committed_epoch})
        return
    completion = validate_completion(
        output_dir, args, current_config, committed_epoch=committed_epoch
    )
    atomic_write_json(
        output_dir / "completed.json",
        {
            "status": "completed",
            "model": MODEL_NAME,
            "seed": args.seed,
            "epochs": args.epochs,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "validation": completion,
        },
    )
    print("COMPLETED_JSON=" + json.dumps(json_ready(completion), sort_keys=True), flush=True)


def repair_best_f1(output_dir: Path, records: dict) -> None:
    """Rebuild F1 selection from committed epoch snapshots, including on resume."""
    selected = max(records, key=lambda e: (float(records[e]["validation"]["best_f1_on_validation_grid"]["f1"]), e))
    payload = torch.load(output_dir / f"epoch{selected:02d}.pt", map_location="cpu", weights_only=False)
    if payload["epoch"] != selected or not _json_equivalent(payload["epoch_metrics"], records[selected]):
        raise RuntimeError("F1 epoch snapshot does not match committed metrics.")
    payload.update(selection_metric="best_f1", selected_f1_epoch=selected,
                   selected_f1=float(records[selected]["validation"]["best_f1_on_validation_grid"]["f1"]))
    atomic_torch_save(payload, output_dir / "best_f1.pt")


if __name__ == "__main__":
    main()
