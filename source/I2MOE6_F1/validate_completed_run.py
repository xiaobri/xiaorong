#!/usr/bin/env python3
"""Strictly validate one completed formal I2MOE9E seed directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torchvision

from i2moe_eeg.data import DEFAULT_CSV_PATH, DEFAULT_DATA_ROOT, DEFAULT_STORED_PREFIX
from i2moe_eeg.model import (
    DISTILLATION_HARD_WEIGHT,
    DISTILLATION_KD_WEIGHT,
    DISTILLATION_TEACHER_AGGREGATION,
    DISTILLATION_TEACHER_TABLE_SHA256,
    DISTILLATION_TEMPERATURE,
    FROZEN_EXPERT_BEST_AUPRC,
    FROZEN_EXPERT_BEST_EPOCHS,
    FROZEN_EXPERT_BEST_SEEDS,
    FROZEN_EXPERT_ENCODER_NAME,
    FROZEN_EXPERT_ENCODER_WEIGHTS,
    default_frozen_checkpoint_paths,
)
from i2moe_eeg.regions import EXPERT_NAMES, region_metadata


MODEL_NAME = "eeg19_distilled_frozen5_redundancy_router6_detach"
REQUIRED_ARTIFACTS = (
    "args.json",
    "run_summary.json",
    "manifest_summary.json",
    "metrics.jsonl",
    "latest_metrics.json",
    "completed.json",
    "resume.pt",
    "last_model.pt",
    "best_auprc.pt",
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is non-finite: {number}.")
    return number


def _state_dict_schema(state: Any, label: str) -> dict[str, tuple[int, ...]]:
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{label} has no non-empty model state.")
    invalid = [name for name, value in state.items() if not isinstance(value, torch.Tensor)]
    if invalid:
        raise ValueError(f"{label} has non-tensor entries: {invalid[:5]}.")
    return {name: tuple(value.shape) for name, value in state.items()}


def _validate_trainable_schema(
    schema: dict[str, tuple[int, ...]], label: str
) -> None:
    if any(name.startswith("frozen_experts.") for name in schema):
        raise ValueError(f"{label} illegally duplicates frozen source weights.")
    if any(not name.startswith(("redundancy_expert.", "router.")) for name in schema):
        raise ValueError(f"{label} contains weights outside the redundancy expert and router.")
    required = {
        "redundancy_expert.position_embedding": (1, 5, 128),
        "redundancy_expert.head.2.weight": (2, 128),
        "router.network.0.weight": (128, 768),
        "router.network.3.weight": (6, 128),
    }
    mismatches = {
        name: {"observed": schema.get(name), "expected": shape}
        for name, shape in required.items()
        if schema.get(name) != shape
    }
    if mismatches:
        raise ValueError(f"{label} trainable I2MOE9E schema mismatch: {mismatches}.")


def _validate_source_identity(config: dict[str, Any]) -> None:
    identity = config.get("frozen_checkpoint_identity")
    defaults = default_frozen_checkpoint_paths()
    if not isinstance(identity, dict) or set(identity) != set(defaults):
        raise ValueError("Frozen source identity set mismatch.")
    for name, expected_path in defaults.items():
        metadata = identity.get(name, {})
        path = Path(expected_path).resolve()
        if metadata.get("path") != str(path):
            raise ValueError(f"{name} source path mismatch.")
        if int(metadata.get("seed", -1)) != FROZEN_EXPERT_BEST_SEEDS[name]:
            raise ValueError(f"{name} is not paired to its selected seed.")
        if int(metadata.get("epoch", -1)) != int(metadata.get("best_epoch", -2)):
            raise ValueError(f"{name} source is not its selected best epoch.")
        if int(metadata.get("best_epoch", -1)) != FROZEN_EXPERT_BEST_EPOCHS[name]:
            raise ValueError(f"{name} source best epoch is not the recorded choice.")
        if not math.isclose(
            float(metadata.get("best_auprc", float("nan"))),
            FROZEN_EXPERT_BEST_AUPRC[name],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{name} source AUPRC is not the recorded choice.")
        if (
            metadata.get("distillation_enabled") is not True
            or not math.isclose(
                float(metadata.get("distillation_hard_weight", float("nan"))),
                DISTILLATION_HARD_WEIGHT,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(metadata.get("distillation_kd_weight", float("nan"))),
                DISTILLATION_KD_WEIGHT,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(metadata.get("distillation_temperature", float("nan"))),
                DISTILLATION_TEMPERATURE,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or metadata.get("teacher_aggregation")
            != DISTILLATION_TEACHER_AGGREGATION
            or metadata.get("teacher_table_sha256")
            != DISTILLATION_TEACHER_TABLE_SHA256
        ):
            raise ValueError(f"{name} source distillation metadata mismatch.")
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{name} source checkpoint is missing/empty: {path}.")
        if int(metadata.get("size_bytes", -1)) != path.stat().st_size:
            raise ValueError(f"{name} source checkpoint size changed.")
        observed_hash = metadata.get("sha256")
        if not isinstance(observed_hash, str) or len(observed_hash) != 64:
            raise ValueError(f"{name} lacks a source SHA-256.")
        if _sha256_file(path) != observed_hash:
            raise ValueError(f"{name} source checkpoint content changed.")
        _finite_number(metadata.get("best_auprc"), f"{name} source AUPRC")


def validate_completed_run(
    run_dir: Path,
    expected_seed: int,
    expected_epochs: int = 20,
    expected_batch_size: int = 128,
    expected_eval_batch_size: int = 128,
    expected_lr: float = 5e-4,
    expected_weight_decay: float = 1e-2,
    expected_auxiliary_task_weight: float = 0.05,
    expected_interaction_weight: float = 0.005,
) -> dict[str, Any]:
    missing = [
        name
        for name in REQUIRED_ARTIFACTS
        if not (run_dir / name).is_file() or (run_dir / name).stat().st_size == 0
    ]
    if missing:
        raise ValueError(f"Missing/empty required artifacts: {missing}.")

    args = _load_json(run_dir / "args.json")
    expected_args = {
        "csv_path": DEFAULT_CSV_PATH,
        "data_root": DEFAULT_DATA_ROOT,
        "stored_prefix": DEFAULT_STORED_PREFIX,
        "seed": expected_seed,
        "epochs": expected_epochs,
        "batch_size": expected_batch_size,
        "eval_batch_size": expected_eval_batch_size,
        "num_workers": 2,
        "lr": expected_lr,
        "weight_decay": expected_weight_decay,
        "auxiliary_task_loss_weight": expected_auxiliary_task_weight,
        "interaction_loss_weight": expected_interaction_weight,
        "target_samples": 2000,
        "eeg_scale_divisor": 0.001,
        "strict_patient_disjoint": False,
        "task_loss": "focal",
        "focal_alpha": 0.85,
        "focal_gamma": 2.0,
        "sampling": "match_cam",
        "samples_per_epoch": 20_000,
        "ratio": 30,
        "hidden_dim": 128,
        "num_heads": 4,
        "fusion_layers": 2,
        "feedforward_multiplier": 2,
        "dropout": 0.3,
        "router_hidden_dim": 128,
        "router_layers": 2,
        "router_dropout": 0.5,
        "router_temperature": 2.0,
        "replacement_std": 1.0,
        "frontal_checkpoint": "",
        "left_temporal_checkpoint": "",
        "right_temporal_checkpoint": "",
        "centro_parietal_checkpoint": "",
        "occipital_checkpoint": "",
        "device": "cuda",
        "amp": False,
        "deterministic": False,
        "max_steps_per_epoch": 0,
        "max_val_samples": 0,
        "threshold_start": 0.05,
        "threshold_end": 0.95,
        "threshold_step": 0.05,
    }
    mismatches = {
        key: {"observed": args.get(key), "expected": expected}
        for key, expected in expected_args.items()
        if args.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Formal configuration mismatch: {mismatches}.")

    summary = _load_json(run_dir / "run_summary.json")
    if summary.get("model") != MODEL_NAME or int(summary.get("world_size", -1)) != 1:
        raise ValueError("run_summary model/world_size mismatch.")
    architecture = summary.get("architecture", {})
    required_architecture = {
        "model": MODEL_NAME,
        "experts": list(EXPERT_NAMES),
        "num_regions": 5,
        "num_experts": 6,
        "hidden_dim": 128,
        "frozen_expert_encoder": FROZEN_EXPERT_ENCODER_NAME,
        "frozen_expert_source": FROZEN_EXPERT_ENCODER_WEIGHTS,
        "frozen_expert_selection": (
            "user-confirmed fixed per-region distilled checkpoints"
        ),
        "frozen_experts_distilled": True,
        "distillation_teacher_table_sha256": DISTILLATION_TEACHER_TABLE_SHA256,
        "frozen_expert_best_seeds": dict(FROZEN_EXPERT_BEST_SEEDS),
        "frozen_sources_shared_across_training_seeds": True,
        "frozen_experts_receive_all_19_channels": True,
        "frozen_experts_eval_mode": True,
        "frozen_experts_fully_frozen": True,
        "interaction_experts": ["redundancy"],
        "interaction_architecture": "redundancy_transformer_encoder",
        "interaction_tokens": "five_frozen_128d_regional_expert_features",
        "interaction_loss_space": "128d_transformer_fused_features",
        "outer_router": "dense_temperature_softmax",
        "outer_router_input": "all_six_expert_128d_representations",
        "outer_router_input_experts": list(EXPERT_NAMES),
        "outer_router_input_shape": "[B,6,128]",
        "outer_router_input_dim": 768,
        "outer_router_includes_interaction_features": True,
        "interaction_features_detached_before_router": True,
        "top_k_routing": False,
        "checkpoint_contains_frozen_source_weights": False,
    }
    architecture_mismatches = {
        key: {"observed": architecture.get(key), "expected": expected}
        for key, expected in required_architecture.items()
        if architecture.get(key) != expected
    }
    if architecture_mismatches:
        raise ValueError(
            f"run_summary architecture mismatch: {architecture_mismatches}."
        )
    if int(architecture.get("parameters_trainable", 0)) <= 0:
        raise ValueError("run_summary reports no trainable parameters.")
    software = summary.get("software", {})
    if software.get("torchvision") != torchvision.__version__:
        raise ValueError("run_summary torchvision version differs from validator.")
    contract = summary.get("comparison_contract", {})
    if (
        contract.get("ecg") is not False
        or contract.get("emg") is not False
        or contract.get("brain_region_experts_frozen") is not True
        or contract.get("same_seed_source_checkpoint_pairing") is not False
        or contract.get("per_region_selected_source_checkpoint_pairing") is not True
        or contract.get("frozen_sources_shared_across_training_seeds") is not True
        or contract.get("frozen_brain_region_experts_distilled") is not True
        or contract.get("final_moe_distillation") is not False
        or int(contract.get("outer_router_input_experts", -1)) != 6
        or contract.get("outer_router_input_shape") != "[B,6,128]"
        or contract.get("outer_router_includes_interaction_features") is not True
        or contract.get("interaction_features_detached_before_router") is not True
    ):
        raise ValueError("Run comparison contract mismatch.")

    config = summary.get("config", {})
    if config.get("model") != MODEL_NAME or config.get("region_contract") != region_metadata():
        raise ValueError("run_summary semantic model/region contract mismatch.")
    if (
        config.get("frozen_expert_best_source_seeds")
        != FROZEN_EXPERT_BEST_SEEDS
        or config.get("frozen_expert_best_source_epochs")
        != FROZEN_EXPERT_BEST_EPOCHS
        or config.get("frozen_expert_best_source_auprc")
        != FROZEN_EXPERT_BEST_AUPRC
        or config.get("frozen_sources_shared_across_training_seeds") is not True
        or config.get("frozen_experts_distilled") is not True
        or config.get("outer_router_input")
        != "all_six_expert_128d_representations"
        or int(config.get("outer_router_input_dim", -1)) != 768
        or config.get("interaction_features_detached_before_router") is not True
    ):
        raise ValueError("Frozen selected-source/router metadata mismatch.")
    expected_source_distillation = {
        "hard_weight": DISTILLATION_HARD_WEIGHT,
        "kd_weight": DISTILLATION_KD_WEIGHT,
        "temperature": DISTILLATION_TEMPERATURE,
        "teacher_aggregation": DISTILLATION_TEACHER_AGGREGATION,
        "teacher_table_sha256": DISTILLATION_TEACHER_TABLE_SHA256,
    }
    if config.get("frozen_source_distillation") != expected_source_distillation:
        raise ValueError("Frozen source distillation contract mismatch.")
    _validate_source_identity(config)
    source_summary = architecture.get("frozen_source_checkpoints", {})
    for name, metadata in config["frozen_checkpoint_identity"].items():
        for key in (
            "path",
            "source_model",
            "seed",
            "epoch",
            "best_epoch",
            "best_auprc",
            "distillation_enabled",
            "distillation_temperature",
            "distillation_hard_weight",
            "distillation_kd_weight",
            "teacher_aggregation",
            "teacher_table_sha256",
        ):
            if source_summary.get(name, {}).get(key) != metadata.get(key):
                raise ValueError(f"Architecture/config source metadata mismatch for {name}/{key}.")

    loss_contract = summary.get("loss", {})
    if (
        float(loss_contract.get("final_router_task_weight", -1)) != 1.0
        or float(loss_contract.get("auxiliary_task_weight", -1))
        != expected_auxiliary_task_weight
        or float(loss_contract.get("interaction_weight", -1))
        != expected_interaction_weight
        or loss_contract.get("interaction_space")
        != "128d Transformer fused features"
        or loss_contract.get("trainable_expert_task_losses")
        != ["redundancy"]
        or loss_contract.get("frozen_expert_auxiliary_task_losses") is not False
        or loss_contract.get("outer_router_balance_or_entropy_loss") is not False
    ):
        raise ValueError("run_summary loss contract mismatch.")

    manifest = _load_json(run_dir / "manifest_summary.json")
    if int(manifest.get("train_val_exact_path_overlap_rows", -1)) != 0:
        raise ValueError("Manifest reports exact Train/Val path overlap.")
    if manifest.get("input_contract") != region_metadata():
        raise ValueError("Manifest brain-region/input contract mismatch.")
    manifest_sha256 = manifest.get("csv_sha256")
    if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64:
        raise ValueError("Manifest lacks a valid CSV SHA-256.")
    if config.get("csv_sha256") != manifest_sha256:
        raise ValueError("run_summary and manifest CSV identities differ.")

    records: dict[int, dict[str, Any]] = {}
    ordered_epochs: list[int] = []
    with (run_dir / "metrics.jsonl").open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed metrics line {line_number}.") from exc
            if record.get("model") != MODEL_NAME or int(record.get("seed", -1)) != expected_seed:
                raise ValueError(f"Foreign metrics record at line {line_number}.")
            epoch = int(record.get("epoch", -1))
            if record.get("config") != config:
                raise ValueError(f"Epoch {epoch} semantic config mismatch.")
            validation = record.get("validation", {})
            _finite_number(validation.get("auprc"), f"epoch {epoch} AUPRC")
            _finite_number(validation.get("auroc"), f"epoch {epoch} AUROC")
            if int(validation.get("samples", -1)) != 166_079:
                raise ValueError(f"Epoch {epoch} did not evaluate the complete Val split.")
            if validation.get("interaction_forward_used") is not False:
                raise ValueError(f"Epoch {epoch} validation used corrupted views.")
            weights = validation.get("interaction_weights", {}).get("mean", {})
            experts = validation.get("expert_metrics", {})
            if set(weights) != set(EXPERT_NAMES) or set(experts) != set(EXPERT_NAMES):
                raise ValueError(f"Epoch {epoch} lacks all six expert diagnostics.")
            training = record.get("train", {})
            if int(training.get("samples", -1)) != 20_000 or int(
                training.get("steps", -1)
            ) != 157:
                raise ValueError(f"Epoch {epoch} training sample/step mismatch.")
            sampling = training.get("sampling", {})
            if int(sampling.get("samples", -1)) != 20_000 or int(
                sampling.get("steps", -1)
            ) != 157:
                raise ValueError(f"Epoch {epoch} sampler ledger mismatch.")
            for key in (
                "loss",
                "final_task_loss",
                "auxiliary_task_loss",
                "interaction_loss",
            ):
                _finite_number(training.get(key), f"epoch {epoch} train {key}")
            if set(training.get("auxiliary_task_components", {})) != {
                "redundancy",
            } or set(training.get("interaction_components", {})) != {
                "redundancy",
            }:
                raise ValueError(f"Epoch {epoch} interaction loss ledger mismatch.")
            if epoch in records:
                raise ValueError(f"Duplicate metrics epoch {epoch}.")
            records[epoch] = record
            ordered_epochs.append(epoch)

    expected_order = list(range(1, expected_epochs + 1))
    if ordered_epochs != expected_order:
        raise ValueError("Metrics ledger must contain exactly one ordered row per epoch.")
    latest = _load_json(run_dir / "latest_metrics.json")
    if latest != records[expected_epochs]:
        raise ValueError("latest_metrics.json differs from final metrics row.")
    completed = _load_json(run_dir / "completed.json")
    if (
        completed.get("status") != "completed"
        or completed.get("model") != MODEL_NAME
        or int(completed.get("seed", -1)) != expected_seed
        or int(completed.get("epochs", -1)) != expected_epochs
    ):
        raise ValueError("completed.json identity/status mismatch.")

    best_epoch = max(
        records,
        key=lambda epoch: (float(records[epoch]["validation"]["auprc"]), epoch),
    )
    best_auprc = float(records[best_epoch]["validation"]["auprc"])
    resume = torch.load(run_dir / "resume.pt", map_location="cpu", weights_only=False)
    best = torch.load(run_dir / "best_auprc.pt", map_location="cpu", weights_only=False)
    if (
        int(resume.get("epoch", -1)) != expected_epochs
        or resume.get("model_name") != MODEL_NAME
        or int(resume.get("seed", -1)) != expected_seed
        or resume.get("config") != config
        or resume.get("epoch_metrics") != records[expected_epochs]
    ):
        raise ValueError("resume.pt identity/config/ledger mismatch.")
    if not all(key in resume for key in ("model", "optimizer", "scaler", "rng_state")):
        raise ValueError("resume.pt lacks full continuation state.")
    if (
        int(best.get("epoch", -1)) != best_epoch
        or int(best.get("best_epoch", -1)) != best_epoch
        or best.get("model_name") != MODEL_NAME
        or int(best.get("seed", -1)) != expected_seed
        or best.get("config") != config
        or best.get("epoch_metrics") != records[best_epoch]
        or not math.isclose(
            float(best.get("best_auprc", float("nan"))),
            best_auprc,
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("best_auprc.pt selection/config/ledger mismatch.")
    if (
        int(resume.get("best_epoch", -1)) != best_epoch
        or not math.isclose(
            float(resume.get("best_auprc", float("nan"))),
            best_auprc,
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("resume.pt best metadata mismatch.")

    resume_state = resume.get("model", {})
    resume_schema = _state_dict_schema(resume_state, "resume.pt")
    _validate_trainable_schema(resume_schema, "resume.pt")
    last_state = torch.load(run_dir / "last_model.pt", map_location="cpu", weights_only=True)
    if _state_dict_schema(last_state, "last_model.pt") != resume_schema:
        raise ValueError("last_model.pt and resume.pt schemas differ.")
    if _state_dict_schema(best.get("model"), "best_auprc.pt") != resume_schema:
        raise ValueError("best_auprc.pt and resume.pt schemas differ.")
    unequal = [name for name in last_state if not torch.equal(last_state[name], resume_state[name])]
    if unequal:
        raise ValueError(f"last_model.pt differs from resume.pt: {unequal[:5]}.")

    proof = completed.get("validation", {})
    if (
        int(proof.get("latest_epoch", -1)) != expected_epochs
        or int(proof.get("metrics_line_count", -1)) != expected_epochs
        or int(proof.get("best_auprc_epoch", -1)) != best_epoch
        or not math.isclose(
            float(proof.get("best_auprc", float("nan"))),
            best_auprc,
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("completed.json validation proof mismatch.")

    return {
        "status": "completed",
        "run_dir": str(run_dir.resolve()),
        "model": MODEL_NAME,
        "seed": expected_seed,
        "epochs": expected_epochs,
        "best_auprc_epoch": best_epoch,
        "best_auprc": best_auprc,
        "latest_auprc": float(records[expected_epochs]["validation"]["auprc"]),
        "metrics_line_count": len(ordered_epochs),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-seed", type=int, required=True)
    parser.add_argument("--expected-epochs", type=int, default=20)
    parser.add_argument("--expected-batch-size", type=int, default=128)
    parser.add_argument("--expected-eval-batch-size", type=int, default=128)
    parser.add_argument("--expected-lr", type=float, default=5e-4)
    parser.add_argument("--expected-weight-decay", type=float, default=1e-2)
    parser.add_argument("--expected-auxiliary-task-weight", type=float, default=0.05)
    parser.add_argument("--expected-interaction-weight", type=float, default=0.005)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = validate_completed_run(
            args.run_dir,
            args.expected_seed,
            expected_epochs=args.expected_epochs,
            expected_batch_size=args.expected_batch_size,
            expected_eval_batch_size=args.expected_eval_batch_size,
            expected_lr=args.expected_lr,
            expected_weight_decay=args.expected_weight_decay,
            expected_auxiliary_task_weight=args.expected_auxiliary_task_weight,
            expected_interaction_weight=args.expected_interaction_weight,
        )
    except Exception as exc:
        if not args.quiet:
            print(f"INVALID: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    if not args.quiet:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
