#!/usr/bin/env python3
"""Run one controlled global-role arm using the original training/commit loop."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (ROOT, ROOT / "source", ROOT / "source" / "I2MOE6_F1"):
    sys.path.insert(0, str(directory))

from experiments.global_role_ablation.protocol import (
    ARMS, VERSION, canonical_sha256, run_directory, validate_formal,
)


def source_hashes():
    paths = [HERE / name for name in ("run.py", "model.py", "protocol.py")]
    paths += [ROOT / "source" / "czb_paths.py", ROOT / "source" / "I2MOE6_F1" / "train_eeg19_i2moe.py"]
    paths += sorted((ROOT / "source" / "I2MOE6_F1" / "i2moe_eeg").glob("*.py"))
    return {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def parse_flags(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--output-root", default="private-runs/global_role_ablation")
    parser.add_argument("--mode", choices=("formal", "debug", "smoke"), default="formal")
    parser.add_argument("--resume-run", action="store_true")
    flags, remaining = parser.parse_known_args(argv)
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    return flags, remaining


def prepare_args(original, flags, remaining):
    # Importantly, the original script's default grid is 19 thresholds. The
    # released F1 experiment instead uses these 99 thresholds in every arm.
    previous = sys.argv
    sys.argv = [str(HERE / "run.py"), "--threshold-start", "0.01",
                "--threshold-end", "0.99", "--threshold-step", "0.01"] + remaining
    try:
        args = original.parse_args()
    finally:
        sys.argv = previous
    if args.resume or args.output_dir != original.DEFAULT_OUTPUT_DIR:
        raise ValueError("Use --output-root and --resume-run; do not pass --output-dir/--resume.")
    args.arm = flags.arm
    args.output_dir = str(run_directory(flags.output_root, flags.arm, args.seed))
    args.resume = str(Path(args.output_dir) / "resume.pt") if flags.resume_run else ""
    if args.resume and not Path(args.resume).is_file():
        raise FileNotFoundError(f"No committed checkpoint to resume: {args.resume}")
    if flags.mode == "formal":
        validate_formal(args)
    elif flags.mode == "debug":
        args.epochs = 1
        args.max_steps_per_epoch = args.max_steps_per_epoch or 2
        args.max_val_samples = args.max_val_samples or 256
    else:
        args.smoke_test = True
    original.validate_args(args)
    return args


def install(original, flags, args):
    """Patch extension points, leaving optimization/evaluation/resume untouched."""
    import numpy as np
    import torch
    from torch.utils.data import SubsetRandomSampler
    from experiments.global_role_ablation.model import GlobalRoleAblation, loss_terms

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json = original.atomic_write_json

    def immutable_json(path, value):
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"Existing experiment identity differs: {path}")
        write_json(path, value)

    frame, _ = original.load_manifest(args.csv_path)
    validation = frame.loc[frame["mode"] == "Val"]
    rows = dict(row_ids=validation["file_path"].astype(str).tolist(),
                labels=validation["target"].astype(int).tolist())
    immutable_json(output / "validation_rows.json", {**rows, "sha256": canonical_sha256(rows)})
    train_labels = frame.loc[frame["mode"] == "Train", "target"].to_numpy(dtype=np.int64)
    del frame, validation
    sample_fn = original.sampled_indices

    def sample_record(labels, count, ratio, sampling, batch_size, seed, epoch):
        indices, stats = sample_fn(labels, count, ratio, sampling, batch_size, seed, epoch)
        generator = torch.Generator().manual_seed(seed + epoch * 10000)
        order = np.asarray(list(SubsetRandomSampler(indices.tolist(), generator=generator)), dtype="<i8")
        record = dict(epoch=epoch, **stats,
                      sampled_indices_sha256=hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
                      actual_batch_order_sha256=hashlib.sha256(order.tobytes()).hexdigest())
        return indices, stats, record

    sampling_records = []
    for epoch in range(1, args.epochs + 1):
        _, stats, record = sample_record(train_labels, args.samples_per_epoch, args.ratio,
                                         args.sampling, args.batch_size, args.seed, epoch)
        if flags.mode == "formal" and stats["samples"] != 20000:
            raise ValueError("Training split has too few examples for 20,000 match_cam draws.")
        sampling_records.append(record)
    immutable_json(output / "sampling_plan.json", sampling_records)
    del train_labels
    code = source_hashes()
    base_contract = dict(protocol_version=VERSION, arm=args.arm, seed=args.seed,
                         formal=flags.mode == "formal", mode=flags.mode,
                         source_sha256=code, sampling_sha256=canonical_sha256(sampling_records),
                         validation_rows_sha256=canonical_sha256(rows),
                         label_use="E auxiliary weighting uses training labels only",
                         independent_test=False, patient_level_split_verified=False,
                         threshold_rule="p > t; within-epoch tie lower t; across-epoch tie later epoch")
    # Smoke mode exits before semantic_config is called. Persist its identity
    # here too, so it cannot be silently promoted into a formal run directory.
    immutable_json(output / "launch_contract.json", base_contract)
    original.parse_args = lambda: args
    original.SixExpertF1 = lambda **kwargs: GlobalRoleAblation(arm=args.arm, **kwargs)
    original.loss_terms = loss_terms
    original.MODEL_NAME = "global_role_ablation"
    build = original.build_model
    state_identity = {}

    def checked_build(actual_args):
        model = build(actual_args)
        digest = hashlib.sha256()
        for name, value in sorted(model.trainable_state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        state_identity["initial_state_sha256"] = digest.hexdigest()
        state_identity["allocated_trainable_parameters"] = sum(p.numel() for p in model.trainable_parameters())
        immutable_json(output / "initialization.json", state_identity)
        return model

    original.build_model = checked_build
    semantic = original.semantic_config

    def checked_config(*values):
        config = semantic(*values)
        # Compare content identities, not location strings, across runs/machines.
        shared = {key: value for key, value in config.items()
                  if key not in {"seed", "csv_path", "data_root", "stored_prefix", "frozen_checkpoint_identity"}}
        shared["frozen_checkpoint_sha256"] = {
            key: value["sha256"] for key, value in config["frozen_checkpoint_identity"].items()}
        shared["source_sha256"] = code
        shared["torch_version"] = torch.__version__
        shared["numpy_version"] = np.__version__
        contract = {**base_contract, **state_identity, "shared_protocol_sha256": canonical_sha256(shared)}
        immutable_json(output / "protocol.json", contract)
        config["global_role"] = contract
        config["outer_router_input"] = ("five_frozen_features_plus_zero_global_token"
                                        if args.arm in {"A", "C"} else "all_six_expert_128d_representations")
        config["global_prediction_enabled"] = args.arm not in {"A", "B"}
        return config

    original.semantic_config = checked_config

    def audited_sample(*values):
        indices, stats, record = sample_record(*values)
        if record != sampling_records[record["epoch"] - 1]:
            raise RuntimeError("Actual training sampler differs from the preregistered plan.")
        write_json(output / f"sampling_epoch{record['epoch']:02d}.json", record)
        return indices, stats

    original.sampled_indices = audited_sample
    evaluate = original.evaluate

    def f1_evaluate(*values):
        metrics = evaluate(*values)
        metrics["selection_metric"] = "best_f1_on_validation_grid"
        metrics["independent_test"] = False
        return metrics

    original.evaluate = f1_evaluate

    def accurate_summary(path, value):
        if Path(path).name == "run_summary.json":
            value["comparison_contract"].update(
                global_role_arm=args.arm,
                outer_router_includes_interaction_features=args.arm not in {"A", "C"},
                global_prediction_enabled=args.arm not in {"A", "B"},
                global_gate_mask_before_softmax=args.arm in {"A", "B"},
                inactive_global_auxiliary_control=args.arm == "A",
            )
            value["loss"]["auxiliary_task"] = (
                "normalized empirical gap weighted focal on global expert"
                if args.arm in {"E", "E_mean"} else "focal on global expert")
        write_json(path, value)

    original.atomic_write_json = accurate_summary


def main(argv=None):
    flags, remaining = parse_flags(argv)
    import train_eeg19_i2moe as original
    args = prepare_args(original, flags, remaining)
    install(original, flags, args)
    original.main()


if __name__ == "__main__":
    main()
