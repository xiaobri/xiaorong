"""Audit and summarize the locked global-role experiment (NumPy/sklearn, CPU only).

All scores are validation-selected descriptive results, not independent-test scores.
The auditor reads every epoch's saved predictions before writing aggregate outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

# Load the dependency-free sibling contract even when invoked as a file from another cwd.
_contract_spec = importlib.util.spec_from_file_location("_global_role_report_contract", Path(__file__).with_name("protocol.py"))
_contract = importlib.util.module_from_spec(_contract_spec)
_contract_spec.loader.exec_module(_contract)
FORMAL = _contract.FORMAL


DEFAULT_ARMS = ("A", "B", "C", "D", "E")
DEFAULT_SEEDS = (3, 42, 0, 1, 2)
GRID = np.asarray([0.01 + i * 0.01 for i in range(99)], dtype=np.float64)
SCORE_FIELDS = ("f1", "precision", "sensitivity", "ap", "brier", "log_loss")
METRIC_ATOL = 1e-10
EXPERT_ATOL = 1e-6
EXPERT_RTOL = 1e-5
WEIGHT_ATOL = 1e-5


class AuditError(ValueError):
    """An incomplete or incomparable experiment must not produce a result table."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()


def _no_constant(value):
    raise AuditError(f"Non-finite JSON constant: {value}")


def read_json(path, expected_type=dict):
    require(path.is_file(), f"Missing required artifact: {path}")
    try:
        result = json.loads(path.read_text(encoding="utf-8"), parse_constant=_no_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"Unreadable JSON: {path}") from exc
    require(isinstance(result, expected_type), f"Unexpected JSON structure: {path}")
    _finite_json(result, str(path))
    return result


def _finite_json(value, context):
    if isinstance(value, dict):
        for item in value.values():
            _finite_json(item, context)
    elif isinstance(value, list):
        for item in value:
            _finite_json(item, context)
    elif isinstance(value, float):
        require(math.isfinite(value), f"Non-finite number: {context}")


def threshold_metrics(labels, probabilities, threshold):
    predicted = probabilities > threshold  # Equality is negative, matching the trainer.
    positive = labels == 1
    tp = int(np.count_nonzero(predicted & positive))
    fp = int(np.count_nonzero(predicted & ~positive))
    fn = int(np.count_nonzero(~predicted & positive))
    tn = int(len(labels) - tp - fp - fn)
    return {"threshold": float(threshold), "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def best_grid_metrics(labels, probabilities):
    """Exact strict-> grid search, with the first threshold winning an F1 tie."""
    order = np.argsort(probabilities, kind="stable")
    sorted_p = probabilities[order]
    cumulative_positive = np.concatenate(([0], np.cumsum(labels[order], dtype=np.int64)))
    negative_count = np.searchsorted(sorted_p, GRID, side="right")
    fn = cumulative_positive[negative_count]
    tp = int(labels.sum()) - fn
    fp = len(labels) - negative_count - tp
    denominator = 2 * tp + fp + fn
    values = np.divide(2 * tp, denominator, out=np.zeros(99, dtype=float), where=denominator != 0)
    return threshold_metrics(labels, probabilities, GRID[int(np.argmax(values))])


def probability_scores(labels, probabilities):
    clipped = np.clip(probabilities, np.finfo(np.float64).eps, 1 - np.finfo(np.float64).eps)
    return {"ap": float(average_precision_score(labels, probabilities)) if labels.sum() else None,
            "brier": float(np.mean((probabilities - labels) ** 2)),
            "log_loss": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)))}


def _check_metric(observed, expected, context):
    require(isinstance(observed, dict), f"Missing validation grid metrics: {context}")
    for name in ("tp", "fp", "tn", "fn"):
        require(type(observed.get(name)) is int and observed[name] == expected[name],
                f"Confusion count mismatch ({name}): {context}")
    for name in ("threshold", "f1", "precision", "sensitivity"):
        value = observed.get(name)
        require(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and math.isclose(value, expected[name], rel_tol=0, abs_tol=METRIC_ATOL),
                f"Recomputed validation metric mismatch ({name}): {context}")


def _normalized_args(args):
    # Only run identity/output location may differ. Hyperparameters and code hashes may not.
    result = {key: value for key, value in args.items()
              if key not in {"arm", "seed", "output_dir", "resume"}}
    if "global_role" in result:
        require(isinstance(result["global_role"], dict), "global_role must be a JSON object")
        result["global_role"] = {key: value for key, value in result["global_role"].items()
                                 if key not in {"arm"}}
    return result


def _hash_field(protocol, key, context):
    value = protocol.get(key)
    require(isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), f"Invalid {key}: {context}")
    return value


def _load_predictions(path, labels, arm):
    require(path.is_file(), f"Missing epoch predictions: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"labels", "probabilities", "expert_probabilities", "weights"}
            require(required <= set(archive.files), f"Missing NPZ arrays: {path}")
            raw_labels = archive["labels"]
            require(raw_labels.shape == labels.shape and np.array_equal(raw_labels, labels),
                    f"NPZ labels/row order mismatch: {path}")
            result = {key: np.asarray(archive[key], dtype=np.float64)
                      for key in required - {"labels"}}
    except (OSError, ValueError, TypeError) as exc:
        if isinstance(exc, AuditError):
            raise
        raise AuditError(f"Unreadable prediction archive: {path}") from exc
    for name, shape in (("probabilities", labels.shape), ("expert_probabilities", (len(labels), 6)),
                        ("weights", (len(labels), 6))):
        value = result[name]
        require(value.shape == shape, f"Wrong {name} shape: {path}")
        require(np.isfinite(value).all() and ((value >= 0) & (value <= 1)).all(),
                f"Non-finite/out-of-range {name}: {path}")
    require(np.allclose(result["weights"].sum(axis=1), 1, rtol=0, atol=WEIGHT_ATOL),
            f"Expert weights are not normalized: {path}")
    if arm in {"A", "B"}:
        require(np.all(result["weights"][:, 5] == 0), f"Masked global weight is nonzero: {path}")
    return result


def audit_run(folder, arm, seed, expected_epochs=20):
    """Return private arrays only to the in-process aggregator, never to report.json."""
    context = f"{arm}/seed{seed}"
    args = read_json(folder / "args.json")
    protocol = read_json(folder / "protocol.json")
    complete = read_json(folder / "completed.json")
    manifest = read_json(folder / "validation_rows.json")
    require(protocol.get("protocol_version") == "global-role-v1", f"Wrong protocol version: {context}")
    require(protocol.get("formal") is True, f"Non-formal run: {context}")
    require(protocol.get("arm") == arm and protocol.get("seed") == seed, f"Protocol identity mismatch: {context}")
    require(args.get("arm", args.get("global_role", {}).get("arm")) == arm
            and args.get("seed") == seed, f"Arguments identity mismatch: {context}")
    require(args.get("epochs") == expected_epochs, f"Wrong epoch budget: {context}")
    for key, expected in {**FORMAL, "epochs": expected_epochs}.items():
        require(args.get(key) == expected, f"Formal training configuration mismatch ({key}): {context}")
    require(complete.get("status") == "completed" and complete.get("epochs") == expected_epochs
            and complete.get("seed", seed) == seed, f"Run is not completed: {context}")
    require(not (folder / "debug_finished.json").exists(), f"Debug completion present: {context}")
    require(not args.get("max_steps_per_epoch", 0) and not args.get("max_val_samples", 0)
            and not args.get("smoke_test", False), f"Debug/truncated arguments: {context}")
    if "global_role" in args:
        require(args["global_role"].get("formal") is True, f"Arguments mark a non-formal run: {context}")
    for name, value in (("threshold_start", .01), ("threshold_end", .99), ("threshold_step", .01)):
        require(args.get(name) == value, f"Threshold grid configuration mismatch ({name}): {context}")
    for name in ("shared_protocol_sha256", "initial_state_sha256", "sampling_sha256", "validation_rows_sha256"):
        _hash_field(protocol, name, context)
    initialization = read_json(folder / "initialization.json")
    require(initialization.get("initial_state_sha256") == protocol["initial_state_sha256"],
            f"Initialization artifact digest mismatch: {context}")
    require(type(initialization.get("allocated_trainable_parameters")) is int
            and initialization["allocated_trainable_parameters"] > 0
            and initialization["allocated_trainable_parameters"] == protocol.get("allocated_trainable_parameters"),
            f"Allocated parameter count mismatch: {context}")
    plan = read_json(folder / "sampling_plan.json", list)
    require(len(plan) == expected_epochs and all(isinstance(r, dict) for r in plan)
            and [r.get("epoch") for r in plan] == list(range(1, expected_epochs + 1)),
            f"Incomplete sampling plan: {context}")
    require(canonical_sha256(plan) == protocol["sampling_sha256"], f"Sampling plan digest mismatch: {context}")
    for item in plan:
        require(item.get("samples") == 20000 and item.get("steps") == 157
                and item.get("full_batches") == 156 and item.get("final_partial_batch_size") == 32,
                f"Formal sampling count/batch mismatch: {context}")
        require(type(item.get("class_0")) is int and type(item.get("class_1")) is int
                and item["class_0"] >= 0 and item["class_1"] >= 0
                and item["class_0"] + item["class_1"] == 20000, f"Sampling class counts mismatch: {context}")
        for key in ("sampled_indices_sha256", "actual_batch_order_sha256"):
            _hash_field(item, key, context)
        require(read_json(folder / f"sampling_epoch{item['epoch']:02d}.json") == item,
                f"Actual sampling record differs from plan: {context}")
    row_ids, raw_labels = manifest.get("row_ids"), manifest.get("labels")
    require(isinstance(row_ids, list) and row_ids and all(isinstance(s, str) for s in row_ids),
            f"Missing validation row identity: {context}")
    require(isinstance(raw_labels, list) and len(raw_labels) == len(row_ids)
            and all(type(y) is int and y in (0, 1) for y in raw_labels), f"Invalid manifest labels: {context}")
    require(manifest.get("sha256") == canonical_sha256({"row_ids": row_ids, "labels": raw_labels}),
            f"Validation identity digest mismatch: {context}")
    require(protocol["validation_rows_sha256"] == manifest["sha256"],
            f"Protocol validation identity digest mismatch: {context}")
    labels = np.asarray(raw_labels, dtype=np.int64)
    require(set(np.unique(labels)) == {0, 1}, f"Validation requires both binary classes: {context}")
    metrics_path = folder / "metrics.jsonl"
    require(metrics_path.is_file(), f"Missing metrics: {context}")
    try:
        lines = metrics_path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line, parse_constant=_no_constant) for line in lines]
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"Unreadable/incomplete metrics: {context}") from exc
    require(len(records) == expected_epochs and all(isinstance(r, dict) for r in records),
            f"Incomplete epoch metrics: {context}")
    require([r.get("epoch") for r in records] == list(range(1, expected_epochs + 1)),
            f"Missing/duplicate/out-of-order epochs: {context}")
    for record in records:
        _finite_json(record, context)
        require(record.get("model") == "global_role_ablation" and record.get("seed") == seed,
                f"Epoch model/seed mismatch: {context}")
        config = record.get("config")
        require(isinstance(config, dict) and config.get("global_role") == protocol,
                f"Epoch global_role protocol mismatch: {context}")
        require(config.get("model") == "global_role_ablation" and config.get("seed") == seed,
                f"Epoch config model/seed mismatch: {context}")
        for key, expected in {**FORMAL, "epochs": expected_epochs}.items():
            require(config.get(key) == expected, f"Epoch formal configuration mismatch ({key}): {context}")
        training = record.get("train")
        require(isinstance(training, dict) and training.get("samples") == 20000 and training.get("steps") == 157,
                f"Incomplete formal training epoch: {context}")
        sampling = {key: value for key, value in plan[record["epoch"] - 1].items()
                    if key not in {"epoch", "sampled_indices_sha256", "actual_batch_order_sha256"}}
        require(training.get("sampling") == sampling, f"Epoch training sampling does not match plan: {context}")
        require(isinstance(record.get("validation"), dict), f"Missing validation object: {context}")
        require(isinstance(record["validation"].get("best_f1_on_validation_grid"), dict),
                f"Missing F1 selection metric: {context}")
        f1 = record["validation"]["best_f1_on_validation_grid"].get("f1")
        require(isinstance(f1, (int, float)) and not isinstance(f1, bool) and 0 <= f1 <= 1,
                f"Invalid F1 selection metric: {context}")
    best_epoch = max(records, key=lambda r: (r["validation"]["best_f1_on_validation_grid"]["f1"], r["epoch"]))["epoch"]
    selected = None
    frozen = None
    for record in records:
        epoch = record["epoch"]
        data = _load_predictions(folder / f"predictions_epoch{epoch:02d}.npz", labels, arm)
        metric = best_grid_metrics(labels, data["probabilities"])
        _check_metric(record["validation"]["best_f1_on_validation_grid"], metric, f"{context}/epoch{epoch}")
        if frozen is None:
            frozen = data["expert_probabilities"][:, :5].copy()
        else:
            require(np.allclose(frozen, data["expert_probabilities"][:, :5], atol=EXPERT_ATOL, rtol=EXPERT_RTOL),
                    f"Frozen regional predictions changed across epochs: {context}")
        if epoch == best_epoch:
            selected = data
            selected_metric = metric
    row = {"arm": arm, "seed": seed, "epoch": best_epoch, "n": len(labels), "positives": int(labels.sum()),
           **selected_metric, **probability_scores(labels, selected["probabilities"])}
    row.update({"fixed_0_5_" + key: value for key, value in
                threshold_metrics(labels, selected["probabilities"], .5).items() if key != "threshold"})
    return {"row": row, "args": _normalized_args(args), "protocol": protocol, "row_ids": row_ids,
            "labels": labels, "predictions": selected, "frozen": frozen}


def _mean_std(values):
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else None


def paired_differences(rows, arms, seeds):
    table = {(r["arm"], r["seed"]): r for r in rows}
    contrasts = [(f"{a}-{b}", {a: 1, b: -1}) for a, b in
                 (("B", "A"), ("C", "A"), ("D", "A"), ("D", "B"), ("D", "C"), ("E", "D"))]
    contrasts.append(("(D-C)-(B-A)", {"D": 1, "C": -1, "B": -1, "A": 1}))
    if "E_mean" in arms:
        contrasts.append(("E_mean-D", {"E_mean": 1, "D": -1}))
        contrasts.append(("E-E_mean", {"E": 1, "E_mean": -1}))
    result = []
    for name, coefficients in contrasts:
        if not set(coefficients) <= set(arms):
            continue
        for metric in SCORE_FIELDS:
            values = [sum(coef * table[(arm, seed)][metric] for arm, coef in coefficients.items()) for seed in seeds]
            for seed, value in zip(seeds, values):
                result.append({"contrast": name, "metric": metric, "seed": seed, "difference": value,
                               "mean_difference": None, "sample_std": None, "n_seeds": 1})
            mean, std = _mean_std(values)
            result.append({"contrast": name, "metric": metric, "seed": "all", "difference": None,
                           "mean_difference": mean, "sample_std": std, "n_seeds": len(seeds)})
    return result


def diagnostics(run, baseline):
    labels = run["labels"]
    pred = run["predictions"]
    expert_correct = (pred["expert_probabilities"] > .5) == labels[:, None]
    region_correct = expert_correct[:, :5]
    any_region = region_correct.any(axis=1)
    selected_index = pred["weights"].argmax(axis=1)
    selected_correct = expert_correct[np.arange(len(labels)), selected_index]
    any_available = expert_correct[:, :5].any(axis=1) if run["row"]["arm"] in {"A", "B"} else expert_correct.any(axis=1)
    global_p = pred["expert_probabilities"][:, 5]
    global_metrics = {"global_f1_fixed_0_5": threshold_metrics(labels, global_p, .5)["f1"],
                      "global_f1_validation_grid": best_grid_metrics(labels, global_p)["f1"],
                      **{"global_" + k: v for k, v in probability_scores(labels, global_p).items()}}
    result = []
    for mode in ("own_validation_threshold", "fixed_0_5"):
        threshold = run["row"]["threshold"] if mode == "own_validation_threshold" else .5
        base_threshold = baseline["row"]["threshold"] if mode == "own_validation_threshold" else .5
        correct = (pred["probabilities"] > threshold) == labels
        base_correct = (baseline["predictions"]["probabilities"] > base_threshold) == labels
        repair = ~base_correct & correct
        damage = base_correct & ~correct
        result.append({"arm": run["row"]["arm"], "seed": run["row"]["seed"], "kind": "repair_damage",
                       "threshold_mode": mode, "threshold": threshold, "baseline_threshold": base_threshold,
                       "n": len(labels), "repair_count": int(repair.sum()), "damage_count": int(damage.sum()),
                       "repair_fraction_all": float(repair.mean()), "damage_fraction_all": float(damage.mean()),
                       "repair_rate_baseline_wrong": float(repair.sum() / (~base_correct).sum()) if (~base_correct).sum() else None,
                       "damage_rate_baseline_correct": float(damage.sum() / base_correct.sum()) if base_correct.sum() else None,
                       "any_region_correct_at_0_5_fraction": float(any_region.mean()),
                       "top_selected_wrong_and_any_region_correct_fraction": float((~selected_correct & any_region).mean()),
                       "top_selected_wrong_and_any_available_correct_fraction": float((~selected_correct & any_available).mean()),
                       "fused_wrong_despite_selected_correct_fraction": float((~correct & selected_correct).mean()),
                       **global_metrics})
    # These are validation-only, label-assisted descriptive bins, never training weights.
    for name, mask in (("all_regions_wrong_at_0_5", ~any_region),
                       ("mixed_regional_correctness_at_0_5", any_region & ~region_correct.all(axis=1)),
                       ("all_regions_correct_at_0_5", region_correct.all(axis=1))):
        row = {"arm": run["row"]["arm"], "seed": run["row"]["seed"], "kind": "label_assisted_bin",
               "bin": name, "threshold_mode": "own_validation_threshold", "threshold": run["row"]["threshold"],
               "n": int(mask.sum())}
        if mask.any():
            row.update(threshold_metrics(labels[mask], pred["probabilities"][mask], run["row"]["threshold"]))
            row.update(probability_scores(labels[mask], pred["probabilities"][mask]))
        result.append(row)
    return result


def build_report(root, arms=DEFAULT_ARMS, seeds=DEFAULT_SEEDS, expected_epochs=20):
    root = Path(root)
    arms, seeds = tuple(arms), tuple(seeds)
    require(arms and len(set(arms)) == len(arms) and set(arms) <= set(DEFAULT_ARMS + ("E_mean",)), "Invalid/duplicate arms")
    require("A" in arms, "Arm A is required as the diagnostic baseline")
    require(seeds and len(set(seeds)) == len(seeds), "Invalid/duplicate seeds")
    require(expected_epochs > 0, "Epoch budget must be positive")
    rows, diagnostic_rows, paired_identity = [], [], {}
    shared_args = shared_hash = row_ids = manifest_labels = None
    for seed in seeds:
        runs = {}
        for arm in arms:
            run = audit_run(root / arm / f"seed{seed}", arm, seed, expected_epochs)
            if shared_args is None:
                shared_args, shared_hash = run["args"], run["protocol"]["shared_protocol_sha256"]
                row_ids, manifest_labels = run["row_ids"], run["labels"]
            require(run["args"] == shared_args, f"Non-shared training arguments: {arm}/seed{seed}")
            require(run["protocol"]["shared_protocol_sha256"] == shared_hash,
                    f"Shared protocol digest mismatch: {arm}/seed{seed}")
            require(run["row_ids"] == row_ids and np.array_equal(run["labels"], manifest_labels),
                    f"Validation row identity/order mismatch: {arm}/seed{seed}")
            if runs:
                reference = next(iter(runs.values()))
                for field in ("initial_state_sha256", "sampling_sha256"):
                    require(run["protocol"][field] == reference["protocol"][field],
                            f"Paired {field} mismatch: {arm}/seed{seed}")
                require(np.allclose(run["frozen"], reference["frozen"], atol=EXPERT_ATOL, rtol=EXPERT_RTOL),
                        f"Frozen regional predictions differ across arms: {arm}/seed{seed}")
            runs[arm] = run
            rows.append(run["row"])
        for run in runs.values():
            diagnostic_rows.extend(diagnostics(run, runs["A"]))
        paired_identity[str(seed)] = {
            field: runs["A"]["protocol"][field]
            for field in ("initial_state_sha256", "sampling_sha256")}
    summary = []
    for arm in arms:
        block = [r for r in rows if r["arm"] == arm]
        row = {"arm": arm, "n_seeds": len(seeds)}
        for field in SCORE_FIELDS + ("threshold", "fixed_0_5_f1", "fixed_0_5_precision", "fixed_0_5_sensitivity"):
            row[field + "_mean"], row[field + "_sample_std"] = _mean_std([r[field] for r in block])
        summary.append(row)
    return {"protocol_version": "global-role-v1", "audit_passed": True, "arms": list(arms), "seeds": list(seeds),
            "shared_protocol_sha256": shared_hash, "paired_identity_sha256": paired_identity,
            "epochs_per_run": expected_epochs, "run_count": len(rows), "validation_samples": len(manifest_labels),
            "validation_positives": int(manifest_labels.sum()), "independent_test": False,
            "threshold_selection": "strict probability > threshold; grid 0.01..0.99 step 0.01; first threshold wins ties",
            "epoch_selection": "maximum (validation grid F1, epoch); later epoch wins ties",
            "std_ddof": 1, "expert_comparison_atol": EXPERT_ATOL, "expert_comparison_rtol": EXPERT_RTOL,
            "notes": ["Mean F1 is the arithmetic mean of per-seed F1, not F1(mean precision, mean recall).",
                      "Threshold and epoch selection use the same validation labels as the reported scores.",
                      "Paired differences are descriptive; no statistical significance is claimed.",
                      "Oracle/repair/gap diagnostics are label-assisted descriptions, not deployable routers, F1 bounds, or Bayes capability estimates.",
                      "Global-alone grid F1 is additionally validation-selected; AP means average precision.",
                      "Validation row identities can be paths and are excluded from every exported report."],
            "summary": summary, "per_seed": rows, "paired_differences": paired_differences(rows, arms, seeds),
            "diagnostics": diagnostic_rows}


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("summary", "per_seed", "paired_differences", "diagnostics"):
        _write_csv(output_dir / f"{name}.csv", report[name])
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Global-role ablation results", "", "Audit passed. These are validation-selected results, not independent-test estimates.",
             "", "Return all six files together: summary.csv, per_seed.csv, paired_differences.csv, diagnostics.csv, report.json, RETURN_RESULTS.md.",
             "", "All rates are fractions (multiply by 100 for percentages); differences in rates multiplied by 100 are percentage points.",
             "Standard deviations use ddof=1. Missing standard deviations for one seed are empty/null, never zero.", "",
             "| Arm | Seeds | Mean F1 | Sample SD | Mean precision | Mean recall |", "|---|---:|---:|---:|---:|---:|"]
    for row in report["summary"]:
        sd = "NA" if row["f1_sample_std"] is None else f"{row['f1_sample_std']:.6f}"
        lines.append(f"| {row['arm']} | {row['n_seeds']} | {row['f1_mean']:.6f} | {sd} | {row['precision_mean']:.6f} | {row['sensitivity_mean']:.6f} |")
    lines += ["", "## Interpretation and required comparisons", "",
              "Read B-A, C-A, D-A, D-B, D-C, E-D and (D-C)-(B-A) as paired differences for the same seed. The interaction contrast is descriptive because each arm selects its own validation threshold and epoch.",
              "", "repair_damage rows compare each arm with A at its own selected validation threshold, and separately at fixed 0.5. Repair/damage counts alone do not determine F1 changes.",
              "Top-selected refers to argmax routing weight (first expert in a tie), not an oracle. Expert correctness in these diagnostics uses strict probability > 0.5. Global-alone results can describe a branch excluded from an arm's final prediction.",
              "", "label_assisted_bin rows partition validation windows using regional correctness at 0.5. They are retrospective descriptions, not training inputs, clinically verified mechanisms, deployment policies, F1 upper bounds, or estimates of irreducible error.",
              "", "The auditor verified all epoch prediction archives, complete epoch histories, paired initialization/sampling hashes, common arguments and protocol hash, exact validation row order and labels, frozen regional probabilities, normalized finite weights, A/B global masking, and recomputed threshold/F1/confusion counts.",
              "", "No paths, patient identifiers, or validation row identifiers are included in these exports. No claim of statistical significance or patient-disjointness is made.", ""]
    (output_dir / "RETURN_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--arms", nargs="+", choices=DEFAULT_ARMS + ("E_mean",), default=list(DEFAULT_ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    flags = parser.parse_args(argv)
    try:
        report = build_report(flags.root, flags.arms, flags.seeds)
        write_report(report, flags.output_dir or flags.root / "report")
    except (AuditError, OSError) as exc:
        parser.exit(2, f"Audit failed: {exc}\n")
    print(f"Audit passed: {report['run_count']} completed runs; aggregate report written.")


if __name__ == "__main__":
    main()
