"""Synthetic CPU tests for result integrity; no clinical data or GPU is required."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


REPORT_PATH = Path(__file__).resolve().parents[1] / "report.py"
spec = importlib.util.spec_from_file_location("global_role_report", REPORT_PATH)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def make_run(root, arm, seed, epochs=2):
    folder = root / arm / f"seed{seed}"
    folder.mkdir(parents=True)
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    experts = np.asarray([[.1, .2, .3, .1, .4, .2], [.7, .6, .8, .4, .9, .8],
                          [.7, .8, .6, .9, .7, .3], [.2, .3, .1, .4, .2, .7],
                          [.1, .6, .2, .3, .4, .4], [.6, .4, .7, .2, .8, .6],
                          [.3, .2, .4, .1, .2, .2], [.8, .7, .9, .6, .7, .9]])
    probabilities = np.asarray([.1, .8, .6, .4, .3, .7, .2, .9])
    if arm in {"D", "E"}:
        probabilities[2] = .3
    if seed == 42:
        probabilities[3] = .2
    weights = np.full((len(labels), 6), 1 / 6)
    if arm in {"A", "B"}:
        weights[:, :5], weights[:, 5] = .2, 0
    args = {**report.FORMAL, "arm": arm, "seed": seed, "epochs": epochs, "batch_size": 128,
            "threshold_start": .01, "threshold_end": .99, "threshold_step": .01,
            "output_dir": str(folder), "resume": "", "max_steps_per_epoch": 0,
            "max_val_samples": 0, "global_role": {"arm": arm, "formal": True, "code_sha256": "a" * 64}}
    stats = {"samples": 20000, "class_0": 19355, "class_1": 645, "steps": 157,
             "full_batches": 156, "final_partial_batch_size": 32}
    sampling = [{"epoch": epoch, **stats, "sampled_indices_sha256": f"{seed + epoch:064x}",
                 "actual_batch_order_sha256": f"{seed + epoch + 10:064x}"} for epoch in range(1, epochs + 1)]
    protocol = {"protocol_version": "global-role-v1", "arm": arm, "seed": seed, "formal": True,
                "shared_protocol_sha256": "b" * 64, "initial_state_sha256": f"{seed + 100:064x}",
                "sampling_sha256": report.canonical_sha256(sampling), "allocated_trainable_parameters": 1000}
    identity = {"row_ids": [f"C:/private/person{i}/window.npy" for i in range(len(labels))],
                "labels": labels.tolist()}
    manifest = {**identity, "sha256": report.canonical_sha256(identity)}
    protocol["validation_rows_sha256"] = manifest["sha256"]
    write_json(folder / "args.json", args)
    write_json(folder / "protocol.json", protocol)
    write_json(folder / "completed.json", {"status": "completed", "epochs": epochs, "seed": seed})
    write_json(folder / "validation_rows.json", manifest)
    write_json(folder / "initialization.json", {"initial_state_sha256": protocol["initial_state_sha256"],
                                                "allocated_trainable_parameters": 1000})
    write_json(folder / "sampling_plan.json", sampling)
    metrics = []
    for epoch in range(1, epochs + 1):
        write_json(folder / f"sampling_epoch{epoch:02d}.json", sampling[epoch - 1])
        np.savez_compressed(folder / f"predictions_epoch{epoch:02d}.npz", labels=labels,
                            probabilities=probabilities, expert_probabilities=experts, weights=weights)
        metrics.append({"model": "global_role_ablation", "seed": seed, "epoch": epoch,
                        "config": {**report.FORMAL, "epochs": epochs, "model": "global_role_ablation",
                                   "seed": seed, "global_role": protocol},
                        "train": {"samples": 20000, "steps": 157, "sampling": stats}, "validation": {
            "best_f1_on_validation_grid": report.best_grid_metrics(labels, probabilities)}})
    (folder / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in metrics), encoding="utf-8")
    return folder


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.seeds = (3, 42)
        for seed in self.seeds:
            for arm in report.DEFAULT_ARMS:
                make_run(self.root, arm, seed)

    def tearDown(self):
        self.temporary.cleanup()

    def build(self):
        return report.build_report(self.root, seeds=self.seeds, expected_epochs=2)

    def change_json(self, relative, callback):
        path = self.root / relative
        value = json.loads(path.read_text(encoding="utf-8"))
        callback(value)
        write_json(path, value)

    def change_npz(self, relative, callback):
        path = self.root / relative
        with np.load(path) as archive:
            value = {k: archive[k] for k in archive.files}
        callback(value)
        np.savez_compressed(path, **value)

    def test_valid_report_ties_statistics_and_privacy(self):
        result = self.build()
        self.assertTrue(result["audit_passed"])
        self.assertEqual(result["run_count"], 10)
        self.assertTrue(all(r["epoch"] == 2 for r in result["per_seed"]))
        a_values = [r["f1"] for r in result["per_seed"] if r["arm"] == "A"]
        a = next(r for r in result["summary"] if r["arm"] == "A")
        self.assertAlmostEqual(a["f1_mean"], np.mean(a_values))
        self.assertAlmostEqual(a["f1_sample_std"], np.std(a_values, ddof=1))
        self.assertEqual(len(result["diagnostics"]), 50)
        self.assertTrue(any(r["contrast"] == "(D-C)-(B-A)" for r in result["paired_differences"]))
        output = self.root / "report"
        report.write_report(result, output)
        self.assertEqual({p.name for p in output.iterdir()}, {"summary.csv", "per_seed.csv", "paired_differences.csv",
                         "diagnostics.csv", "report.json", "RETURN_RESULTS.md"})
        for path in output.iterdir():
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("C:/private/", text)
            self.assertNotIn(str(self.root), text)
            self.assertNotIn("NaN", text)

    def test_grid_matches_bruteforce_with_equal_threshold_values(self):
        rng = np.random.default_rng(7)
        labels = rng.integers(0, 2, 100)
        probabilities = rng.choice(report.GRID, 100)
        expected = max([report.threshold_metrics(labels, probabilities, t) for t in report.GRID], key=lambda r: r["f1"])
        self.assertEqual(report.best_grid_metrics(labels, probabilities), expected)
        metric = report.threshold_metrics(np.array([0, 1]), np.array([.5, .5]), .5)
        self.assertEqual((metric["tp"], metric["fp"], metric["tn"], metric["fn"]), (0, 0, 1, 1))

    def test_first_threshold_tie(self):
        metric = report.best_grid_metrics(np.array([0, 1]), np.array([0., 1.]))
        self.assertEqual(metric["threshold"], .01)

    def test_missing_run_rejected(self):
        (self.root / "E/seed42/completed.json").unlink()
        with self.assertRaisesRegex(report.AuditError, "Missing required"):
            self.build()

    def test_missing_nonselected_epoch_predictions_rejected(self):
        (self.root / "B/seed3/predictions_epoch01.npz").unlink()
        with self.assertRaisesRegex(report.AuditError, "Missing epoch"):
            self.build()

    def test_nonformal_and_debug_rejected(self):
        self.change_json("B/seed3/protocol.json", lambda x: x.update(formal=False))
        with self.assertRaisesRegex(report.AuditError, "Non-formal"):
            self.build()
        self.change_json("B/seed3/protocol.json", lambda x: x.update(formal=True))
        self.change_json("B/seed3/args.json", lambda x: x.update(max_val_samples=8))
        with self.assertRaisesRegex(report.AuditError, "Formal training configuration"):
            self.build()

    def test_incomplete_status_rejected(self):
        self.change_json("A/seed3/completed.json", lambda x: x.update(status="running"))
        with self.assertRaisesRegex(report.AuditError, "not completed"):
            self.build()

    def test_configuration_mismatch_rejected(self):
        self.change_json("C/seed3/args.json", lambda x: x.update(batch_size=32))
        with self.assertRaisesRegex(report.AuditError, "Formal training configuration"):
            self.build()

    def test_grid_configuration_rejected(self):
        self.change_json("C/seed3/args.json", lambda x: x.update(threshold_step=.05))
        with self.assertRaisesRegex(report.AuditError, "configuration mismatch"):
            self.build()

    def test_initialization_or_sampling_mismatch_rejected(self):
        self.change_json("C/seed3/protocol.json", lambda x: x.update(sampling_sha256="c" * 64))
        with self.assertRaisesRegex(report.AuditError, "Sampling plan digest mismatch"):
            self.build()

    def test_common_protocol_hash_rejected(self):
        self.change_json("C/seed3/protocol.json", lambda x: x.update(shared_protocol_sha256="c" * 64))
        with self.assertRaisesRegex(report.AuditError, "global_role protocol mismatch"):
            self.build()

    def test_initialization_artifact_mismatch_rejected(self):
        self.change_json("C/seed3/initialization.json", lambda x: x.update(initial_state_sha256="c" * 64))
        with self.assertRaisesRegex(report.AuditError, "Initialization artifact digest mismatch"):
            self.build()

    def test_actual_sampling_differs_from_plan_rejected(self):
        self.change_json("C/seed3/sampling_epoch01.json", lambda x: x.update(actual_batch_order_sha256="c" * 64))
        with self.assertRaisesRegex(report.AuditError, "Actual sampling record differs"):
            self.build()

    def test_missing_actual_sampling_rejected(self):
        (self.root / "C/seed3/sampling_epoch01.json").unlink()
        with self.assertRaisesRegex(report.AuditError, "Missing required artifact"):
            self.build()

    def test_nonformal_setting_in_all_arms_rejected(self):
        for arm in report.DEFAULT_ARMS:
            self.change_json(f"{arm}/seed3/args.json", lambda x: x.update(lr=.001))
        with self.assertRaisesRegex(report.AuditError, "Formal training configuration mismatch"):
            self.build()

    def test_epoch_identity_and_training_budget_rejected(self):
        path = self.root / "D/seed3/metrics.jsonl"
        records = [json.loads(s) for s in path.read_text().splitlines()]
        records[0]["seed"] = 999
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        with self.assertRaisesRegex(report.AuditError, "Epoch model/seed mismatch"):
            self.build()
        records[0]["seed"] = 3
        records[0]["train"]["samples"] = 19968
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        with self.assertRaisesRegex(report.AuditError, "Incomplete formal training epoch"):
            self.build()

    def test_epoch_scientific_configuration_rejected(self):
        path = self.root / "D/seed3/metrics.jsonl"
        records = [json.loads(s) for s in path.read_text().splitlines()]
        records[0]["config"]["focal_alpha"] = .5
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        with self.assertRaisesRegex(report.AuditError, "Epoch formal configuration mismatch"):
            self.build()

    def test_manifest_hash_rejected(self):
        self.change_json("B/seed3/validation_rows.json", lambda x: x["row_ids"].reverse())
        with self.assertRaisesRegex(report.AuditError, "identity digest mismatch"):
            self.build()

    def test_matching_labels_but_different_row_identity_rejected(self):
        def change(value):
            value["row_ids"][0] = "different-window"
            value["sha256"] = report.canonical_sha256({"row_ids": value["row_ids"], "labels": value["labels"]})
        self.change_json("B/seed3/validation_rows.json", change)
        with self.assertRaisesRegex(report.AuditError, "validation identity digest mismatch"):
            self.build()

    def test_prediction_label_order_rejected(self):
        self.change_npz("B/seed3/predictions_epoch01.npz", lambda x: x.update(labels=x["labels"][::-1]))
        with self.assertRaisesRegex(report.AuditError, "labels/row order mismatch"):
            self.build()

    def test_frozen_predictions_across_arms_rejected(self):
        for epoch in (1, 2):
            self.change_npz(f"B/seed3/predictions_epoch{epoch:02d}.npz",
                            lambda x: x["expert_probabilities"].__setitem__((0, 0), .4))
        with self.assertRaisesRegex(report.AuditError, "differ across arms"):
            self.build()

    def test_masked_global_weight_rejected(self):
        def change(value):
            value["weights"][0, 5] = .1
            value["weights"][0, 0] -= .1
        self.change_npz("B/seed3/predictions_epoch01.npz", change)
        with self.assertRaisesRegex(report.AuditError, "Masked global weight"):
            self.build()

    def test_unnormalized_weights_rejected(self):
        self.change_npz("D/seed3/predictions_epoch01.npz", lambda x: x["weights"].__setitem__((0, 5), .7))
        with self.assertRaisesRegex(report.AuditError, "not normalized"):
            self.build()

    def test_nan_predictions_rejected(self):
        self.change_npz("D/seed3/predictions_epoch01.npz", lambda x: x["probabilities"].__setitem__(0, np.nan))
        with self.assertRaisesRegex(report.AuditError, "Non-finite/out-of-range"):
            self.build()

    def test_nan_json_rejected(self):
        self.change_json("D/seed3/args.json", lambda x: x.update(lr=float("nan")))
        with self.assertRaisesRegex(report.AuditError, "Non-finite JSON"):
            self.build()

    def test_false_metrics_rejected(self):
        path = self.root / "D/seed3/metrics.jsonl"
        records = [json.loads(s) for s in path.read_text().splitlines()]
        records[0]["validation"]["best_f1_on_validation_grid"]["f1"] = .999
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        with self.assertRaisesRegex(report.AuditError, "Recomputed validation metric"):
            self.build()

    def test_single_seed_sd_is_missing_not_zero(self):
        value = report.build_report(self.root, seeds=(3,), expected_epochs=2)
        self.assertTrue(all(r["f1_sample_std"] is None for r in value["summary"]))

    def test_paired_interaction_contrast(self):
        rows = [{"arm": arm, "seed": seed, **{field: number + seed for field in report.SCORE_FIELDS}}
                for seed in (0, 1) for arm, number in zip(report.DEFAULT_ARMS, (.2, .3, .4, .7, .8))]
        contrasts = report.paired_differences(rows, report.DEFAULT_ARMS, (0, 1))
        row = next(r for r in contrasts if r["contrast"] == "(D-C)-(B-A)" and r["metric"] == "f1" and r["seed"] == "all")
        self.assertAlmostEqual(row["mean_difference"], .2)


if __name__ == "__main__":
    unittest.main()
