"""Exercise the real trainer/ledger on synthetic EEG and tiny frozen experts."""
from __future__ import annotations

import contextlib
import importlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.global_role_ablation import run, suite
from experiments.global_role_ablation.model import GlobalRoleAblation
from experiments.global_role_ablation.protocol import FORMAL, validate_formal
import train_eeg19_i2moe


class TinyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(19, 8)
        self.head = nn.Linear(8, 2)

    def forward(self, eeg):
        features = self.encoder(eeg.mean(-1))
        return features, self.head(features)


class RunnerTests(unittest.TestCase):
    def test_formal_rejects_changed_budget_grid_or_debug(self):
        args = SimpleNamespace(**FORMAL, seed=3, smoke_test=False)
        validate_formal(args)
        for key, value in (("epochs", 1), ("threshold_step", 0.05), ("amp", True)):
            changed = SimpleNamespace(**vars(args))
            setattr(changed, key, value)
            with self.assertRaises(ValueError):
                validate_formal(changed)

    def test_suite_has_25_unique_runs_without_private_dependencies(self):
        args = SimpleNamespace(seeds=[3, 42, 0, 1, 2], arms=list("ABCDE"),
                               output_root="private-runs/example", resume_existing=False)
        commands = list(suite.commands(args, ["--device", "cpu"]))
        self.assertEqual(len(commands), 25)
        self.assertEqual(len({str(directory) for directory, _ in commands}), 25)
        self.assertTrue(all(command[-2:] == ["--device", "cpu"] for _, command in commands))

    def test_real_training_loop_all_arms_and_resume(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(0)
            rows = []
            for mode, count in (("Train", 8), ("Val", 4)):
                folder = root / mode
                folder.mkdir()
                for index in range(count):
                    path = folder / f"{index}.npy"
                    np.save(path, (rng.normal(size=(19, 40)) * 0.001).astype(np.float32))
                    rows.append(dict(file_path=path.as_posix(), target=index % 2, mode=mode))
            manifest = root / "manifest.csv"
            pd.DataFrame(rows).to_csv(manifest, index=False)
            initial_hashes, sampling_hashes = [], []
            for arm in ("A", "B", "C", "D", "E", "E_mean"):
                for resume in (False, True):
                    original = importlib.reload(train_eeg19_i2moe)
                    flags = SimpleNamespace(arm=arm, output_root=str(root / "runs"),
                                            mode="debug", resume_run=resume)
                    args = run.prepare_args(original, flags, [
                        "--csv-path", str(manifest), "--data-root", str(root),
                        "--device", "cpu", "--seed", "3", "--num-workers", "0",
                        "--batch-size", "4", "--eval-batch-size", "4",
                        "--samples-per-epoch", "8", "--ratio", "1", "--target-samples", "40",
                        "--max-val-samples", "4", "--preflight-samples", "2",
                        "--hidden-dim", "8", "--num-heads", "2", "--fusion-layers", "1",
                        "--router-hidden-dim", "12",
                    ])
                    original.build_model = lambda actual: GlobalRoleAblation(
                        arm=actual.arm, hidden_dim=8, num_heads=2, fusion_layers=1,
                        router_hidden_dim=12, frozen_expert_factory=lambda _: TinyExpert())
                    original.frozen_checkpoint_identity = lambda *unused: {
                        f"region{i}": {"sha256": str(i) * 64} for i in range(5)}
                    run.install(original, flags, args)
                    with contextlib.redirect_stdout(io.StringIO()):
                        original.main()
                    directory = Path(args.output_dir)
                    contract = json.loads((directory / "protocol.json").read_text())
                    self.assertFalse(contract["formal"])
                    self.assertEqual(contract["arm"], arm)
                    self.assertTrue((directory / "best_f1.pt").is_file())
                    self.assertTrue((directory / "debug_finished.json").is_file())
                    self.assertFalse((directory / "completed.json").exists())
                    self.assertEqual(len((directory / "metrics.jsonl").read_text().splitlines()), 1)
                    if not resume:
                        initial_hashes.append(contract["initial_state_sha256"])
                        sampling_hashes.append(contract["sampling_sha256"])
            self.assertEqual(len(set(initial_hashes)), 1)
            self.assertEqual(len(set(sampling_hashes)), 1)
        importlib.reload(train_eeg19_i2moe)

    def test_d_and_e_interrupted_resume_match_two_continuous_epochs(self):
        """Actually train after resume, including dropout, noise and Adam state."""
        torch.set_num_threads(2)

        class PlannedInterruption(RuntimeError):
            pass

        def assert_state_equal(actual, expected, location):
            if isinstance(expected, torch.Tensor):
                self.assertEqual(actual.dtype, expected.dtype, location)
                self.assertTrue(torch.equal(actual, expected), location)
            elif isinstance(expected, np.ndarray):
                np.testing.assert_array_equal(actual, expected, err_msg=location)
            elif isinstance(expected, dict):
                self.assertEqual(actual.keys(), expected.keys(), location)
                for key in expected:
                    assert_state_equal(actual[key], expected[key], f"{location}.{key}")
            elif isinstance(expected, (list, tuple)):
                self.assertEqual(type(actual), type(expected), location)
                self.assertEqual(len(actual), len(expected), location)
                for index, (got, wanted) in enumerate(zip(actual, expected)):
                    assert_state_equal(got, wanted, f"{location}[{index}]")
            else:
                self.assertEqual(actual, expected, location)

        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                rng = np.random.default_rng(19)
                rows = []
                for mode, count in (("Train", 12), ("Val", 4)):
                    folder = root / mode
                    folder.mkdir()
                    for index in range(count):
                        path = folder / f"{index}.npy"
                        np.save(path, (rng.normal(size=(19, 40)) * 0.001).astype(np.float32))
                        rows.append(dict(file_path=path.as_posix(), target=index % 2, mode=mode))
                manifest = root / "manifest.csv"
                pd.DataFrame(rows).to_csv(manifest, index=False)

                def execute(arm, output_root, resume=False, interrupt=False):
                    original = importlib.reload(train_eeg19_i2moe)
                    flags = SimpleNamespace(arm=arm, output_root=str(output_root),
                                            mode="debug", resume_run=resume)
                    args = run.prepare_args(original, flags, [
                        "--csv-path", str(manifest), "--data-root", str(root),
                        "--device", "cpu", "--seed", "3", "--num-workers", "0",
                        "--batch-size", "4", "--eval-batch-size", "4",
                        "--samples-per-epoch", "8", "--ratio", "1", "--target-samples", "40",
                        "--max-val-samples", "4", "--preflight-samples", "2",
                        "--hidden-dim", "8", "--num-heads", "2", "--fusion-layers", "1",
                        "--router-hidden-dim", "12",
                    ])
                    # Extend only this synthetic test's debug target. Both the
                    # interrupted and continuous runs preregister two epochs.
                    args.epochs = 2
                    original.build_model = lambda actual: GlobalRoleAblation(
                        arm=actual.arm, hidden_dim=8, num_heads=2, fusion_layers=1,
                        router_hidden_dim=12, dropout=actual.dropout,
                        router_dropout=actual.router_dropout,
                        frozen_expert_factory=lambda _: TinyExpert())
                    original.frozen_checkpoint_identity = lambda *unused: {
                        f"region{i}": {"sha256": str(i) * 64} for i in range(5)}
                    run.install(original, flags, args)
                    train_epoch = original.train_one_epoch
                    trained_epochs = []

                    def tracked_train(*values):
                        epoch = values[-1]
                        if interrupt and epoch == 2:
                            # Epoch 1 has already committed its model, Adam,
                            # scaler, RNG and ledger. No epoch-2 step runs yet.
                            raise PlannedInterruption("after committing epoch 1")
                        result = train_epoch(*values)
                        trained_epochs.append(epoch)
                        return result

                    original.train_one_epoch = tracked_train
                    with contextlib.redirect_stdout(io.StringIO()):
                        if interrupt:
                            with self.assertRaisesRegex(PlannedInterruption, "epoch 1"):
                                original.main()
                        else:
                            original.main()
                    return Path(args.output_dir), trained_epochs

                for arm in ("D", "E"):
                    with self.subTest(arm=arm):
                        continuous_dir, continuous_epochs = execute(arm, root / "continuous")
                        interrupted_dir, prefix_epochs = execute(
                            arm, root / "resumed", interrupt=True)
                        self.assertEqual(continuous_epochs, [1, 2])
                        self.assertEqual(prefix_epochs, [1])
                        first_commit = torch.load(interrupted_dir / "resume.pt",
                                                  map_location="cpu", weights_only=False)
                        self.assertEqual(first_commit["epoch"], 1)
                        self.assertFalse((interrupted_dir / "sampling_epoch02.json").exists())
                        self.assertFalse((interrupted_dir / "debug_finished.json").exists())

                        resumed_dir, resumed_epochs = execute(arm, root / "resumed", resume=True)
                        self.assertEqual(resumed_dir, interrupted_dir)
                        self.assertEqual(resumed_epochs, [2])
                        actual = torch.load(resumed_dir / "resume.pt", map_location="cpu", weights_only=False)
                        expected = torch.load(continuous_dir / "resume.pt", map_location="cpu", weights_only=False)
                        self.assertEqual(actual["epoch"], 2)
                        for key in ("model", "optimizer", "scaler", "rng_state"):
                            assert_state_equal(actual[key], expected[key], f"{arm}.{key}")
                        # Ensure the comparison really includes a second update
                        # interval, rather than reopening a completed checkpoint.
                        self.assertTrue(any(not torch.equal(actual["model"][name], tensor)
                                            for name, tensor in first_commit["model"].items()))
                        for filename in ("initialization.json", "sampling_plan.json", "sampling_epoch02.json"):
                            got = json.loads((resumed_dir / filename).read_text())
                            wanted = json.loads((continuous_dir / filename).read_text())
                            self.assertEqual(got, wanted, f"{arm}.{filename}")
                        ledger = [json.loads(line) for line in
                                  (resumed_dir / "metrics.jsonl").read_text().splitlines()]
                        self.assertEqual([record["epoch"] for record in ledger], [1, 2])
                        self.assertTrue((resumed_dir / "debug_finished.json").is_file())
                        for directory in (continuous_dir, resumed_dir):
                            selected = torch.load(directory / "best_f1.pt", map_location="cpu", weights_only=False)
                            if directory == continuous_dir:
                                best_continuous = selected
                            else:
                                self.assertEqual(selected["selected_f1_epoch"], best_continuous["selected_f1_epoch"])
                                self.assertEqual(selected["selected_f1"], best_continuous["selected_f1"])
                                assert_state_equal(selected["model"], best_continuous["model"], f"{arm}.best_f1")
        finally:
            importlib.reload(train_eeg19_i2moe)


if __name__ == "__main__":
    unittest.main()
