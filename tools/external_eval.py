"""Evaluate the four locked checkpoints on contiguous native batches at each hospital."""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "source/I2MOE/test333/fair_refreshed_pairs_20261120"))
import bootstrap
import train_eeg19_i2moe as trainer
from refresh_models import RefreshedPairMoE
from i2moe_eeg.regions import EXPERT_NAMES
from i2moe_eeg.utils import atomic_write_json, set_seed

HOSPITALS = {"qilu": "test_qilu_checkout_source", "tiantan": "test_tiantan_checkout_source",
             "chifeng": "test_chifeng", "xian": "test_xian"}
KEYS = ("baseline_3", "baseline_42", "fresh_both_3", "fresh_both_42")
BATCH = 128


def sha(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def models_and_identity(args, device):
    with (ROOT / "results/selected.csv").open() as stream:
        selected = {(r["method"], int(r["seed"])): r for r in csv.DictReader(stream)}
    models, identity = {}, {}
    verified = {}
    for key in KEYS:
        method, seed = key.rsplit("_", 1)
        seed = int(seed)
        folder = args.baseline if method == "baseline" else args.candidate
        path = folder / f"seed{seed}" / "best_f1.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        config = payload["config"]
        row = selected[method, seed]
        if (payload["seed"] != seed or payload["epoch"] != int(row["epoch"])
                or config["strict_extension"]["method"] != method
                or payload["validation"]["best_f1_on_validation_grid"]["threshold"] != float(row["threshold"])):
            raise ValueError(f"Checkpoint does not match the locked selection for {key}")
        flags = argparse.Namespace(**payload["args"])
        for name, source in trainer.frozen_checkpoint_paths(flags).items():
            if source not in verified:
                verified[source] = sha(source)
            if verified[source] != config["frozen_checkpoint_identity"][name]["sha256"]:
                raise ValueError(f"Frozen expert differs for {key} / {name}")
        set_seed(seed, flags.deterministic)
        constructor = trainer.SixExpertF1 if method == "baseline" else lambda **kw: RefreshedPairMoE("fresh_both", **kw)
        with patch.object(trainer, "SixExpertF1", constructor):
            model = trainer.build_model(flags)
        model.load_trainable_state_dict(payload["model"])
        models[key] = model.to(device).eval()
        identity[key] = dict(epoch=payload["epoch"], threshold=float(row["threshold"]), sha256=sha(path))
    if len(verified) != 5 or tuple(EXPERT_NAMES)[-1] != "redundancy":
        raise ValueError("Expected the five shared frozen sources and the global expert")
    return models, identity


def outputs(model, eeg):
    out = model(eeg, compute_interactions=False)
    array = torch.cat([out.logits.softmax(1)[:, 1:2], out.interaction_weights,
                       out.expert_logits.softmax(2)[:, :, 1]], 1)
    if array.shape != (len(eeg), 13):
        raise ValueError("Invalid model output")
    return array


@torch.inference_mode()
def predict(models, eeg, direct_check):
    # Only convolution outputs are shared. All model-specific hooks, heads, routers and
    # analytic corrections execute normally, including hooks on the convolution module.
    first = models[KEYS[0]]
    maps = {}
    with ExitStack() as hooks:
        for name, expert in first.frozen_experts.items():
            def capture(module, inputs, result, name=name):
                maps[name] = result
            hooks.enter_context(expert.encoder.features.register_forward_hook(capture))
        result = [outputs(first, eeg)]
    errors = {}
    for key in KEYS[1:]:
        model = models[key]
        with ExitStack() as reuse:
            for name, expert in model.frozen_experts.items():
                reuse.enter_context(patch.object(expert.encoder.features, "forward", lambda x, name=name: maps[name]))
            current = outputs(model, eeg)
        if direct_check:
            direct = outputs(model, eeg)
            torch.testing.assert_close(current, direct, rtol=0, atol=0)
            errors[key] = float((current - direct).abs().max().item())
        result.append(current)
    array = torch.stack(result, 1).cpu().numpy()
    if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
        raise ValueError("Nonfinite output or probability outside [0,1]")
    np.testing.assert_allclose(array[:, :, 1:7].sum(2), 1, rtol=0, atol=1e-5)
    return array, errors


def dataset_for(site, storage):
    collection = HOSPITALS[site]
    root = storage / f"test-{site}-dataset"
    manifest = root / f"{collection}_with_mode.csv"
    frame = pd.read_csv(manifest, usecols=["file_path", "mode", "target"])
    marker = "/" + collection + "/"
    head, sep, _ = frame.file_path.iloc[0].partition(marker)
    prefix = head + "/" + collection
    if (not sep or frame.isna().any().any() or not frame.file_path.str.startswith(prefix + "/").all()
            or not frame["mode"].eq("Val").all() or set(frame["target"].unique()) != {0, 1}):
        raise ValueError(f"Manifest disagrees with the expected external data contract for {site}")
    data = trainer.EEG19NpyDataset(frame, "Val", prefix, str(root), target_samples=2000, eeg_scale_divisor=1e-3)
    return data, sha(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Private working directory for window scores")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expected one visible GPU and a valid partition")
    torch.set_num_threads(2)
    device = torch.device("cuda:0")
    models, checkpoints = models_and_identity(args, device)
    for site in HOSPITALS:
        data, manifest_hash = dataset_for(site, args.storage)
        n = len(data)
        batches = math.ceil(n / BATCH)
        start = min(n, (batches * args.shard // args.shards) * BATCH)
        stop = min(n, (batches * (args.shard + 1) // args.shards) * BATCH)
        if stop <= start:
            raise ValueError("Empty partition")
        dest = args.output / site / f"shard{args.shard}"
        dest.mkdir(parents=True, exist_ok=True)
        contract = dict(hospital=site, shard=args.shard, shards=args.shards, start=start, stop=stop,
                        total=n, batch_size=BATCH, manifest_sha256=manifest_hash, checkpoints=checkpoints,
                        script_sha256=sha(__file__), experts=list(EXPERT_NAMES), models=list(KEYS))
        finished = dest / "complete.json"
        if finished.exists():
            done = json.loads(finished.read_text())
            if done["contract"] != contract or done["sha256"] != sha(dest / "scores.npz"):
                raise RuntimeError("Existing partition has a different contract or digest")
            continue
        loader = DataLoader(Subset(data, range(start, stop)), batch_size=BATCH, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
        all_scores = np.empty((stop - start, 4, 13), dtype=np.float32)
        offset = 0
        audits = []
        for eeg, labels in loader:
            check = offset == 0 or len(eeg) != BATCH
            scores, errors = predict(models, eeg.to(device, non_blocking=True), check)
            np.testing.assert_array_equal(data.labels[start + offset:start + offset + len(eeg)], labels.numpy())
            if check:
                audits.append(dict(offset=offset, rows=len(eeg), maximum_direct_difference=errors))
            all_scores[offset:offset + len(eeg)] = scores
            offset += len(eeg)
            if offset == len(eeg) or offset % (BATCH * 50) == 0:
                print(f"{site} shard {args.shard} {offset}/{stop-start}", flush=True)
        if offset != stop - start:
            raise RuntimeError("Incomplete evaluation")
        temp = dest / "scores.tmp.npz"
        np.savez(temp, scores=all_scores, labels=data.labels[start:stop].astype(np.uint8))
        os.replace(temp, dest / "scores.npz")
        atomic_write_json(finished, dict(contract=contract, direct_checks=audits, sha256=sha(dest / "scores.npz")))
        print(f"Completed {site} shard {args.shard} ({offset} windows)", flush=True)


if __name__ == "__main__":
    main()
