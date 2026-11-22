#!/usr/bin/env python3
"""One bounded-memory job per GPU, with explicit smoke/debug/formal outputs."""
from __future__ import annotations
import czb_paths as _paths

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


HERE = Path(__file__).resolve().parent
PYTHON = f"{_paths.PYTHON}"
OUTPUT = Path(f"{_paths.STORAGE}/I2MOE6_F1/screen_20261112")
JOBS = [(gpu, variant, seed) for gpu, (variant, seed) in enumerate(
    (variant, seed) for variant in ("baseline", "sampling", "competence", "temporal_mil")
    for seed in (3, 42))]


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def common_arguments(variant: str, seed: int) -> list[str]:
    args = ["--variant", variant, "--seed", str(seed), "--batch-size", "128",
            "--eval-batch-size", "128", "--samples-per-epoch", "20000",
            "--lr", "0.0005", "--epochs", "20", "--num-workers", "2",
            "--threshold-start", "0.01", "--threshold-end", "0.99", "--threshold-step", "0.01",
            "--memory-cap-gib", "4", "--ratio", "10" if variant == "sampling" else "30"]
    if variant in {"competence", "temporal_mil"}:
        args += ["--interaction-loss-weight", "0"]
    if variant == "competence":
        args += ["--auxiliary-task-loss-weight", "0.1"]
    return args


def free_memory(gpu: int) -> float:
    result = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"
    ], text=True)
    return float(result.strip()) / 1024


def worker(job: tuple[int, str, int], mode: str) -> dict:
    gpu, variant, seed = job
    root = OUTPUT if mode == "formal" else HERE / mode
    out = root / variant / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "job_status.json"
    state = dict(gpu=gpu, variant=variant, seed=seed, mode=mode, output=str(out))
    lock_file = Path(f"{_paths.LOCK_PREFIX}{gpu}.lock").open("a")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return {**state, "status": "already_running"}
    if mode == "formal" and (out / "completed.json").exists():
        result = {**state, "status": "completed_previously"}
        atomic_json(state_path, result)
        return result
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1", PROFILE_CAP_GIB="4")
    args = common_arguments(variant, seed)
    script = "profile_memory.py" if mode == "profiles" else "train_eeg19_i2moe.py"
    if mode == "integration":
        args[args.index("--epochs") + 1] = "1"
        args += ["--max-steps-per-epoch", "2", "--max-val-samples", "512"]
    if mode != "profiles":
        args += ["--output-dir", str(out)]
        if (out / "resume.pt").exists():
            args += ["--resume", str(out / "resume.pt")]
    command = [PYTHON, "-u", script, *args]
    atomic_json(out / "command.json", {"argv": command, "cuda_visible_devices": str(gpu)})
    while True:
        free = free_memory(gpu)
        if free >= 6:
            break
        atomic_json(state_path, {**state, "status": "waiting_for_6_gib_free", "free_gib": free})
        time.sleep(30)
    started = time.time()
    with (out / "run.log").open("a") as log:
        process = subprocess.Popen(command, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
        atomic_json(state_path, {**state, "status": "running", "pid": process.pid, "started": started})
        return_code = process.wait()
    status = "finished" if return_code == 0 else "failed"
    if mode == "formal" and return_code == 0:
        status = "completed" if (out / "completed.json").exists() else "missing_completion"
    result = {**state, "status": status, "return_code": return_code, "elapsed_seconds": time.time() - started}
    atomic_json(state_path, result)
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("profiles", "integration", "formal"), required=True)
    args = parser.parse_args()
    sources = sorted(HERE.glob("*.py")) + sorted((HERE / "i2moe_eeg").glob("*.py"))
    hashes = {str(path.relative_to(HERE)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    if args.mode == "formal":
        verified = json.loads((HERE / "integration" / "all_jobs.json").read_text())
        if verified.get("source_sha256") != hashes or any(r["status"] != "finished" for r in verified["jobs"]):
            raise RuntimeError("Current source must pass all integration jobs before formal launch.")
        for gpu, variant, _ in JOBS:
            profile = json.loads((HERE / "profiles" / f"{variant}_gpu{gpu}.json").read_text())
            if not profile["all_finite"] or profile["missing_gradient_parameters"]:
                raise RuntimeError(f"Invalid GPU profile for {variant} on {gpu}")
        plan = dict(jobs=JOBS, source_sha256=hashes, total_epochs=20, samples_per_epoch=20000,
                    batch_size=128, allocator_cap_gib=4, min_free_gib=6,
                    selection="save best AP and best validation F1 separately; report fine and original coarse grids")
        plan_path = OUTPUT / "plan.json"
        if plan_path.exists() and json.loads(plan_path.read_text()) != json.loads(json.dumps(plan)):
            raise RuntimeError("Existing formal plan differs; refusing mixed-protocol resume.")
        atomic_json(plan_path, plan)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda job: worker(job, args.mode), JOBS))
    root = OUTPUT if args.mode == "formal" else HERE / args.mode
    atomic_json(root / "all_jobs.json", {"jobs": results, "source_sha256": hashes})
    if any(r["status"] in {"failed", "missing_completion"} for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
