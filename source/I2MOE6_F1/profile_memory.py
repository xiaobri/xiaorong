#!/usr/bin/env python3
"""Profile a real B128 training step with a bounded CUDA allocator."""
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import torch

import train_eeg19_i2moe as trainer


def main() -> None:
    profile_dir = Path(__file__).resolve().parent / "profiles"
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "unknown")
    variant = sys.argv[sys.argv.index("--variant") + 1] if "--variant" in sys.argv else "baseline"
    stem = f"{variant}_gpu{gpu}"
    free, total = torch.cuda.mem_get_info()
    cap_gib = float(os.environ.get("PROFILE_CAP_GIB", "8"))
    if free / 1024**3 < cap_gib + 2:
        raise RuntimeError(f"Need {cap_gib + 2:.1f} GiB free; currently {free / 1024**3:.2f}")
    torch.cuda.set_per_process_memory_fraction(cap_gib * 1024**3 / total)
    if "--smoke-test" not in sys.argv:
        sys.argv.append("--smoke-test")
    started = time.monotonic()
    log = profile_dir / f"{stem}.log"
    with log.open("w") as stream, contextlib.redirect_stdout(stream):
        trainer.main()
    result = next(json.loads(line.split("=", 1)[1]) for line in log.read_text().splitlines()
                  if line.startswith("SMOKE_TEST_JSON="))
    result.update(gpu=gpu, allocator_cap_gib=cap_gib, free_before_gib=free / 1024**3,
                  elapsed_seconds=time.monotonic() - started)
    path = profile_dir / f"{stem}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "gpu", "all_finite", "missing_gradient_parameters", "optimizer_step_completed",
        "peak_allocated_gib", "peak_reserved_gib", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
