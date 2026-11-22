"""Reproducibility and durable-output helpers."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, payloads: list[Any]) -> None:
    """Atomically replace a JSONL ledger with fully flushed records."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(
                json.dumps(json_ready(payload), sort_keys=True, allow_nan=False) + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    # Windows fsync requires a writable descriptor. The file was already
    # serialized above; reopening without truncation preserves its bytes.
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Any, device: torch.device) -> None:
    if not isinstance(state, dict):
        raise ValueError("Checkpoint has no valid rng_state mapping.")
    missing = [key for key in ("python", "numpy", "torch") if key not in state]
    if missing:
        raise ValueError(f"Checkpoint is missing RNG states: {missing}.")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    if isinstance(numpy_state, list):
        numpy_state = tuple(numpy_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda":
        cuda_states = state.get("cuda")
        if not isinstance(cuda_states, (tuple, list)):
            raise ValueError("CUDA resume requires saved CUDA RNG states.")
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "Visible CUDA device count changed across resume: "
                f"saved={len(cuda_states)}, current={torch.cuda.device_count()}."
            )
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
