#!/usr/bin/env python3
"""Serial single-GPU launcher; dry-run does not import torch or private data."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.global_role_ablation.protocol import ARMS, MAIN_ARMS, SEEDS, run_directory


def commands(args, passthrough):
    for seed in args.seeds:
        for arm in args.arms:
            directory = run_directory(args.output_root, arm, seed)
            command = [sys.executable, str(HERE / "run.py"), "--arm", arm, "--seed", str(seed),
                       "--output-root", str(Path(args.output_root).resolve())]
            if args.resume_existing and (directory / "resume.pt").is_file():
                # Even completed runs are reopened: the original trainer checks
                # the ledger, checkpoint configuration and completion contract.
                command.append("--resume-run")
            yield directory, command + passthrough


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-root", default="private-runs/global_role_ablation")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(MAIN_ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, passthrough = parser.parse_known_args(argv)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    reserved = {"--arm", "--seed", "--output-root", "--resume-run", "--output-dir", "--resume"}
    if any(value.split("=", 1)[0] in reserved for value in passthrough):
        parser.error("Do not override matrix identity/output options after --.")
    if len(set(args.arms)) != len(args.arms) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate arms or seeds.")
    planned = list(commands(args, passthrough))
    print(json.dumps({"runs": len(planned), "arms": args.arms, "seeds": args.seeds}), flush=True)
    for directory, command in planned:
        print(shlex.join(command), flush=True)
        if args.dry_run:
            continue
        directory.mkdir(parents=True, exist_ok=True)
        # No shell: arguments, paths and redirection work on Linux and Windows.
        with (directory / "launcher.log").open("a", encoding="utf-8") as stream:
            stream.write("\nCOMMAND " + json.dumps(command) + "\n")
            stream.flush()
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f"Run failed ({result.returncode}); inspect {directory / 'launcher.log'}")
    if not args.dry_run:
        print("Matrix finished. Run report.py to audit and export results.")


if __name__ == "__main__":
    main()
