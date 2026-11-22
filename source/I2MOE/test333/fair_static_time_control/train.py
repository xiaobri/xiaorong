"""Train a fresh_both model varying only the temporal scorer input."""
import argparse
import sys
from pathlib import Path

from time_control import ARMS, configure, parameter_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", required=True, type=int, choices=(3, 42))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("formal", "integration"), default="formal")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    sys.path.insert(0, str(folder.parent / "fair_refreshed_pairs_20261120"))
    import train_strict as parent
    from i2moe_eeg.utils import atomic_write_json

    constructor = parent.RefreshedPairMoE
    original_contract = parent.method_contract
    original_identity = parent.source_identity

    def construct(method, **kwargs):
        model = configure(constructor(method, **kwargs), args.arm)
        atomic_write_json(args.output_dir / "paired_initialization.json", {
            "arm": args.arm, "seed": args.seed, "all_trainable_sha256": parameter_digest(model),
            "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
        return model

    def contract(method):
        return dict(original_contract(method), temporal_scorer_input=args.arm,
                    positional_control="Eight fixed DCT rows on scorer inputs only")

    def identity():
        files = (folder / "time_control.py", folder / "train.py")
        return {**original_identity(), **{str(p): parent.sha256(p) for p in files}}

    parent.RefreshedPairMoE = construct
    parent.method_contract = contract
    parent.source_identity = identity
    sys.argv = [str(Path(parent.__file__)), "--method", "fresh_both", "--seed", str(args.seed),
                "--output-dir", str(args.output_dir), "--mode", args.mode]
    if args.resume:
        sys.argv.append("--resume")
    parent.main()


if __name__ == "__main__":
    main()
