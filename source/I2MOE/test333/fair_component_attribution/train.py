"""Run one matched first-stage correction fit with the original formal trainer."""
import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fair_refreshed_pairs_20261120"))
import bootstrap
import train_eeg19_i2moe as trainer
from complement_execution import install
from complement_models import ComplementMoE
from protected_evaluation import install as install_evaluation
from i2moe_eeg.utils import atomic_write_json
from components import ARMS, configure, loss_terms


def digest(named):
    h = hashlib.sha256()
    for name, p in sorted(named):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", required=True, type=int, choices=(3, 42))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    folder = args.output_dir
    folder.mkdir(exist_ok=True, parents=True)
    baseline = bootstrap.CACHE.parent / "I2MOE6_F1/screen_20261112/baseline" / f"seed{args.seed}"
    flags = argparse.Namespace(**json.loads((baseline / "args.json").read_text()))
    flags.output_dir = str(folder)
    flags.resume = str(folder / "resume.pt") if args.resume else ""
    flags.memory_cap_gib = 4.0
    expected = torch.load(baseline / "best_f1.pt", map_location="cpu", weights_only=False)["config"]
    contract = install(trainer, args.seed, "risk_power")
    install_evaluation(trainer)
    trainer.parse_args = lambda: flags
    trainer.loss_terms = lambda out,y,a,f: loss_terms(args.arm,out,y,a,f)
    trainer.SixExpertF1 = lambda **kw: configure(ComplementMoE("risk_power", **kw), args.arm)
    original_builder = trainer.build_model

    def build(flags):
        model = original_builder(flags)
        metadata = contract["train_cohort_metadata"]
        model.correction.group_probabilities.copy_(torch.tensor(metadata["probabilities"]))
        model.correction.group_base_weights.copy_(torch.tensor(metadata["weights"]))
        params = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
        core = [(n,p) for n,p in model.trainable_state_dict().items() if not n.startswith("correction.")]
        atomic_write_json(folder / "paired_initialization.json", dict(all_trainable=digest(params),
                core=digest(core), parameters=sum(p.numel() for _,p in params)))
        return model
    trainer.build_model = build
    original_config = trainer.semantic_config

    def check(*values):
        config = original_config(*values)
        differences = [k for k,v in config.items() if expected.get(k) != v]
        if differences:
            raise ValueError(f"Baseline configuration changed: {differences}")
        extension = dict(arm=args.arm, settings=asdict(ARMS[args.arm]), seed=args.seed,
            weights=contract["cohort_metadata_sha256"],
            experiment_code={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (HERE/"train.py", HERE/"components.py")})
        atomic_write_json(folder / "configuration_check.json", dict(passed=True, differences=[],formal=True,
                    fields=list(config), **extension))
        return dict(config, component_control=extension)
    trainer.semantic_config = check
    sample = trainer.sampled_indices
    def sampled(*v, **kw):
        ix, stats = sample(*v, **kw)
        seed, epoch = v[-2:]
        sampler = SubsetRandomSampler(ix.tolist(), generator=torch.Generator().manual_seed(seed+10000*epoch))
        order = np.asarray(list(sampler), dtype=np.int64)
        atomic_write_json(folder / f"sampling_epoch{epoch:02d}.json", dict(**stats,
            actual_batch_order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),
            sampled_indices_sha256=hashlib.sha256(ix.tobytes()).hexdigest()))
        return ix, stats
    trainer.sampled_indices = sampled
    trainer.main()


if __name__ == "__main__":
    main()
