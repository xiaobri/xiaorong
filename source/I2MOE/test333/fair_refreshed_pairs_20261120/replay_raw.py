#!/usr/bin/env python3
"""Replay a cached-execution checkpoint from unchanged raw EEG inputs."""
import bootstrap
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import train_eeg19_i2moe as original
from i2moe_eeg.losses import MulticlassFocalLoss
from i2moe_eeg.utils import set_seed, atomic_write_json
from refresh_models import RefreshedPairMoE
from pair_evaluation import install as install_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    flags = parser.parse_args()
    install_evaluation(original)
    payload = torch.load(flags.run/'best_f1.pt', map_location='cpu', weights_only=False)
    extension = payload['config']['strict_extension']
    seed, method = extension['seed'], extension['method']
    args = argparse.Namespace(**json.loads(Path(extension['original_args']).read_text()))
    destination = flags.run/'raw_replay'
    destination.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(destination)
    torch.set_num_threads(2)
    device = original.initialize_device(args)
    set_seed(seed, args.deterministic)
    if method != 'baseline':
        original.SixExpertF1 = lambda **kwargs: RefreshedPairMoE(method, **kwargs)
    model = original.build_model(args).to(device)
    model.load_trainable_state_dict(payload['model'])
    frame, _ = original.load_manifest(args.csv_path)
    data = original.EEG19NpyDataset(frame, 'Val', args.stored_prefix, args.data_root,
        target_samples=args.target_samples, eeg_scale_divisor=args.eeg_scale_divisor)
    metrics = original.evaluate(model, data, MulticlassFocalLoss(args.focal_alpha, args.focal_gamma),
                                args, device, payload['epoch'])
    with (np.load(flags.run/f"predictions_epoch{payload['epoch']:02d}.npz") as cached,
          np.load(destination/f"predictions_epoch{payload['epoch']:02d}.npz") as raw):
        np.testing.assert_array_equal(cached['labels'], raw['labels'])
        threshold = payload['validation']['best_f1_on_validation_grid']['threshold']
        original_prob = cached['probabilities'].astype(np.float64)
        current_prob = raw['probabilities'].astype(np.float64)
        changed = int(np.sum((original_prob > threshold) != (current_prob > threshold)))
        error = float(np.max(np.abs(original_prob-current_prob)))
    proof = dict(seed=seed, method=method, rows=len(data),
                 cached_metric=payload['validation']['best_f1_on_validation_grid'],
                 raw_metric=metrics['best_f1_on_validation_grid'],
                 changed_threshold_decisions=changed, maximum_probability_error=error,
                 full_original_six_experts_recomputed=True,
                 exact_f1=metrics['best_f1_on_validation_grid']['f1'] ==
                          payload['validation']['best_f1_on_validation_grid']['f1'])
    atomic_write_json(destination/'replay_result.json', proof)
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
