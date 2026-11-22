#!/usr/bin/env python3
"""Delegate training/evaluation to the unchanged original baseline trainer."""
import bootstrap
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler

import train_eeg19_i2moe as original
from complement_models import METHODS, ComplementMoE, method_contract, method_loss_terms
from complement_execution import install
from protected_evaluation import install as install_evaluation

from i2moe_eeg.utils import atomic_write_json


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity() -> dict:
    paths = list(bootstrap.ROOT.glob('*.py'))
    for module in tuple(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file and file.endswith('.py'):
            path = Path(file).resolve()
            if path.is_relative_to(bootstrap.BASELINE) or path.is_relative_to(bootstrap.SOURCE):
                paths.append(path)
    return {str(path): sha256(path) for path in sorted(set(paths))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--seed', type=int, choices=(3, 42), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('formal', 'integration', 'smoke'), default='formal')
    parser.add_argument('--resume', action='store_true')
    flags = parser.parse_args()
    baseline_dir = bootstrap.CACHE.parent / f'I2MOE6_F1/screen_20261112/baseline/seed{flags.seed}'
    original_args = json.loads((baseline_dir/'args.json').read_text())
    args = argparse.Namespace(**original_args)
    args.output_dir = str(flags.output_dir)
    args.resume = str(flags.output_dir/'resume.pt') if flags.resume else ''
    # A resource ceiling, not a training hyperparameter. Original B128 stays.
    args.memory_cap_gib = 4.0 if flags.method == 'baseline' else 32.0
    if flags.mode == 'integration':
        args.epochs, args.max_steps_per_epoch, args.max_val_samples = 1, 2, 256
    args.smoke_test = flags.mode == 'smoke'
    baseline_config = torch.load(baseline_dir/'best_f1.pt', map_location='cpu', weights_only=False)['config']
    flags.output_dir.mkdir(parents=True, exist_ok=True)
    cache_contract = install(original, flags.seed, flags.method)
    install_evaluation(original)
    identity = source_identity()
    contract = dict(cached_execution=cache_contract, method=flags.method, seed=flags.seed, mode=flags.mode,
                    original_args=str(baseline_dir/'args.json'), source_sha256=identity,
                    fresh_sixth_and_router=True, prior_trained_model_reused=False,
                    frozen_regional_checkpoints='exact original five baseline sources',
                    unchanged_loss=False, method_loss_and_fusion=method_contract(flags.method),
                    equal_training_draw_budget=True,
                    same_raw_input_and_frozen_features=True,
                    execution_input='FP32 packed frozen-source features, replacing only repeated frozen forwards',
                    outer_tta=False, independent_test=False,
                    validation_selection='original best F1 checkpoint and threshold grid',
                    resource_exception=dict(memory_cap_gib=args.memory_cap_gib),
                    architecture_note='Original core gradients preserved; cross-region or4independent moment heads; cap16power0.5 vs cap64power1 Traincohortweights; mean member corrected focal and0.03top16rank')
    contract_path = flags.output_dir/'strict_contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise RuntimeError('Source or strict experiment contract changed')
    atomic_write_json(contract_path, contract)
    original.parse_args = lambda: args
    original.loss_terms = lambda output, labels, args, focal: method_loss_terms(flags.method, output, labels, args, focal)
    build_model = original.build_model
    if flags.method != 'baseline':
        original.SixExpertF1 = lambda **kwargs: ComplementMoE(flags.method, **kwargs)

    def checked_build(actual_args):
        model = build_model(actual_args)
        metadata = cache_contract['train_cohort_metadata']
        model.correction.group_probabilities.copy_(torch.tensor(metadata['probabilities']))
        model.correction.group_base_weights.copy_(torch.tensor(metadata['weights']))
        initial = model.trainable_state_dict()
        digest = hashlib.sha256()
        for name, tensor in sorted(initial.items()):
            if not name.startswith(('correction.',)):
                digest.update(name.encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        atomic_write_json(flags.output_dir/'initialization.json', dict(
            original_trainable_state_sha256=digest.hexdigest(),
            parameters_trainable=sum(p.numel() for p in model.trainable_parameters()),
            correction_initialization='zero final projection; exact baseline logits at initialization'))
        return model

    original.build_model = checked_build
    semantic_config = original.semantic_config

    def checked_config(*values):
        config = semantic_config(*values)
        ignored = {'epochs', 'max_steps_per_epoch', 'max_val_samples'} if flags.mode == 'integration' else set()
        differences = {k: dict(baseline=baseline_config.get(k), current=v)
                       for k, v in config.items() if k not in ignored and baseline_config.get(k) != v}
        if differences:
            raise RuntimeError('Baseline configuration mismatch: '+json.dumps(differences))
        atomic_write_json(flags.output_dir/'configuration_check.json', dict(
            passed=True, checked_fields=list(config), differences=differences,
            debug_exceptions=sorted(ignored), formal=flags.mode == 'formal'))
        config['strict_extension'] = contract
        return config

    original.semantic_config = checked_config
    sample = original.sampled_indices

    def audited_sample(*values, **kwargs):
        indices, stats = sample(*values, **kwargs)
        seed, epoch = values[-2:]
        generator = torch.Generator().manual_seed(seed + epoch * 10000)
        order = np.asarray(list(SubsetRandomSampler(indices.tolist(), generator=generator)), dtype=np.int64)
        atomic_write_json(flags.output_dir/f'sampling_epoch{epoch:02d}.json', dict(
            **stats, sampled_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
            actual_batch_order_sha256=hashlib.sha256(order.tobytes()).hexdigest()))
        return indices, stats

    original.sampled_indices = audited_sample
    original.main()


if __name__ == '__main__':
    main()
