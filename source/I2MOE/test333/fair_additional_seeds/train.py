"""Run the original training loop with new seeds, native tails and a matched baseline."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from additional import HERE, METHOD, OUT, REFERENCE, SEEDS, SUPPLEMENT, bootstrap, digest, ordered, original, read
from cohort_execution import make_metadata
from i2moe_eeg.utils import atomic_write_json
from pair_evaluation import install as paired_evaluation
from refresh_execution import EPOCH_FIELD, ROW_FIELD, install as install_full
from refresh_models import RefreshedPairMoE, method_loss_terms
import train_strict


def install_supplement(seed, plan, full):
    contract = install_full(original, seed, METHOD)
    parent = original.EEG19NpyDataset
    mapping = np.load(SUPPLEMENT / 'mapping.npy', mmap_mode='r')
    regular = np.load(SUPPLEMENT / 'packets.npy', mmap_mode='r')
    tails = {item['epoch']: item for item in plan['tails'] if item['seed'] == seed}
    if set(tails) != set(range(1, 21)):
        raise ValueError('Incomplete native batch plan')

    class SupplementedDataset(parent):
        def __init__(self, frame, mode, *args, **kwargs):
            super().__init__(frame, mode, *args, **kwargs)
            if mode == 'Train':
                metadata, self.groups = make_metadata(frame)
                assert metadata['groups_sha256'] == contract['train_cohort_metadata']['groups_sha256']
                self.extra_tail_lookup = {}

        def set_epoch(self, epoch):
            self.original_epoch = epoch
            self.extra_tail_lookup = {row: i for i, row in enumerate(tails[epoch]['rows'])}
            self.extra_tail = np.load(SUPPLEMENT / f'tail_{seed}_{epoch}.npy', mmap_mode='r')

        def __getitem__(self, index):
            if self.mode == 'Val':
                return super().__getitem__(index)
            if index in self.extra_tail_lookup:
                value = self.extra_tail[self.extra_tail_lookup[index]]
            elif mapping[index] >= 0:
                value = regular[mapping[index]]
            else:
                return super().__getitem__(index)
            packet = torch.from_numpy(value.copy())
            packet[0, 3, 130] = int(self.groups[index])
            packet[0, 3, ROW_FIELD] = int(index)
            packet[0, 3, EPOCH_FIELD] = self.original_epoch
            return packet, int(self.labels[index])

    original.EEG19NpyDataset = SupplementedDataset
    if full:
        paired_evaluation(original)
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--method', choices=(METHOD, 'baseline'), required=True)
    flags = parser.parse_args()
    seed, method = flags.seed, flags.method
    full = method == METHOD
    plan = read(SUPPLEMENT / 'plan.json')
    plan_hash = digest(SUPPLEMENT / 'plan.json')
    markers = [read(SUPPLEMENT / f'shard{i}.json') for i in range(plan['shards'])]
    if any(m['plan_sha256'] != plan_hash for m in markers) or sum(m['rows'] for m in markers) != plan['missing_rows']:
        raise RuntimeError('Incomplete frozen feature supplement')
    output = OUT / 'formal' / method / f'seed{seed}'
    output.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(**read(REFERENCE / 'args.json'))
    args.seed, args.output_dir = seed, str(output)
    args.resume = str(output / 'resume.pt') if (output / 'resume.pt').exists() else ''
    baseline = torch.load(REFERENCE / 'best_f1.pt', map_location='cpu', weights_only=False)['config']
    assert baseline['seed'] == 3
    baseline['seed'] = seed
    if full:
        reference = read(bootstrap.OUTPUT / 'formal' / METHOD / 'seed3' / 'strict_contract.json')
        for file, expected in reference['source_sha256'].items():
            if digest(Path(file)) != expected:
                raise RuntimeError(f'Published source changed: {file}')
    cache_contract = install_supplement(seed, plan, full)
    contract = dict(seed=seed, method=method, original_args=str(REFERENCE / 'args.json'),
                    supplement_plan_sha256=plan_hash, cached_execution=cache_contract,
                    reference_seed=3, fresh_trainable_initialization=True,
                    source_sha256=train_strict.source_identity())
    proof = output / 'strict_contract.json'
    if proof.exists() and read(proof) != contract:
        raise RuntimeError('Existing training contract differs')
    atomic_write_json(proof, contract)
    orig_config = original.semantic_config
    def config(*values):
        value = orig_config(*values)
        if value != baseline:
            differences = [k for k in set(value) | set(baseline) if value.get(k) != baseline.get(k)]
            raise ValueError(f'Baseline configuration mismatch: {differences}')
        atomic_write_json(output / 'configuration_check.json',
                          dict(passed=True, seed=seed, reference_seed=3, fields=sorted(value)))
        return dict(value, strict_extension=contract)
    original.semantic_config = config
    orig_build = original.build_model
    def build(actual_args):
        model = orig_build(actual_args)
        if full:
            meta = cache_contract['train_cohort_metadata']
            model.correction.group_probabilities.copy_(torch.tensor(meta['probabilities']))
            model.correction.group_base_weights.copy_(torch.tensor(meta['weights']))
            model.correction.refinement_weights.copy_(torch.tensor(meta['weights']))
            model.correction.analytic.support.copy_((torch.tensor(meta['counts']) >= 50).all(dim=0))
        else:
            model.moments = {}
        h = hashlib.sha256()
        for name, value in sorted(model.trainable_state_dict().items()):
            if not name.startswith('correction.'):
                h.update(name.encode())
                h.update(value.detach().cpu().contiguous().numpy().tobytes())
        atomic_write_json(output / 'initialization.json', dict(original_trainable_state_sha256=h.hexdigest()))
        return model
    original.build_model = build
    if full:
        original.SixExpertF1 = lambda **kw: RefreshedPairMoE(method, **kw)
        original.loss_terms = lambda output, labels, args, focal: method_loss_terms(method, output, labels, args, focal)
    sample = original.sampled_indices
    expected_order = {t['epoch']: t['order_sha256'] for t in plan['tails'] if t['seed'] == seed}
    def audited(*values):
        indices, stats = sample(*values)
        actual_seed, epoch = values[-2:]
        order_hash = hashlib.sha256(ordered(indices, actual_seed, epoch).tobytes()).hexdigest()
        if actual_seed != seed or order_hash != expected_order[epoch]:
            raise ValueError('Native sampler order mismatch')
        atomic_write_json(output / f'sampling_epoch{epoch:02d}.json',
                          dict(**stats, actual_batch_order_sha256=order_hash))
        return indices, stats
    original.sampled_indices = audited
    original.parse_args = lambda: args
    original.main()


if __name__ == '__main__':
    main()
