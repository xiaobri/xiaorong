"""Delegate an isolated fit to the original baseline trainer and audit each draw."""
import argparse
import hashlib
import sys

import torch

from single import ARMS, HERE, OUT, PAIR, SEEDS, bootstrap, build, loss
sys.path.insert(0, str(HERE.parent / 'fair_additional_seeds'))
from additional import OUT as REPLICATION, REFERENCE, SUPPLEMENT, digest, ordered, original, read
from train import install_supplement
from refresh_execution import install
from i2moe_eeg.utils import atomic_write_json


def reference_dir(seed):
    if seed in (3, 42):
        return bootstrap.OUTPUT.parent / f'I2MOE_test333_strict_baseline_20261115/formal/baseline/seed{seed}'
    return REPLICATION / 'formal/baseline' / f'seed{seed}'


def run(arm, seed):
    folder = OUT / 'formal' / arm / f'seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(**read(REFERENCE / 'args.json'))
    args.seed, args.output_dir = seed, str(folder)
    args.resume = str(folder / 'resume.pt') if (folder / 'resume.pt').exists() else ''
    expected = torch.load(REFERENCE / 'best_f1.pt', map_location='cpu', weights_only=False)['config']
    expected['seed'] = seed
    if seed in (3, 42):
        cache = install(original, seed, 'fresh_both')
    else:
        plan = read(SUPPLEMENT / 'plan.json')
        plan_hash = digest(SUPPLEMENT / 'plan.json')
        assert all(read(SUPPLEMENT / f'shard{i}.json')['plan_sha256'] == plan_hash for i in range(plan['shards']))
        cache = install_supplement(seed, plan, False)
    meta = cache['train_cohort_metadata']
    ref = reference_dir(seed)
    original.parse_args = lambda: args
    original.loss_terms = lambda out,y,a,f: loss(arm,out,y,a,f)
    original.SixExpertF1 = lambda **kw: build(arm, **kw)
    builder = original.build_model
    def checked_builder(args):
        model = builder(args)
        for name, key in [('group_probabilities','probabilities'), ('group_base_weights','weights')]:
            if hasattr(model.correction, name):
                getattr(model.correction, name).copy_(torch.tensor(meta[key]))
        if arm in PAIR:
            model.correction.analytic.support.copy_((torch.tensor(meta['counts']) >= 50).all(0))
        h = hashlib.sha256()
        for n,t in sorted(model.trainable_state_dict().items()):
            if not n.startswith('correction.'):
                h.update(n.encode())
                h.update(t.detach().cpu().contiguous().numpy().tobytes())
        if h.hexdigest() != read(ref/'initialization.json')['original_trainable_state_sha256']:
            raise ValueError('Initial core changed')
        atomic_write_json(folder/'initialization.json',dict(core=h.hexdigest(),
                          parameters=sum(p.numel() for p in model.trainable_parameters())))
        return model
    original.build_model = checked_builder
    config = original.semantic_config
    def checked_config(*values):
        actual = config(*values)
        if actual != expected:
            raise ValueError(f'Config differs: {[k for k in actual if expected.get(k)!=actual[k]]}')
        design=dict(arm=arm, seed=seed, code={f:digest(HERE/f) for f in ('fit.py','single.py')})
        atomic_write_json(folder/'configuration_check.json', dict(passed=True, **design))
        return dict(actual, single_component=design)
    original.semantic_config = checked_config
    sample = original.sampled_indices
    def audited(*v):
        ix, stats = sample(*v)
        order = ordered(ix, v[-2], v[-1])
        sha = hashlib.sha256(order.tobytes()).hexdigest()
        fname = f'sampling_epoch{v[-1]:02d}.json'
        base = read(ref / fname)
        if sha != base['actual_batch_order_sha256'] or any(stats[k] != base[k] for k in stats):
            raise ValueError('Different baseline sampling')
        atomic_write_json(folder/fname, dict(**stats, actual_batch_order_sha256=sha))
        return ix, stats
    original.sampled_indices = audited
    original.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--seed', choices=SEEDS, type=int, required=True)
    a = parser.parse_args()
    run(a.arm, a.seed)
