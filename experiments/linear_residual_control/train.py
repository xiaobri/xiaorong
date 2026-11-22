"""Train the original core with detached readouts and a matched trajectory audit."""
import argparse
import copy
import hashlib
import time
from unittest.mock import patch

import numpy as np
import torch

from study import (ARMS, RATES, HERE, OUT, Learner, Observations, corrected, digest,
                   fit, folder, load, name, original, read, state_hash)
from metrics import scan


def training_pass(train, values, learners, support):
    model, dataset, optimizer, scaler, focal, args, device, epoch = values
    features = Observations(model)
    inputs = {}
    totals = {k: dict(loss=0., examples=0, steps=0, seconds=0.) for k in learners}
    base_loss = original.loss_terms

    def observe(module, batch, output):
        packet = batch[0]
        inputs['x'] = features(packet, output)
        inputs['eligible'] = support[packet[:, 0, 3, 130].long()]

    def loss(output, labels, flags, criterion):
        parts = base_loss(output, labels, flags, criterion)
        for key, learner in learners.items():
            row = totals[key]
            torch.cuda.synchronize()
            start = time.perf_counter()
            step = learner.step(output.logits, inputs['x'], labels, inputs['eligible'], criterion)
            if step:
                loss_value, n = step
                row['loss'] += float(loss_value) * n
                row['examples'] += n
                row['steps'] += 1
            torch.cuda.synchronize()
            row['seconds'] += time.perf_counter() - start
        return parts

    try:
        with model.register_forward_hook(observe), patch.object(original, 'loss_terms', loss):
            training = train(*values)
    finally:
        features.close()
    return training, totals


def main(seed):
    dest = folder(seed)
    dest.mkdir(parents=True, exist_ok=True)
    ref = fit.reference_dir(seed)
    args = argparse.Namespace(**read(fit.REFERENCE / 'args.json'))
    args.seed, args.output_dir = seed, str(dest)
    args.resume = str(dest / 'resume.pt') if (dest / 'resume.pt').is_file() else ''
    if seed in (3, 42):
        settings = [(a, lr) for a in ARMS for lr in RATES]
        cache = fit.install(original, seed, 'fresh_both')
    else:
        lock = read(OUT / 'selection.json')
        settings = [(a, lock['rates'][a]) for a in ARMS]
        plan = read(fit.SUPPLEMENT / 'plan.json')
        assert all(read(fit.SUPPLEMENT / f'shard{i}.json')['plan_sha256'] ==
                   digest(fit.SUPPLEMENT / 'plan.json') for i in range(plan['shards']))
        cache = fit.install_supplement(seed, plan, False)
    meta = cache['train_cohort_metadata']
    assert meta['source'] == 'Train rows only'
    support = (torch.tensor(meta['counts'], device='cuda') >= 50).all(0)
    learners = {name(a, lr): Learner(a, lr, 'cuda') for a, lr in settings}
    expected = load(fit.REFERENCE / 'best_f1.pt')['config']
    expected['seed'] = seed
    config = original.semantic_config
    builder = original.build_model
    sample = original.sampled_indices
    train, evaluate = original.train_one_epoch, original.evaluate
    design = dict(seed=seed, settings=[list(s) for s in settings],
                  code={f:digest(HERE / f) for f in ('train.py','study.py','metrics.py')},
                  train_group_sha256=meta['groups_sha256'])
    last = load(dest / 'resume.pt')['epoch'] if args.resume else 0
    if last:
        saved = load(dest / f'heads{last:02d}.pt')
        assert saved['design'] == design and saved['epoch'] == last
        for k, learner in learners.items():
            learner.restore(saved['learners'][k])
    shadow = {}
    epoch_data = {}

    def checked_config(*v):
        actual = config(*v)
        if actual != expected:
            raise ValueError(f'Semantic protocol differs: {[k for k in actual if actual[k] != expected.get(k)]}')
        return dict(actual, linear_control=design)

    def checked_build(flags):
        model = builder(flags)
        model.moments = {}
        h = state_hash(model.trainable_state_dict())
        assert h == read(ref/'initialization.json')['original_trainable_state_sha256']
        original.atomic_write_json(dest / 'init.json', dict(sha256=h, eligible_groups=int(support.sum()),
                                 eligible_rule='Train each label >= 50', design=design))
        return model

    def audited(*values):
        ix, stats = sample(*values)
        order = fit.ordered(ix, *values[-2:])
        h = hashlib.sha256(order.tobytes()).hexdigest()
        baseline = read(ref / f'sampling_epoch{values[-1]:02d}.json')
        assert h == baseline['actual_batch_order_sha256'] and all(baseline[k] == v for k,v in stats.items())
        epoch_data['order'] = h
        return ix, stats

    def train_epoch(*values):
        model, dataset, optimizer, scaler, focal, flags, device, epoch = values
        if not shadow:
            shadow['core'] = copy.deepcopy(model)
            shadow['opt'] = torch.optim.AdamW(shadow['core'].trainable_parameters(),
                                              lr=flags.lr, weight_decay=flags.weight_decay)
            shadow['opt'].load_state_dict(copy.deepcopy(optimizer.state_dict()))
            shadow['scaler'] = copy.deepcopy(scaler)
        active = shadow['core']
        rng = original.capture_rng_state()
        started = time.monotonic()
        stats, totals = training_pass(train, values, learners, support)
        after = original.capture_rng_state()
        original.restore_rng_state(rng, device)
        comparison = (active, dataset, shadow['opt'], shadow['scaler'], focal, flags, device, epoch)
        if epoch == 1:
            arm, lr = settings[0]
            independent = Learner(arm, lr, device)
            training_pass(train, comparison, {'isolated': independent}, support)
            peer = learners[name(arm, lr)]
            torch.testing.assert_close(independent.head.state_dict(), peer.head.state_dict(), rtol=0, atol=0)
            torch.testing.assert_close(independent.opt.state_dict()['state'], peer.opt.state_dict()['state'],
                                       rtol=0, atol=0)
        else:
            train(*comparison)
        original.restore_rng_state(after, device)
        live = model.trainable_state_dict()
        h = state_hash(live)
        audit_hash = state_hash(active.trainable_state_dict())
        assert h == audit_hash, 'Independent core trajectory diverged'
        archive = load(ref / f'epoch{epoch:02d}.pt')['model']
        dif = max((t.cpu() - archive[k]).abs().max().item() for k,t in live.items())
        for row in totals.values():
            assert np.isfinite(row['loss']) and row['steps'] > 0
        epoch_data.update(training=stats, totals=totals, core=h, shadow=audit_hash,
                          archived_max_abs=dif, start=started)
        return stats

    def evaluate_epoch(model, dataset, focal, flags, device, epoch):
        features = Observations(model)
        probabilities = []
        def record(module, inputs, output):
            x = features(inputs[0], output)
            probabilities.append(torch.stack([corrected(output.logits, x, l.head).softmax(1)[:,1]
                                               for l in learners.values()], 1).detach().cpu().numpy())
        try:
            with model.register_forward_hook(record):
                validation = evaluate(model, dataset, focal, flags, device, epoch)
        finally:
            features.close()
        scores = np.concatenate(probabilities)
        labels = np.asarray(dataset.labels)
        assert len(labels) == 166079 and labels.sum() == 3471
        stats = epoch_data['training']
        rows = {}
        for j, (k, learner) in enumerate(learners.items()):
            t = epoch_data['totals'][k]
            row = dict(arm=learner.arm, seed=seed, epoch=epoch, head_lr=learner.lr,
                       **scan(labels,scores[:,j],original.threshold_grid(flags)),
                       core_loss=stats['loss'], core_primary=stats['final_task_loss'],
                       core_auxiliary=stats['auxiliary_task_loss'], core_interaction=stats['interaction_loss'],
                       head_loss=t['loss']/t['examples'], draws=stats['samples'], core_steps=stats['steps'],
                       head_steps=t['steps'], head_examples=t['examples'], head_seconds=t['seconds'],
                       weight_norm=float(learner.head.weight.norm()), bias=float(learner.head.bias.detach()),
                       batch_order_sha256=epoch_data['order'], core_state_sha256=epoch_data['core'],
                       matched_core_sha256=epoch_data['shadow'], archive_core_max_abs=epoch_data['archived_max_abs'],
                       shared_train_seconds=stats['seconds'], shared_epoch_seconds=time.monotonic()-epoch_data['start'],
                       peak_gpu_bytes=torch.cuda.max_memory_allocated(device))
            assert np.isfinite(row['weight_norm']) and row['weight_norm'] > 0
            rows[k] = row
        np.save(dest / f'scores{epoch:02d}.npy', scores)
        original.atomic_write_json(dest / f'rows{epoch:02d}.json', rows)
        original.atomic_torch_save(dict(epoch=epoch, design=design,
            learners={k:l.save() for k,l in learners.items()}), dest / f'heads{epoch:02d}.pt')
        return validation

    original.parse_args = lambda: args
    original.build_model, original.semantic_config, original.sampled_indices = checked_build, checked_config, audited
    original.train_one_epoch, original.evaluate = train_epoch, evaluate_epoch
    original.main()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', required=True, type=int, choices=(3,42,0,1,2))
    main(p.parse_args().seed)
