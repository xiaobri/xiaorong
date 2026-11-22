"""Recalculate the public ablation measurements with the Python standard library."""
import argparse
import csv
from collections import Counter
from math import isclose
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2, 3, 42)
COUNTS = ('tp', 'fp', 'fn', 'tn')


def read(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def same(actual, expected):
    if not isclose(float(actual), float(expected), abs_tol=1e-9, rel_tol=0):
        raise ValueError(f'Measurements differ: {actual} versus {expected}')


def score(row):
    tp, fp, fn, tn = [int(row[k]) for k in COUNTS]
    if min(tp, fp, fn, tn) < 0 or tp + fn != 3471 or tn + fp != 162608:
        raise ValueError('Unexpected Xiehe validation confusion counts')
    return 200 * tp / (2 * tp + fp + fn)


def verify(only=None):
    data = ROOT / 'results'
    selected = read(data / 'single_component_seeds.csv')
    summaries = read(data / 'single_component_summary.csv')
    table = {(r['arm'], int(r['seed'])):r for r in selected}
    arms = {d.name for d in (ROOT / 'experiments').iterdir() if d.is_dir() and (d / 'run.py').exists()}
    all_arms = arms | {'baseline', 'fresh_both'}
    if len(arms) != 13 or len(selected) != 75 or set(table) != {(a,s) for a in all_arms for s in SEEDS}:
        raise ValueError('Incomplete or duplicated arm/seed results')
    if len(summaries) != len(all_arms) or {r['arm'] for r in summaries} != all_arms:
        raise ValueError('Incomplete summary')
    if only is not None and only not in arms:
        raise ValueError(f'Unknown intervention: {only}')
    reference_orders = read(data / 'baseline_pairing.csv')
    identities = read(data / 'fit_identities.csv')
    orders = {(int(r['seed']),int(r['epoch'])):r for r in reference_orders}
    if len(reference_orders) != 100 or set(orders) != {(s,e) for s in SEEDS for e in range(1,21)}:
        raise ValueError('Incomplete baseline sampling proof')
    proof = {(r['arm'],int(r['seed'])):r for r in identities}
    if len(identities) != 65 or set(proof) != {(a,s) for a in arms for s in SEEDS}:
        raise ValueError('Incomplete selected-fit identities')
    archived = Counter()
    checked = 0
    for summary in summaries:
        a = summary['arm']
        if only is not None and a != only:
            continue
        folder = ROOT / ('experiments' if a in arms else 'controls') / a
        local = read(folder / 'results.csv')
        if local != [table[a,s] for s in SEEDS]:
            raise ValueError(f'Experiment folder differs from central table: {a}')
        values, gains = [], []
        for s, r in zip(SEEDS, local):
            val = score(r)
            same(r['f1_points'], val)
            same(summary[f'seed{s}_F1'], val)
            gain = val - score(table['baseline',s])
            same(gain, r['delta_vs_baseline'])
            gains.append(gain)
            values.append(val)
        for col, expected in [('mean_F1',mean(values)),('sd_F1',stdev(values)),('mean_gain',mean(gains)),
                              ('sd_gain',stdev(gains)),('positive_seeds',sum(g>0 for g in gains)),
                              ('development_gain',mean(gains[3:])),('additional_gain',mean(gains[:3]))]:
            same(summary[col], expected)
        if a not in arms:
            continue
        records = read(folder / 'epochs.csv')
        if len(records) != 100:
            raise ValueError(f'Incomplete training records in {a}')
        for s in SEEDS:
            rows = [r for r in records if int(r['seed']) == s]
            if [int(r['epoch']) for r in rows] != list(range(1,21)):
                raise ValueError(f'Missing or out-of-order epoch in {a}/{s}')
            for row in rows:
                same(score(row), row['cached_f1_points'])
                t = float(row['threshold'])
                if not 1 <= round(t*100) <= 99:
                    raise ValueError('Threshold outside the selection grid')
                same(t*100, round(t*100))
                if (int(row['training_draws']),int(row['optimizer_steps'])) != (20000,157):
                    raise ValueError('Changed training budget')
                key = (s,int(row['epoch']))
                order = row['batch_order_sha256']
                if order != orders[key]['batch_order_sha256']:
                    raise ValueError(f'Different paired sampling order at {key}')
            if any(proof[a,s]['core_init_sha256'] != orders[s,e]['baseline_core_init_sha256']
                   for e in range(1,21)):
                raise ValueError(f'Core initialization differs for {a}/{s}')
            best = max(rows,key=lambda r:(float(r['cached_f1_points']),int(r['epoch'])))
            raw = table[a,s]
            same(best['epoch'], raw['epoch'])
            same(best['threshold'], raw['threshold'])
            moved = int(raw['changed_cached_decisions'])
            minimum = abs(int(best['tp'])-int(raw['tp'])) + abs(int(best['fp'])-int(raw['fp']))
            if moved < minimum or (moved - minimum) % 2:
                raise ValueError('Inconsistent cached/raw decisions')
            historical = a in ('moment_plain','moment_recipe') and s in (3,42)
            expected_source = ('original locked single-head replay' if historical
                               else 'new isolated fit and locked replay')
            if raw['source'] != expected_source:
                raise ValueError(f'Different training provenance in {a}/{s}')
            archived[historical] += 1
        checked += len(records)
    contrasts = read(data / 'dependent_contrasts.csv')
    for r in contrasts:
        deltas = [score(table[r['arm'],s])-score(table[r['reference'],s]) for s in SEEDS]
        same(r['mean_difference'], mean(deltas))
        for s, val in zip(SEEDS,deltas):
            same(r[f'seed{s}'], val)
    if only is None and archived != {True:4, False:61}:
        raise ValueError('Expected 61 fresh and four historical ablation fits')
    print(f'OK: {only or "13 ablations + 2 controls"}; {checked} epoch records; '
          f'{archived[False]} new and {archived[True]} historical fits.')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm')
    verify(p.parse_args().arm)
