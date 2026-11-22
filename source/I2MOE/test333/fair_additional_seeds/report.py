"""Summarize completed runs at their locked thresholds and verify paired training histories."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

from additional import METHOD, OUT, SEEDS, digest, read
from i2moe_eeg.utils import atomic_write_json


def results(study_root):
    raw = [read(OUT / 'raw_replay' / f'shard{i}.json') for i in range(8)]
    spans = [(r['contract']['start'], r['contract']['stop']) for r in raw]
    assert spans[0][0] == 0 and spans[-1][1] == 166079
    assert all(a[1] == b[0] and a[1] % 128 == 0 for a, b in zip(spans, spans[1:]))
    locks = raw[0]['contract']['locks']
    assert all(r['contract']['locks'] == locks and
               r['contract']['manifest'] == raw[0]['contract']['manifest'] for r in raw)
    out = []
    pair_checks = []
    for seed in SEEDS:
        folders = [OUT / 'formal' / method / f'seed{seed}' for method in ('baseline', METHOD)]
        pairs = []
        for folder in folders:
            method = folder.parent.name
            key = f'{method}_{seed}'
            lock = locks[key]
            assert digest(folder / 'best_f1.pt') == lock['checkpoint_sha256']
            config = read(folder / 'configuration_check.json')
            assert config['passed'] and config['seed'] == seed
            samples = [read(folder / f'sampling_epoch{epoch:02d}.json') for epoch in range(1, 21)]
            assert all(s['samples'] == 20000 and s['final_partial_batch_size'] == 32 for s in samples)
            pairs.append((read(folder / 'initialization.json'), samples))
            c = {col: sum(r['counts'][key][col] for r in raw) for col in ('tn', 'fp', 'fn', 'tp')}
            assert sum(c.values()) == 166079 and c['fn'] + c['tp'] == 3471
            f1 = 2*c['tp'] / (2*c['tp'] + c['fp'] + c['fn'])
            epochs = [json.loads(s) for s in (folder / 'metrics.jsonl').read_text().splitlines()]
            metric = epochs[lock['epoch'] - 1]['validation']['best_f1_on_validation_grid']
            changed = sum(r['changed_decisions'][key] for r in raw)
            if changed == 0:
                assert all(metric[n] == c[n] for n in c)
            out.append(dict(seed=seed, phase='additional', method=method, epoch=lock['epoch'],
                            threshold=lock['threshold'], windows=sum(c.values()), **c, raw_f1=f1,
                            cached_f1=metric['f1'], changed_threshold_decisions=changed))
        assert pairs[0] == pairs[1]
        pair_checks.append(dict(seed=seed, matched_original_core_initialization=True,
                                matched_sample_order_epochs=20))
    published = list(csv.DictReader((study_root / 'results/selected.csv').open()))
    # The original component audit also contains the locked raw confusion counts.
    with (study_root / 'results/component_xiehe.csv').open() as f:
        component = list(csv.DictReader(f))
    for r in published:
        if r['method'] not in ('baseline', METHOD):
            continue
        audit = next(a for a in component if a['arm'] == r['method'] and a['seed'] == r['seed'])
        c = {name: int(audit[name]) for name in ('tp', 'tn', 'fp', 'fn')}
        assert all(int(r[k]) == c[k] for k in c)
        out.append(dict(seed=int(r['seed']), phase='development', method=r['method'],
                        epoch=int(r['epoch']), threshold=float(r['threshold']), windows=sum(c.values()),
                        **c, raw_f1=2*c['tp']/(2*c['tp']+c['fp']+c['fn']), cached_f1=float(r['f1']),
                        changed_threshold_decisions=int(audit['changed_cached_decisions'])))
    return out, pair_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, required=True)
    flags = parser.parse_args()
    rows, checks = results(flags.study_root)
    with (OUT / 'seed_results.csv').open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for phase in ('additional', 'development', 'all'):
        block = [r for r in rows if phase == 'all' or r['phase'] == phase]
        table = {method: sorted((r for r in block if r['method'] == method), key=lambda r:r['seed'])
                 for method in ('baseline', METHOD)}
        values = {m: [100*r['raw_f1'] for r in rr] for m, rr in table.items()}
        delta = [f - b for f, b in zip(values[METHOD], values['baseline'])]
        summary[phase] = {m: dict(mean=mean(v), sample_sd=stdev(v)) for m, v in values.items()}
        summary[phase]['paired_gain_points'] = mean(delta)
    atomic_write_json(OUT / 'seed_summary.json', dict(summary=summary, pair_checks=checks,
                      raw_windows=166079, positive_windows=3471, independent_test=False))
    print(summary)


if __name__ == '__main__':
    main()
