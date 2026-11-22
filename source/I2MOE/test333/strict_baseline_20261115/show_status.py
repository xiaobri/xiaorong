#!/usr/bin/env python3
"""Read the strict sweep without touching running processes or checkpoints."""
import czb_paths as _paths
import json
from pathlib import Path

OUTPUT = Path(f'{_paths.RUNS}/I2MOE_test333_strict_baseline_20261115')


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


if __name__ == '__main__':
    coordinator = read(OUTPUT/'coordinator_status.json')
    rows = []
    for gpu, (method, seed) in enumerate((method, seed)
            for method in ('baseline', 'channel', 'orbit', 'joint') for seed in (3, 42)):
        folder = OUTPUT/'formal'/method/f'seed{seed}'
        job, metrics = read(folder/'job_status.json'), read(folder/'latest_metrics.json')
        records = []
        if (folder/'metrics.jsonl').exists():
            records = [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        best = max((row['validation']['best_f1_on_validation_grid']['f1'] for row in records), default=None)
        rows.append(dict(gpu=gpu, method=method, seed=seed, status=job.get('status', 'not_started'),
                         completed_epoch=metrics.get('epoch', 0), best_validation_f1=best,
                         configuration_check=read(folder/'configuration_check.json').get('passed')))
    print(json.dumps(dict(coordinator=coordinator, jobs=rows), indent=2))

