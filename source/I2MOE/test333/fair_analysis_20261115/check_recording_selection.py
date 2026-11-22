"""Measure threshold and epoch transfer across existing validation recording folders."""
import czb_paths as _paths
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from compare_raw_operating_points import DEFAULT_CSV_PATH, atomic_write_json, load_manifest
from train_eeg19_i2moe import threshold_grid


def summarize_counts(tp, fp, fn):
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def check(run: Path, labels: np.ndarray, folds: np.ndarray) -> dict:
    threshold = threshold_grid(argparse.Namespace(**json.loads((run/'args.json').read_text())))
    history = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    if len(history) != 20:
        raise ValueError(f'Incomplete training at {run}')
    counts = np.zeros((20, len(threshold), 5, 3), dtype=np.int64)
    for index, record in enumerate(history):
        epoch = record['epoch']
        with np.load(run/f'predictions_epoch{epoch:02d}.npz') as saved:
            np.testing.assert_array_equal(saved['labels'], labels)
            probability = saved['probabilities'].astype(np.float64)
        for f in range(5):
            held = folds == f
            positive = np.sort(probability[held & (labels == 1)])
            negative = np.sort(probability[held & (labels == 0)])
            missing = np.searchsorted(positive, threshold, side='right')
            tp = len(positive) - missing
            fp = len(negative) - np.searchsorted(negative, threshold, side='right')
            counts[index, :, f] = np.stack((tp, fp, missing), axis=-1)
        full = counts[index].sum(1).T
        known = record['validation']['best_f1_on_validation_grid']
        values = summarize_counts(*full)
        np.testing.assert_allclose(values.max(), known['f1'], atol=1e-12, rtol=0.)
    total = counts.sum(2)
    held_counts = []
    rows = []
    for f in range(5):
        selection = (total - counts[:, :, f]).transpose(2, 0, 1)
        metrics = summarize_counts(*selection)
        t = np.argmax(metrics, axis=1)
        m = metrics[np.arange(20), t]
        # The original threshold rule keeps the first grid tie, and the epoch rule keeps the last tie.
        best_epoch = int(np.flatnonzero(m == m.max())[-1])
        best_threshold = t[best_epoch]
        held = counts[best_epoch, best_threshold, f]
        held_counts.append(held)
        rows.append(dict(fold=f, selected_epoch=best_epoch+1, threshold=float(threshold[best_threshold]),
            selection_f1=float(m.max()), held_f1=float(summarize_counts(*held)),
            held_counts=held.tolist()))
    return dict(folds=rows, held_pooled_f1=float(summarize_counts(*np.sum(held_counts, axis=0))),
        full_selected_f1=float(summarize_counts(*total.transpose(2,0,1)).max()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('method')
    args = parser.parse_args()
    frame, _ = load_manifest(DEFAULT_CSV_PATH)
    val = frame.loc[frame['mode'] == 'Val']
    labels = val.target.to_numpy()
    folders = val.file_path.str.rsplit('/', n=1).str[0].to_numpy()
    folds = np.full(len(labels), -1)
    for i, (_, test) in enumerate(StratifiedGroupKFold(5).split(labels[:, None], labels, folders)):
        folds[test] = i
    assert min(folds) == 0 and all(len(np.unique(folds[folders == g])) == 1 for g in np.unique(folders))
    ref = Path(f'{_paths.RUNS}/I2MOE_test333_strict_baseline_20261115/formal/baseline')
    result = {}
    for s in (3, 42):
        result[s] = {n:check(d/f'seed{s}', labels, folds) for n,d in
            [('baseline',ref),('candidate',args.output/'formal'/args.method)]}
        print(s, {n:round(100*v['held_pooled_f1'],4) for n,v in result[s].items()})
    atomic_write_json(args.output/f'{args.method}_recording_selection.json', dict(method=args.method, seeds=result,
        grouping='Five deterministic stratified folds by full recording folder, shared by both seeds and models. Each window is scored once at a checkpoint and threshold selected without its recording folder.',
        scope='Sensitivity to epoch and threshold selection on the reused original validation set. The architecture was already developed on these data. This is neither an independent evaluation nor the primary success metric.'))


if __name__ == '__main__':
    main()
