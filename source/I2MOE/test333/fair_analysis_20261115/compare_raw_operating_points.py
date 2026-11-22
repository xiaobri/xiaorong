"""Compare the confirmed correction with raw baseline runs at fixed selected thresholds."""
import czb_paths as _paths
import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, f'{_paths.BASELINE}')
from i2moe_eeg.data import load_manifest, DEFAULT_CSV_PATH
from i2moe_eeg.utils import atomic_write_json

HOME_DIR = Path(f'{_paths.RUNS}')
TARGET = HOME_DIR/'I2MOE_test333_fair_complement_20261118'


def read_run(folder, labels, raw_replay=False):
    epochs = [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
    best = max(epochs,key=lambda e:(e['validation']['best_f1_on_validation_grid']['f1'],e['epoch']))
    by_ap = max(epochs,key=lambda e:(e['validation']['auprc'],e['epoch']))
    val = best['validation']['best_f1_on_validation_grid']
    pred_folder = folder/'raw_replay' if raw_replay else folder
    with np.load(pred_folder/f"predictions_epoch{best['epoch']:02d}.npz") as data:
        np.testing.assert_array_equal(data['labels'],labels)
        predicted = data['probabilities'].astype(np.float64)>val['threshold']
    tp = np.count_nonzero(predicted & (labels==1))
    fixed = 2 * tp / (np.count_nonzero(predicted) + np.count_nonzero(labels==1))
    info = dict(best_f1_epoch=best['epoch'],selected_threshold=val['threshold'],fixed_f1=float(fixed),
                selection_grid_f1=val['f1'],best_ap_epoch=by_ap['epoch'],
                f1_at_best_ap_epoch=by_ap['validation']['best_f1_on_validation_grid']['f1'])
    if raw_replay:
        proof = json.loads((pred_folder/'replay_result.json').read_text())
        assert proof['full_original_six_experts_recomputed'] and proof['rows']==len(labels)
        info['raw_grid_best_f1'] = proof['raw_metric']['f1']
        if proof['raw_metric']['threshold'] == val['threshold']:
            np.testing.assert_allclose(fixed,info['raw_grid_best_f1'],atol=1e-12,rtol=0)
    else:
        np.testing.assert_allclose(fixed,val['f1'],atol=1e-12,rtol=0)
    return predicted, info


def counts(pred, label, groups):
    return np.stack([np.bincount(groups,weights=mask) for mask in
                    ((label==1)&pred,(label==0)&pred,(label==1)&~pred)],axis=1)


def f1(count):
    tp, fp, fn = count.T
    return 2*tp/np.maximum(1,2*tp+fp+fn)


def paired_intervals(candidate, baseline, repeats=3000):
    rng = np.random.default_rng(20261119)
    gaps = []
    for i in range(0,repeats,50):
        draws = rng.integers(0,len(baseline),size=(min(50,repeats-i),len(baseline)))
        gaps.extend(f1(candidate[draws].sum(1))-f1(baseline[draws].sum(1)))
    return np.quantile(gaps,[.025,.975]).tolist()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=TARGET)
    parser.add_argument('--method',default='risk_power')
    flags=parser.parse_args()
    frame, _ = load_manifest(DEFAULT_CSV_PATH)
    val = frame[frame['mode']=='Val']
    labels = val['target'].to_numpy()
    # Full directory path distinguishes recording folders with repeated basenames between acquisition sources.
    directories = val['file_path'].str.rsplit('/',n=1).str[0]
    names, groups = np.unique(directories.to_numpy(),return_inverse=True)
    results = []
    for seed in (3,42):
        run = f'seed{seed}'
        pred, current = read_run(flags.output/'formal'/flags.method/run,labels,True)
        candidate = counts(pred,labels,groups)
        for label, path, raw in (
            ('historical_raw',Path(f'{_paths.STORAGE}/I2MOE6_F1/screen_20261112/baseline')/run,False),
            ('strict_raw_rerun',HOME_DIR/'I2MOE_test333_strict_baseline_20261115/formal/baseline'/run,False),
            ('paired_cache_raw_replay',HOME_DIR/'I2MOE_test333_fair_cached_20261115/formal/baseline'/run,True)):
            other, reference = read_run(path,labels,raw)
            base = counts(other,labels,groups)
            delta = f1(candidate.sum(0))-f1(base.sum(0))
            np.testing.assert_allclose(delta,current['fixed_f1']-reference['fixed_f1'],atol=1e-10)
            results.append(dict(seed=seed,reference_type=label,baseline=reference,candidate=current,
                fixed_point_paired_delta=delta,folder_bootstrap_95_interval=paired_intervals(candidate,base),
                change_true_positives=int(candidate[:,0].sum()-base[:,0].sum()),
                change_false_positives=int(candidate[:,1].sum()-base[:,1].sum())))
    atomic_write_json(flags.output/'paired_raw_operating_points.json',dict(method=flags.method,rows=results,
        distinct_validation_recording_directories=len(names),resamples=3000,
        grouping='Full parent directory, a recording proxy. Patient identity not established.',
        scope='Paired empirical window-F1 difference resampling whole recording folders. Conditions on selected checkpoints, grid thresholds and the reused Val set; excludes checkpoint/model selection and training uncertainty.'))


if __name__ == '__main__':
    main()
