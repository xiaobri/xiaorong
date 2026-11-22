"""Apply the original operating points after excluding potential cross-split subject overlaps."""
import czb_paths as _paths
import argparse
from pathlib import Path

import numpy as np

from compare_raw_operating_points import DEFAULT_CSV_PATH, atomic_write_json, load_manifest, read_run


def metrics(labels, predicted):
    tp = int(np.sum((labels==1)&predicted))
    fp = int(np.sum((labels==0)&predicted))
    fn = int(np.sum((labels==1)&~predicted))
    return dict(rows=len(labels),positives=int(labels.sum()),tp=tp,fp=fp,fn=fn,
                f1=2*tp/max(1,2*tp+fp+fn))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('method')
    args = parser.parse_args()
    frame, summary = load_manifest(DEFAULT_CSV_PATH)
    val = frame.loc[frame['mode']=='Val']
    labels = val.target.to_numpy()
    names = val.file_path.str.rsplit('/',n=2).str[-2]
    overlap = summary['inferred_parent_subject_audit']['overlap_ids']
    retained = ~names.isin(overlap).to_numpy()
    baseline = Path(f'{_paths.RUNS}/I2MOE_test333_strict_baseline_20261115/formal/baseline')
    results = []
    for seed in (3,42):
        candidate, c = read_run(args.output/'formal'/args.method/f'seed{seed}',labels,raw_replay=True)
        reference, b = read_run(baseline/f'seed{seed}',labels)
        results.append(dict(seed=seed,candidate_threshold=c['selected_threshold'],baseline_threshold=b['selected_threshold'],
            candidate_all=metrics(labels,candidate),baseline_all=metrics(labels,reference),
            candidate_excluding_overlap=metrics(labels[retained],candidate[retained]),
            baseline_excluding_overlap=metrics(labels[retained],reference[retained])))
    atomic_write_json(args.output/f'{args.method}_overlap_sensitivity.json',dict(rows=results,
        possible_overlapping_subject_names=len(overlap),excluded_windows=int((~retained).sum()),
        excluded_positive_windows=int(labels[~retained].sum()),exact_train_val_path_overlap=summary['train_val_exact_path_overlap_rows'],
        scope='Sensitivity only. Original selected operating points remain locked. Recording-parent basenames serve as possible subject identifiers. The available manifest does not establish patient-disjoint evaluation.'))
    print([(r['seed'],r['baseline_excluding_overlap']['f1'],r['candidate_excluding_overlap']['f1']) for r in results])


if __name__=='__main__':
    main()
