"""Join verified private predictions and publish only aggregate counts and digests."""
import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np

from metrics import measure, scan
from raw_scores import STAGES
from selection import lock, trajectory, winners
from study import ARMS, HERE, OUT, REFS, ROOT, SEEDS, digest, fit, folder, load, name, read


def write(directory, filename, records):
    with (directory/filename).open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def private_scores(stage):
    path = OUT/'raw'/stage
    files = sorted(path.glob('shard*.json'),key=lambda f:int(f.stem[5:]))
    cols, labels, offset, first = [], [], 0, None
    for i,file in enumerate(files):
        mark = read(file)
        c = mark['contract']
        if first is None:
            first = c
        assert (c['start'] == offset and c['stop'] > offset and file.stem == f'shard{i}'
                and len(files)==c['shards'] and all(c[k]==first[k] for k in c if k not in ('start','stop')))
        npz = file.with_suffix('.npz')
        assert digest(npz) == mark['score_sha256']
        with np.load(npz) as z:
            scores, y = z['scores'],z['labels']
        assert scores.shape == (c['stop']-offset,len(c['keys'])) and len(y)==len(scores)
        cols.append(scores)
        labels.append(y)
        offset=c['stop']
    assert offset==166079
    return first, np.concatenate(cols), np.concatenate(labels)


def matched_archive(seed, epoch):
    core=load(folder(seed)/f'epoch{epoch:02d}.pt')['model']
    errs={}
    for ref in REFS[1:]:
        old=load(fit.OUT/'formal'/ref/f'seed{seed}'/f'epoch{epoch:02d}.pt')['model']
        errs[f'{ref}_core_max_abs']=max((v-old[k]).abs().max().item() for k,v in core.items())
    return errs


def generate(dest):
    dest.mkdir(exist_ok=True,parents=True)
    rates=lock()
    historical={(r['arm'],int(r['seed'])):r for r in csv.DictReader((ROOT/'results/single_component_seeds.csv').open())}
    seed_rows,diagnostics,epochs,tuning,runtime,sensitivity=[],[],[],[],[],[]
    public_probe=read(OUT/'probe.json')
    assert public_probe['eval_state_unchanged']
    for stage in STAGES:
        c,scores,y=private_scores(stage)
        for j,ident in enumerate(c['keys']):
            locked=c['locks'][ident]
            seed,arm,epoch=locked['seed'],locked['arm'],locked['epoch']
            threshold=locked['threshold']
            t=scan(y,scores[:,j],[threshold])
            exact=scan(y,scores[:,j])
            if arm in ARMS:
                d=folder(seed)
                key=name(arm,rates[arm])
                snapshot=load(d/f'heads{epoch:02d}.pt')
                index=list(snapshot['learners']).index(key)
                cache=np.load(d/f'scores{epoch:02d}.npy',mmap_mode='r')[:,index]
                with np.load(d/f'predictions_epoch{epoch:02d}.npz') as z:
                    np.testing.assert_array_equal(z['labels'],y)
                expected=winners(seed)[key]
                source='new head, locked raw EEG replay'
            else:
                d=fit.reference_dir(seed) if arm=='baseline' else fit.OUT/'formal'/arm/f'seed{seed}'
                with np.load(d/f'predictions_epoch{epoch:02d}.npz') as z:
                    np.testing.assert_array_equal(z['labels'],y)
                    cache=z['probabilities']
                expected=load(d/'best_f1.pt')['validation']['best_f1_on_validation_grid']
                source='archived checkpoint, new raw EEG replay'
            change=np.count_nonzero((cache.astype(np.float64)>threshold)!=(scores[:,j].astype(np.float64)>threshold))
            cached=scan(y,cache,[threshold])
            assert all(cached[k]==expected[k] for k in ('tp','fp','fn','tn','threshold'))
            seed_rows.append(dict(arm=arm,seed=seed,head_lr=locked['head_lr'],head_weight_decay=.01 if arm in ARMS else '',
                            epoch=epoch,threshold_protocol='99-point Val grid, raw locked replay',**t,
                            changed_cached_decisions=int(change),source=source,
                            checkpoint_sha256=locked['checkpoint_sha256'],
                            archived_f1_points=historical[arm,seed]['f1_points'] if arm in REFS else ''))
            diag=dict(arm=arm,seed=seed,epoch=epoch,checkpoint_sha256=locked['checkpoint_sha256'])
            diag.update({f'grid_{k}':v for k,v in t.items()})
            diag.update({f'exact_{k}':v for k,v in exact.items()})
            diag.update(exact_minus_grid_points=exact['f1_points']-t['f1_points'],
                        distinct_probabilities=len(np.unique(scores[:,j])))
            diagnostics.append(diag)
    order=REFS+ARMS
    seed_rows.sort(key=lambda r:(order.index(r['arm']),r['seed']))
    diagnostics.sort(key=lambda r:(order.index(r['arm']),r['seed']))
    references={(r['arm'],r['seed']):r['f1_points'] for r in seed_rows if r['arm'] in REFS}
    for r in seed_rows:
        for ref in REFS:
            r['delta_vs_'+ref]=r['f1_points']-references[ref,r['seed']]
    for seed in SEEDS:
        full=trajectory(seed)
        wins=winners(seed)
        d=folder(seed)
        ledger=[json.loads(s) for s in (d/'metrics.jsonl').read_text().splitlines()]
        native=max(ledger,key=lambda l:(l['validation']['best_f1_on_validation_grid']['f1'],l['epoch']))
        core=native['validation']['best_f1_on_validation_grid']
        by_epoch={ep:matched_archive(seed,ep) for ep in range(1,21)}
        for key,win in wins.items():
            arm,lr=win['arm'],win['head_lr']
            chosen=lr==rates[arm]
            ph='development' if seed in (3,42) else 'additional'
            tuning.append(dict(arm=arm,seed=seed,phase=ph,head_lr=lr,selected_lr=chosen,completed=True,
                       epoch=win['epoch'],threshold=win['threshold'],tp=win['tp'],fp=win['fp'],fn=win['fn'],
                       tn=win['tn'],f1_points=win['f1_points'],run_id=f'{seed}.{key}',status='completed'))
            entries=[rows[key] for rows in full]
            for row in entries:
                row.update(by_epoch[row['epoch']])
            if chosen:
                epochs.extend(entries)
                sensitivity.append(dict(arm=arm,seed=seed,epoch=win['epoch'],linear_cached_f1=win['f1_points'],
                    matched_core_epoch=native['epoch'],matched_core_threshold=core['threshold'],
                    matched_core_cached_f1=100*core['f1'],cached_gain=win['f1_points']-100*core['f1']))
            end=load(d/'heads20.pt')['learners'][key]['optimizer']['state']
            state_size=sum(v.numel() for item in end.values() for v in item.values())
            runtime.append(dict(arm=arm,seed=seed,head_lr=lr,phase=ph,selected_lr=chosen,head_parameters=196,
                adamw_state_scalars=state_size,shared_heads=len(wins),head_seconds=sum(r['head_seconds'] for r in entries),
                shared_training_wall_seconds=sum(r['shared_train_seconds'] for r in entries),
                shared_train_validation_audit_seconds=sum(r['epoch_seconds'] for r in ledger),
                peak_gpu_bytes=max(r['peak_gpu_gib']*1024**3 for r in ledger),
                core_steps=sum(r['core_steps'] for r in entries),head_steps=sum(r['head_steps'] for r in entries),
                training_draws=sum(r['draws'] for r in entries),head_examples=sum(r['head_examples'] for r in entries)))
    summaries=[]
    for arm in order:
        for group, seeds in [('all',SEEDS),('development',(3,42)),('additional',(0,1,2))]:
            rows=[r for r in seed_rows if r['arm']==arm and r['seed'] in seeds]
            f=[r['f1_points'] for r in rows]
            item=dict(arm=arm,phase=group,n=len(f),mean_f1=statistics.mean(f),sd_f1=statistics.stdev(f))
            for ref in REFS:
                values=[r['delta_vs_'+ref] for r in rows]
                item.update({f'mean_delta_vs_{ref}':statistics.mean(values),
                             f'sd_delta_vs_{ref}':statistics.stdev(values),
                             f'positive_vs_{ref}':sum(v>0 for v in values)})
            summaries.append(item)
    for filename,records in [('seed_results.csv',seed_rows),('threshold_diagnostics.csv',diagnostics),
                    ('epochs.csv',epochs),('tuning_results.csv',tuning),('runtime.csv',runtime),
                    ('summary.csv',summaries),('matched_core_sensitivity.csv',sensitivity)]:
        write(dest,filename,records)
    (dest/'selection.json').write_text(json.dumps(read(OUT/'selection.json'),indent=2)+'\n')
    (dest/'feature_check.json').write_text(json.dumps(public_probe,indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=HERE/'results')
    generate(p.parse_args().output)
