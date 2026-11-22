"""Audit the published aggregate tables with the Python standard library."""
import argparse
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import statistics

ARMS = ('linear_focal_all','linear_bce_all','linear_focal_eligible','linear_bce_eligible')
REFS = ('baseline','pairs','pairs_refresh')
GROUPS = {'development':(3,42), 'additional':(0,1,2), 'all':(0,1,2,3,42)}
RATES = (.0001,.0005,.002)


def close(a,b):
    assert math.isclose(float(a),float(b),abs_tol=1e-9,rel_tol=0), (a,b)


def counts(row,prefix=''):
    tp,fp,fn,tn=(int(row[prefix+k]) for k in ('tp','fp','fn','tn'))
    assert min(tp,fp,fn,tn)>=0 and tp+fn==3471 and tn+fp==162608
    f=Fraction(200*tp,2*tp+fp+fn)
    close(row[prefix+'f1_points'],f)
    if prefix+'precision_points' in row:
        close(row[prefix+'precision_points'],100*tp/(tp+fp) if tp+fp else 0)
        close(row[prefix+'recall_points'],100*tp/(tp+fn))
    return f


def audit(directory):
    def table(name):
        with (directory/(name+'.csv')).open() as stream:
            return list(csv.DictReader(stream))
    locked=json.loads((directory/'selection.json').read_text())
    rates=locked['rates']
    assert locked['seeds']==[3,42] and tuple(locked['candidates'])==RATES
    assert set(locked['development_ledger_sha256'])=={'3','42'}
    for hashes in locked['development_ledger_sha256'].values():
        assert len(hashes)==20 and all(len(bytes.fromhex(h))==32 for h in hashes)
    summary=table('summary')
    main=table('seed_results')
    curves=table('epochs')
    fits=table('tuning_results')
    diagnostics=table('threshold_diagnostics')
    times=table('runtime')
    assert [len(t) for t in (main,curves,fits,summary,diagnostics,times)]==[35,400,36,21,35,36]
    key=lambda r:(r['arm'],int(r['seed']))
    fit_key=lambda r:(*key(r),float(r['head_lr']))
    seeds=GROUPS['all']
    metrics={key(r):counts(r) for r in main}
    assert set(metrics)=={(a,s) for a in ARMS+REFS for s in seeds}
    by_key={key(r):r for r in main}
    for r in main:
        a,s=key(r)
        assert 1<=int(r['epoch'])<=20 and int(r['changed_cached_decisions'])>=0
        assert len(bytes.fromhex(r['checkpoint_sha256']))==32
        for ref in REFS:
            close(r['delta_vs_'+ref],metrics[a,s]-metrics[ref,s])
        if a in ARMS:
            close(r['head_lr'],rates[a])
            close(r['head_weight_decay'],.01)
    assert {key(r) for r in diagnostics}==set(metrics)
    for r in diagnostics:
        a,s=key(r)
        m=by_key[a,s]
        assert r['checkpoint_sha256']==m['checkpoint_sha256'] and r['epoch']==m['epoch']
        assert all(r['grid_'+k]==m[k] for k in ('tp','fp','fn','tn','threshold'))
        coarse,exact=counts(r,'grid_'),counts(r,'exact_')
        assert exact>=coarse and 1<=int(r['distinct_probabilities'])<=166079
        close(r['exact_minus_grid_points'],exact-coarse)
    assert {(r['arm'],r['phase']) for r in summary}=={(a,p) for a in ARMS+REFS for p in GROUPS}
    for r in summary:
        a=r['arm']
        subset=GROUPS[r['phase']]
        vals=[float(metrics[a,s]) for s in subset]
        assert len(vals)==int(r['n'])
        close(r['mean_f1'],statistics.mean(vals))
        close(r['sd_f1'],statistics.stdev(vals))
        for ref in REFS:
            delta=[float(metrics[a,s]-metrics[ref,s]) for s in subset]
            close(r['mean_delta_vs_'+ref],statistics.mean(delta))
            close(r['sd_delta_vs_'+ref],statistics.stdev(delta))
            assert int(r['positive_vs_'+ref])==sum(v>0 for v in delta)
    fitted={fit_key(r):counts(r) for r in fits}
    expected={(a,s,lr) for a in ARMS for s in seeds for lr in (RATES if s in (3,42) else (rates[a],))}
    assert set(fitted)==expected and {fit_key(r) for r in times}==expected
    for r in fits:
        a,s,lr=fit_key(r)
        assert r['selected_lr']==str(lr==rates[a]) and r['completed']=='True'
        assert s in GROUPS[r['phase']] and r['phase']!='all'
    for arm in ARMS:
        winner=max(RATES,key=lambda lr:(sum(fitted[arm,s,lr] for s in (3,42)),-lr))
        close(rates[arm],winner)
    assert {(*key(r),int(r['epoch'])) for r in curves}=={(a,s,e) for a in ARMS for s in seeds for e in range(1,21)}
    for s in seeds:
        for epoch in range(1,21):
            same=[r for r in curves if int(r['seed'])==s and int(r['epoch'])==epoch]
            for col in ('batch_order_sha256','core_state_sha256','matched_core_sha256'):
                assert len({r[col] for r in same})==1 and len(bytes.fromhex(same[0][col]))==32
    for r in curves:
        counts(r)
        assert r['core_state_sha256']==r['matched_core_sha256'] and float(r['weight_norm'])>0
        assert int(r['draws'])==20000 and int(r['core_steps'])==157
        assert 0<int(r['head_steps'])<=157 and 0<int(r['head_examples'])<=20000
        if r['arm'].endswith('_all'):
            assert int(r['head_examples'])==20000 and int(r['head_steps'])==157
        for col in ('head_loss','core_loss','core_primary','core_auxiliary','core_interaction','head_seconds'):
            assert math.isfinite(float(r[col])) and float(r[col])>=0
    for a in ARMS:
        for s in seeds:
            rows=[r for r in curves if key(r)==(a,s)]
            win=max(rows,key=lambda r:(counts(r),int(r['epoch'])))
            f=next(r for r in fits if fit_key(r)==(a,s,rates[a]))
            assert all(win[k]==f[k] for k in ('epoch','threshold','tp','fp','fn','tn'))
            primary=by_key[a,s]
            assert win['epoch']==primary['epoch'] and win['threshold']==primary['threshold']
            assert (abs(int(win['tp'])-int(primary['tp']))+abs(int(win['fp'])-int(primary['fp']))
                    <=int(primary['changed_cached_decisions']))
    for r in times:
        assert int(r['head_parameters'])==196 and int(r['adamw_state_scalars'])==394
        assert int(r['training_draws'])==400000 and int(r['core_steps'])==3140
        assert float(r['peak_gpu_bytes'])>0 and float(r['head_seconds'])>0
        if r['selected_lr']=='True':
            ep=[e for e in curves if key(e)==key(r)]
            for k in ('head_examples','head_steps'):
                assert int(r[k])==sum(int(e[k]) for e in ep)
    sens=table('matched_core_sensitivity')
    assert len(sens)==20 and {key(r) for r in sens}=={(a,s) for a in ARMS for s in seeds}
    for r in sens:
        close(r['linear_cached_f1'],fitted[(r['arm'],int(r['seed']),rates[r['arm']])])
        close(r['cached_gain'],float(r['linear_cached_f1'])-float(r['matched_core_cached_f1']))
    audit_record=json.loads((directory/'protocol_audit.json').read_text())
    assert set(audit_record['seeds'])=={str(s) for s in seeds} and audit_record['all_fits_completed']==36
    digests=[audit_record['manifest_sha256'],*audit_record['frozen_checkpoint_sha256'].values()]
    assert len(digests)==6
    for record in audit_record['seeds'].values():
        digests.extend((record['init_sha256'],record['ordered_400000_sha256']))
        assert record['eligible_groups']==9
    assert all(len(bytes.fromhex(h))==32 for h in digests)
    probe=json.loads((directory/'feature_check.json').read_text())
    assert (probe['windows'],probe['features'])==(128,195)
    assert all(probe[k]==0 for k in ('pair_feature_max_abs','zero_head_logit_max_abs','metadata_feature_max_abs'))
    assert probe['train_support_unchanged_after_val_mutation'] and probe['eval_state_unchanged']
    for row in curves:
        assert float(row['pairs_core_max_abs'])==float(row['pairs_refresh_core_max_abs'])==0
    print('Verified 35 seed results, 35 fixed-checkpoint scans, 400 epoch rows, 36 fits and 21 summaries.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path(__file__).resolve().parent/'results')
    audit(p.parse_args().results)
