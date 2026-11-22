"""Aggregate locked raw confusion counts, then calculate within-seed contrasts."""
import argparse
import csv
from pathlib import Path
from statistics import mean, stdev

from single import ARMS, OUT, SEEDS
from fit import REPLICATION, read
from queue_fits import jobs


def save(name, rows):
    with (OUT/name).open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def result(arm,seed,epoch,threshold,counts,changed,source):
    c={k:int(counts[k]) for k in ('tp','fp','fn','tn')}
    assert c['tp']+c['fn']==3471 and sum(c.values())==166079
    return dict(arm=arm,seed=int(seed),epoch=int(epoch),threshold=float(threshold),**c,
                f1_points=200*c['tp']/(2*c['tp']+c['fp']+c['fn']),
                changed_cached_decisions=int(changed),source=source)


def main(study):
    rows=[]
    with (REPLICATION/'seed_results.csv').open() as f:
        for r in csv.DictReader(f):
            rows.append(result(r['method'],r['seed'],r['epoch'],r['threshold'],r,
                               r['changed_threshold_decisions'],'matched baseline or complete model replication'))
    with (study/'results/component_xiehe.csv').open() as f:
        for r in csv.DictReader(f):
            if r['arm'] in ('risk_power','plain'):
                arm={'risk_power':'moment_recipe','plain':'moment_plain'}[r['arm']]
                rows.append(result(arm,r['seed'],r['epoch'],r['threshold'],r,
                                   r['changed_cached_decisions'],'original locked single-head replay'))
    for seed in SEEDS:
        parts=[read(OUT/'raw'/f'seed{seed}_shard{i}.json') for i in range(8)]
        locks=parts[0]['contract']['locks']
        assert all(p['contract']['locks']==locks and p['contract']['seed']==seed for p in parts)
        assert parts[0]['contract']['start']==0 and parts[-1]['contract']['stop']==166079
        assert all(p['contract']['stop']==q['contract']['start'] for p,q in zip(parts,parts[1:]))
        for arm,s in jobs():
            if s!=seed:
                continue
            counts={k:sum(p['counts'][arm][k] for p in parts) for k in ('tp','fp','fn','tn')}
            changed=sum(p['changes'][arm] for p in parts)
            lock=locks[arm]
            row=result(arm,seed,lock['epoch'],lock['threshold'],counts,changed,'new isolated fit and locked replay')
            if not changed:
                assert abs(row['f1_points']-100*lock['cached_f1'])<1e-9
            rows.append(row)
    table={(r['arm'],r['seed']):r['f1_points'] for r in rows}
    arms=('baseline',*ARMS,'fresh_both')
    assert len(table)==len(rows)==len(arms)*len(SEEDS)
    for r in rows:
        r['delta_vs_baseline']=r['f1_points']-table['baseline',r['seed']]
    save('single_component_seeds.csv',rows)
    summaries=[]
    for arm in arms:
        values=[table[arm,s] for s in SEEDS]
        gains=[table[arm,s]-table['baseline',s] for s in SEEDS]
        summaries.append(dict(arm=arm,**{f'seed{s}_F1':table[arm,s] for s in sorted(SEEDS)},
                              mean_F1=mean(values),sd_F1=stdev(values),
                              mean_gain=mean(gains),sd_gain=stdev(gains),positive_seeds=sum(g>0 for g in gains),
                              development_gain=mean(gains[:2]),additional_gain=mean(gains[2:])))
    save('single_component_summary.csv',summaries)
    contrasts=[]
    for a,b in [('time','regional_time'),('time_aux','time_native'),('pairs_refresh','pairs'),
                ('moment_recipe','moment_plain'),('peak_native','peak'),('time_native','time'),
                ('fresh_both','pairs_refresh'),('fresh_both','moment_recipe')]:
        v=[table[a,s]-table[b,s] for s in SEEDS]
        contrasts.append(dict(arm=a,reference=b,mean_difference=mean(v),**{f'seed{s}':d for s,d in zip(SEEDS,v)}))
    save('dependent_contrasts.csv',contrasts)
    for s in summaries:
        print(s['arm'],round(s['mean_F1'],3),round(s['mean_gain'],3),s['positive_seeds'])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--study',type=Path,required=True)
    main(p.parse_args().study)
