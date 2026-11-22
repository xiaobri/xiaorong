"""Run the preregistered phases; completed epoch checkpoints support restarts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import sys
import time

from selection import lock
from study import HERE, OUT, folder


def dispatch(script, flags, label, gpu):
    logs = OUT/'logs'
    logs.mkdir(parents=True,exist_ok=True)
    with (logs/f'{label}.log').open('a') as log:
        subprocess.run([sys.executable,'-u',str(HERE/script),*flags], check=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                                MKL_NUM_THREADS='2'), stdout=log, stderr=subprocess.STDOUT)


def fits(seeds, gpus):
    with ThreadPoolExecutor(len(seeds)) as pool:
        jobs = [pool.submit(dispatch,'train.py',['--seed',str(s)],f'train_{s}',g)
                for s,g in zip(seeds,gpus) if not (folder(s)/'completed.json').exists()]
        for job in jobs:
            job.result()


def replay(stage, gpus):
    with ThreadPoolExecutor(len(gpus)) as pool:
        jobs = [pool.submit(dispatch,'raw_scores.py',['--stage',stage,'--shard',str(i),
                  '--shards',str(len(gpus))],f'raw_scores_{stage}{i}',g) for i,g in enumerate(gpus)]
        for job in jobs:
            job.result()


def run(phase):
    if phase == 'references':
        replay('references', (2,3,4,5,6,7))
    elif phase == 'development':
        fits((3,42),(0,1))
    elif phase == 'after-development':
        while not all((folder(s)/'completed.json').is_file() for s in (3,42)):
            time.sleep(15)
        print('Locked',lock(),flush=True)
        with ThreadPoolExecutor(2) as pool:
            training = pool.submit(fits,(0,1,2),(1,2,4))
            raw = pool.submit(replay,'development',(0,3,5,6,7))
            training.result()
            raw.result()
        replay('additional',tuple(range(8)))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True,choices=('references','development','after-development'))
    run(p.parse_args().phase)
