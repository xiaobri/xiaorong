"""One independent fit per GPU. Retain the original checkpoints and evaluation records."""
import csv
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from single import ARMS, HERE, OUT, SEEDS


def jobs():
    # The two published first-correction fits and the original paired baselines
    # already have complete source selections and locked raw replays.
    return [(a,s) for s in SEEDS for a in ARMS
            if not (s in (3,42) and a in ('moment_plain','moment_recipe'))]


def main():
    out = OUT / 'formal'
    out.mkdir(parents=True, exist_ok=True)
    stream = queue.Queue()
    for arm, seed in jobs():
        stream.put((arm, seed))
    def work(gpu):
        failures = []
        while True:
            try:
                arm, seed = stream.get_nowait()
            except queue.Empty:
                return failures
            folder = out / arm / f'seed{seed}'
            folder.mkdir(parents=True, exist_ok=True)
            if (folder/'completed.json').is_file():
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
            with (folder/'run.log').open('a') as log:
                p = subprocess.run([sys.executable,'-u',str(HERE/'fit.py'),'--arm',arm,'--seed',str(seed)],
                                   env=env,stdout=log,stderr=subprocess.STDOUT)
            print(arm, seed, gpu, p.returncode, flush=True)
            if p.returncode or not (folder/'completed.json').is_file():
                failures.append((arm,seed))
                # Stop this worker immediately on a systematic failure.
                return failures
    with ThreadPoolExecutor(8) as workers:
        failed = sum(workers.map(work,range(8)), [])
    if failed or not stream.empty():
        raise RuntimeError(f'Fit failures {failed}; pending {stream.qsize()}')


if __name__ == '__main__':
    main()
