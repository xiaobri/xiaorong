#!/usr/bin/env python3
"""Eight isolated jobs, with mandatory real-data gates before formal training."""
import czb_paths as _paths
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import train_strict
from bootstrap import ROOT, OUTPUT
from i2moe_eeg.utils import atomic_write_json

PYTHON = f'{_paths.PYTHON}'
JOBS = [(gpu, method, seed) for gpu, (method, seed) in enumerate(
    (method, seed) for method in train_strict.METHODS for seed in (3, 42))]


def worker(job, mode):
    gpu, method, seed = job
    out = OUTPUT/mode/method/f'seed{seed}'
    out.mkdir(parents=True, exist_ok=True)
    state = dict(gpu=gpu, method=method, seed=seed, mode=mode, output=str(out))
    if (out/'job_status.json').exists():
        previous = json.loads((out/'job_status.json').read_text())
        if previous.get('status') == 'passed':
            contract = json.loads((out/'strict_contract.json').read_text())
            if contract['source_sha256'] != train_strict.source_identity():
                raise RuntimeError('Completed job source changed')
            return previous
    command = [PYTHON, '-u', str(ROOT/'train_strict.py'), '--method', method,
               '--seed', str(seed), '--output-dir', str(out), '--mode', mode]
    if mode == 'formal' and (out/'resume.pt').exists():
        command.append('--resume')
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                       MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    atomic_write_json(out/'command.json', dict(argv=command, cuda_visible_devices=str(gpu)))
    free = float(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    if free < 34*1024:
        raise RuntimeError(f'GPU {gpu} no longer has 34 GiB free; refusing to collide with another job')
    started = time.time()
    with (out/'run.log').open('a') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT)
        atomic_write_json(out/'job_status.json', dict(state, status='running',
                          pid=process.pid, started=started))
        return_code = process.wait()
    passed = return_code == 0
    if mode == 'formal':
        passed = passed and (out/'completed.json').exists()
    if mode == 'integration':
        passed = passed and (out/'debug_finished.json').exists()
    if mode == 'smoke' and passed:
        lines = [line for line in (out/'run.log').read_text().splitlines()
                 if line.startswith('SMOKE_TEST_JSON=')]
        if not lines:
            passed = False
        else:
            smoke = json.loads(lines[-1].split('=', 1)[1])
            atomic_write_json(out/'smoke_passed.json', smoke)
    result = dict(state, status='passed' if passed else 'failed',
                  return_code=return_code, seconds=time.time()-started)
    atomic_write_json(out/'job_status.json', result)
    print(json.dumps(result), flush=True)
    return result


def check_pairing():
    for seed in (3, 42):
        init, sample = None, None
        for method in train_strict.METHODS:
            out = OUTPUT/'integration'/method/f'seed{seed}'
            config = json.loads((out/'configuration_check.json').read_text())
            if not config['passed'] or config['differences']:
                raise RuntimeError('Configuration check failed')
            current_init = json.loads((out/'initialization.json').read_text())['original_trainable_state_sha256']
            current_sample = json.loads((out/'sampling_epoch01.json').read_text())
            if init is not None and (current_init != init or current_sample != sample):
                raise RuntimeError('Original initialization or actual sample order differs across methods')
            init, sample = current_init, current_sample
    atomic_write_json(OUTPUT/'paired_protocol_passed.json', dict(
        same_initial_sixth_and_router=True, same_actual_sampling_order=True,
        exact_original_loss=True, all_eight_real_data_smoke_and_integration_passed=True))


def summarize():
    import torch
    rows = []
    for gpu, method, seed in JOBS:
        out = OUTPUT/'formal'/method/f'seed{seed}'
        best = torch.load(out/'best_f1.pt', map_location='cpu', weights_only=False)
        rows.append(dict(gpu=gpu, method=method, seed=seed, epoch=best['epoch'],
                         metric=best['validation']['best_f1_on_validation_grid'],
                         checkpoint=str(out/'best_f1.pt')))
    atomic_write_json(OUTPUT/'RESULTS.json', dict(
        validation_only=True, independent_test=False, rows=rows,
        shared_training=dict(epochs=20, samples_per_epoch=20000, batch_size=128,
                             learning_rate=5e-4, precision='FP32', schedule='constant'),
        method_difference='Fresh waveform correction architecture; joint trained once, not separately pretrained ensemble'))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT/'launcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Claim all eight experiment locks for the lifetime of this sweep.
    gpu_locks = []
    for gpu, _, _ in JOBS:
        handle = Path(f'/tmp/test333_strict_gpu{gpu}.lock').open('a')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_locks.append(handle)
    identity = train_strict.source_identity()
    atomic_write_json(OUTPUT/'plan.json', dict(jobs=JOBS, source_sha256=identity,
                      original_baseline=f'{_paths.STORAGE}/I2MOE6_F1/screen_20261112/baseline',
                      all_new_trainable_components_fresh=True, total_draws_per_job=400000,
                      no_historical_cbra=True, loss='original baseline unchanged',
                      runtime_resource_change='4 GiB baseline allocator cap; 32 GiB new-method cap; batch remains 128'))
    for mode in ('smoke', 'integration', 'formal'):
        atomic_write_json(OUTPUT/'coordinator_status.json', dict(stage=mode, timestamp=time.time()))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda job: worker(job, mode), JOBS))
        atomic_write_json(OUTPUT/f'{mode}_jobs.json', dict(jobs=results, source_sha256=identity))
        if any(item['status'] != 'passed' for item in results):
            atomic_write_json(OUTPUT/'coordinator_status.json', dict(stage='blocked_on_failed_check', mode=mode))
            raise RuntimeError(f'{mode} failed; no later-stage jobs launched')
        if mode == 'integration':
            check_pairing()
    summarize()
    atomic_write_json(OUTPUT/'coordinator_status.json', dict(stage='completed', timestamp=time.time()))


if __name__ == '__main__':
    main()
