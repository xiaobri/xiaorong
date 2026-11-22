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
    while True:
        free = float(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        if free >= 6*1024:
            break
        atomic_write_json(out/'job_status.json', dict(state, status='waiting_for_memory',
            observed_free_mib=free, required_free_mib=6*1024, timestamp=time.time()))
        time.sleep(30)
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
        reference = Path(f'{_paths.RUNS}/I2MOE_test333_strict_baseline_20261115/formal/baseline')/f'seed{seed}'
        init = json.loads((reference/'initialization.json').read_text())['original_trainable_state_sha256']
        sample = json.loads((reference/'sampling_epoch01.json').read_text())
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
        original_core_loss_retained=True, explicit_independent_correction_loss=True, all_eight_real_data_smoke_and_integration_passed=True, compared_to_original_baseline_run=True))


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
        method_difference='Same-acquisition quantile comparisons with collection-time or recomputed parent margins on cases and controls.'))


def check_cache_reference():
    reference = Path(f'{_paths.RUNS}/I2MOE_test333_fair_cached_20261115')
    while True:
        paths = [reference/f'formal/baseline/seed{seed}/raw_replay/replay_result.json' for seed in (3, 42)]
        if all(path.exists() for path in paths):
            reports = [json.loads(path.read_text()) for path in paths]
            for report in reports:
                if (report['rows'] != 166079 or not report['full_original_six_experts_recomputed']
                        or report['maximum_probability_error'] > .001
                        or report['changed_threshold_decisions'] > 5
                        or abs(report['cached_metric']['f1']-report['raw_metric']['f1']) > .001):
                    raise RuntimeError('Frozen cache reference discrepancy requires investigation: '+json.dumps(report))
            atomic_write_json(OUTPUT/'cache_reference_check.json', dict(
                controls=reports, candidate_target_requires_its_own_raw_replay=True,
                numerical_acceptance='FP32 cache/reference max probability error <=0.001; <=5 threshold decision differences; F1 difference <=0.001'))
            return
        atomic_write_json(OUTPUT/'coordinator_status.json', dict(stage='waiting_for_raw_cache_reference', timestamp=time.time()))
        print('Waiting for both full raw-input cache-control replays', flush=True)
        time.sleep(30)



def wait_for_moment_cache():
    from bootstrap import MOMENT_CACHE
    while not (MOMENT_CACHE/'ready.json').exists():
        atomic_write_json(OUTPUT/'coordinator_status.json', dict(stage='waiting_for_frozen_moment_cache', timestamp=time.time()))
        time.sleep(30)


def wait_for_prior_rounds():
    rounds = ('fair_conditional_quantiles_20261120','fair_patch_distill_20261120')
    for name in rounds:
        output = OUTPUT.with_name('I2MOE_test333_'+name)
        while not (output/'completed_round_audits.json').exists():
            state = output/'coordinator_status.json'
            stage = json.loads(state.read_text()).get('stage') if state.exists() else None
            if stage == 'skipped_target_already_verified':
                atomic_write_json(OUTPUT/'coordinator_status.json', dict(stage=stage))
                raise SystemExit(0)
            if stage == 'blocked_on_failed_check':
                raise RuntimeError(f'Check prior failure in {output}')
            atomic_write_json(OUTPUT/'coordinator_status.json', dict(stage='queued_after_prior_audit', round=name, timestamp=time.time()))
            time.sleep(30)
        audit = json.loads((output/'completed_round_audits.json').read_text())[name]
        if audit['paired_raw_target_methods']:
            atomic_write_json(OUTPUT/'coordinator_status.json', dict(stage='skipped_target_already_verified', target=audit))
            raise SystemExit(0)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT/'launcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Claim all eight experiment locks for the lifetime of this sweep.
    gpu_locks = []
    for gpu, _, _ in JOBS:
        handle = Path(f'/tmp/test333_refreshed_pairs_gpu{gpu}.lock').open('a')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_locks.append(handle)
    wait_for_prior_rounds()
    wait_for_moment_cache()
    from bootstrap import BLOCK_CACHE
    while not (BLOCK_CACHE/'ready.json').exists():
        atomic_write_json(OUTPUT/'coordinator_status.json',dict(stage='waiting_for_segment_features',timestamp=time.time()))
        time.sleep(20)
    parity=json.loads((OUTPUT/'gpu_raw_cache_parity.json').read_text())
    if not parity['passed'] or not json.loads((OUTPUT/'actual_history_preflight.json').read_text())['verified']:
        raise RuntimeError('Pairwise evidence raw/cache check failed')
    identity = train_strict.source_identity()
    atomic_write_json(OUTPUT/'plan.json', dict(jobs=JOBS, source_sha256=identity,
                      original_baseline=f'{_paths.STORAGE}/I2MOE6_F1/screen_20261112/baseline',
                      all_new_trainable_components_fresh=True, regional_head_initialization='All paired evidence statistics and negative reference slots initialized empty. Original six-expert independent_aux parent.', total_draws_per_job=400000,
                      no_historical_cbra=True, loss='Original independent_aux AdamW objectives unchanged. Extra pairwise ranking statistics use current rare positives and prior negative features from the same original stream.',
                      runtime_resource_change='Same 4 GiB allocator cap as baseline; wait for at least 6 GiB free; batch remains128; GPU sharing affects wall time only'))
    for mode in ('smoke', 'integration', 'formal'):
        if mode == 'formal':
            check_cache_reference()
            wait_for_prior_rounds()
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
