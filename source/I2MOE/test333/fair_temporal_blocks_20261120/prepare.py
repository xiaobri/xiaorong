"""Eight pre-normalization segment means from the original five intact CNNs."""
import czb_paths as _paths
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

BASELINE = Path(f'{_paths.BASELINE}')
sys.path.insert(0, str(BASELINE))
from i2moe_eeg.f1_variants import SixExpertF1
from i2moe_eeg.model import default_frozen_checkpoint_paths, FROZEN_EXPERT_BEST_SEEDS
from i2moe_eeg.regions import EXPERT_NAMES
from block_features import SEGMENTS,base_weights,register_blocks
from i2moe_eeg.utils import atomic_write_json, set_seed

STORE = Path(f'{_paths.STORAGE}')
RAW = STORE / 'I2MOE_test333_fair_raw_cache'
TEMPORAL = STORE / 'I2MOE_test333_fair_temporal_cache'
OUTPUT = Path(f'{_paths.RUNS}/I2MOE_test333_fair_temporal_blocks_20261120')
GPUS = (0, 2, 3, 5)
SHARDS = 8


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text())


def source_model():
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    set_seed(3, deterministic=False)
    model = SixExpertF1(variant='baseline', checkpoint_paths=default_frozen_checkpoint_paths(),
                        expected_source_seeds=FROZEN_EXPERT_BEST_SEEDS).cuda().eval().requires_grad_(False)
    assert tuple(model.frozen_experts.keys()) == EXPERT_NAMES[:5]
    return model


@torch.no_grad()
def segment_packet(model, captured, eeg):
    captured.clear()
    features,logits = model._encode_frozen(eeg)
    result = torch.stack([captured[i][0] for i in range(5)],1)
    if result.shape != (len(eeg),5,SEGMENTS,128) or any(length!=63 for _,length in captured.values()):
        raise ValueError('Unexpected original EEG projection or time width')
    return result,features,logits


def prepare_plan():
    raw, temporal = read(RAW / 'ready.json'), read(TEMPORAL / 'ready.json')
    mapping_hash = temporal['row_mapping_sha256']
    if raw['row_mapping_sha256'] != mapping_hash or digest(TEMPORAL/'row_to_cache.npy') != mapping_hash:
        raise RuntimeError('Frozen and waveform cache row mapping differs')
    verified = dict(temporal['plan']['sources'])
    for file, value in raw['plan']['sources'].items():
        if file in verified and verified[file] != value:
            raise RuntimeError(f'Source hashes disagree: {file}')
        verified[file] = value
    for file, value in verified.items():
        if digest(file) != value:
            raise RuntimeError(f'Source identity changed: {file}')
    original = temporal['plan']
    verified[str(Path(__file__).with_name('block_features.py').resolve())] = digest(Path(__file__).with_name('block_features.py'))
    shape = [original['shape'][0], 5, SEGMENTS, 128]
    assert raw['plan']['shape'] == [shape[0],19,2000]
    plan = dict(shape=shape, dtype='float32', segments=SEGMENTS, native_length=63, descriptor='Projection affine output before LayerNorm, each consecutive segment mean',
        sources=verified, extractor=digest(__file__), row_mapping_sha256=mapping_hash,
        raw_eeg_sha256=raw['eeg_sha256'], tails=original['tails'], val_tail=original['validation_tail_rows'],
        raw_input='Original preprocessed EEG, intact five source inputs at each depth.',
        original_window_rows=shape[0], training_or_fitting=False)
    dest = OUTPUT/'plan.json'
    if dest.exists():
        if read(dest) != plan:
            raise RuntimeError('Requested segment cache differs from saved plan')
    else:
        array = np.lib.format.open_memmap(OUTPUT/'packets.npy', mode='w+', dtype='float32', shape=tuple(shape))
        array.flush()
        atomic_write_json(dest, plan)
    return plan


@torch.no_grad()
def validate(model,captured,raw,temporal):
    mapping = np.load(TEMPORAL/'row_to_cache.npy',mmap_mode='r')
    index = np.flatnonzero(mapping[:731610]>=0)[::3000][:128]
    slots = mapping[index]
    eeg = torch.from_numpy(raw[slots]).cuda()
    packet,features,logits = segment_packet(model,captured,eeg)
    ref = torch.from_numpy(np.array(temporal[slots,:,63:65])).cuda()
    error = float((ref[:,:,1,:2]-logits).abs().max())
    if error>.002 or not packet.isfinite().all():
        raise RuntimeError('Intermediate extraction failed frozen-source identity or finite-value checks')
    weights = base_weights(63,packet)
    uniform = (packet*weights[None,None,:,None]).sum(2)
    recreated = torch.stack([source.encoder.projection[1](uniform[:,idx]) for idx,source in enumerate(model.frozen_experts.values())],1)
    pool_error = float((recreated-features).abs().max())
    if pool_error>.001: raise RuntimeError(f'Original pooling is not recovered: {pool_error}')
    return dict(train_rows=len(index),original_pooled_feature_max_error=pool_error,source_logit_max_error=error,
                source_feature_max_error=float((ref[:,:,0]-features).abs().max()),shape=list(packet.shape))


def extract(shard):
    plan = read(OUTPUT/'plan.json')
    marker = OUTPUT/f'shard{shard}_complete.json'
    if marker.exists():
        assert read(marker)['extractor'] == plan['extractor']
        return
    model = source_model()
    captured,handles = register_blocks(model.frozen_experts)
    raw = np.load(RAW/'eeg.npy', mmap_mode='r')
    temporal = np.load(TEMPORAL/'packets.npy', mmap_mode='r')
    if shard == 0:
        atomic_write_json(OUTPUT/'source_check.json', validate(model, captured, raw, temporal))
    dest = np.load(OUTPUT/'packets.npy', mmap_mode='r+')
    blocks = (len(raw)+127)//128
    lo, hi = min(len(raw),128*(blocks*shard//SHARDS)), min(len(raw),128*(blocks*(shard+1)//SHARDS))
    started = time.monotonic()
    for start in range(lo, hi, 128):
        value = torch.from_numpy(np.array(raw[start:min(start+128,hi)])).cuda()
        size = len(value)
        if size != 128:
            value = torch.cat((value,value[:1].expand(128-size,-1,-1)))
        packet,_,_ = segment_packet(model,captured,value)
        if not packet.isfinite().all():
            raise FloatingPointError(f'Nonfinite intermediate features for cache slot {start}')
        dest[start:start+size] = packet[:size].cpu().numpy()
        if (start-lo)//128 % 64 == 0:
            atomic_write_json(OUTPUT/f'shard{shard}_progress.json', dict(rows=start-lo+size,total=hi-lo))
    dest.flush()
    mapping = np.load(TEMPORAL/'row_to_cache.npy',mmap_mode='r')
    tails = [(f"tail_seed{e['seed']}_epoch{e['epoch']:02d}.npy", e['rows'])
             for j,e in enumerate(plan['tails']) if j % SHARDS == shard]
    if shard == 0:
        tails.append(('validation_tail.npy', plan['val_tail']))
    for filename, indices in tails:
        batch = torch.from_numpy(raw[mapping[indices]]).cuda()
        packet,_,_ = segment_packet(model,captured,batch)
        assert packet.isfinite().all()
        np.save(OUTPUT/filename, packet.cpu().numpy())
    atomic_write_json(marker, dict(start=lo,end=hi,rows=hi-lo,extractor=plan['extractor'],
        seconds=time.monotonic()-started,peak_gib=torch.cuda.max_memory_allocated()/2**30))


def worker(slot):
    gpu = GPUS[slot]
    for shard in range(slot, SHARDS, len(GPUS)):
        if (OUTPUT/f'shard{shard}_complete.json').exists():
            continue
        while True:
            free = int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free',
                                                 '--format=csv,noheader,nounits'],text=True).strip())
            if free >= 6144:
                break
            time.sleep(20)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        subprocess.run([sys.executable,'-u',__file__,'--shard',str(shard)],env=env,check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard',type=int,choices=range(SHARDS))
    args = parser.parse_args()
    if args.shard is not None:
        extract(args.shard)
        return
    OUTPUT.mkdir(exist_ok=True)
    while not Path(f'{_paths.RUNS}/I2MOE_test333_fair_depth_cache_20261120/ready.json').exists():
        time.sleep(20)
    with (OUTPUT/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan = prepare_plan()
        if (OUTPUT/'ready.json').exists():
            assert read(OUTPUT/'ready.json')['plan'] == plan
            return
        with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
            list(pool.map(worker,range(len(GPUS))))
        shards = [read(OUTPUT/f'shard{i}_complete.json') for i in range(SHARDS)]
        offset = 0
        for shard in shards:
            assert shard['start'] == offset and shard['extractor'] == plan['extractor']
            offset = shard['end']
        assert offset == plan['shape'][0]
        atomic_write_json(OUTPUT/'ready.json',dict(plan=plan,shards=shards,
            packets_sha256=digest(OUTPUT/'packets.npy'), row_mapping_sha256=plan['row_mapping_sha256']))


if __name__ == '__main__':
    main()
