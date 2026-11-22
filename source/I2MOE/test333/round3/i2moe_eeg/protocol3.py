"""Explicitly locked comparison protocol and wall-clock authorization."""
import czb_paths as _paths
from datetime import datetime, timezone
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
OUTPUT = Path(f'{_paths.RUNS}/I2MOE_test333_round3_20261114')
BASE_CACHE = Path(f'{_paths.STORAGE}/I2MOE_test333/cache')
STARTED_UTC = '2026-11-13T19:04:23+00:00'
DEADLINE_UTC = '2026-11-14T05:04:23+00:00'
DEADLINE = datetime.fromisoformat(DEADLINE_UTC).timestamp()
VARIANTS = ('labram_lora', 'bsr_video', 'graph_gcmae', 'patchcode')
PROTOCOL = {
    'seeds': [3, 42], 'channels': 19, 'raw_samples': 2000,
    'raw_hz': 500, 'train_rows': 731610, 'val_rows': 166079,
    'split': 'unchanged original Train/Val; all original rows retained',
    'epochs': 30, 'samples_per_epoch': 100000, 'negative_positive_ratio': 30,
    'batch_size': 128, 'optimizer': 'AdamW', 'weight_decay': 0.01,
    'head_lr': 3e-4, 'pretrained_lr': 1e-5, 'gradient_clip': 1.0,
    'primary_loss': 'original binary focal alpha=0.85 gamma=2',
    'thresholds': '0.01..0.99 step 0.01; strict probability > threshold',
    'selection': 'original round1 full-Val F1 screening; NOT independent test',
    'old_experts': 'immutable same-seed six-expert baseline cache',
    'auxiliary_pretraining': 'Train only; separate budget and provenance ledger',
    'no_cross_seed_ensemble': True,
    'deadline_utc': DEADLINE_UTC,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def source_identity() -> dict:
    paths = list(ROOT.glob('*.py')) + list(ROOT.glob('*.sh'))
    paths += list((ROOT / 'i2moe_eeg').glob('*.py'))
    paths += list((ROOT / 'tests').glob('*.py'))
    paths += [PARENT / 'i2moe_eeg' / name for name in
              ('data.py', 'metrics.py', 'foundation_data.py', 'foundation_models.py', 'utils.py', 'regions.py')]
    return {str(p): sha256(p) for p in sorted(paths)}

