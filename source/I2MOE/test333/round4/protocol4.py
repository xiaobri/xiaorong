"""Renewed authorization: retain the scientific protocol, remove old deadline."""
import czb_paths as _paths
from pathlib import Path
import hashlib
from i2moe_eeg.protocol3 import PROTOCOL as PREVIOUS_PROTOCOL

ROOT = Path(__file__).resolve().parent
OUTPUT = Path(f'{_paths.RUNS}/I2MOE_test333_round4_20261114')
INPUT_ROOT = Path(f'{_paths.RUNS}/I2MOE_test333_round3_20261114')
HISTORICAL = Path(f'{_paths.STORAGE}/I2MOE_test333/runs/embrace_cbramod')
PROTOCOL = {k:v for k,v in PREVIOUS_PROTOCOL.items() if k != 'deadline_utc'}
PROTOCOL.update(authorization='User renewed continued experimentation until F1 72; no new deadline specified',
                initialization='Same-seed historical best EmBrace-CBraMod; old six experts remain immutable',
                anchor_f1={'3': .7043320205998183, '42': .706379260122342})
VARIANTS = ('local_mil', 'hard_rank', 'sam_residual', 'wave_detail')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8<<20), b''):
            digest.update(block)
    return digest.hexdigest()


def source_identity() -> dict:
    names=('bootstrap.py','protocol4.py','anchor_model.py','models4.py','data4.py',
           'functional_sam.py','train4.py')
    return {str(ROOT/name):sha256(ROOT/name) for name in names}

