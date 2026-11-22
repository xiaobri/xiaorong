"""Use the original baseline trainer and audited waveform encoders."""
import czb_paths as _paths
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent
BASELINE = Path(f'{_paths.BASELINE}')
OUTPUT = Path(f'{_paths.RUNS}/I2MOE_test333_strict_baseline_20261115')
for folder in ('round4_channelset', 'round4_refine', 'round4_softshape',
               'round4_domains', 'round4_raw', 'round4', 'round3', '.deps'):
    sys.path.append(str(SOURCE / folder))
sys.path.insert(0, str(BASELINE))
import i2moe_eeg
i2moe_eeg.__path__.append(str(SOURCE / 'round3/i2moe_eeg'))

