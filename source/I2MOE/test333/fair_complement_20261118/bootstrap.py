import czb_paths as paths
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent
BASELINE = paths.BASELINE
OUTPUT = paths.RUNS / 'I2MOE_test333_fair_complement_20261118'
CACHE = paths.STORAGE / 'I2MOE_test333_fair_temporal_cache'
sys.path.insert(0, str(BASELINE))
sys.path.append(str(SOURCE/'fair_arch_20261115'))
sys.path.append(str(SOURCE/'fair_cached_20261115'))


MOMENT_CACHE = paths.STORAGE / 'I2MOE_test333_fair_moment_cache'

sys.path.append(str(SOURCE/'fair_moments_20261115'))
sys.path.append(str(SOURCE/'fair_routing_20261115'))

sys.path.append(str(SOURCE/'fair_protected_20261115'))

sys.path.append(str(SOURCE/'fair_cohorts_20261115'))

sys.path.append(str(SOURCE/'fair_riskrefine_20261115'))
