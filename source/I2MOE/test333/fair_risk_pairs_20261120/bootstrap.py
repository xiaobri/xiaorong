import czb_paths as _paths
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent
BASELINE = Path(f'{_paths.BASELINE}')
OUTPUT = Path(f'{_paths.RUNS}/I2MOE_test333_fair_risk_pairs_20261120')
CACHE = Path(f'{_paths.STORAGE}/I2MOE_test333_fair_temporal_cache')
sys.path.insert(0, str(BASELINE))
sys.path.append(str(SOURCE/'fair_arch_20261115'))
sys.path.append(str(SOURCE/'fair_cached_20261115'))


MOMENT_CACHE = Path(f'{_paths.STORAGE}/I2MOE_test333_fair_moment_cache')

sys.path.append(str(SOURCE/'fair_moments_20261115'))
sys.path.append(str(SOURCE/'fair_routing_20261115'))

sys.path.append(str(SOURCE/'fair_protected_20261115'))

sys.path.append(str(SOURCE/'fair_cohorts_20261115'))

sys.path.append(str(SOURCE/'fair_riskrefine_20261115'))
sys.path.append(str(SOURCE/'fair_complement_20261118'))

sys.path.append(str(SOURCE/'fair_source_overlap_20261119'))

sys.path.append(str(SOURCE/'fair_peakshape_20261119'))
TIME_CACHE=Path(f'{_paths.RUNS}/I2MOE_test333_fair_time_scores_20261119')

sys.path.append(str(SOURCE/'fair_time_alignment_20261119'))

sys.path.append(str(SOURCE/'fair_temporal_blocks_20261120'))
BLOCK_CACHE=Path(f'{_paths.RUNS}/I2MOE_test333_fair_temporal_blocks_20261120')

sys.path.append(str(SOURCE/'fair_expert_allocation_20261120'))
sys.path.append(str(SOURCE/'fair_field_fusion_20261120'))

sys.path.append(str(SOURCE/'fair_analytic_regions_20261120'))
