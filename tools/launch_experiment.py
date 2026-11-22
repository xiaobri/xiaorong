"""Start one isolated fit using the entry point that produced its published row."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'source'
VARIANTS = SOURCE / 'I2MOE/test333'
RUN = VARIANTS / 'fair_single_components'


def main(arm: str) -> None:
    parser = argparse.ArgumentParser(description=f'Xiehe single-component ablation: {arm}')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--seed', type=int, choices=(0, 1, 2, 3, 42), help='train the selected seed')
    group.add_argument('--verify', action='store_true', help='check published results without private EEG or PyTorch')
    args = parser.parse_args()
    if args.verify:
        from verify_results import verify
        verify(arm)
        return

    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(filter(None, (str(SOURCE), env.get('PYTHONPATH'))))
    # The older first-stage records were produced with two different entry points.
    if args.seed in (3, 42) and arm in ('moment_plain', 'moment_recipe'):
        storage = Path(env.get('CZB_STORAGE', ROOT / 'private-data')).resolve()
        output = storage / 'I2MOE_single_components_20261122' / 'historical' / arm / f'seed{args.seed}'
        if arm == 'moment_plain':
            command = [VARIANTS / 'fair_component_attribution/train.py', '--arm', 'plain']
        else:
            command = [VARIANTS / 'fair_complement_20261118/train_strict.py', '--method', 'risk_power', '--mode', 'formal']
        command += ['--output-dir', output]
        if (output / 'resume.pt').is_file():
            command.append('--resume')
    else:
        command = [RUN / 'fit.py', '--arm', arm]
    command += ['--seed', str(args.seed)]
    os.execve(sys.executable, [sys.executable, '-u', *map(str, command)], env)
