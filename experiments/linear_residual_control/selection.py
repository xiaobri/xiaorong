"""Lock learning rates using the two completed development seeds only."""
import argparse
from fractions import Fraction

from study import ARMS, RATES, OUT, digest, folder, name, original, read


def fraction(row):
    return Fraction(2 * row['tp'], 2 * row['tp'] + row['fp'] + row['fn'])


def trajectory(seed):
    d = folder(seed)
    assert read(d/'completed.json')['epochs'] == 20
    return [read(d / f'rows{ep:02d}.json') for ep in range(1,21)]


def winners(seed):
    rows = trajectory(seed)
    return {key: max((r[key] for r in rows), key=lambda r:(fraction(r), r['epoch'])) for key in rows[0]}


def lock():
    wins = {s:winners(s) for s in (3,42)}
    rates = {}
    for arm in ARMS:
        def criterion(lr):
            return sum(fraction(wins[s][name(arm, lr)]) for s in wins), -lr
        rates[arm] = max(RATES, key=criterion)
    payload = dict(rates=rates, seeds=[3,42], candidates=list(RATES),
                   development_ledger_sha256={str(s):[digest(folder(s)/f'rows{i:02d}.json')
                       for i in range(1,21)] for s in wins})
    path = OUT/'selection.json'
    if path.exists():
        assert read(path) == payload
    else:
        original.atomic_write_json(path, payload)
    return rates


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(lock())
