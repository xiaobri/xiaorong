"""Paths, registered observations and the four pre-specified readouts."""
import hashlib
import os
from dataclasses import replace
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'source'))
sys.path.insert(0, str(ROOT / 'source/I2MOE/test333/fair_single_components'))
import fit
from single import AnalyticRegionMoE, register_blocks

OUT = Path(os.environ.get('CZB_LINEAR_OUTPUT', fit.bootstrap.CACHE.parent / 'issue1_linear_residual'))
ARMS = ('linear_focal_all', 'linear_bce_all', 'linear_focal_eligible', 'linear_bce_eligible')
RATES = (.0001, .0005, .002)
REFS = ('baseline', 'pairs', 'pairs_refresh')
SEEDS = (0, 1, 2, 3, 42)
original = fit.original
read = fit.read
digest = fit.digest


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def folder(seed):
    return OUT / f'seed{seed}'


def name(arm, lr):
    return f'{arm}_{lr:g}'


class Observations:
    def __init__(self, core):
        self.frozen_experts = core.frozen_experts
        self.width = min(32, core.hidden_dim)
        self.temporal = True
        self.block_capture, self.handles = register_blocks(self.frozen_experts)

    @torch.no_grad()
    def __call__(self, eeg, output):
        x = AnalyticRegionMoE.observations(self, eeg, output)
        if x.shape != (len(eeg), 195):
            raise ValueError('Unexpected regional feature schema')
        return x.detach().clamp(-32, 32)

    def close(self):
        for handle in self.handles:
            handle.remove()


def new_head(device):
    with torch.random.fork_rng(devices=[]):
        layer = nn.Linear(195, 1)
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
    return layer.to(device)


def corrected(logits, x, head):
    delta = head(x.detach()).squeeze(1)
    return logits.detach() + torch.stack((-delta / 2, delta / 2), dim=1)


class Learner:
    def __init__(self, arm, lr, device):
        self.arm, self.lr = arm, lr
        self.head = new_head(device)
        self.opt = torch.optim.AdamW([{'params': [self.head.weight], 'weight_decay': .01},
                                      {'params': [self.head.bias], 'weight_decay': 0.}], lr=lr)

    def save(self):
        return dict(arm=self.arm, lr=self.lr, head=self.head.state_dict(), optimizer=self.opt.state_dict())

    def restore(self, state):
        if (state['arm'], state['lr']) != (self.arm, self.lr):
            raise ValueError('Head resume configuration changed')
        self.head.load_state_dict(state['head'])
        self.opt.load_state_dict(state['optimizer'])

    def step(self, logits, x, labels, mask, focal):
        if self.arm.endswith('eligible'):
            logits, x, labels = logits[mask], x[mask], labels[mask]
        if not len(labels):
            return None
        self.opt.zero_grad(set_to_none=True)
        pred = corrected(logits, x, self.head)
        if 'focal' in self.arm:
            loss = focal(pred, labels)
        else:
            loss = F.binary_cross_entropy_with_logits(pred[:, 1] - pred[:, 0], labels.float())
        loss.backward()
        self.opt.step()
        return loss.detach(), len(labels)


class RawLinear(nn.Module):
    def __init__(self, core, head):
        super().__init__()
        self.core, self.head = core, head
        self.features = Observations(core)

    @property
    def frozen_experts(self):
        return self.core.frozen_experts

    def forward(self, eeg, compute_interactions=False):
        out = self.core(eeg, compute_interactions=compute_interactions)
        return replace(out, logits=corrected(out.logits, self.features(eeg, out), self.head))
