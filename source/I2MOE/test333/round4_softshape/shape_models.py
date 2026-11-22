"""SoftShape-inspired local EEG shapes, with an event-MIL alternative.

Independent adaptation: 19 channels, four internal top-2 experts, normalized
background merge, and sparsification active from the first of 30 epochs.
The event variant additionally makes the new shape encoder invariant to global
polarity and the canonical left-right electrode reflection.
"""
import czb_paths as _paths
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from models4 import focal_each
from raw_data import RawDataset
from raw_models import supervised_contrastive

OUTPUT = Path(f'{_paths.RUNS}/I2MOE_test333_round4_softshape_20261114')
VARIANTS = ('shape_sparse', 'shape_event')
VENDOR_COMMIT = '79b3b47abbcfd027bed1754f9c7238f1ebe9ee72'


class RMSNorm(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = values.float() * values.float().square().mean(-1, keepdim=True).add(1e-6).rsqrt()
        return normalized * self.weight


def sparsify(values: torch.Tensor, scores: torch.Tensor, ratio: float) -> torch.Tensor:
    """Keep ordered high-score tokens and summarize all remaining tokens."""
    if ratio == 1:
        return values * scores
    count = max(1, min(values.shape[1] - 1, math.ceil(values.shape[1] * ratio)))
    indices = scores.squeeze(-1).topk(count, dim=1).indices.sort(dim=1).values
    retained = values.gather(1, indices[:, :, None].expand(-1, -1, values.shape[-1]))
    retained_scores = scores.gather(1, indices[:, :, None])
    remaining = torch.ones_like(scores).scatter(1, indices[:, :, None], 0)
    weights = scores * remaining
    background = (values * weights).sum(1, keepdim=True) / weights.sum(1, keepdim=True).clamp_min(1e-6)
    return torch.cat((retained * retained_scores, background), 1)


class SparseShapes(nn.Module):
    """Actually dispatch selected token rows; never evaluate inactive experts."""
    def __init__(self, width: int = 128, experts: int = 4, topk: int = 2):
        super().__init__()
        self.topk = topk
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Dropout(.1),
                          nn.Linear(2 * width, width)) for _ in range(experts)])
        self.router = nn.Linear(width, experts, bias=False)
        nn.init.normal_(self.router.weight, std=.02)
        self.norm = RMSNorm(width)

    def forward(self, values: torch.Tensor):
        flat = values.reshape(-1, values.shape[-1])
        probabilities = self.router(flat).float().softmax(-1)
        weights, selected = probabilities.topk(self.topk, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        result = torch.zeros_like(flat, dtype=torch.float32)
        loads = []
        for index, expert in enumerate(self.experts):
            rows, slots = (selected == index).nonzero(as_tuple=True)
            loads.append((selected == index).float().mean())
            if rows.numel():
                updates = expert(flat[rows]).float() * weights[rows, slots, None]
                result = result.index_add(0, rows, updates)
        fraction = torch.stack(loads).detach()
        # Switch-style differentiable load objective. The tiny denominator
        # gradient of a normalized top-1 gate is avoided by choosing top-2.
        balance = len(self.experts) * (probabilities.mean(0) * fraction).sum()
        return F.gelu(self.norm(values + result.reshape_as(values))), balance, fraction


class ShapeContext(nn.Module):
    def __init__(self, width: int = 128):
        super().__init__()
        self.bottleneck = nn.Conv1d(width, 32, 1, bias=False)
        self.convolutions = nn.ModuleList([nn.Conv1d(32, 32, k, padding=k // 2, bias=False)
                                           for k in (39, 19, 9)])
        self.pool = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1),
                                  nn.Conv1d(width, 32, 1, bias=False))
        self.norm = nn.InstanceNorm1d(width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        sequence = values.transpose(1, 2)
        reduced = self.bottleneck(sequence)
        combined = torch.cat([conv(reduced) for conv in self.convolutions] + [self.pool(sequence)], 1)
        return F.gelu(self.norm(combined)).transpose(1, 2)


class ShapeEncoder(nn.Module):
    def __init__(self, invariant: bool):
        super().__init__()
        self.invariant = invariant
        frequency = torch.fft.rfftfreq(2000, d=1 / 500)
        mask = (frequency >= 1) & (frequency <= 200)
        for line in (50, 100, 150, 200):
            mask &= (frequency - line).abs() > 1
        self.register_buffer('frequency_mask', mask.float()[None, None])
        self.register_buffer('mirror', torch.tensor((1, 0, 3, 2, 5, 4, 7, 6, 9, 8,
                                                    11, 10, 13, 12, 15, 14, 16, 17, 18)))
        # 196 overlapping 100 ms shapes cover all 2000 original samples.
        self.patch = nn.Conv1d(19, 128, 50, stride=10, bias=False)
        self.position = nn.Parameter(torch.randn(1, 196, 128) * .02)
        self.norms = nn.ModuleList([RMSNorm(128) for _ in range(3)])
        self.score = nn.Sequential(nn.Linear(128, 32), nn.Tanh(), nn.Linear(32, 1), nn.Sigmoid())
        self.context = nn.ModuleList([ShapeContext() for _ in range(3)])
        self.mixture = SparseShapes()
        self.project = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU())

    def forward(self, raw: torch.Tensor):
        with torch.autocast(device_type=raw.device.type, enabled=False):
            signal = torch.fft.irfft(torch.fft.rfft(raw.float(), dim=-1) * self.frequency_mask,
                                     n=2000, dim=-1)
            signal = signal / signal.square().mean(-1, keepdim=True).sqrt().clamp_min(.01)
        patches = self.patch(signal).transpose(1, 2)
        if self.invariant:
            mirrored = self.patch(signal[:, self.mirror]).transpose(1, 2)
            original_even = .5 * (F.gelu(patches) + F.gelu(-patches))
            mirrored_even = .5 * (F.gelu(mirrored) + F.gelu(-mirrored))
            patches = .5 * (original_even + mirrored_even)
        tokens = patches + self.position
        balance = tokens.sum() * 0
        loads = []
        for norm, context, ratio in zip(self.norms, self.context, (1., .75, .5)):
            tokens = norm(tokens)
            tokens = sparsify(tokens, self.score(tokens), ratio)
            local, penalty, fraction = self.mixture(tokens)
            tokens = F.gelu(tokens + local + context(tokens))
            balance = balance + penalty / 3
            loads.append(fraction)
        embedding = self.project(torch.cat((tokens.mean(1), tokens.amax(1)), -1))
        return embedding, tokens, balance, torch.stack(loads)


class ShapeCorrection(nn.Module):
    def __init__(self, variant: str):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.encoder = ShapeEncoder(invariant=variant == 'shape_event')
        self.auxiliary = nn.Linear(128, 1)
        self.head = nn.Sequential(nn.LayerNorm(265), nn.Linear(265, 64), nn.GELU(),
                                  nn.Dropout(.1), nn.Linear(64, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, batch: dict) -> dict:
        embedding, tokens, balance, loads = self.encoder(batch['raw'])
        if self.variant == 'shape_event':
            # Fixed local top-four bag rule; labels never choose locations.
            instance = self.auxiliary(tokens).squeeze(-1).float()
            standalone = instance.topk(4, dim=1).values.mean(1)
        else:
            standalone = self.auxiliary(embedding).squeeze(-1).float()
        features = torch.cat((embedding.float(), batch['features'].float(),
                              batch['base'].float() / 4, batch['anchor'].float()[:, None] / 4,
                              standalone[:, None] / 4), -1)
        delta = .75 * torch.tanh(self.head(features).squeeze(-1).float() / .75)
        return dict(fused=batch['anchor'].float() + delta, delta=delta, embedding=embedding,
                    standalone=standalone, route_balance=balance, route_fraction=loads)


def objective(output: dict, batch: dict, variant: str) -> torch.Tensor:
    labels = batch['label']
    return (focal_each(output['fused'], labels).mean()
            + .02 * output['delta'].square().mean()
            + .1 * focal_each(output['standalone'], labels).mean()
            + .003 * supervised_contrastive(output['embedding'], labels)
            + .001 * output['route_balance'])
