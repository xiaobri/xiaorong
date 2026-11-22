"""Controlled refinements of local event shapes under the fixed EEG protocol.

Four fresh paired-seed runs isolate cohort weighting, multiscale shapes,
late group pooling, and hard-example ranking. No validation metadata enters
the model or loss. The frozen original experts and input window are unchanged.
"""
import czb_paths as _paths
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from domain_models import DomainDataset
from models4 import focal_each, hard_negative_rank
from raw_models import supervised_contrastive
from shape_models import ShapeEncoder, sparsify

OUTPUT = Path(f'{_paths.RUNS}/I2MOE_test333_round4_refine_20261114')
VARIANTS = ('event_cohort', 'event_multiscale', 'event_late_orbit', 'event_rank')


class RefinedEncoder(ShapeEncoder):
    def __init__(self, variant: str):
        super().__init__(invariant=True)
        self.variant = variant
        if variant == 'event_multiscale':
            self.patch = nn.ModuleList([
                nn.Conv1d(19, 128, size, stride=10, padding=size // 2, bias=False)
                for size in (25, 51, 101)])
            self.position = nn.Parameter(torch.randn(1, 200, 128) * .02)

    def filtered(self, raw: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=raw.device.type, enabled=False):
            signal = torch.fft.irfft(torch.fft.rfft(raw.float(), dim=-1)
                                     * self.frequency_mask, n=2000, dim=-1)
            return signal / signal.square().mean(-1, keepdim=True).sqrt().clamp_min(.01)

    def contextualize(self, patches: torch.Tensor):
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

    @staticmethod
    def even(values: torch.Tensor) -> torch.Tensor:
        return .5 * (F.gelu(values) + F.gelu(-values))

    def forward(self, raw: torch.Tensor):
        if self.variant not in ('event_multiscale', 'event_late_orbit'):
            return super().forward(raw)
        signal = self.filtered(raw)
        mirrored = signal[:, self.mirror]
        if self.variant == 'event_multiscale':
            scales = [.5 * (self.even(layer(signal)) + self.even(layer(mirrored)))
                      for layer in self.patch]
            return self.contextualize(torch.stack(scales).mean(0).transpose(1, 2))
        # Preserve signed morphology through nonlinear contextualization;
        # aggregate the four group elements only after event extraction.
        # Canonical group order: identity, polarity, reflection, both.
        views = torch.cat((signal, -signal, mirrored, -mirrored), 0)
        embedding, tokens, balance, loads = self.contextualize(self.patch(views).transpose(1, 2))
        return embedding.reshape(4, len(raw), -1), tokens.reshape(4, len(raw), *tokens.shape[1:]), balance, loads


class RefinedCorrection(nn.Module):
    def __init__(self, variant: str):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.encoder = RefinedEncoder(variant)
        self.auxiliary = nn.Linear(128, 1)
        self.head = nn.Sequential(nn.LayerNorm(265), nn.Linear(265, 64), nn.GELU(),
                                  nn.Dropout(.1), nn.Linear(64, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, batch: dict) -> dict:
        embedding, tokens, balance, loads = self.encoder(batch['raw'])
        instance = self.auxiliary(tokens).squeeze(-1).float()
        standalone = instance.topk(4, dim=-1).values.mean(-1)
        if self.variant == 'event_late_orbit':
            # Pair sums make group permutations stable in floating point.
            embedding = ((embedding[0] + embedding[1]) + (embedding[2] + embedding[3])) * .25
            standalone = ((standalone[0] + standalone[1]) + (standalone[2] + standalone[3])) * .25
        features = torch.cat((embedding.float(), batch['features'].float(),
                              batch['base'].float() / 4, batch['anchor'].float()[:, None] / 4,
                              standalone[:, None] / 4), -1)
        delta = .75 * torch.tanh(self.head(features).squeeze(-1).float() / .75)
        return dict(fused=batch['anchor'].float() + delta, delta=delta, embedding=embedding,
                    standalone=standalone, route_balance=balance, route_fraction=loads)


def objective(output: dict, batch: dict, variant: str) -> torch.Tensor:
    labels = batch['label']
    weights = batch['row_weight'].float()
    loss = ((weights * focal_each(output['fused'], labels)).mean()
            + .02 * output['delta'].square().mean()
            + .1 * (weights * focal_each(output['standalone'], labels)).mean()
            + .003 * supervised_contrastive(output['embedding'], labels)
            + .001 * output['route_balance'])
    if variant == 'event_rank':
        loss = loss + .005 * hard_negative_rank(output['fused'], labels)
    return loss
