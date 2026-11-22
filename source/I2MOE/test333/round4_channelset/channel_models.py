"""Shared channel morphology with CN/ACN and a PCN-inspired normalization.

CN/ACN follow the official ICML 2025 implementation, with an epsilon for
cosine normalization of zero vectors. PCN follows paper equations 7--9,
but uses small independent prototype initialization noise to avoid exact
prototype symmetry and a zero cosine denominator. This is an adaptation,
not a reproduction of the original time-series forecasting experiment.
"""
import czb_paths as _paths
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from domain_models import DomainDataset
from models4 import focal_each
from raw_models import supervised_contrastive

OUTPUT = Path(f'{_paths.RUNS}/I2MOE_test333_round4_channelset_20261114')
VARIANTS = ('channel_plain', 'channel_cn', 'channel_acn', 'channel_pcn')


class ChannelNorm(nn.Module):
    def __init__(self, kind: str, width: int = 128, channels: int = 19):
        super().__init__()
        self.kind, self.temperature, self.eps = kind, .1, 1e-5
        if kind == 'plain':
            self.weight = nn.Parameter(torch.ones(width))
            self.bias = nn.Parameter(torch.zeros(width))
        elif kind in ('cn', 'acn'):
            self.weight = nn.Parameter(torch.ones(channels, width))
            self.bias = nn.Parameter(torch.zeros(channels, width))
            if kind == 'acn':
                self.weight_global = nn.Parameter(torch.ones(channels, width))
                self.bias_global = nn.Parameter(torch.ones(channels, width))
        elif kind == 'pcn':
            self.projection = nn.Linear(2000, width)
            self.weight = nn.Parameter(torch.ones(5, width)+.02*torch.randn(5, width))
            self.bias = nn.Parameter(.02*torch.randn(5, width))
        else:
            raise ValueError(kind)

    def forward(self, z: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        # Keep variance and cosine calculations in full precision under AMP.
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            normalized = (z-z.mean(-1, keepdim=True))/torch.sqrt(
                z.var(-1, keepdim=True, unbiased=False)+self.eps)
            if self.kind in ('plain', 'cn'):
                return normalized*self.weight+self.bias
            if self.kind == 'acn':
                unit = F.normalize(normalized, dim=-1, eps=1e-6)
                weights = (unit@unit.transpose(-1, -2)/self.temperature).softmax(-1)
                scale = (weights@self.weight)*self.weight_global
                shift = (weights@self.bias)*self.bias_global
            else:
                context = F.normalize(self.projection(raw.float()), dim=-1, eps=1e-6)
                scale_weights = (context@F.normalize(self.weight, dim=-1, eps=1e-6).T
                                 / self.temperature).softmax(-1)
                shift_weights = (context@F.normalize(self.bias, dim=-1, eps=1e-6).T
                                 / self.temperature).softmax(-1)
                scale, shift = scale_weights@self.weight, shift_weights@self.bias
            return scale*normalized+shift


class ChannelBlock(nn.Module):
    def __init__(self, kind: str):
        super().__init__()
        self.qkv, self.projection = nn.Linear(128, 384), nn.Linear(128, 128)
        self.norm1, self.norm2 = ChannelNorm(kind), ChannelNorm(kind)
        self.dropout = nn.Dropout(.1)
        self.feed_forward = nn.Sequential(nn.Linear(128, 256), nn.GELU(),
                                         nn.Dropout(.1), nn.Linear(256, 128))

    def forward(self, z: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        batch, channels, _ = z.shape
        q, k, v = self.qkv(z).reshape(batch, channels, 3, 4, 32).permute(2, 0, 3, 1, 4)
        attention = ((q.float()@k.float().transpose(-1, -2))*(32**-.5)).softmax(-1)
        context = (attention.to(v.dtype)@v).transpose(1, 2).reshape(batch, channels, 128)
        z = self.norm1(z+self.dropout(self.projection(context)), raw)
        return self.norm2(z+self.dropout(self.feed_forward(z)), raw)


class ChannelEncoder(nn.Module):
    def __init__(self, kind: str):
        super().__init__()
        frequency = torch.fft.rfftfreq(2000, d=1/500)
        mask = (frequency >= 1) & (frequency <= 200)
        for line in (50, 100, 150, 200):
            mask &= (frequency-line).abs() > 1
        self.register_buffer('frequency_mask', mask.float()[None, None])
        self.banks = nn.ModuleList([nn.Conv1d(1, 16, k, stride=10, padding=k//2)
                                   for k in (7, 31, 101)])
        self.mix = nn.Sequential(nn.GroupNorm(6, 48), nn.GELU(), nn.Conv1d(48, 48, 1))
        self.context = nn.ModuleList([
            nn.Sequential(nn.Conv1d(48, 48, 9, padding=4*d, dilation=d, groups=48),
                          nn.GroupNorm(6, 48), nn.GELU(), nn.Conv1d(48, 48, 1))
            for d in (1, 4)])
        self.attention = nn.Sequential(nn.Conv1d(48, 16, 1), nn.Tanh(), nn.Conv1d(16, 1, 1))
        self.channel_embedding = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, 128), nn.GELU())
        self.blocks = nn.ModuleList([ChannelBlock(kind) for _ in range(2)])
        self.output = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU())
        self.local_auxiliary = nn.Conv1d(48, 1, 1)

    def forward(self, raw: torch.Tensor):
        batch, channels, length = raw.shape
        if channels != 19 or length != 2000:
            raise ValueError('Expected unchanged 19-channel, 2000-sample input')
        with torch.autocast(device_type=raw.device.type, enabled=False):
            x = torch.fft.irfft(torch.fft.rfft(raw.float(), dim=-1)*self.frequency_mask,
                                n=2000, dim=-1)
            x = x/x.square().mean(-1, keepdim=True).sqrt().clamp_min(.01)
        local = x.reshape(batch*channels, 1, length)
        local = self.mix(torch.cat([bank(local) for bank in self.banks], 1))
        for block in self.context:
            local = F.gelu(local+block(local))
        pooled = (local*self.attention(local).softmax(-1)).sum(-1)
        z = self.channel_embedding(torch.cat((pooled, local.amax(-1)), -1))
        z = z.reshape(batch, channels, 128)
        for block in self.blocks:
            z = block(z, x)
        embedding = self.output(torch.cat((z.mean(1), z.amax(1)), -1))
        local_scores = self.local_auxiliary(local).reshape(batch, -1).float()
        standalone = local_scores.topk(4, dim=-1).values.mean(-1)
        return embedding, standalone


class ChannelCorrection(nn.Module):
    def __init__(self, variant: str):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.encoder = ChannelEncoder(variant.removeprefix('channel_'))
        self.head = nn.Sequential(nn.LayerNorm(265), nn.Linear(265, 64), nn.GELU(),
                                  nn.Dropout(.1), nn.Linear(64, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, batch: dict) -> dict:
        embedding, standalone = self.encoder(batch['raw'])
        features = torch.cat((embedding.float(), batch['features'].float(),
                              batch['base'].float()/4, batch['anchor'].float()[:, None]/4,
                              standalone[:, None]/4), -1)
        delta = .75*torch.tanh(self.head(features).squeeze(-1).float()/.75)
        return dict(fused=batch['anchor'].float()+delta, delta=delta,
                    embedding=embedding, standalone=standalone)


def objective(output: dict, batch: dict, variant: str):
    labels, weights = batch['label'], batch['row_weight'].float()
    return ((weights*focal_each(output['fused'], labels)).mean()
            + .02*output['delta'].square().mean()
            + .1*(weights*focal_each(output['standalone'], labels)).mean()
            + .003*supervised_contrastive(output['embedding'], labels))
