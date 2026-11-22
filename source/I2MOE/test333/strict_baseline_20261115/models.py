"""Fresh correction branches trained within the original baseline budget.

No historical six-expert, CBra, or correction checkpoint is loaded. The five
frozen regional sources are exactly those used by the baseline. The unchanged
baseline loss supervises the corrected final logits and original sixth expert.
"""
import bootstrap
import torch
from torch import nn

from i2moe_eeg.f1_variants import SixExpertF1
from channel_models import ChannelEncoder
from refine_models import RefinedEncoder

METHODS = ('baseline', 'channel', 'orbit', 'joint')


class WaveformBranch(nn.Module):
    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind
        if kind == 'channel':
            self.encoder = ChannelEncoder('plain')
        elif kind == 'orbit':
            self.encoder = RefinedEncoder('event_late_orbit')
            self.event_head = nn.Linear(128, 1)
        else:
            raise ValueError(kind)
        # Own 128 features, seven baseline margins, and own event score.
        self.head = nn.Sequential(nn.LayerNorm(136), nn.Linear(136, 64),
                                  nn.GELU(), nn.Dropout(.1), nn.Linear(64, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, eeg: torch.Tensor, margins: torch.Tensor) -> torch.Tensor:
        # Input remains the exact baseline-preprocessed tensor (volts / 1e-3).
        # The fixed x10 adapter matches the encoder's amplitude-floor units;
        # it neither reads a second raw cache nor changes the selected window.
        raw = eeg * 10
        if self.kind == 'channel':
            embedding, event = self.encoder(raw)
        else:
            embedding, tokens, _, _ = self.encoder(raw)
            event = self.event_head(tokens).squeeze(-1).topk(4, dim=-1).values.mean(-1)
            embedding = ((embedding[0] + embedding[1]) +
                         (embedding[2] + embedding[3])) * .25
            event = ((event[0] + event[1]) + (event[2] + event[3])) * .25
        features = torch.cat((embedding, margins.detach() / 4, event[:, None] / 4), -1)
        return .75 * torch.tanh(self.head(features).squeeze(-1) / .75)


class StrictCorrectionMoE(SixExpertF1):
    def __init__(self, method: str, **kwargs):
        super().__init__(**kwargs)
        if method not in METHODS[1:]:
            raise ValueError(method)
        self.method = method
        kinds = ('channel', 'orbit') if method == 'joint' else (method,)
        # Do not shift the baseline's initial weights or random-number stream.
        with torch.random.fork_rng(devices=[]):
            self.corrections = nn.ModuleDict({kind: WaveformBranch(kind) for kind in kinds})

    def forward(self, eeg: torch.Tensor, compute_interactions=None):
        result = super().forward(eeg, compute_interactions)
        margins = torch.cat(((result.logits[:, 1] - result.logits[:, 0])[:, None],
                             result.expert_logits[..., 1] - result.expert_logits[..., 0]), -1)
        devices = [eeg.device.index] if eeg.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            deltas = torch.stack([branch(eeg, margins) for branch in self.corrections.values()])
        delta = deltas.mean(0)  # Fixed mean; no validation alpha or weight search.
        result.logits = result.logits + torch.stack((-delta / 2, delta / 2), -1)
        return result

    def architecture_summary(self):
        result = super().architecture_summary()
        result.update(strict_method=self.method,
                      added_correction_branches=list(self.corrections),
                      original_six_router_unchanged=True,
                      trainable_sixth_and_router_fresh=True,
                      prior_cbra_or_six_checkpoint_loaded=False,
                      correction_bound=.75, correction_combination='fixed arithmetic mean of margins',
                      loss='exact original baseline loss, no new auxiliary terms',
                      outer_test_time_augmentation=False,
                      internal_orbit_views=4 if 'orbit' in self.corrections else 0)
        return result

