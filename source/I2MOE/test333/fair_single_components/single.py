"""Isolated interventions on the original six-expert predictor."""
from dataclasses import replace
from pathlib import Path
import sys

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'fair_refreshed_pairs_20261120'))
import bootstrap
from alignment_models import matched_rank
from allocation_models import AllocationLayer
from analytic_models import AnalyticRegionMoE
from block_features import base_weights, register_blocks
from cohort_models import MAX_GROUPS, focal_per_row, group_weights
from complement_models import ComplementMoE, method_loss_terms as moment_recipe_loss
from i2moe_eeg.f1_variants import SixExpertF1, loss_terms as core_loss
from i2moe_eeg.losses import task_loss
from moment_models import source_moments
from peak_models import PeakRefinement, corrected, descriptors
from refresh_models import HistoryState, RefreshedPairMoE
from risk_models import hard_rank

SEEDS = (3, 42, 0, 1, 2)
ARMS = ('moment_plain', 'moment_recipe', 'weight', 'rank', 'peak', 'regional_time',
        'time', 'pairs', 'pairs_refresh', 'aux_only', 'peak_native', 'time_native', 'time_aux')
PAIR = ('pairs', 'pairs_refresh')
TIME = ('regional_time', 'time', 'time_native', 'time_aux')
MOMENT = ('moment_plain', 'moment_recipe')
OUT = bootstrap.CACHE.parent / 'I2MOE_single_components_20261122'


class SingleComponent(SixExpertF1):
    def __init__(self, arm, **kwargs):
        super().__init__(**kwargs)
        self.arm = arm
        self.moments = {}
        self.correction = nn.Module()
        with torch.random.fork_rng(devices=[]):
            if arm in ('peak', 'peak_native'):
                width = next(iter(self.frozen_experts.values())).encoder.projection[0].in_features
                self.correction.refiner = PeakRefinement(width)
                for i, expert in enumerate(self.frozen_experts.values()):
                    def capture(module, inputs, value, index=i):
                        self.moments[index] = source_moments(value).detach()
                    expert.encoder.features.register_forward_hook(capture)
            if arm in TIME:
                self.correction.allocation = AllocationLayer(self.hidden_dim)
                if arm == 'regional_time':
                    self.correction.allocation.redundancy.requires_grad_(False)
            if arm in TIME + PAIR:
                self.block_capture, self._handles = register_blocks(self.frozen_experts)
            if arm in PAIR:
                self.width, self.temporal = min(32, self.hidden_dim), True
                self.refresh_method = 'fresh_both' if arm == 'pairs_refresh' else 'collection'
                self.correction.analytic = HistoryState(5*(self.width + 7))
        self.correction.register_buffer('group_base_weights', torch.zeros(2, MAX_GROUPS))
        if arm == 'weight':
            self.correction.register_buffer('group_probabilities', torch.zeros(2, MAX_GROUPS))
            self.correction.register_buffer('group_risk', torch.ones(2, MAX_GROUPS))
            self.correction.register_buffer('group_seen', torch.zeros(2, MAX_GROUPS, dtype=torch.bool))

    @torch.no_grad()
    def current_parent(self, packet):
        modes = [(m, m.training) for m in self.modules()]
        try:
            self.eval()
            return super().forward(packet, False).logits.diff(dim=-1).flatten()
        finally:
            for m, mode in modes:
                m.training = mode

    def time_output(self, eeg, core):
        blocks = (eeg[:, :, 3, 256:].reshape(len(eeg), 5, 8, 128) if eeg.ndim == 4
                  else torch.stack([self.block_capture[i][0] for i in range(5)], 1))
        sources = list(self.frozen_experts.values())
        old = core.expert_logits.detach()
        regions, change, _ = self.correction.allocation(blocks, base_weights(63, blocks),
                            core.frozen_features.detach(), old[:, :5], sources, False)
        if self.arm == 'regional_time':
            change = torch.zeros_like(change)
        global_score = old[:, 5] + torch.stack((-change / 2, change / 2), 1)
        experts = torch.cat((regions, global_score[:, None]), 1)
        shift = ((experts - old) * core.interaction_weights.detach()[..., None]).sum(1)
        return replace(core, logits=core.logits.detach() + shift, expert_logits=experts)

    def forward(self, eeg, compute_interactions=None):
        core = super().forward(eeg, compute_interactions)
        if self.arm in ('peak', 'peak_native'):
            context = torch.cat((core.expert_logits.detach().diff(dim=-1).squeeze(-1),
                                 core.logits.detach().diff(dim=-1)), 1)
            shapes = descriptors(torch.stack([self.moments[i].detach() for i in range(5)], 1))
            out = replace(core, logits=corrected(core.logits, self.correction.refiner(shapes, context)))
        elif self.arm in TIME:
            out = self.time_output(eeg, core)
        elif self.arm in PAIR:
            x = AnalyticRegionMoE.observations(self, eeg, core)
            margin = core.logits.detach().diff(dim=-1).flatten()
            delta = self.correction.analytic(x, margin)
            out = replace(core, logits=core.logits + torch.stack((-delta / 2, delta / 2), 1))
            out.analytic_features, out.parent_margin = x, margin
            out.allocation_output = core
            if self.training:
                out.history_packet = eeg
        else:
            out = replace(core)
        out.core_output, out.cohort_state = core, self.correction
        out.training_groups = (eeg[:, 0, 3, 130].long() if eeg.ndim == 4 else
                               torch.full((len(eeg),), -1, dtype=torch.long, device=eeg.device))
        if self.arm in PAIR and self.training:
            out.refresh_history = lambda y: RefreshedPairMoE.refresh_history(self, out, y)
        return out


def build(arm, **kwargs):
    if arm not in ARMS:
        raise ValueError(arm)
    return ComplementMoE('risk_power', **kwargs) if arm in MOMENT else SingleComponent(arm, **kwargs)


def loss(arm, out, labels, args, focal):
    if arm == 'moment_recipe':
        return moment_recipe_loss('risk_power', out, labels, args, focal)
    parts = core_loss(out.core_output, labels, args, focal)
    per_row = focal_per_row(out.logits, labels, focal)
    extra = per_row.new_zeros(())
    if arm == 'weight':
        weights = group_weights('cohort_dro', out, labels, per_row)
        extra = (per_row * weights).mean() - parts['primary']
    elif arm == 'rank':
        extra = .03 * hard_rank(out.logits, labels, torch.ones_like(per_row), 16)
    elif arm in PAIR:
        if arm == 'pairs_refresh':
            out.refresh_history(labels)
        risk = out.core_output.expert_logits[:, :5].detach().diff(dim=-1).squeeze(-1).mean(1)
        out.cohort_state.analytic.update(out.analytic_features, out.parent_margin, labels, out.training_groups, risk)
    elif arm in ('peak_native', 'time_native', 'time_aux', 'aux_only'):
        w = out.cohort_state.group_base_weights[labels, out.training_groups]
        if arm != 'aux_only':
            risk = out.core_output.expert_logits[:, :5].detach().diff(dim=-1).squeeze(-1).mean(1)
            rank = (hard_rank(out.logits, labels, w, 16) if arm == 'peak_native' else
                    matched_rank(out.logits, labels, w, risk))
            extra = (w * per_row).mean() + .03 * rank
        if arm in ('aux_only', 'time_aux'):
            reg = focal_per_row(out.expert_logits[:, :5].reshape(-1, 2), labels.repeat_interleave(5), focal)
            extra = extra + .1 * (w * reg.reshape(-1, 5).mean(1)).mean()
    else:
        extra = per_row.mean()
    fused = task_loss(out.logits, labels, args.task_loss, focal)
    parts['total'] = parts['total'] + extra
    parts['extra'] = parts['extra'] + parts['primary'] + extra - fused
    parts['primary'] = fused
    return parts
