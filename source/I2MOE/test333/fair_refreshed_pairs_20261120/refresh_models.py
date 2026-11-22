"""Refresh stale parent margins in the within-acquisition regional pair ranker."""
import torch

from allocation_models import ExpertAllocationMoE
from pair_models import PairState, RiskPairMoE, method_contract as pair_contract
from pair_models import method_loss_terms as pair_loss
from refresh_execution import EPOCH_FIELD, ROW_FIELD, history_batch

METHODS = ('collection', 'fresh_cases', 'fresh_controls', 'fresh_both')


def method_contract(method: str) -> dict:
    result = pair_contract('within_quantiles')
    result.update(method=method, parent='within_quantiles', positive_margin_refresh=method in ('fresh_cases','fresh_both'),
        negative_margin_refresh=method in ('fresh_controls','fresh_both'),
        hypothesis='Training dropout and optimizer updates make the parent margin used at control collection differ from the parent at the time of comparison.',
        procedure='Keep the previously observed control risk, descriptors, bank size and sixteen within-acquisition quantiles. For the chosen arms, re-evaluate existing positive or prior negative packets under the current six-expert parent in evaluation mode before the update. A refreshed control retains the updated margin until eviction.',
        positive_calibration='The all-window calibration also gets the refreshed positive margin when enabled. All other windows retain the parent score at collection.',
        cost='Up to sixteen extra frozen-feature parent-head predictions per eligible drawn positive, plus its own prediction in the case-refresh arms. No new unique training row, optimizer step, foundation model or inference branch.')
    return result


class HistoryState(PairState):
    def __init__(self, width, capacity=2048):
        super().__init__(width, within=True, nearest=False, capacity=capacity)
        self.register_buffer('row_ids', torch.full((capacity,), -1, dtype=torch.long))
        self.register_buffer('epochs', torch.full((capacity,), -1, dtype=torch.long))
        self.register_buffer('refreshed_controls', torch.zeros((), dtype=torch.long))
        self.register_buffer('refreshed_cases', torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def remember_positions(self, labels, groups, ids, epochs):
        eligible = (groups >= 0) & self.support[groups.clamp_min(0)] & (labels == 0)
        kept = torch.where(eligible)[0][-self.capacity:]
        slots = (self.cursor+torch.arange(len(kept), device=labels.device)) % self.capacity
        # Called immediately before PairState.update writes the corresponding negative feature slots.
        self.row_ids[slots] = ids[kept]
        self.epochs[slots] = epochs[kept]


class RefreshedPairMoE(RiskPairMoE):
    def __init__(self, method: str, **kwargs):
        if method not in METHODS:
            raise ValueError(method)
        super().__init__('within_quantiles', **kwargs)
        self.refresh_method = method
        self.correction.analytic = HistoryState(5*(self.width+7))

    def forward(self, eeg, compute_interactions=None):
        out = super().forward(eeg, compute_interactions)
        if self.training and eeg.ndim == 4:
            out.history_packet = eeg
            out.refresh_history = self.refresh_history
        return out

    @torch.no_grad()
    def current_parent(self, packet):
        modes = [(module,module.training) for module in self.modules()]
        try:
            self.eval()
            result = ExpertAllocationMoE.forward(self, packet, compute_interactions=False)
            return result.logits.diff(dim=-1).flatten()
        finally:
            for module, mode in modes:
                module.training = mode

    @torch.no_grad()
    def refresh_history(self, out, labels):
        packet = out.history_packet
        groups = out.training_groups
        state = self.correction.analytic
        positive = torch.where((groups >= 0) & state.support[groups.clamp_min(0)] & (labels == 1))[0]
        if len(positive) and self.refresh_method in ('fresh_cases','fresh_both'):
            margin = out.parent_margin.clone()
            margin[positive] = self.current_parent(packet[positive])
            out.parent_margin = margin
            state.refreshed_cases.add_(len(positive))
        if len(positive) and self.refresh_method in ('fresh_controls','fresh_both'):
            adjusted = out.allocation_output.expert_logits[:, :5]
            risk = (adjusted[...,1]-adjusted[...,0]).mean(1)
            slots = torch.cat([state.select(groups[i],risk[i]) for i in positive]).unique()
            if len(slots):
                stored = history_batch(state.row_ids[slots].cpu().tolist(), state.epochs[slots].cpu().tolist())
                state.negative_margin[slots] = self.current_parent(stored.to(packet.device))
                state.refreshed_controls.add_(len(slots))
        state.remember_positions(labels,groups,packet[:,0,3,ROW_FIELD].long(),packet[:,0,3,EPOCH_FIELD].long())

    def architecture_summary(self):
        result = super().architecture_summary()
        result.update(method=self.refresh_method, method_contract=method_contract(self.refresh_method))
        return result


def method_loss_terms(method, output, labels, args, focal):
    if output.cohort_state.analytic.training and torch.is_grad_enabled():
        output.refresh_history(output, labels)
    return pair_loss('within_quantiles', output, labels, args, focal)
