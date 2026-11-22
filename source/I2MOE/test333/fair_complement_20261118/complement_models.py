"""Cross-region and independent-head corrections with Train cohort robustness."""
import bootstrap
import torch
from torch import nn

from cohort_models import CohortCorrectionMoE, focal_per_row, group_weights
from protected_models import CorrectiveHead
from moment_models import PerViewTrainMeanCenter
from risk_models import hard_rank, method_contract as risk_contract
from i2moe_eeg.f1_variants import loss_terms as baseline_loss_terms
from i2moe_eeg.losses import task_loss

METHODS = ('risk_power', 'risk_attention', 'risk_ensemble', 'risk_ensemble_power')


def method_contract(method):
    result = risk_contract('risk_cap16')
    power = 'power' in method
    heads = 4 if 'ensemble' in method else 1
    result.update(method=method, correction_heads=heads,
        static_weights=f'Train-only class-conditional inverse-frequency power{1. if power else .5}; cap{64 if power else 16}; population mean1',
        cohort_weighting='Train-only conditional static weights and EMA risk',
        correction_architecture='Independent moment MLP heads' if heads == 4 else
            ('Cross-region moment attention' if method == 'risk_attention' else 'Original moment MLP'),
        additional_head_loss='Mean member corrected focal plus0.03 weighted top16-negative ranking; one AdamW update',
        final_fusion='Mean member bounded margin corrections, plus detached original six-expert margin',
        heads_train_on_same_original_draws=True, extra_training_draws=0, extra_optimizer_steps=0,
        baseline_path_receives_only_original_loss_gradients=True,
        core_trajectory_expected_equal_to_baseline=True)
    return result


class AttentionCorrection(nn.Module):
    def __init__(self, width, source_width, dropout):
        super().__init__()
        self.encoders = nn.ModuleList([nn.Sequential(nn.LayerNorm(source_width*3),
            nn.Linear(source_width*3, 128), nn.GELU(), nn.LayerNorm(128)) for _ in range(5)])
        self.positions = nn.Parameter(torch.randn(5, 128)*.01)
        self.attention = nn.TransformerEncoderLayer(128, 4, 256, dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.predictor = nn.Sequential(nn.LayerNorm(256+6*width+7), nn.Linear(256+6*width+7, 128),
            nn.GELU(), nn.Dropout(dropout), PerViewTrainMeanCenter(128), nn.Linear(128, 1, bias=False))
        nn.init.zeros_(self.predictor[-1].weight)

    def forward(self, core, local_tokens, moments):
        tokens = torch.stack([encoder(moments[i].detach().flatten(1)) for i, encoder in enumerate(self.encoders)], 1)
        tokens = self.attention(tokens+self.positions[None])
        logits, base = core.expert_logits.detach(), core.logits.detach()
        context = torch.cat((tokens.mean(1), tokens.amax(1), core.router_features.detach().flatten(1),
            (base[:, 1]-base[:, 0])[:, None], logits[..., 1]-logits[..., 0]), -1)
        return torch.tanh(self.predictor(context).squeeze(-1))


class EnsembleCorrection(nn.Module):
    def __init__(self, width, source_width, dropout, classifiers):
        super().__init__()
        self.heads = nn.ModuleList([CorrectiveHead('protected_moments', width, source_width, dropout, classifiers)
                                   for _ in range(4)])
        self.member_margins = None

    def forward(self, core, local_tokens, moments):
        self.member_margins = torch.stack([head(core, local_tokens, moments) for head in self.heads], 1)
        return self.member_margins.mean(1)


class ComplementMoE(CohortCorrectionMoE):
    def __init__(self, method, **kwargs):
        if method not in METHODS:
            raise ValueError(method)
        super().__init__('cohort_dro', **kwargs)
        self.complement_method = method
        sources = [getattr(expert, 'module', expert) for expert in self.frozen_experts.values()]
        if method != 'risk_power':
            with torch.random.fork_rng(devices=[]):
                if method == 'risk_attention':
                    replacement = AttentionCorrection(self.hidden_dim, sources[0].encoder.projection[0].in_features,
                        kwargs.get('dropout', .3))
                else:
                    replacement = EnsembleCorrection(self.hidden_dim, sources[0].encoder.projection[0].in_features,
                        kwargs.get('dropout', .3), [source.classifier for source in sources])
            for name in ('group_risk', 'group_seen', 'group_probabilities', 'group_base_weights'):
                replacement.register_buffer(name, getattr(self.correction, name).clone())
            self.correction = replacement

    def forward(self, eeg, compute_interactions=None):
        output = super().forward(eeg, compute_interactions)
        if 'ensemble' in self.complement_method:
            output.member_margins = 2*self.correction.member_margins
        else:
            output.member_margins = output.correction_margin[:, None]
        return output

    def architecture_summary(self):
        result = super().architecture_summary()
        result.update(method=self.complement_method, method_contract=method_contract(self.complement_method))
        return result


def method_loss_terms(method, output, labels, args, focal):
    parts = baseline_loss_terms(output.core_output, labels, args, focal)
    core_primary = parts['primary']
    per_row = focal_per_row(output.logits, labels, focal)
    weights = group_weights('cohort_dro', output, labels, per_row)
    losses = []
    for margin in output.member_margins.unbind(1):
        logits = output.core_output.logits.detach()+torch.stack((-margin/2, margin/2), -1)
        losses.append((focal_per_row(logits, labels, focal)*weights).mean()+.03*hard_rank(logits, labels, weights, 16))
    correction = torch.stack(losses).mean()
    fused = task_loss(output.logits, labels, args.task_loss, focal)
    parts['total'] = parts['total']+correction
    parts['primary'] = fused
    parts['extra'] = parts['extra']+core_primary+correction-fused
    return parts
