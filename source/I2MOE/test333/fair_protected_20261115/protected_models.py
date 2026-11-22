"""Train the original six experts independently from a corrective branch.

Both paths start fresh and use the same original batches/optimizer steps.
Only the original objective sends gradients into the six-expert baseline.
The added fused-task objective, and optional ranking term, train a detached
correction path. This is an explicit loss-method change, not extra pretraining.
"""
from dataclasses import replace

import bootstrap
import torch
from torch import nn

from i2moe_eeg.f1_variants import SixExpertF1, loss_terms as baseline_loss_terms
from i2moe_eeg.losses import task_loss
from moment_models import PerViewTrainMeanCenter, source_moments
from routing_models import within_batch_rank_loss

METHODS = ('protected_scores', 'protected_temporal', 'protected_moments', 'protected_moments_rank')
MARGIN_BOUND = 1.


def method_contract(method: str) -> dict:
    return dict(original_six_expert_objective_retained=True, unchanged_total_loss=False,
        extra_fused_task_loss_weight=1., extra_rank_weight=.01 if method.endswith('_rank') else 0.,
        baseline_path_receives_only_original_loss_gradients=True,
        correction_inputs_and_base_logits_detached=True,
        baseline_and_correction_initialized_fresh=True,
        historical_baseline_checkpoint_reused=False,
        extra_training_draws=0, extra_optimizer_steps=0,
        same_batches_and_adamw_updates=True,
        bounded_margin_correction=MARGIN_BOUND,
        evaluation_center='Fixed running mean from training batches',
        final_f1_noninferiority_guaranteed=False)


class CorrectiveHead(nn.Module):
    def __init__(self, method: str, width: int, source_width: int, dropout: float, classifiers):
        super().__init__()
        self.method = method
        object.__setattr__(self, 'source_classifiers', tuple(classifiers))
        self.encoders = None
        dimension = 7
        if method != 'protected_scores':
            input_width = width*2 if method == 'protected_temporal' else source_width*3
            self.encoders = nn.ModuleList([nn.Sequential(nn.LayerNorm(input_width),
                nn.Linear(input_width, 64), nn.GELU()) for _ in range(5)])
            dimension += width*6 + 64*5
        self.predictor = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 128),
            nn.GELU(), nn.Dropout(dropout), PerViewTrainMeanCenter(128), nn.Linear(128, 1, bias=False))
        nn.init.zeros_(self.predictor[-1].weight)

    def forward(self, core, local_tokens, moments):
        logits = core.expert_logits.detach()
        base = core.logits.detach()
        margins = logits[..., 1]-logits[..., 0]
        context = torch.cat(((base[:, 1]-base[:, 0])[:, None], margins), -1)
        if self.encoders is not None:
            features = []
            for index, encoder in enumerate(self.encoders):
                if self.method == 'protected_temporal':
                    tokens = local_tokens[index].detach()
                    with torch.no_grad():
                        source = self.source_classifiers[index](tokens)
                        evidence = (source[..., 1]-source[..., 0])/2
                        positive = (tokens*evidence.softmax(1)[..., None]).sum(1)
                        negative = (tokens*(-evidence).softmax(1)[..., None]).sum(1)
                    descriptor = torch.cat((positive, positive-negative), -1)
                else:
                    descriptor = moments[index].detach().flatten(1)
                features.append(encoder(descriptor))
            context = torch.cat((context, core.router_features.detach().flatten(1), *features), -1)
        return MARGIN_BOUND*torch.tanh(self.predictor(context).squeeze(-1)/MARGIN_BOUND)


class ProtectedCorrectionMoE(SixExpertF1):
    def __init__(self, method: str, **kwargs):
        super().__init__(**kwargs)
        if method not in METHODS:
            raise ValueError(method)
        self.method, self.local_tokens, self.moments = method, {}, {}
        sources = [getattr(expert, 'module', expert) for expert in self.frozen_experts.values()]
        source_width = sources[0].encoder.projection[0].in_features
        with torch.random.fork_rng(devices=[]):
            self.correction = CorrectiveHead(method, self.hidden_dim, source_width,
                kwargs.get('dropout', .3), [source.classifier for source in sources])
        self._source_hooks = []
        if method != 'protected_scores':
            for index, source in enumerate(sources):
                def capture(module, inputs, output, index=index, source=source):
                    if self.method == 'protected_temporal':
                        self.local_tokens[index] = source.encoder.projection(output.mean(-2).transpose(1, 2)).detach()
                    else:
                        self.moments[index] = source_moments(output).detach()
                self._source_hooks.append(source.encoder.features.register_forward_hook(capture))

    def _encode_frozen(self, eeg):
        self.local_tokens.clear()
        self.moments.clear()
        result = super()._encode_frozen(eeg)
        if self.method == 'protected_temporal' and len(self.local_tokens) != 5:
            raise RuntimeError('Missing frozen temporal descriptors')
        if self.method.startswith('protected_moments') and len(self.moments) != 5:
            raise RuntimeError('Missing frozen pre-projection moments')
        return result

    def forward(self, eeg, compute_interactions=None):
        core = super().forward(eeg, compute_interactions)
        devices = [eeg.device.index] if eeg.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            margin = self.correction(core, self.local_tokens, self.moments)
        result = replace(core, logits=core.logits.detach()+torch.stack((-margin/2, margin/2), -1))
        result.core_output = core
        result.correction_margin = margin
        return result

    def architecture_summary(self):
        result = super().architecture_summary()
        result.update(method=self.method, retained_experts=list(self.expert_names),
            original_five_source_experts_unmodified=True,
            original_redundancy_expert_and_router_unmodified=True,
            method_contract=method_contract(self.method),
            added_module='Detached bounded residual prediction branch',
            expert_metric_columns='Original six; correction margin is separately recorded',
            full_system_passes=1, outer_tta=False, external_pretraining_added=False)
        return result


def method_loss_terms(method, output, labels, args, focal):
    parts = baseline_loss_terms(output.core_output, labels, args, focal)
    core_primary = parts['primary']
    correction_task = task_loss(output.logits, labels, args.task_loss, focal)
    ranking = .01*within_batch_rank_loss(output.logits, labels) if method.endswith('_rank') else core_primary.new_zeros(())
    parts['total'] = parts['total']+correction_task+ranking
    # Preserve the trainer's accounting identity: primary is fused task,
    # auxiliary/interaction belong to core, extra includes core primary.
    parts['primary'] = correction_task
    parts['extra'] = parts['extra']+core_primary+ranking
    return parts
