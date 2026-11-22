"""Adapt pre-projection source information while retaining original predictions.

Each source CNN is frozen. New low-rank projection or moment adapters expose
information before its 128-D bottleneck. Regional log-odds corrections are
bounded, and global corrections are centered using training intact views only.
"""
import bootstrap
import torch
from torch import nn
from torch.nn import functional as F

from i2moe_eeg.f1_variants import SixExpertF1

METHODS = ('projection_adapter', 'moment_regional', 'moment_global', 'moment_joint')
FEATURE_BOUND = .25
REGIONAL_MARGIN_BOUND = .5


class PerViewTrainMeanCenter(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.register_buffer('running_mean', torch.zeros(width))
        self.register_buffer('batches', torch.zeros((), dtype=torch.long))

    def forward(self, values: torch.Tensor):
        if self.training:
            # Separate means prevent intact-view statistics from disclosing
            # an excluded region to that region's corrupted training view.
            center = values.mean(0)
            reference = center if values.ndim == 2 else values[:, 0].mean(0)
            with torch.no_grad():
                if self.batches.item() == 0:
                    self.running_mean.copy_(reference.detach())
                else:
                    self.running_mean.lerp_(reference.detach(), .1)
                self.batches.add_(1)
        else:
            center = self.running_mean
        return values-center


def source_moments(feature_map: torch.Tensor) -> torch.Tensor:
    return torch.stack((F.adaptive_avg_pool2d(feature_map, (1, 1)).flatten(1),
                        feature_map.amax(dim=(-2, -1)),
                        feature_map.flatten(2).std(-1, unbiased=False)), 1)


class ProjectionAdapter(nn.Module):
    def __init__(self, source_width: int, width: int, dropout: float, projection):
        super().__init__()
        object.__setattr__(self, 'source_projection', projection)
        self.down = nn.Linear(source_width, 16, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(16, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, moments: torch.Tensor):
        mean = moments[:, 0].contiguous()
        original = self.source_projection[0](mean)
        delta = self.up(self.dropout(self.down(mean)))
        return self.source_projection[1](original+delta)-self.source_projection[1](original)


class MomentAdapter(nn.Module):
    def __init__(self, source_width: int, width: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(source_width*3),
            nn.Linear(source_width*3, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, width))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, moments: torch.Tensor):
        return self.network(moments.flatten(1))


class MomentGlobal(nn.Module):
    def __init__(self, source_width: int, width: int, dropout: float):
        super().__init__()
        self.encoders = nn.ModuleList([nn.Sequential(nn.LayerNorm(source_width*3),
            nn.Linear(source_width*3, width), nn.GELU()) for _ in range(5)])
        self.attention = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.projection = nn.Sequential(nn.LayerNorm(width*3), nn.Linear(width*3, width),
            nn.GELU(), nn.Dropout(dropout), PerViewTrainMeanCenter(width), nn.Linear(width, width, bias=False))
        nn.init.zeros_(self.projection[-1].weight)

    def forward(self, moments: torch.Tensor, originals: torch.Tensor):
        count = 1 if originals.ndim == 2 else originals.shape[1]
        if count not in (1, 6):
            raise ValueError('Expected original intact/five-corruption views')
        encoded = torch.stack([encoder(moments[:, index].flatten(1))
                               for index, encoder in enumerate(self.encoders)], 1)
        scores = self.attention(encoded).squeeze(-1)
        views = []
        for index in range(count):
            current = scores
            maximum_input = encoded
            if index:
                mask = torch.arange(5, device=encoded.device) == index-1
                current = scores.masked_fill(mask[None], -torch.inf)
                maximum_input = encoded.masked_fill(mask[None, :, None], -torch.inf)
            pooled = (encoded*current.softmax(1)[..., None]).sum(1)
            original = originals if count == 1 else originals[:, index]
            views.append(torch.cat((original, pooled, maximum_input.amax(1)), -1))
        features = views[0] if count == 1 else torch.stack(views, 1)
        delta = self.projection(features)
        return FEATURE_BOUND * torch.tanh(delta/FEATURE_BOUND)


class MomentArchitectureMoE(SixExpertF1):
    def __init__(self, method: str, **kwargs):
        super().__init__(**kwargs)
        if method not in METHODS:
            raise ValueError(method)
        self.method, self.moments = method, {}
        sources = [getattr(expert, 'module', expert) for expert in self.frozen_experts.values()]
        source_width = sources[0].encoder.projection[0].in_features
        self.source_width = source_width
        width, dropout = self.hidden_dim, kwargs.get('dropout', .3)
        with torch.random.fork_rng(devices=[]):
            self.regional_adapter = None
            if method != 'moment_global':
                self.regional_adapter = nn.ModuleList([
                    ProjectionAdapter(source_width, width, dropout, source.encoder.projection)
                    if method == 'projection_adapter' else MomentAdapter(source_width, width, dropout)
                    for source in sources])
            self.moment_global = MomentGlobal(source_width, width, dropout) if method in ('moment_global', 'moment_joint') else None
        self._moment_hooks = []
        for index, source in enumerate(sources):
            def capture(module, inputs, output, index=index):
                self.moments[index] = source_moments(output).detach()
            self._moment_hooks.append(source.encoder.features.register_forward_hook(capture))
        if self.moment_global is not None:
            self._global_hook = self.redundancy_expert.register_forward_hook(self._enhance_global)

    def _encode_frozen(self, eeg):
        self.moments.clear()
        features, logits = super()._encode_frozen(eeg)
        if len(self.moments) != 5:
            raise RuntimeError('All five original source moments are required')
        if self.regional_adapter is None:
            return features, logits
        devices = [eeg.device.index] if eeg.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            deltas = torch.stack([adapter(self.moments[index])
                                  for index, adapter in enumerate(self.regional_adapter)], 1)
        adapted = features + FEATURE_BOUND * torch.tanh(deltas/FEATURE_BOUND)
        corrections = []
        for index, expert in enumerate(self.frozen_experts.values()):
            classifier = getattr(expert, 'module', expert).classifier
            original = classifier(features[:, index])
            changed = classifier(adapted[:, index])
            margin = (changed[:, 1]-changed[:, 0])-(original[:, 1]-original[:, 0])
            margin = REGIONAL_MARGIN_BOUND * torch.tanh(margin/REGIONAL_MARGIN_BOUND)
            corrections.append(torch.stack((-margin/2, margin/2), -1))
        return adapted, logits+torch.stack(corrections, 1)

    def _enhance_global(self, module, inputs, output):
        features, logits = output
        moments = torch.stack([self.moments[index] for index in range(5)], 1)
        devices = [moments.device.index] if moments.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            delta = self.moment_global(moments, features)
        return features+delta, logits+F.linear(delta, module.head[-1].weight, bias=None)

    def architecture_summary(self):
        result = super().architecture_summary()
        result.update(method=self.method, retained_experts=list(self.expert_names),
            original_source_weights_and_classifiers_frozen=True,
            pre_projection_descriptors=['mean', 'maximum', 'population_std'],
            descriptor_width_per_source=self.source_width,
            projection_adapter_rank=16 if self.method == 'projection_adapter' else None,
            regional_adapter=self.regional_adapter is not None, global_moment_adapter=self.moment_global is not None,
            feature_component_correction_bound=FEATURE_BOUND,
            regional_margin_correction_bound=REGIONAL_MARGIN_BOUND,
            global_centering='Separate batch mean per training view; only intact running mean saved for evaluation',
            corrupted_global_views_exclude_corresponding_source_moments=True,
            original_loss_unchanged=True, original_redundancy_transformer_preserved=True,
            external_pretraining_added=False, full_system_passes=1, outer_tta=False)
        return result
