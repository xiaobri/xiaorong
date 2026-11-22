"""Compare EEG-conditioned and position-only inputs to the existing temporal scorers."""
import hashlib
import math

import torch

ARMS = ("dynamic", "fixed")


def scorer_template(segments: int, width: int) -> torch.Tensor:
    """Eight distinct zero-mean, unit-RMS cosine codes, with no EEG or labels."""
    position = torch.arange(1, segments + 1, dtype=torch.float64)[:, None]
    channel = torch.arange(width, dtype=torch.float64) + 0.5
    return (math.sqrt(2) * torch.cos(math.pi * position * channel / width)).float()


def position_only(scorer, inputs):
    (values,) = inputs
    if values.shape[1:] != scorer.position_input.shape:
        raise ValueError("Unexpected temporal scorer input dimensions")
    return (scorer.position_input.expand_as(values),)


def configure(model, arm: str):
    if arm not in ARMS:
        raise ValueError(arm)
    if arm == "fixed":
        for scorer in model.correction.allocation.scorers:
            width, = scorer[0].normalized_shape
            scorer.register_buffer("position_input", scorer_template(8, width), persistent=False)
            scorer.register_forward_pre_hook(position_only)
    return model


def parameter_digest(model) -> str:
    digest = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        if param.requires_grad:
            digest.update(name.encode())
            digest.update(param.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
