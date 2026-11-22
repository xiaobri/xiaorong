"""Matched interventions on the first detached regional correction in the full model."""
from dataclasses import dataclass

import torch

import complement_models as parent


@dataclass(frozen=True)
class Spec:
    weighting: str
    ranking: bool
    features: str = "all"


ARMS = {
    "plain": Spec("unit", False),
    "rank": Spec("unit", True),
    "weight": Spec("adaptive", False),
    "full": Spec("adaptive", True),
    "static_rank": Spec("static", True),
    "no_moments": Spec("adaptive", True, "router"),
    "scores_only": Spec("adaptive", True, "scores"),
    "mean_only": Spec("adaptive", True, "mean"),
}


def configure(model, arm):
    feature_set = ARMS[arm].features
    if feature_set == "all":
        return model
    def input_mask(module, inputs):
        values, = inputs
        if feature_set == "mean":
            moments = values.reshape(len(values), 3, -1).clone()
            moments[:, 1:] = 0
            return (moments.reshape_as(values),)
        return (torch.zeros_like(values),)
    for encoder in model.correction.encoders:
        encoder.register_forward_pre_hook(input_mask)
    if feature_set == "scores":
        width = 6 * model.hidden_dim
        def remove_router(module, inputs):
            values = inputs[0].clone()
            values[:, 7:7+width] = 0
            return (values,)
        model.correction.predictor.register_forward_pre_hook(remove_router)
    return model


def loss_terms(arm, out, labels, args, focal):
    spec = ARMS[arm]
    if spec.weighting == "adaptive" and spec.ranking:
        return parent.method_loss_terms("risk_power", out, labels, args, focal)
    parts = parent.baseline_loss_terms(out.core_output, labels, args, focal)
    original = parts["primary"]
    per_row = parent.focal_per_row(out.logits, labels, focal)
    weighting = {"unit": "rank_control", "static": "cohort_balanced", "adaptive": "cohort_dro"}
    weights = parent.group_weights(weighting[spec.weighting], out, labels, per_row)
    corrected = (per_row * weights).mean()
    if spec.ranking:
        corrected = corrected + .03 * parent.hard_rank(out.logits, labels, weights, 16)
    fused = parent.task_loss(out.logits, labels, args.task_loss, focal)
    parts["total"] = parts["total"] + corrected
    parts["primary"] = fused
    parts["extra"] = parts["extra"] + original + corrected - fused
    return parts
