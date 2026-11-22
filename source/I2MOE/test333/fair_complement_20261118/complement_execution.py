"""The sample stream is unchanged; only Train-derived loss weights differ."""
import hashlib
import json
import numpy as np
from cohort_execution import install as install_cohorts


def conditional_weights(probabilities, power, cap):
    weights = np.zeros_like(probabilities)
    for label in (0, 1):
        active = probabilities[label] > 0
        raw = np.zeros(probabilities.shape[1])
        raw[active] = probabilities[label, active]**(-power)
        lower, upper = 0., 100.
        for _ in range(80):
            midpoint = (lower+upper)/2
            if np.sum(probabilities[label]*np.minimum(cap, midpoint*raw)) < 1:
                lower = midpoint
            else:
                upper = midpoint
        weights[label] = np.minimum(cap, (lower+upper)/2*raw)
    np.testing.assert_allclose((probabilities*weights).sum(-1), [1, 1], atol=1e-12, rtol=0)
    return weights


def install(original, seed, method):
    contract = install_cohorts(original, seed, method)
    metadata = contract['train_cohort_metadata']
    power, cap = (1., 64.) if 'power' in method else (.5, 16.)
    probabilities = np.asarray(metadata['probabilities'])
    weights = conditional_weights(probabilities, power, cap)
    metadata.update(weights=weights.tolist(), maximum_weight=float(weights.max()), weight_cap=cap,
        inverse_frequency_power=power, class_expected_weights=(probabilities*weights).sum(-1).tolist())
    contract['cohort_metadata_sha256'] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    return contract
