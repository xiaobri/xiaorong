"""Train-only cohort IDs in an otherwise unused feature-packet field."""
import hashlib
import json

import bootstrap
import numpy as np
from cohort_models import MAX_GROUPS
from moment_cached_execution import install as install_moments


def make_metadata(frame):
    train = frame.loc[frame['mode'] == 'Train', ['file_path', 'target']]
    cohorts = train['file_path'].str.rsplit('/', n=3).str[-3].to_numpy()
    names = sorted(set(cohorts))
    if len(names) > MAX_GROUPS:
        raise ValueError('Too many Train cohorts')
    mapping = {name: index for index, name in enumerate(names)}
    groups = np.asarray([mapping[name] for name in cohorts], dtype=np.int64)
    labels = train['target'].to_numpy(dtype=np.int64)
    counts = np.zeros((2, MAX_GROUPS), dtype=np.int64)
    np.add.at(counts, (labels, groups), 1)
    probabilities = counts/counts.sum(-1, keepdims=True)
    weights = np.zeros_like(probabilities)
    for label in (0, 1):
        present = counts[label] > 0
        raw = np.zeros(MAX_GROUPS)
        raw[present] = 1/np.sqrt(probabilities[label, present])
        lower, upper = 0., 100.
        for _ in range(80):
            middle = (lower+upper)/2
            average = np.sum(probabilities[label]*np.minimum(4., middle*raw))
            if average < 1:
                lower = middle
            else:
                upper = middle
        weights[label] = np.minimum(4., (lower+upper)/2*raw)
    np.testing.assert_allclose((probabilities*weights).sum(-1), 1, rtol=0, atol=1e-12)
    return dict(names=names, counts=counts.tolist(), probabilities=probabilities.tolist(),
                weights=weights.tolist(), groups_sha256=hashlib.sha256(groups.tobytes()).hexdigest(),
                source='Train rows only', maximum_weight=float(weights.max()),
                class_expected_weights=(probabilities*weights).sum(-1).tolist()), groups


def install(original, seed, method):
    contract = install_moments(original, seed, method)
    frame, _ = original.load_manifest(original.DEFAULT_CSV_PATH)
    metadata, groups = make_metadata(frame)
    source = original.EEG19NpyDataset

    class CohortDataset(source):
        def __getitem__(self, index):
            packet, label = super().__getitem__(index)
            packet[0, 3, 130] = int(groups[index]) if self.mode == 'Train' else -1
            return packet, label

    original.EEG19NpyDataset = CohortDataset
    contract['train_cohort_metadata'] = metadata
    contract['cohort_metadata_sha256'] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    contract['cohort_id_not_used_by_inference_modules'] = True
    return contract
