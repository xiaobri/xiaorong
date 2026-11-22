"""Strict probability thresholds, grouped ties, both ends of the exact scan."""
import numpy as np


def measure(tp, fp, pos, neg, threshold):
    tp, fp = int(tp), int(fp)
    fn, tn = int(pos) - tp, int(neg) - fp
    return dict(threshold=float(threshold), tp=tp, fp=fp, fn=fn, tn=tn,
                f1_points=200 * tp / (tp + fp + pos),
                precision_points=100 * tp / (tp + fp) if tp + fp else 0.,
                recall_points=100 * tp / pos)


def scan(labels, probs, thresholds=None):
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape != labels.shape or not np.isfinite(probs).all():
        raise ValueError('Score shape or finite-value check failed')
    positive = np.sort(probs[labels == 1])
    negative = np.sort(probs[labels == 0])
    if len(positive) + len(negative) != len(labels) or not min(len(positive), len(negative)):
        raise ValueError('Both binary labels required')
    if thresholds is None:
        unique = np.unique(probs)
        thresholds = np.r_[unique[::-1], np.nextafter(unique[0], -np.inf)]
    tp = len(positive) - np.searchsorted(positive, thresholds, side='right')
    fp = len(negative) - np.searchsorted(negative, thresholds, side='right')
    index = np.argmax(2 * tp / (tp + fp + len(positive)))
    return measure(tp[index], fp[index], len(positive), len(negative), thresholds[index])
