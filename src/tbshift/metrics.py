"""Evaluation metrics for a binary TB classifier.

Everything here is plain NumPy so each metric can be read and checked by hand.
See ``tbshift/__init__.py`` for the label and threshold convention.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np


class ConfusionCounts(NamedTuple):
    """The four cells of the confusion matrix."""

    tp: int  # TB, flagged          (caught)
    fp: int  # not TB, flagged      (unnecessary follow-up)
    tn: int  # not TB, ruled out    (correctly sent home)
    fn: int  # TB, ruled out        (MISSED - the dangerous error)


class BootstrapResult(NamedTuple):
    """A point estimate with a percentile bootstrap confidence interval."""

    point: float
    lower: float
    upper: float


def _validate(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to 1-D arrays and check they describe a binary problem."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("y_true and y_score must be 1-D")
    if y_true.shape != y_score.shape:
        raise ValueError(f"length mismatch: {y_true.shape[0]} labels vs {y_score.shape[0]} scores")
    if y_true.size == 0:
        raise ValueError("inputs are empty")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("y_true must contain only 0 and 1")
    if np.isnan(y_score).any():
        raise ValueError("y_score contains NaN")
    return y_true.astype(int), y_score


def confusion_counts(y_true, y_score, threshold: float) -> ConfusionCounts:
    """Count TP, FP, TN, FN when flagging every case with ``score >= threshold``."""
    y_true, y_score = _validate(y_true, y_score)
    flagged = y_score >= threshold
    is_tb = y_true == 1
    return ConfusionCounts(
        tp=int(np.sum(flagged & is_tb)),
        fp=int(np.sum(flagged & ~is_tb)),
        tn=int(np.sum(~flagged & ~is_tb)),
        fn=int(np.sum(~flagged & is_tb)),
    )


def sensitivity(y_true, y_score, threshold: float) -> float:
    """Fraction of TB cases that were flagged: TP / (TP + FN).

    Returns NaN if there are no TB cases, because the question has no answer.
    """
    c = confusion_counts(y_true, y_score, threshold)
    n_pos = c.tp + c.fn
    return c.tp / n_pos if n_pos > 0 else float("nan")


def specificity(y_true, y_score, threshold: float) -> float:
    """Fraction of non-TB cases that were ruled out: TN / (TN + FP).

    Returns NaN if there are no non-TB cases.
    """
    c = confusion_counts(y_true, y_score, threshold)
    n_neg = c.tn + c.fp
    return c.tn / n_neg if n_neg > 0 else float("nan")


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Rank values from 1..n; tied values share the average of their ranks.

    Example: [10, 20, 20, 30] -> [1, 2.5, 2.5, 4].
    """
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    last_rank = np.cumsum(counts)  # rank of the last member of each tie group
    first_rank = last_rank - counts + 1
    return ((first_rank + last_rank) / 2.0)[inverse]


def auc_roc(y_true, y_score) -> float:
    """Area under the ROC curve, computed from ranks (Mann-Whitney U).

    Meaning: pick one TB image and one non-TB image at random. AUC is the
    probability that the TB image gets the higher score (ties count as half).
    It does not depend on any threshold.
    """
    y_true, y_score = _validate(y_true, y_score)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs at least one TB and one non-TB case")
    ranks = _average_ranks(y_score)
    rank_sum_pos = ranks[y_true == 1].sum()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Expected Calibration Error for the TB probability, equal-width bins.

    Group predictions into bins by predicted probability. In each bin compare
    the average predicted probability with the actual fraction of TB cases.
    ECE is the average gap, weighted by how many predictions fall in each bin.
    0 means perfectly calibrated.

    Note: this is the "positive-class" form used for binary problems. The
    multi-class form in Guo et al. (2017) bins by the confidence of the
    predicted class instead; the two differ, so always say which one you report.
    """
    y_true, y_prob = _validate(y_true, y_prob)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if (y_prob < 0).any() or (y_prob > 1).any():
        raise ValueError("y_prob must lie in [0, 1] (did you pass logits?)")

    # Bin index 0..n_bins-1. A probability of exactly 1.0 goes in the last bin.
    bin_idx = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)
    n = y_true.size
    ece = 0.0
    for b in range(n_bins):
        in_bin = bin_idx == b
        count = int(in_bin.sum())
        if count == 0:
            continue
        gap = abs(y_true[in_bin].mean() - y_prob[in_bin].mean())
        ece += (count / n) * gap
    return float(ece)


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true,
    y_score,
    n_boot: int = 2000,
    confidence: float = 0.95,
    stratified: bool = True,
    seed: int | None = 0,
) -> BootstrapResult:
    """Percentile bootstrap confidence interval for any metric.

    Idea: we only have one test set, so we imitate "collecting it again" by
    resampling it with replacement many times and recomputing the metric.
    The spread of those values estimates our uncertainty.

    ``stratified=True`` resamples TB and non-TB cases separately so every
    resample keeps the original class counts. This matters with few TB cases:
    an unstratified resample could contain almost none.

    ``metric_fn`` takes ``(y_true, y_score)`` and returns a float. For a metric
    that needs a threshold, wrap it:
    ``lambda y, s: sensitivity(y, s, threshold=0.3)``.
    """
    y_true, y_score = _validate(y_true, y_score)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")

    rng = np.random.default_rng(seed)
    n = y_true.size
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)

    values = np.empty(n_boot)
    for b in range(n_boot):
        if stratified:
            idx = np.concatenate(
                [
                    rng.choice(pos_idx, size=pos_idx.size, replace=True),
                    rng.choice(neg_idx, size=neg_idx.size, replace=True),
                ]
            )
        else:
            idx = rng.integers(0, n, size=n)
        values[b] = metric_fn(y_true[idx], y_score[idx])

    tail = (1.0 - confidence) / 2.0
    lower, upper = np.nanquantile(values, [tail, 1.0 - tail])
    return BootstrapResult(
        point=float(metric_fn(y_true, y_score)), lower=float(lower), upper=float(upper)
    )


def violation_rate(achieved, target: float) -> float:
    """Fraction of repeats whose achieved value fell below the target.

    Used for the project's headline number: across many random repeats, how
    often was achieved sensitivity below the promised sensitivity?
    NaN entries (repeats where the metric was undefined) are ignored.
    """
    achieved = np.asarray(achieved, dtype=float)
    valid = achieved[~np.isnan(achieved)]
    if valid.size == 0:
        raise ValueError("no valid values to evaluate")
    return float(np.mean(valid < target))
