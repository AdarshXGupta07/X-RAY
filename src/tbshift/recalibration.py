"""Restoring the guarantee at a new site (plan phase P6).

Two routes:

Label-free (quantile matching, in the spirit of unsupervised prediction alignment)
    Map each target score to the source score at the same quantile, so the
    target score distribution lines up with the source one, then reuse the
    source threshold. Needs no labels, but silently assumes the MIX of TB and
    non-TB cases is the same at both sites. If TB is more common at the new
    site, the target scores are higher for an honest reason, matching pulls
    them down, and TB cases get ruled out.

Few-label re-certification
    Spend a small number of labelled target cases and recompute the threshold
    from the target TB-positives among them. The scarce resource is the number
    of POSITIVES drawn, not the number of labels.
"""

from __future__ import annotations

import math

import numpy as np

from tbshift.conformal import certified_threshold, conformal_threshold


def _as_sample(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if np.isnan(a).any():
        raise ValueError(f"{name} contains NaN")
    return a


def quantile_match(target_scores, source_scores, values=None) -> np.ndarray:
    """Map target scores onto the source score distribution, preserving their order.

    Step 1: find where each value sits within the target sample, as a
            fraction u in (0, 1). We use the "mid-rank" position, so tied
            values get the same u and the smallest of n values gets 0.5 / n.
    Step 2: return the source score at that same fraction (np.quantile).

    ``values`` lets you map new scores with a mapping learned from
    ``target_scores``; by default the target scores themselves are mapped.
    """
    unsorted_target = _as_sample(target_scores, "target_scores")
    target = np.sort(unsorted_target)
    source = _as_sample(source_scores, "source_scores")
    v = unsorted_target if values is None else _as_sample(values, "values")

    n = target.size
    below = np.searchsorted(target, v, side="left")
    at_or_below = np.searchsorted(target, v, side="right")
    u = (below + at_or_below) / (2.0 * n)  # mid-rank position, in [0, 1]
    return np.quantile(source, u)


def label_free_threshold(source_threshold: float, source_scores, target_scores) -> float:
    """The target-score threshold equivalent to quantile matching + source threshold.

    Instead of moving every target score, move the threshold once: if the
    source threshold rules out a fraction q of (unlabelled) source images,
    rule out the same fraction q of the target batch.
    """
    if source_threshold == -math.inf:
        return -math.inf
    source = _as_sample(source_scores, "source_scores")
    target = _as_sample(target_scores, "target_scores")
    q = float(np.mean(source < source_threshold))
    return float(np.quantile(target, q))


def few_label_threshold(
    target_scores, target_labels, alpha: float, delta: float | None = None
) -> float:
    """Threshold recomputed from labelled target cases.

    ``delta=None`` gives the marginal conformal threshold; a float gives the
    certified one. Only the TB-positive cases are used. Returns ``-inf`` when
    the labelled sample has too few TB-positives.
    """
    scores = _as_sample(target_scores, "target_scores")
    labels = np.asarray(target_labels)
    if labels.shape != scores.shape:
        raise ValueError("target_scores and target_labels must have the same length")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("target_labels must contain only 0 and 1")

    positives = scores[labels == 1]
    if delta is None:
        return conformal_threshold(positives, alpha)
    return certified_threshold(positives, alpha, delta)
