"""Label-free shift detection (plan phase P5).

Question: given a batch of UNLABELLED images from a new site, can we tell
that it differs from the source sites? Two views of the same batch:

* scores      - the model's 1-D output; compare distributions with a
                two-sample Kolmogorov-Smirnov (KS) test.
* embeddings  - the model's feature vectors; train a small classifier to tell
                source from target. If it cannot (AUC near 0.5) there is no
                detectable shift; if it can (AUC near 1) there is.

A detector firing does NOT prove the sensitivity guarantee broke, and a
silent detector does not prove it held. Measuring how often the two coincide
is part of research question 2 (see experiments/synthetic_shift.py).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tbshift.metrics import auc_roc


class ShiftResult(NamedTuple):
    statistic: float
    p_value: float
    shifted: bool  # True when p_value < significance level


def _as_sample(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if np.isnan(a).any():
        raise ValueError(f"{name} contains NaN")
    return a


def ks_statistic(a, b) -> float:
    """Largest vertical gap between the empirical CDFs of two 1-D samples (0..1).

    The empirical CDF of a sample at x is the fraction of its values <= x.
    The biggest gap can only occur at one of the observed values, so we
    evaluate both CDFs at every observed value and take the maximum difference.
    """
    a = np.sort(_as_sample(a, "a"))
    b = np.sort(_as_sample(b, "b"))
    points = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, points, side="right") / a.size
    cdf_b = np.searchsorted(b, points, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def detect_score_shift(source_scores, target_scores, significance: float = 0.05) -> ShiftResult:
    """Two-sample KS test on model scores.

    The statistic is our own ``ks_statistic``; the p-value comes from SciPy,
    which knows the statistic's null distribution. With many tied scores the
    p-value is conservative (the test fires a little less often than it should).
    """
    if not 0 < significance < 1:
        raise ValueError("significance must be strictly between 0 and 1")
    source = _as_sample(source_scores, "source_scores")
    target = _as_sample(target_scores, "target_scores")
    statistic = ks_statistic(source, target)
    p_value = float(ks_2samp(source, target).pvalue)
    return ShiftResult(statistic=statistic, p_value=p_value, shifted=p_value < significance)


def domain_classifier_auc(
    source_embeddings, target_embeddings, n_splits: int = 5, seed: int = 0
) -> float:
    """Cross-validated AUC of a simple classifier separating source from target.

    Label source images 0 and target images 1, fit a standardised logistic
    regression, and score every image with a model that never saw it
    (out-of-fold predictions). Scoring on the training images instead would
    report a shift that is really just memorisation.

    About 0.5: the two sites look the same in feature space.
    Near 1:    a linear model can tell them apart, so there is a shift.
    Whether the AUC is "significantly" above 0.5 must be judged against a
    null reference (for example source-vs-source splits), not a fixed number.
    """
    source = np.asarray(source_embeddings, dtype=float)
    target = np.asarray(target_embeddings, dtype=float)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("embeddings must be 2-D arrays (images x features)")
    if source.shape[1] != target.shape[1]:
        raise ValueError(
            f"feature size mismatch: {source.shape[1]} vs {target.shape[1]}"
        )
    if min(len(source), len(target)) < n_splits:
        raise ValueError(f"need at least {n_splits} images per site for {n_splits}-fold CV")

    x = np.vstack([source, target])
    y = np.concatenate([np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    p_target = cross_val_predict(model, x, y, cv=folds, method="predict_proba")[:, 1]
    return auc_roc(y, p_target)
