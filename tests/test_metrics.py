"""Tests for tbshift.metrics.

Each test checks a metric against a value worked out by hand or against a
trusted reference (scikit-learn), so a wrong formula cannot pass silently.
"""

import math

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from tbshift.metrics import (
    auc_roc,
    bootstrap_ci,
    confusion_counts,
    expected_calibration_error,
    sensitivity,
    specificity,
    violation_rate,
)

# A tiny dataset small enough to count on your fingers.
#   index : 0    1    2    3    4    5    6    7
Y_TRUE = [1, 1, 1, 1, 0, 0, 0, 0]
Y_SCORE = [0.9, 0.8, 0.6, 0.2, 0.7, 0.4, 0.3, 0.1]
# At threshold 0.5: flagged = scores 0.9, 0.8, 0.6 (TB) and 0.7 (not TB).
#   TP = 3, FN = 1 (the 0.2), FP = 1 (the 0.7), TN = 3.


# ---------------------------------------------------------------- confusion


def test_confusion_counts_hand_example():
    c = confusion_counts(Y_TRUE, Y_SCORE, threshold=0.5)
    assert (c.tp, c.fp, c.tn, c.fn) == (3, 1, 3, 1)


def test_score_equal_to_threshold_is_flagged():
    # Convention: score >= threshold means flagged.
    c = confusion_counts([1, 0], [0.5, 0.5], threshold=0.5)
    assert (c.tp, c.fp, c.tn, c.fn) == (1, 1, 0, 0)


def test_sensitivity_and_specificity_hand_example():
    assert sensitivity(Y_TRUE, Y_SCORE, 0.5) == pytest.approx(3 / 4)
    assert specificity(Y_TRUE, Y_SCORE, 0.5) == pytest.approx(3 / 4)


def test_threshold_extremes():
    # Flag everyone: no TB missed, but no one ruled out.
    assert sensitivity(Y_TRUE, Y_SCORE, -math.inf) == 1.0
    assert specificity(Y_TRUE, Y_SCORE, -math.inf) == 0.0
    # Flag no one: everything ruled out.
    assert sensitivity(Y_TRUE, Y_SCORE, math.inf) == 0.0
    assert specificity(Y_TRUE, Y_SCORE, math.inf) == 1.0


def test_lower_threshold_never_reduces_sensitivity():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = rng.random(300)
    thresholds = np.linspace(1, 0, 21)
    sens = np.array([sensitivity(y, s, t) for t in thresholds])
    assert (np.diff(sens) >= 0).all()


def test_undefined_metrics_return_nan():
    assert math.isnan(sensitivity([0, 0], [0.1, 0.9], 0.5))  # no TB cases
    assert math.isnan(specificity([1, 1], [0.1, 0.9], 0.5))  # no non-TB cases


@pytest.mark.parametrize(
    "y, s",
    [
        ([1, 0, 1], [0.1, 0.2]),  # different lengths
        ([], []),  # empty
        ([0, 2], [0.1, 0.2]),  # label that is not 0/1
        ([0, 1], [0.1, float("nan")]),  # NaN score
    ],
)
def test_bad_inputs_raise(y, s):
    with pytest.raises(ValueError):
        confusion_counts(y, s, 0.5)


# ---------------------------------------------------------------------- AUC


def test_auc_hand_example():
    # 16 (TB, non-TB) pairs. TB wins unless: 0.6 < 0.7, and 0.2 < 0.7, 0.4, 0.3.
    # 4 losses -> 12 wins -> 12/16.
    assert auc_roc(Y_TRUE, Y_SCORE) == pytest.approx(12 / 16)


def test_auc_perfect_reversed_and_constant():
    assert auc_roc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auc_roc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auc_roc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5  # all ties


@pytest.mark.parametrize("seed", range(5))
def test_auc_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 500)
    s = rng.random(500) + 0.3 * y
    assert auc_roc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_auc_matches_sklearn_with_many_ties():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.random(500) + 0.3 * y, 1)  # only ~14 distinct values
    assert auc_roc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_auc_unchanged_by_monotonic_transform():
    # AUC depends only on the ORDER of scores. This is why temperature
    # scaling (which keeps the order) cannot change AUC.
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 400)
    logits = rng.normal(size=400) + y
    probs = 1 / (1 + np.exp(-logits / 3.0))
    assert auc_roc(y, logits) == pytest.approx(auc_roc(y, probs))


def test_auc_needs_both_classes():
    with pytest.raises(ValueError):
        auc_roc([1, 1, 1], [0.1, 0.5, 0.9])


# ---------------------------------------------------------------------- ECE


def test_ece_hand_example():
    # Two bins of width 0.5.
    # Bin [0, 0.5):  probs 0.2, 0.2 -> mean 0.2; labels 0, 1 -> mean 0.5; gap 0.3
    # Bin [0.5, 1]:  probs 0.8, 0.8 -> mean 0.8; labels 1, 1 -> mean 1.0; gap 0.2
    # ECE = 0.5 * 0.3 + 0.5 * 0.2 = 0.25
    ece = expected_calibration_error([0, 1, 1, 1], [0.2, 0.2, 0.8, 0.8], n_bins=2)
    assert ece == pytest.approx(0.25)


def test_ece_near_zero_when_calibrated():
    # Labels are drawn so that P(TB) really equals the stated probability.
    rng = np.random.default_rng(0)
    p = rng.random(200_000)
    y = (rng.random(200_000) < p).astype(int)
    assert expected_calibration_error(y, p, n_bins=10) < 0.01


def test_ece_large_when_overconfident():
    # The model says 0.99 for everything but is right only 70% of the time.
    rng = np.random.default_rng(0)
    y = (rng.random(50_000) < 0.7).astype(int)
    p = np.full(50_000, 0.99)
    assert expected_calibration_error(y, p) == pytest.approx(0.29, abs=0.01)


def test_ece_handles_probability_exactly_one():
    assert expected_calibration_error([1, 1], [1.0, 1.0]) == 0.0


def test_ece_rejects_logits():
    with pytest.raises(ValueError):
        expected_calibration_error([0, 1], [-2.0, 3.5])


# ---------------------------------------------------------------- bootstrap


def _sens_at_half(y, s):
    return sensitivity(y, s, threshold=0.5)


def _make_data(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    s = np.clip(rng.normal(0.35 + 0.3 * y, 0.2), 0, 1)
    return y, s


def test_bootstrap_interval_contains_point():
    y, s = _make_data(400, seed=0)
    r = bootstrap_ci(_sens_at_half, y, s, n_boot=500)
    assert r.lower <= r.point <= r.upper
    assert r.point == pytest.approx(_sens_at_half(y, s))


def test_bootstrap_is_reproducible_with_seed():
    y, s = _make_data(200, seed=1)
    a = bootstrap_ci(_sens_at_half, y, s, n_boot=300, seed=42)
    b = bootstrap_ci(_sens_at_half, y, s, n_boot=300, seed=42)
    assert a == b


def test_bootstrap_interval_narrows_with_more_data():
    y_small, s_small = _make_data(60, seed=2)
    y_big, s_big = _make_data(6000, seed=2)
    small = bootstrap_ci(_sens_at_half, y_small, s_small, n_boot=500)
    big = bootstrap_ci(_sens_at_half, y_big, s_big, n_boot=500)
    assert (big.upper - big.lower) < (small.upper - small.lower)


def test_bootstrap_stratified_keeps_class_counts():
    y, s = _make_data(100, seed=3)
    n_pos = int(y.sum())
    seen = []

    def count_pos(y_b, s_b):
        seen.append(int(y_b.sum()))
        return 0.0

    bootstrap_ci(count_pos, y, s, n_boot=50, stratified=True)
    assert set(seen) == {n_pos}


def test_bootstrap_wider_confidence_gives_wider_interval():
    y, s = _make_data(150, seed=4)
    r90 = bootstrap_ci(_sens_at_half, y, s, n_boot=1000, confidence=0.90)
    r99 = bootstrap_ci(_sens_at_half, y, s, n_boot=1000, confidence=0.99)
    assert (r99.upper - r99.lower) >= (r90.upper - r90.lower)


# ----------------------------------------------------------- violation rate


def test_violation_rate_hand_example():
    # Target 0.90. Below target: 0.85 and 0.89 -> 2 of 5.
    assert violation_rate([0.95, 0.90, 0.85, 0.89, 1.0], target=0.90) == pytest.approx(0.4)


def test_violation_rate_exactly_on_target_is_not_a_violation():
    assert violation_rate([0.90, 0.90], target=0.90) == 0.0


def test_violation_rate_ignores_nan():
    assert violation_rate([0.8, float("nan"), 0.95], target=0.9) == pytest.approx(0.5)


def test_violation_rate_all_nan_raises():
    with pytest.raises(ValueError):
        violation_rate([float("nan")], target=0.9)
