"""Specification for tbshift.conformal (phase P3).

The simulations use TB-positive scores drawn from a standard normal, so the
TRUE sensitivity of any threshold tau is known exactly: 1 - Phi(tau).
"""

import math

import numpy as np
import pytest
from scipy.stats import norm

from tbshift.conformal import (
    TRIVIAL_THRESHOLD,
    certified_rank,
    certified_threshold,
    conformal_threshold,
    marginal_rank,
    min_positives_certified,
    min_positives_marginal,
)
from tbshift.metrics import violation_rate

ALPHA = 0.10  # target sensitivity 90%
DELTA = 0.10  # allowed chance that a calibration set misses the target
N_POS = 50
REPEATS = 2000


def _true_sensitivities(threshold_fn):
    """Calibrate on fresh positives many times; return the exact sensitivity each time."""
    rng = np.random.default_rng(0)
    return np.array(
        [1 - norm.cdf(threshold_fn(rng.normal(size=N_POS))) for _ in range(REPEATS)]
    )


def test_marginal_guarantee_holds_on_average():
    sens = _true_sensitivities(lambda s: conformal_threshold(s, ALPHA))
    assert sens.mean() >= (1 - ALPHA) - 0.01


def test_marginal_threshold_is_not_wastefully_conservative():
    sens = _true_sensitivities(lambda s: conformal_threshold(s, ALPHA))
    assert sens.mean() <= (1 - ALPHA) + 0.03


def test_marginal_threshold_violates_often_even_without_shift():
    # The key finding behind the design: an average guarantee still leaves a
    # large share of individual calibration sets below target.
    sens = _true_sensitivities(lambda s: conformal_threshold(s, ALPHA))
    assert violation_rate(sens, 1 - ALPHA) > 0.25


def test_certified_threshold_controls_violation_rate():
    sens = _true_sensitivities(lambda s: certified_threshold(s, ALPHA, DELTA))
    assert violation_rate(sens, 1 - ALPHA) <= DELTA + 0.02


def test_certified_is_at_least_as_cautious_as_marginal():
    scores = np.random.default_rng(1).normal(size=200)
    assert certified_threshold(scores, ALPHA, DELTA) <= conformal_threshold(scores, ALPHA)


def test_threshold_is_one_of_the_calibration_scores():
    scores = np.random.default_rng(2).normal(size=100)
    assert conformal_threshold(scores, ALPHA) in scores


def test_stricter_alpha_gives_lower_threshold():
    scores = np.random.default_rng(3).normal(size=500)
    assert conformal_threshold(scores, 0.05) <= conformal_threshold(scores, 0.20)


def test_monotonic_transform_commutes_with_threshold():
    # Thresholding depends only on score order, so calibrating on logits or on
    # sigmoid(logits) must rule out exactly the same images.
    logits = np.random.default_rng(4).normal(size=100)
    probs = 1 / (1 + np.exp(-logits))
    tau_logit = conformal_threshold(logits, ALPHA)
    assert conformal_threshold(probs, ALPHA) == pytest.approx(1 / (1 + math.exp(-tau_logit)))


def test_too_few_positives_gives_trivial_threshold():
    few = np.random.default_rng(5).normal(size=5)
    assert conformal_threshold(few, ALPHA) == -math.inf
    assert certified_threshold(few, ALPHA, DELTA) == -math.inf


@pytest.mark.parametrize("alpha, expected", [(0.20, 4), (0.10, 9), (0.05, 19)])
def test_label_floor_marginal(alpha, expected):
    assert min_positives_marginal(alpha) == expected


@pytest.mark.parametrize("alpha, delta, expected", [(0.10, 0.10, 22), (0.10, 0.05, 29)])
def test_label_floor_certified(alpha, delta, expected):
    assert min_positives_certified(alpha, delta) == expected


def test_floors_agree_with_threshold_functions():
    rng = np.random.default_rng(6)
    n = min_positives_marginal(ALPHA)
    assert conformal_threshold(rng.normal(size=n), ALPHA) > -math.inf
    assert conformal_threshold(rng.normal(size=n - 1), ALPHA) == -math.inf
    m = min_positives_certified(ALPHA, DELTA)
    assert certified_threshold(rng.normal(size=m), ALPHA, DELTA) > -math.inf
    assert certified_threshold(rng.normal(size=m - 1), ALPHA, DELTA) == -math.inf


# ------------------------------------------------------------- edge cases


def test_marginal_rank_hand_values():
    # n = 50, alpha = 0.10: floor(0.1 * 51) = 5 -> the 5th lowest TB score.
    assert marginal_rank(50, 0.10) == 5
    assert marginal_rank(8, 0.10) == 0  # floor(0.9) = 0 -> trivial
    assert marginal_rank(9, 0.10) == 1


def test_marginal_rank_survives_float_rounding():
    # 0.29 * 100 is 28.999999999999996 in floating point; the answer is 29.
    assert marginal_rank(99, 0.29) == 29


def test_certified_rank_matches_binomial_definition():
    from scipy.stats import binom

    n, k = 200, certified_rank(200, ALPHA, DELTA)
    assert binom.cdf(k - 1, n, ALPHA) <= DELTA < binom.cdf(k, n, ALPHA)


def test_trivial_threshold_flags_everyone():
    assert TRIVIAL_THRESHOLD == -math.inf
    assert conformal_threshold([], ALPHA) == -math.inf
    assert certified_threshold([], ALPHA, DELTA) == -math.inf


def test_input_order_does_not_matter():
    scores = np.random.default_rng(7).normal(size=60)
    assert conformal_threshold(scores, ALPHA) == conformal_threshold(scores[::-1], ALPHA)


def test_ties_are_handled():
    scores = np.array([0.5] * 30)
    assert conformal_threshold(scores, ALPHA) == 0.5  # flags all 30 (score >= 0.5)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_bad_alpha_raises(alpha):
    with pytest.raises(ValueError):
        conformal_threshold([0.1, 0.2], alpha)


def test_nan_scores_raise():
    with pytest.raises(ValueError):
        conformal_threshold([0.1, float("nan")], ALPHA)
