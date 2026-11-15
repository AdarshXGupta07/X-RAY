"""Specification for tbshift.calibration (phase P3)."""

import numpy as np
import pytest

from tbshift.calibration import T_MAX, apply_temperature, binary_nll, fit_temperature
from tbshift.metrics import auc_roc, expected_calibration_error


def _overconfident_data(n=20_000, true_temperature=2.5, seed=0):
    """Logits that are `true_temperature` times too large to be calibrated."""
    rng = np.random.default_rng(seed)
    honest_logits = rng.normal(0, 1.5, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-honest_logits))).astype(int)
    return honest_logits * true_temperature, y


def test_temperature_one_is_plain_sigmoid():
    z = np.array([-2.0, 0.0, 3.0])
    assert apply_temperature(z, 1.0) == pytest.approx(1 / (1 + np.exp(-z)))


def test_probabilities_stay_in_unit_interval():
    p = apply_temperature(np.array([-50.0, 0.0, 50.0]), 0.5)
    assert ((p >= 0) & (p <= 1)).all()


def test_fit_recovers_known_temperature():
    logits, y = _overconfident_data(true_temperature=2.5)
    assert fit_temperature(logits, y) == pytest.approx(2.5, rel=0.1)


def test_scaling_reduces_ece():
    logits, y = _overconfident_data()
    before = expected_calibration_error(y, apply_temperature(logits, 1.0))
    after = expected_calibration_error(y, apply_temperature(logits, fit_temperature(logits, y)))
    assert after < before


def test_scaling_does_not_change_auc():
    # Dividing by T keeps the order of scores, so ranking metrics cannot move.
    logits, y = _overconfident_data(n=2000)
    assert auc_roc(y, apply_temperature(logits, 3.0)) == pytest.approx(auc_roc(y, logits))


# ------------------------------------------------------------- edge cases


def test_nll_hand_value():
    # A logit of 0 means p = 0.5 for every image, so NLL = log 2 whatever the labels.
    assert binary_nll([0.0, 0.0], [0, 1]) == pytest.approx(np.log(2))


def test_nll_is_stable_for_huge_logits():
    # A naive log(sigmoid(x)) gives -inf here; logaddexp does not.
    assert np.isfinite(binary_nll([1000.0, -1000.0], [0, 1]))


def test_already_calibrated_logits_give_t_near_one():
    logits, y = _overconfident_data(true_temperature=1.0)
    assert fit_temperature(logits, y) == pytest.approx(1.0, abs=0.05)


def test_fitted_t_is_never_worse_than_t_one():
    logits, y = _overconfident_data(n=3000, seed=3)
    t = fit_temperature(logits, y)
    assert binary_nll(logits, y, t) <= binary_nll(logits, y, 1.0) + 1e-12


def test_useless_logits_hit_the_upper_bound():
    # Logits with no signal: the best probability is a constant, i.e. T -> infinity.
    rng = np.random.default_rng(4)
    t = fit_temperature(rng.normal(size=2000), rng.integers(0, 2, 2000))
    assert t > T_MAX / 2


@pytest.mark.parametrize("bad_t", [0.0, -1.0, float("nan"), float("inf")])
def test_bad_temperature_raises(bad_t):
    with pytest.raises(ValueError):
        apply_temperature([0.1, 0.2], bad_t)


def test_fit_needs_both_classes():
    with pytest.raises(ValueError):
        fit_temperature([0.1, 0.5, 2.0], [1, 1, 1])
