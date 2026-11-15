"""Specification for tbshift.calibration (phase P3). Expected to fail until implemented."""

import numpy as np
import pytest

from tbshift.calibration import apply_temperature, fit_temperature
from tbshift.metrics import auc_roc, expected_calibration_error

pytestmark = pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="phase P3 not written")


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
