"""Specification for tbshift.shift (phase P5)."""

import numpy as np
import pytest
from scipy.stats import ks_2samp

from tbshift.shift import detect_score_shift, domain_classifier_auc, ks_statistic


def test_ks_identical_samples_is_zero():
    x = np.random.default_rng(0).normal(size=200)
    assert ks_statistic(x, x) == pytest.approx(0.0)


def test_ks_disjoint_samples_is_one():
    assert ks_statistic([0.1, 0.2, 0.3], [0.7, 0.8, 0.9]) == pytest.approx(1.0)


def test_ks_matches_scipy():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=300), rng.normal(0.4, 1.2, size=250)
    assert ks_statistic(a, b) == pytest.approx(ks_2samp(a, b).statistic)


def test_detector_fires_on_clear_shift():
    rng = np.random.default_rng(2)
    result = detect_score_shift(rng.normal(size=500), rng.normal(1.0, 1.0, size=200))
    assert result.shifted and result.p_value < 0.05


def test_detector_false_alarm_rate_is_controlled():
    # Same distribution on both sides: it should fire about 5% of the time.
    rng = np.random.default_rng(3)
    fired = [
        detect_score_shift(rng.normal(size=300), rng.normal(size=100), significance=0.05).shifted
        for _ in range(400)
    ]
    assert np.mean(fired) < 0.10


def test_domain_classifier_near_chance_without_shift():
    rng = np.random.default_rng(4)
    auc = domain_classifier_auc(rng.normal(size=(300, 16)), rng.normal(size=(300, 16)))
    assert 0.40 < auc < 0.60


def test_domain_classifier_high_with_shift():
    rng = np.random.default_rng(5)
    auc = domain_classifier_auc(rng.normal(size=(300, 16)), rng.normal(0.8, 1.0, size=(300, 16)))
    assert auc > 0.90


# ------------------------------------------------------------- edge cases


def test_ks_is_symmetric():
    rng = np.random.default_rng(6)
    a, b = rng.normal(size=80), rng.normal(0.3, 1.0, size=120)
    assert ks_statistic(a, b) == pytest.approx(ks_statistic(b, a))


def test_ks_matches_scipy_with_ties():
    rng = np.random.default_rng(7)
    a = np.round(rng.normal(size=300), 1)
    b = np.round(rng.normal(0.2, 1.0, size=200), 1)
    assert ks_statistic(a, b) == pytest.approx(ks_2samp(a, b).statistic)


def test_detector_statistic_is_our_ks():
    rng = np.random.default_rng(8)
    a, b = rng.normal(size=100), rng.normal(0.5, 1.0, size=100)
    assert detect_score_shift(a, b).statistic == pytest.approx(ks_statistic(a, b))


def test_small_batches_have_little_power():
    # A real but modest shift is hard to see with 10 images: the detector
    # should miss it most of the time. This is why batch size matters.
    rng = np.random.default_rng(9)
    fired = [
        detect_score_shift(rng.normal(size=500), rng.normal(0.3, 1.0, size=10)).shifted
        for _ in range(300)
    ]
    assert np.mean(fired) < 0.3


@pytest.mark.parametrize("bad", [[], [0.1, float("nan")]])
def test_detector_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        detect_score_shift([0.1, 0.2, 0.3], bad)


def test_domain_classifier_rejects_feature_mismatch():
    with pytest.raises(ValueError):
        domain_classifier_auc(np.zeros((20, 4)), np.zeros((20, 5)))


def test_domain_classifier_is_reproducible():
    rng = np.random.default_rng(10)
    a, b = rng.normal(size=(100, 8)), rng.normal(0.3, 1.0, size=(100, 8))
    assert domain_classifier_auc(a, b, seed=3) == domain_classifier_auc(a, b, seed=3)
