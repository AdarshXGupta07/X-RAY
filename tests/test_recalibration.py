"""Specification for tbshift.recalibration (phase P6)."""

import math

import numpy as np
import pytest

from tbshift.recalibration import few_label_threshold, label_free_threshold, quantile_match
from tbshift.shift import ks_statistic


def test_quantile_match_preserves_order():
    rng = np.random.default_rng(0)
    target, source = rng.normal(2, 3, 200), rng.normal(size=500)
    mapped = quantile_match(target, source)
    assert (np.argsort(mapped, kind="stable") == np.argsort(target, kind="stable")).all()


def test_quantile_match_aligns_distributions():
    rng = np.random.default_rng(1)
    target, source = rng.normal(2, 3, 2000), rng.normal(size=2000)
    assert ks_statistic(quantile_match(target, source), source) < 0.05


def test_quantile_match_undoes_a_pure_score_shift():
    # The new site's scores are the source scores pushed down by a constant.
    # After matching, the source threshold should flag the same fraction again.
    rng = np.random.default_rng(2)
    source = rng.normal(size=5000)
    target = rng.normal(size=5000) - 1.0
    tau = np.quantile(source, 0.3)
    assert np.mean(quantile_match(target, source) >= tau) == pytest.approx(0.7, abs=0.03)


def test_few_label_uses_only_tb_positives():
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.2] * 10)
    labels = np.array([1, 1, 1, 0, 0] * 10)
    tau = few_label_threshold(scores, labels, alpha=0.10)
    assert 0.7 <= tau <= 0.9  # the non-TB scores 0.1 and 0.2 must not drag it down


def test_few_label_trivial_when_positives_run_out():
    scores = np.linspace(0, 1, 10)
    labels = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])  # only 3 TB cases
    assert few_label_threshold(scores, labels, alpha=0.10) == -math.inf


# ------------------------------------------------------------- edge cases


def test_quantile_match_values_uses_learned_mapping():
    rng = np.random.default_rng(3)
    target, source = rng.normal(5, 1, 1000), rng.normal(size=1000)
    # The target median should land near the source median.
    assert quantile_match(target, source, values=[np.median(target)])[0] == pytest.approx(
        np.median(source), abs=0.1
    )


def test_quantile_match_ties_map_to_the_same_value():
    mapped = quantile_match([1.0, 2.0, 2.0, 3.0], [10.0, 20.0, 30.0, 40.0])
    assert mapped[1] == mapped[2]


def test_label_free_threshold_agrees_with_quantile_match():
    # Moving the threshold once must flag the same share of target images as
    # moving every target score and keeping the source threshold.
    rng = np.random.default_rng(4)
    source, target = rng.normal(size=4000), rng.normal(-0.8, 1.3, size=4000)
    tau = np.quantile(source, 0.25)
    by_scores = np.mean(quantile_match(target, source) >= tau)
    by_threshold = np.mean(target >= label_free_threshold(tau, source, target))
    assert by_scores == pytest.approx(by_threshold, abs=0.01)


def test_label_free_keeps_trivial_threshold_trivial():
    assert label_free_threshold(-math.inf, [0.1, 0.2], [0.3, 0.4]) == -math.inf


def test_quantile_match_wrongly_lowers_scores_when_tb_is_more_common():
    # Same model, same scanner, but TB is far more common at the new site.
    # Higher target scores are honest; matching pulls them back down and the
    # source threshold then misses real TB cases.
    rng = np.random.default_rng(5)

    def site(n, prevalence):
        is_tb = rng.random(n) < prevalence
        return np.where(is_tb, rng.normal(2, 1, n), rng.normal(0, 1, n)), is_tb

    source, src_tb = site(20_000, 0.5)
    target, tgt_tb = site(20_000, 0.8)
    tau = np.quantile(source[src_tb], 0.10)  # 90% sensitivity at the source
    assert np.mean(target[tgt_tb] >= tau) == pytest.approx(0.90, abs=0.02)  # still fine
    assert np.mean(target[tgt_tb] >= label_free_threshold(tau, source, target)) < 0.85


def test_few_label_certified_variant():
    scores = np.random.default_rng(6).normal(size=200)
    labels = np.ones(200, dtype=int)
    assert few_label_threshold(scores, labels, 0.10, delta=0.10) <= few_label_threshold(
        scores, labels, 0.10
    )


def test_few_label_rejects_bad_labels():
    with pytest.raises(ValueError):
        few_label_threshold([0.1, 0.2], [1, 2], alpha=0.1)
