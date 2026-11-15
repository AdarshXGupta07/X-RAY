"""Specification for tbshift.recalibration (phase P6). Expected to fail until implemented."""

import math

import numpy as np
import pytest

from tbshift.recalibration import few_label_threshold, quantile_match
from tbshift.shift import ks_statistic

pytestmark = pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="phase P6 not written")


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
