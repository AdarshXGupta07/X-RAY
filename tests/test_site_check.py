"""Tests for the site-check verdict logic (plain numbers, no model)."""

import math

import numpy as np
import pytest

from tbshift.conformal import certified_threshold, min_positives_certified
from tbshift.site_check import MIN_IMAGES, UNLABELLED, build_reference, site_check


@pytest.fixture(scope="module")
def ref():
    rng = np.random.default_rng(0)
    z = np.concatenate([rng.normal(2.0, 1.0, 150), rng.normal(-2.0, 1.0, 150)])
    y = np.concatenate([np.ones(150, int), np.zeros(150, int)])
    return build_reference(z, y, "toy", "toysite")


def _site(rng, n, tb_shift=0.0, normal_shift=0.0):
    z = np.concatenate([rng.normal(2.0 + tb_shift, 1.0, n), rng.normal(-2.0 + normal_shift, 1.0, n)])
    y = np.concatenate([np.ones(n, int), np.zeros(n, int)])
    return z, y


def test_reference_threshold_is_the_certified_one(ref):
    assert ref.threshold == certified_threshold(ref.logits[ref.labels == 1], 0.10, 0.10)
    s = ref.summary()
    assert s["tb_labels_needed"] == 22 and s["n_tb"] == 150
    assert 0 < s["threshold_probability"] < 1


def test_reference_needs_enough_tb_cases():
    with pytest.raises(ValueError, match="too few TB"):
        build_reference(np.arange(10.0), np.r_[np.ones(5, int), np.zeros(5, int)], "m", "s")


def test_too_few_images_defers(ref):
    out = site_check(ref, np.zeros(MIN_IMAGES - 1))
    assert out.verdict == "defer" and out.shift_detected is None and out.ks_p_value is None


def test_same_distribution_is_plausible(ref):
    z, _ = _site(np.random.default_rng(1), 40)
    out = site_check(ref, z)
    assert out.verdict == "plausible" and out.shift_detected is False
    assert out.threshold_in_force_logit == ref.threshold
    assert out.recertified_threshold_logit is None


def test_shift_without_labels_asks_for_labels(ref):
    z, _ = _site(np.random.default_rng(2), 40, tb_shift=-2.0)
    out = site_check(ref, z)
    assert out.verdict == "recalibrate" and out.shift_detected
    assert out.tb_labels_needed == min_positives_certified(0.10, 0.10)


def test_shift_with_too_few_labels_counts_what_is_missing(ref):
    z, y = _site(np.random.default_rng(3), 40, tb_shift=-2.0)
    shown = np.where(np.arange(z.size) < 10, y, UNLABELLED)   # 10 labelled TB
    out = site_check(ref, z, shown)
    assert out.verdict == "recalibrate"
    assert out.n_labelled_tb == 10 and out.tb_labels_needed == 12
    assert out.labelled_tb_missed == int(np.sum(z[:10] < ref.threshold))


def test_a_few_labels_can_already_prove_failure(ref):
    rng = np.random.default_rng(8)
    z, _ = _site(rng, 40)                          # no score shift overall
    tb_idx = np.arange(15)
    z[tb_idx[:6]] = ref.threshold - 1.0            # 6 of 15 labelled TB fall below
    shown = np.full(z.size, UNLABELLED)
    shown[tb_idx] = 1
    out = site_check(ref, z, shown)
    assert out.labelled_tb_missed >= 6
    assert out.verdict == "recalibrate" and "already show" in out.headline


def test_one_miss_in_a_few_labels_is_not_proof(ref):
    rng = np.random.default_rng(9)
    z, _ = _site(rng, 40)
    shown = np.full(z.size, UNLABELLED)
    shown[:10] = 1
    z[:10] = ref.threshold + 1.0
    z[0] = ref.threshold - 1.0                     # 1 miss of 10 is consistent with 90%
    out = site_check(ref, z, shown)
    assert out.labelled_tb_missed == 1 and out.verdict == "plausible"


def test_enough_tb_labels_recertify_even_without_shift(ref):
    z, y = _site(np.random.default_rng(4), 40)
    out = site_check(ref, z, y)          # 40 labelled TB >= 22
    assert out.verdict == "recertified"
    assert out.recertified_threshold_logit == certified_threshold(z[y == 1], 0.10, 0.10)
    assert out.threshold_in_force_logit == out.recertified_threshold_logit
    assert out.ruled_out == [bool(v) for v in z < out.recertified_threshold_logit]


def test_recertified_threshold_restores_sensitivity_on_harmful_shift(ref):
    rng = np.random.default_rng(5)
    z, y = _site(rng, 200, tb_shift=-2.5)
    out = site_check(ref, z, y)
    sens_source = np.mean(z[y == 1] >= ref.threshold)
    sens_new = np.mean(z[y == 1] >= out.recertified_threshold_logit)
    assert sens_source < 0.9 <= sens_new


def test_histogram_shares_sum_to_one(ref):
    z, _ = _site(np.random.default_rng(6), 30)
    h = site_check(ref, z).histogram
    assert len(h["edges"]) == len(h["source"]) + 1 == len(h["site"]) + 1
    assert math.isclose(sum(h["source"]), 1) and math.isclose(sum(h["site"]), 1)
    assert h["new_threshold"] is None


def test_bad_inputs_are_rejected(ref):
    with pytest.raises(ValueError):
        site_check(ref, np.array([]))
    with pytest.raises(ValueError):
        site_check(ref, np.array([np.nan] * 30))
    with pytest.raises(ValueError):
        site_check(ref, np.zeros(30), np.full(30, 2))
    with pytest.raises(ValueError):
        site_check(ref, np.zeros(30), np.zeros(29))


def test_to_dict_has_no_infinite_numbers(ref):
    z, _ = _site(np.random.default_rng(7), 30)
    d = site_check(ref, z).to_dict()
    flat = [v for v in d.values() if isinstance(v, float)]
    assert all(math.isfinite(v) for v in flat)
