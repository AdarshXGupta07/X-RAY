"""Tests for the P7 analysis stage, on a synthetic degradation cache."""

import pickle

import numpy as np
import pytest

from tbshift.experiments.p7_degradations import DegradeFn, analyze


def _write(path, logits, labels):
    np.savez(path, logits=logits.astype(np.float32), labels=labels.astype(np.int8),
             image_ids=np.array([f"i{k}" for k in range(labels.size)]),
             splits=np.array(["test"] * labels.size), embeddings=np.zeros((labels.size, 2), np.float16))


@pytest.fixture(scope="module")
def summary(tmp_path_factory):
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(300), np.zeros(300)].astype(int)
    clean = np.where(y == 1, rng.normal(2, 1, y.size), rng.normal(-2, 1, y.size))
    d = tmp_path_factory.mktemp("p7")
    _write(d / "clean.npz", clean, y)
    _write(d / "blur_s5.npz", clean - 1.5, y)  # every score lower: a harmful shift
    return analyze(d, repeats=80).set_index(["degradation", "severity", "method"])


def test_clean_control_keeps_the_promise(summary):
    assert summary.loc[("clean", 0, "certified"), "violation_rate"] <= 0.3
    assert summary.loc[("clean", 0, "certified"), "detector_fires"] <= 0.15


def test_harmful_degradation_breaks_promise_and_is_detected(summary):
    row = summary.loc[("blur", 5, "certified")]
    assert row["violation_rate"] > 0.9
    assert row["detector_fires"] > 0.9
    assert row["mean_logit_shift_tb"] == pytest.approx(-1.5, abs=1e-5)


def test_repairs_restore_the_promise(summary):
    # Same prevalence on both sides, so label-free matching is valid here.
    assert summary.loc[("blur", 5, "label_free"), "violation_rate"] < 0.5
    assert summary.loc[("blur", 5, "few_label_certified"), "violation_rate"] < 0.4
    assert summary.loc[("blur", 5, "detect_defer"), "violation_rate"] < 0.15


def test_degrade_fn_pickles_and_varies_by_image():
    fn = pickle.loads(pickle.dumps(DegradeFn("noise", 3, seed=1)))
    img = np.full((16, 16), 0.5, np.float32)
    assert not np.array_equal(fn(img, 0), fn(img, 1))  # per-image seed
    assert np.array_equal(fn(img, 0), DegradeFn("noise", 3, seed=1)(img, 0))  # reproducible
