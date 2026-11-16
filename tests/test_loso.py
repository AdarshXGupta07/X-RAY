"""Tests for the P4 leave-one-site-out module, on a synthetic score cache."""

import numpy as np
import pytest

from tbshift.experiments.loso import run, stratified_half, summarize


def _site(rng, n_tb, n_normal, shift=0.0):
    labels = np.r_[np.ones(n_tb), np.zeros(n_normal)].astype(np.int8)
    logits = np.where(labels == 1, rng.normal(2, 1, labels.size), rng.normal(-2, 1, labels.size)) + shift
    splits = rng.choice(["train", "val", "cal", "test"], size=labels.size, p=[0.4, 0.1, 0.25, 0.25])
    return {"logits": logits.astype(np.float32), "labels": labels, "splits": splits.astype(str),
            "image_ids": np.array([f"i{k}" for k in range(labels.size)]),
            "embeddings": np.zeros((labels.size, 4), np.float16)}


@pytest.fixture
def cache(tmp_path):
    """Fold trained on 'big' (unshifted); unseen site 'small' has every logit 1.5 lower."""
    rng = np.random.default_rng(0)
    fold = tmp_path / "toy" / "train_big__none"
    fold.mkdir(parents=True)
    np.savez(fold / "big.npz", **_site(rng, 400, 400))
    np.savez(fold / "small.npz", **_site(rng, 300, 300, shift=-1.5))
    return tmp_path


def test_stratified_half_keeps_class_balance():
    labels = np.r_[np.ones(40), np.zeros(60)].astype(int)
    pick = stratified_half(labels, np.random.default_rng(0))
    assert labels[pick].sum() == 20 and (~labels[pick].astype(bool)).sum() == 30


def test_columns_and_repeats(cache):
    df = run("toy", repeats=20, scores_root=cache)
    assert set(df["method"]) == {"fixed", "marginal", "certified"}
    assert set(df["evaluated_on"]) == {"source_test", "target"}
    assert df["repeat"].nunique() == 20 and len(df) == 20 * 2 * 3


def test_control_holds_and_shift_breaks_it(cache):
    s = summarize(run("toy", repeats=200, scores_root=cache)).set_index(["evaluated_on", "method"])
    # In-distribution: certified keeps its promise (allowance for finite test halves).
    assert s.loc[("source_test", "certified"), "violation_rate"] <= 0.25
    # Unseen site with every score 1.5 lower: the same thresholds miss many TB cases.
    assert s.loc[("target", "certified"), "violation_rate"] > 0.9


def test_reproducible_with_seed(cache):
    a = run("toy", repeats=10, seed=3, scores_root=cache)
    b = run("toy", repeats=10, seed=3, scores_root=cache)
    assert a.equals(b)


def test_too_few_source_positives_gives_trivial_threshold(tmp_path):
    rng = np.random.default_rng(1)
    fold = tmp_path / "toy" / "train_tiny__none"
    fold.mkdir(parents=True)
    tiny = _site(rng, 30, 30)
    tiny["splits"] = np.array(["cal"] * 60)  # pool of 30 TB -> about 15 per calibration half
    np.savez(fold / "tiny.npz", **tiny)
    np.savez(fold / "other.npz", **_site(rng, 50, 50))
    s = summarize(run("toy", repeats=30, scores_root=tmp_path)).set_index(["evaluated_on", "method"])
    assert s.loc[("target", "certified"), "trivial_rate"] == 1.0  # 15 < 22 positives
    assert s.loc[("target", "marginal"), "trivial_rate"] == 0.0  # 15 >= 9 positives
