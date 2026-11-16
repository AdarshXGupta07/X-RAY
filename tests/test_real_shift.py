"""Tests for the real-data P5/P6 module, on a synthetic score cache."""

import numpy as np
import pytest

from tbshift.experiments.real_shift import run, summarize


def _site(rng, n_tb, n_normal, shift=0.0, emb_shift=0.0):
    labels = np.r_[np.ones(n_tb), np.zeros(n_normal)].astype(np.int8)
    logits = np.where(labels == 1, rng.normal(2, 1, labels.size), rng.normal(-2, 1, labels.size)) + shift
    splits = rng.choice(["train", "val", "cal", "test"], size=labels.size, p=[0.4, 0.1, 0.25, 0.25])
    return {"logits": logits.astype(np.float32), "labels": labels, "splits": splits.astype(str),
            "image_ids": np.array([f"i{k}" for k in range(labels.size)]),
            "embeddings": (rng.normal(size=(labels.size, 8)) + emb_shift).astype(np.float16)}


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """Unseen site: every logit 1.5 lower (harmful) and clearly different embeddings."""
    root = tmp_path_factory.mktemp("cache")
    rng = np.random.default_rng(0)
    fold = root / "toy" / "train_big__none"
    fold.mkdir(parents=True)
    np.savez(fold / "big.npz", **_site(rng, 400, 400))
    np.savez(fold / "small.npz", **_site(rng, 200, 200, shift=-1.5, emb_shift=1.0))
    methods, detection, domains = run("toy", repeats=60, scores_root=root)
    meth, det = summarize(methods, detection)
    return meth.set_index(["method", "budget"]), det.set_index("batch_size"), domains


def test_shift_breaks_source_threshold_and_label_free_repairs_it(results):
    meth, _, _ = results
    assert meth.loc[("source_certified", 0), "violation_rate"] > 0.9
    assert meth.loc[("label_free", 0), "violation_rate"] < 0.5


def test_detector_has_power_and_few_false_alarms(results):
    _, det, _ = results
    assert det["power"].iloc[-1] > 0.9  # largest batch
    assert det["false_alarm"].max() < 0.2


def test_few_label_certified_needs_enough_positives(results):
    meth, _, _ = results
    assert meth.loc[("few_label_certified", 10), "flags_everyone_rate"] == 1.0  # ~5 TB < 22
    assert meth.loc[("few_label_certified", 200), "violation_rate"] <= 0.25


def test_domain_classifier_separates_shifted_embeddings(results):
    _, _, domains = results
    row = domains.iloc[0]
    assert row["domain_auc_source_vs_target"] > 0.9
    assert 0.3 < row["domain_auc_source_vs_source"] < 0.7


def test_methods_never_score_on_adaptation_images(results):
    meth, _, _ = results
    # Budgets are capped at the adaptation half (200 of 400 target images).
    assert max(b for _, b in meth.index) == 200
