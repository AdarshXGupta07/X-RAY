"""Smoke test for the synthetic shift experiment.

Small repeat counts keep it fast, so the bounds are loose. They lock in the
qualitative findings: if a code change flips one of these, investigate.
"""

import pytest

from tbshift.experiments.synthetic_shift import run_detection, run_methods


@pytest.fixture(scope="module")
def methods():
    df = run_methods(budgets=(100,), repeats=150, seed=1)
    return df.set_index(["scenario", "method"])["violation_rate"]


def test_source_certified_holds_without_shift(methods):
    assert methods["none", "source_certified"] <= 0.20


def test_harmful_shift_breaks_source_and_label_free_repairs_it(methods):
    assert methods["harmful_shift", "source_certified"] > 0.60
    assert methods["harmful_shift", "label_free"] < 0.35


def test_label_free_fails_when_tb_becomes_more_common(methods):
    assert methods["prevalence_rise", "source_certified"] <= 0.20
    assert methods["prevalence_rise", "label_free"] > 0.80


def test_few_labels_repair_the_subtle_shift(methods):
    assert methods["tb_spread", "source_certified"] > 0.80
    assert methods["tb_spread", "few_label_certified_n100"] <= 0.20


def test_detector_fires_on_harmless_shift_too():
    df = run_detection(batch_sizes=(200,), repeats=100, seed=2).set_index("scenario")["fire_rate"]
    assert df["none"] < 0.12
    assert df["harmless_shift"] > 0.70
