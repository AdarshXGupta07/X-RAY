"""Tests for the multi-seed aggregation helpers."""

import pandas as pd
import pytest

from tbshift.experiments.multi_seed import find_runs, mean_sd, paired_mask_effect


def _p2(rows):
    return pd.DataFrame(rows, columns=["fold", "mask", "seed", "ft_auc_target"])


def test_mean_sd_per_group():
    df = _p2([("a", "none", 0, 0.7), ("a", "none", 1, 0.9), ("b", "none", 0, 0.5)])
    out = mean_sd(df, ["fold"], ["ft_auc_target"]).set_index("fold")
    assert out.loc["a", "ft_auc_target_mean"] == pytest.approx(0.8)
    assert out.loc["a", "ft_auc_target_std"] == pytest.approx(0.1414, abs=1e-3)
    assert out.loc["a", "seeds"] == 2 and out.loc["b", "seeds"] == 1


def test_paired_mask_effect_counts_seeds_where_mask_helps():
    df = _p2([
        ("f", "none", 0, 0.80), ("f", "corners", 0, 0.85),   # +0.05
        ("f", "none", 1, 0.90), ("f", "corners", 1, 0.88),   # -0.02
        ("f", "none", 2, 0.70), ("f", "corners", 2, 0.75),   # +0.05
    ])
    row = paired_mask_effect(df, "ft_auc_target").iloc[0]
    assert row["mean_diff"] == pytest.approx(0.08 / 3)
    assert row["seeds_mask_better"] == 2 and row["seeds"] == 3


def test_find_runs_needs_summary_and_scores(tmp_path):
    for name, has_scores in [("p2_densenet", True), ("p2_densenet_s3", True), ("p2_densenet_s4", False),
                             ("p2_densenet_old", True)]:
        (tmp_path / "results" / name).mkdir(parents=True)
        (tmp_path / "results" / name / "summary.csv").write_text("x\n", encoding="utf-8")
        if has_scores:
            (tmp_path / "scores" / name).mkdir(parents=True)
    runs = find_runs(tmp_path / "results", tmp_path / "scores")
    assert runs == {0: "p2_densenet", 3: "p2_densenet_s3"}
