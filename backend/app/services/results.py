"""Read the study's result tables (aggregate numbers only, no per-patient data)."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

TABLES = {
    "p2_per_seed": "multi_seed/p2_per_seed.csv",
    "p4_per_seed": "multi_seed/p4_per_seed.csv",
    "p56_detection_summary": "multi_seed/p56_detection_summary.csv",
    "p56_methods_summary": "multi_seed/p56_methods_summary.csv",
    "mask_effect": "multi_seed/mask_effect_summary.csv",
    "p7_degradations": "p7_degradations/train_shenzhen__none.csv",
}


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}
            for row in df.to_dict(orient="records")]


def available(results_dir: Path) -> list[str]:
    return [name for name, rel in TABLES.items() if (results_dir / rel).exists()]


def table(results_dir: Path, name: str) -> list[dict] | None:
    rel = TABLES.get(name)
    if rel is None or not (results_dir / rel).exists():
        return None
    return _records(pd.read_csv(results_dir / rel))


def headline(results_dir: Path, source: str = "shenzhen", mask: str = "none") -> list[dict] | None:
    """Per training seed: the certified threshold at the source control and at the unseen site."""
    p4_path = results_dir / TABLES["p4_per_seed"]
    p2_path = results_dir / TABLES["p2_per_seed"]
    if not (p4_path.exists() and p2_path.exists()):
        return None
    p4 = pd.read_csv(p4_path)
    p4 = p4[(p4.source == source) & (p4["mask"] == mask) & (p4.method == "certified")]
    ctrl = p4[p4.evaluated_on == "source_test"].set_index("seed")
    tgt = p4[p4.evaluated_on == "target"].set_index("seed")
    p2 = pd.read_csv(p2_path)
    p2 = p2[(p2.source == source) & (p2["mask"] == mask)].set_index("seed")
    seeds = tgt.index
    out = pd.DataFrame({
        "seed": seeds.to_numpy(),
        "target": tgt["target"].to_numpy(),
        "control_violation": ctrl["violation_rate"].reindex(seeds).to_numpy(),
        "target_auc": p2["ft_auc_target"].reindex(seeds).to_numpy(),
        "target_violation": tgt["violation_rate"].to_numpy(),
        "target_sensitivity": tgt["mean_sensitivity"].to_numpy(),
        "target_specificity": tgt["mean_specificity"].to_numpy(),
    }).sort_values("seed")
    return _records(out)
