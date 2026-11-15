"""Deterministic, stratified, patient-level splits (plan phase P1).

Each site's images are divided into:

    train  fit the network
    val    early stopping / model selection
    cal    temperature scaling and the rule-out threshold
    test   evaluation, touched once per experiment

Three rules:

1. Patient-level: all images of one patient land in the same split, so the
   model is never tested on a person it trained on. (The NLM sets have one
   image per patient, but the code does not rely on that.)
2. Stratified: each class is split separately, so every split keeps the
   site's TB share. With only 58 TB cases at Montgomery, an unstratified
   split could leave the calibration split nearly empty of TB.
3. Deterministic: the same manifest and seed always give the same split,
   whatever order the rows are in.

Run:  python -m tbshift.data.splits --site montgomery --seed 0
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "cal", "test")
DEFAULT_FRACTIONS = {"train": 0.60, "val": 0.10, "cal": 0.15, "test": 0.15}


def allocate(n: int, fractions: dict[str, float]) -> dict[str, int]:
    """Split n items into whole-number counts that follow ``fractions`` and sum to n.

    Largest-remainder method: give each split floor(n * fraction), then hand
    the leftover items to the splits with the largest fractional parts.
    Example: n = 10, fractions 0.6/0.1/0.15/0.15 -> 6/1/2/1 or 6/1/1/2.
    """
    exact = {name: n * f for name, f in fractions.items()}
    counts = {name: math.floor(x) for name, x in exact.items()}
    leftover = n - sum(counts.values())
    by_remainder = sorted(fractions, key=lambda k: (exact[k] - counts[k], fractions[k]), reverse=True)
    for name in by_remainder[:leftover]:
        counts[name] += 1
    return counts


def _check_fractions(fractions: dict[str, float]) -> None:
    unknown = set(fractions) - set(SPLITS)
    if unknown:
        raise ValueError(f"unknown split names: {sorted(unknown)}")
    if any(f < 0 for f in fractions.values()):
        raise ValueError("fractions must be non-negative")
    if not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-9):
        raise ValueError(f"fractions must sum to 1, got {sum(fractions.values())}")


def assign_splits(
    manifest: pd.DataFrame,
    fractions: dict[str, float] | None = None,
    seed: int = 0,
) -> pd.Series:
    """Return a ``split`` value for every manifest row (same index as the manifest).

    A patient counts as TB if any of their images is TB.
    """
    fractions = dict(fractions or DEFAULT_FRACTIONS)
    _check_fractions(fractions)
    for column in ("patient_id", "label"):
        if column not in manifest.columns:
            raise ValueError(f"manifest needs a {column!r} column")

    patient_label = manifest.groupby("patient_id")["label"].max().sort_index()
    rng = np.random.default_rng(seed)
    split_of = {}
    for label in (0, 1):
        patients = rng.permutation(patient_label[patient_label == label].index.to_numpy())
        start = 0
        for name, count in allocate(len(patients), fractions).items():
            for pid in patients[start : start + count]:
                split_of[pid] = name
            start += count
    return manifest["patient_id"].map(split_of).rename("split")


def summarize(manifest: pd.DataFrame, split: pd.Series) -> pd.DataFrame:
    """Table of TB / normal counts per split, for printing and the notebook."""
    table = pd.crosstab(split, manifest["label"]).reindex(SPLITS).fillna(0).astype(int)
    table.columns = [{0: "normal", 1: "TB"}.get(c, c) for c in table.columns]
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a 'split' column to a site manifest")
    parser.add_argument("--site", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest-dir", default="data/manifests")
    args = parser.parse_args()

    path = Path(args.manifest_dir) / f"{args.site}.csv"
    df = pd.read_csv(path)
    df["split"] = assign_splits(df, seed=args.seed)
    df.to_csv(path, index=False)
    print(f"{args.site}, seed {args.seed}:")
    print(summarize(df, df["split"]).to_string())
    print(f"updated {path}")


if __name__ == "__main__":
    main()
