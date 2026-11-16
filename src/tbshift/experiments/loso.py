"""Phase P4: leave-one-site-out evaluation of the rule-out promise (RQ1).

Reads only the score cache written in P2. For each fold (model trained on one
site), each corner-mask setting and each of many repeats:

1. Draw a fresh calibration set: a random, class-stratified half of the
   source site's held-out images (its ``cal`` and ``test`` splits pooled).
   The other half is the in-distribution test set for that repeat.
2. Fit temperature scaling on the calibration half (both classes).
3. Set three thresholds on the logits:
       fixed      P(TB) >= 0.5 after temperature scaling (logit >= 0)
       marginal   split-conformal, 90% target sensitivity
       certified  sensitivity >= 90% with probability >= 90% (delta = 0.10)
4. Measure sensitivity and specificity on
       source_test  the other half of the source pool (the control)
       target       every image of the unseen site

A repeat "violates" when measured sensitivity is below 90%. On the control,
the certified threshold should violate in about 10% of repeats or fewer,
plus some extra from the finite size of the test half. Anything much worse
on the control is a bug, not a finding.

Run:  python -m tbshift.experiments.loso [--repeats 1000] [--run p2_densenet]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tbshift.calibration import apply_temperature, fit_temperature
from tbshift.conformal import certified_threshold, conformal_threshold
from tbshift.metrics import expected_calibration_error, sensitivity, specificity

ALPHA = 0.10
DELTA = 0.10
METHODS = ("fixed", "marginal", "certified")


def load_scores(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in ("logits", "labels", "splits")}


def stratified_half(labels: np.ndarray, rng: np.random.Generator, fraction: float = 0.5):
    """Boolean mask choosing ``fraction`` of each class (rounded) for calibration."""
    chosen = np.zeros(labels.size, dtype=bool)
    for cls in (0, 1):
        idx = rng.permutation(np.flatnonzero(labels == cls))
        chosen[idx[: round(fraction * idx.size)]] = True
    return chosen


def one_fold(
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    repeats: int,
    rng: np.random.Generator,
    cal_fraction: float = 0.5,
) -> pd.DataFrame:
    """Per-repeat results for one trained model."""
    pool = np.isin(source["splits"], ("cal", "test"))
    z_pool, y_pool = source["logits"][pool], source["labels"][pool].astype(int)
    z_tgt, y_tgt = target["logits"], target["labels"].astype(int)

    rows = []
    for r in range(repeats):
        cal = stratified_half(y_pool, rng, cal_fraction)
        z_cal, y_cal = z_pool[cal], y_pool[cal]
        temperature = fit_temperature(z_cal, y_cal)
        cal_pos = z_cal[y_cal == 1]
        thresholds = {
            "fixed": 0.0,
            "marginal": conformal_threshold(cal_pos, ALPHA),
            "certified": certified_threshold(cal_pos, ALPHA, DELTA),
        }
        evals = {"source_test": (y_pool[~cal], z_pool[~cal]), "target": (y_tgt, z_tgt)}
        for where, (y, z) in evals.items():
            ece_raw = expected_calibration_error(y, apply_temperature(z, 1.0))
            ece_ts = expected_calibration_error(y, apply_temperature(z, temperature))
            for method, tau in thresholds.items():
                rows.append({
                    "repeat": r, "evaluated_on": where, "method": method,
                    "threshold": tau, "trivial": tau == -math.inf,
                    "sensitivity": sensitivity(y, z, tau), "specificity": specificity(y, z, tau),
                    "n_cal_tb": int(cal_pos.size), "n_eval_tb": int(y.sum()),
                    "temperature": temperature, "ece_raw": ece_raw, "ece_scaled": ece_ts,
                })
    return pd.DataFrame(rows)


def run(
    run_name: str = "p2_densenet", repeats: int = 1000, seed: int = 0, cal_fraction: float = 0.5,
    scores_root: Path | str = "artifacts/scores",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for fold_dir in sorted((Path(scores_root) / run_name).iterdir()):
        fold, mask = fold_dir.name.split("__")
        source_site = fold.removeprefix("train_")
        target_site = next(p.stem for p in fold_dir.glob("*.npz") if p.stem != source_site)
        df = one_fold(load_scores(fold_dir / f"{source_site}.npz"),
                      load_scores(fold_dir / f"{target_site}.npz"), repeats, rng, cal_fraction)
        df.insert(0, "mask", mask)
        df.insert(0, "target", target_site)
        df.insert(0, "source", source_site)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarize(per_repeat: pd.DataFrame) -> pd.DataFrame:
    """One row per (source, mask, evaluated_on, method)."""
    keys = ["source", "target", "mask", "evaluated_on", "method"]
    g = per_repeat.groupby(keys, sort=False)
    out = g.agg(
        repeats=("repeat", "nunique"),
        cal_tb=("n_cal_tb", "median"),
        eval_tb=("n_eval_tb", "median"),
        mean_sensitivity=("sensitivity", "mean"),
        sens_p05=("sensitivity", lambda s: s.quantile(0.05)),
        mean_specificity=("specificity", "mean"),
        trivial_rate=("trivial", "mean"),
        temperature=("temperature", "median"),
        ece_raw=("ece_raw", "mean"),
        ece_scaled=("ece_scaled", "mean"),
    ).reset_index()
    out.insert(out.columns.get_loc("mean_sensitivity") + 1, "violation_rate",
               g["sensitivity"].apply(lambda s: float(np.mean(s.dropna() < 1 - ALPHA))).to_numpy())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="P4 leave-one-site-out on the score cache")
    parser.add_argument("--run", default="p2_densenet")
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    per_repeat = run(args.run, args.repeats, args.seed)
    summary = summarize(per_repeat)
    out = Path("results") / "p4_loso"
    out.mkdir(parents=True, exist_ok=True)
    per_repeat.to_csv(out / "repeats.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    cols = ["source", "mask", "evaluated_on", "method", "cal_tb", "mean_sensitivity", "violation_rate",
            "sens_p05", "mean_specificity", "trivial_rate"]
    print(f"{args.repeats} calibration draws per model, seed {args.seed}; target sensitivity 90%, delta {DELTA}\n")
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    cal = summary[summary.method == "fixed"][["source", "mask", "evaluated_on", "temperature", "ece_raw", "ece_scaled"]]
    print("\nCalibration (temperature fitted on each calibration half):")
    print(cal.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out / 'repeats.csv'} and {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
