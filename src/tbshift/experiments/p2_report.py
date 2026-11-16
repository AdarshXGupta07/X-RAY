"""Report for phase P2, read only from the score cache.

For every fold and mask setting:

* AUC with a stratified bootstrap 95% interval, on the source test split and
  on the whole unseen target site.
* A first look at research question 1 (the full study is phase P4): set a
  90%-sensitivity rule-out threshold from the source CALIBRATION split's TB
  cases, then measure the sensitivity and specificity it actually achieves on
  the source test split and on the unseen site.

This is ONE calibration split per fold, so it shows the direction of the
effect, not its frequency. P4 repeats it over many random calibration draws.

Run:  python -m tbshift.experiments.p2_report [--run p2_densenet]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tbshift.conformal import certified_threshold, conformal_threshold
from tbshift.metrics import auc_roc, bootstrap_ci, sensitivity, specificity

ALPHA = 0.10
DELTA = 0.10


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def report(run: str = "p2_densenet", n_boot: int = 2000) -> pd.DataFrame:
    rows = []
    for fold_dir in sorted((Path("artifacts/scores") / run).iterdir()):
        fold, mask = fold_dir.name.split("__")
        source = fold.removeprefix("train_")
        target = next(p.stem for p in fold_dir.glob("*.npz") if p.stem != source)
        src, tgt = load(fold_dir / f"{source}.npz"), load(fold_dir / f"{target}.npz")

        cal = src["splits"] == "cal"
        cal_pos = src["logits"][cal & (src["labels"] == 1)]
        thresholds = {
            "marginal": conformal_threshold(cal_pos, ALPHA),
            "certified": certified_threshold(cal_pos, ALPHA, DELTA),
        }
        test = src["splits"] == "test"
        evals = {
            "source_test": (src["labels"][test], src["logits"][test]),
            "target_all": (tgt["labels"], tgt["logits"]),
        }
        for where, (y, z) in evals.items():
            auc = bootstrap_ci(auc_roc, y, z, n_boot=n_boot)
            row = {
                "fold": fold, "mask": mask, "evaluated_on": where,
                "n": len(y), "n_tb": int(y.sum()), "cal_tb": len(cal_pos),
                "auc": auc.point, "auc_lo": auc.lower, "auc_hi": auc.upper,
            }
            for name, tau in thresholds.items():
                sens = bootstrap_ci(lambda yy, zz, t=tau: sensitivity(yy, zz, t), y, z, n_boot=n_boot)
                row[f"{name}_sens"] = sens.point
                row[f"{name}_sens_lo"] = sens.lower
                row[f"{name}_sens_hi"] = sens.upper
                row[f"{name}_spec"] = specificity(y, z, tau)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="P2 report from the score cache")
    parser.add_argument("--run", default="p2_densenet")
    args = parser.parse_args()
    df = report(args.run)
    out = Path("results") / args.run / "report.csv"
    df.to_csv(out, index=False)
    cols = ["fold", "mask", "evaluated_on", "n", "n_tb", "cal_tb", "auc", "auc_lo", "auc_hi",
            "marginal_sens", "marginal_spec", "certified_sens", "certified_sens_lo", "certified_sens_hi",
            "certified_spec"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
