"""Validity simulation for the two rule-out thresholds, with NO site shift.

This is the control experiment for the whole project: if the guarantees do
not hold here, any failure at a new hospital could just be a bug.

Synthetic setup: TB-positive scores are standard normal, so the TRUE
sensitivity of a threshold tau is known exactly, 1 - Phi(tau). For each
number of calibration positives we draw many calibration sets, compute both
thresholds, and record:

* mean_sensitivity  - average true sensitivity (what the marginal promise is about)
* violation_rate    - share of calibration sets with true sensitivity below target
* trivial_rate      - share of calibration sets where the threshold was -inf
                      (rule out nobody: safe but useless)

Run:
    python -m tbshift.experiments.validity_simulation
    python -m tbshift.experiments.validity_simulation --out results/validity_simulation.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm

from tbshift.conformal import (
    certified_threshold,
    conformal_threshold,
    min_positives_certified,
    min_positives_marginal,
)
from tbshift.metrics import violation_rate

DEFAULT_N_POS = (5, 9, 10, 20, 22, 30, 50, 100, 200)


def simulate(
    n_pos_grid=DEFAULT_N_POS,
    alpha: float = 0.10,
    delta: float = 0.10,
    repeats: int = 5000,
    seed: int = 0,
) -> pd.DataFrame:
    """Return one row per (n_pos, method) with the three summary numbers."""
    rng = np.random.default_rng(seed)
    methods = {
        "marginal": lambda s: conformal_threshold(s, alpha),
        "certified": lambda s: certified_threshold(s, alpha, delta),
    }
    rows = []
    for n_pos in n_pos_grid:
        taus = {name: np.empty(repeats) for name in methods}
        for r in range(repeats):
            scores = rng.normal(size=n_pos)  # same calibration set for both methods
            for name, fn in methods.items():
                taus[name][r] = fn(scores)
        for name, tau in taus.items():
            true_sens = norm.sf(tau)  # P(score >= tau); equals 1 when tau = -inf
            rows.append(
                {
                    "n_pos": n_pos,
                    "method": name,
                    "mean_sensitivity": float(true_sens.mean()),
                    "violation_rate": violation_rate(true_sens, 1 - alpha),
                    "trivial_rate": float(np.mean(np.isneginf(tau))),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="optional CSV path")
    args = parser.parse_args()

    df = simulate(alpha=args.alpha, delta=args.delta, repeats=args.repeats, seed=args.seed)
    print(
        f"target sensitivity {1 - args.alpha:.0%}, delta {args.delta}, "
        f"{args.repeats} calibration sets per row, seed {args.seed}"
    )
    print(
        f"label floors: marginal {min_positives_marginal(args.alpha)} positives, "
        f"certified {min_positives_certified(args.alpha, args.delta)} positives\n"
    )
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
