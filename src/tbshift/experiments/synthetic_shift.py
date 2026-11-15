"""Synthetic dress rehearsal for research questions 1-3 (no real images).

Real experiments need cached model scores from real hospitals. Until then,
this script runs the full comparison on synthetic scores where the TRUE
sensitivity and specificity of any threshold are known exactly. It answers:

* Detection table:  how often does the label-free KS detector fire, by
  scenario and batch size?
* Methods table:    for each scenario, what happens to the sensitivity
  promise under each strategy?

Synthetic world
---------------
Scores are the model's TB logit. Non-TB ~ N(mu_neg, 1), TB ~ N(mu_pos, sd_pos).
The source site is N(0, 1) vs N(2, 1) with 50% TB. Each scenario changes the
target site in one way:

    none             nothing changes                     (control)
    harmful_shift    every score drops by 0.5            (scanner hides TB)
    harmless_shift   every score rises by 0.5            (detector fires, no harm)
    prevalence_rise  TB becomes 80% of cases             (scores unchanged)
    tb_spread        TB scores become more variable      (subtle, harmful)

Methods (all aim at 90% sensitivity)
------------------------------------
    fixed_p0.5         flag if P(TB) >= 0.5 (logit >= 1); not tuned for sensitivity
    source_marginal    split-conformal threshold from source TB cases
    source_certified   certified threshold from source TB cases (delta = 0.10)
    label_free         source_certified moved by quantile matching to the target batch
    detect_defer       source_certified, but defer EVERY case if the KS detector fires
    few_label_*        threshold recomputed from n labelled target images

Run:
    python -m tbshift.experiments.synthetic_shift
    python -m tbshift.experiments.synthetic_shift --repeats 2000 --out-dir results
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from tbshift.conformal import certified_threshold, conformal_threshold
from tbshift.metrics import violation_rate
from tbshift.recalibration import few_label_threshold, label_free_threshold
from tbshift.shift import detect_score_shift

ALPHA = 0.10
DELTA = 0.10
FIXED_LOGIT = 1.0  # P(TB) = 0.5 for the source world (log-odds = 2x - 2)


@dataclass(frozen=True)
class Scenario:
    mu_neg: float = 0.0
    mu_pos: float = 2.0
    sd_pos: float = 1.0
    prevalence: float = 0.5

    def sample(self, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        """n images: (scores, labels)."""
        labels = (rng.random(n) < self.prevalence).astype(int)
        scores = np.where(
            labels == 1,
            rng.normal(self.mu_pos, self.sd_pos, n),
            rng.normal(self.mu_neg, 1.0, n),
        )
        return scores, labels

    def sensitivity(self, tau) -> np.ndarray:
        """Exact P(score >= tau | TB). Equals 1 when tau = -inf."""
        return norm.sf((np.asarray(tau) - self.mu_pos) / self.sd_pos)

    def specificity(self, tau) -> np.ndarray:
        """Exact P(score < tau | not TB). Equals 0 when tau = -inf."""
        return norm.cdf(np.asarray(tau) - self.mu_neg)


SOURCE = Scenario()
SCENARIOS = {
    "none": SOURCE,
    "harmful_shift": Scenario(mu_neg=-0.5, mu_pos=1.5),
    "harmless_shift": Scenario(mu_neg=0.5, mu_pos=2.5),
    "prevalence_rise": Scenario(prevalence=0.8),
    "tb_spread": Scenario(sd_pos=1.6),
}


def run_detection(
    batch_sizes=(10, 20, 50, 100, 200),
    repeats: int = 1000,
    n_source_ref: int = 500,
    significance: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Share of target batches on which the KS detector fires."""
    rng = np.random.default_rng(seed)
    rows = []
    for name, scenario in SCENARIOS.items():
        for batch in batch_sizes:
            fired = np.empty(repeats, dtype=bool)
            for r in range(repeats):
                source_ref, _ = SOURCE.sample(rng, n_source_ref)
                target_batch, _ = scenario.sample(rng, batch)
                fired[r] = detect_score_shift(source_ref, target_batch, significance).shifted
            rows.append({"scenario": name, "batch_size": batch, "fire_rate": float(fired.mean())})
    return pd.DataFrame(rows)


def run_methods(
    budgets=(10, 20, 50, 100),
    repeats: int = 1000,
    n_source_cal_pos: int = 100,
    n_source_ref: int = 500,
    target_batch: int = 200,
    significance: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Achieved sensitivity/specificity of every method in every scenario."""
    rng = np.random.default_rng(seed)
    rows = []
    for name, scenario in SCENARIOS.items():
        log: list[tuple[str, float, float]] = []  # (method, threshold, positives used)
        for _ in range(repeats):
            # What the source site gives us: labelled TB cases for calibration,
            # plus an unlabelled reference sample of mixed images.
            cal_pos = rng.normal(SOURCE.mu_pos, SOURCE.sd_pos, n_source_cal_pos)
            source_ref, _ = SOURCE.sample(rng, n_source_ref)
            tau_marginal = conformal_threshold(cal_pos, ALPHA)
            tau_certified = certified_threshold(cal_pos, ALPHA, DELTA)

            # What the new site gives us for free: an unlabelled batch.
            batch, _ = scenario.sample(rng, target_batch)
            fired = detect_score_shift(source_ref, batch, significance).shifted

            log += [
                ("fixed_p0.5", FIXED_LOGIT, math.nan),
                ("source_marginal", tau_marginal, math.nan),
                ("source_certified", tau_certified, math.nan),
                ("label_free", label_free_threshold(tau_certified, source_ref, batch), math.nan),
                ("detect_defer", -math.inf if fired else tau_certified, math.nan),
            ]

            # What the new site gives us if we pay for labels.
            for n in budgets:
                scores, labels = scenario.sample(rng, n)
                n_pos = int(labels.sum())
                log += [
                    (f"few_label_marginal_n{n}", few_label_threshold(scores, labels, ALPHA), n_pos),
                    (
                        f"few_label_certified_n{n}",
                        few_label_threshold(scores, labels, ALPHA, DELTA),
                        n_pos,
                    ),
                ]

        runs = pd.DataFrame(log, columns=["method", "tau", "n_pos"])
        for method, group in runs.groupby("method", sort=False):
            tau = group["tau"].to_numpy()
            sens = scenario.sensitivity(tau)
            rows.append(
                {
                    "scenario": name,
                    "method": method,
                    "mean_sensitivity": float(sens.mean()),
                    "violation_rate": violation_rate(sens, 1 - ALPHA),
                    "mean_specificity": float(scenario.specificity(tau).mean()),
                    # -inf means "flag everyone": trivial threshold or deferral.
                    "flag_all_rate": float(np.mean(np.isneginf(tau))),
                    "mean_positives": float(group["n_pos"].mean()),  # NaN if no labels used
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic site-shift experiment")
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", help="optional folder for CSV output")
    args = parser.parse_args()

    detection = run_detection(repeats=args.repeats, seed=args.seed)
    methods = run_methods(repeats=args.repeats, seed=args.seed)

    fmt = {"float_format": lambda x: f"{x:.3f}"}
    print(f"SYNTHETIC DATA ONLY. target sensitivity {1 - ALPHA:.0%}, delta {DELTA}, "
          f"{args.repeats} repeats, seed {args.seed}\n")
    print("KS detector fire rate (rows: scenario, columns: unlabelled batch size)")
    print(detection.pivot(index="scenario", columns="batch_size", values="fire_rate")
          .reindex(list(SCENARIOS)).to_string(**fmt))
    print("\nMethods (violation = share of repeats with true sensitivity below 90%)")
    print(methods.to_string(index=False, **fmt))

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        detection.to_csv(out / "synthetic_detection.csv", index=False)
        methods.to_csv(out / "synthetic_methods.csv", index=False)
        print(f"\nwrote {out / 'synthetic_detection.csv'} and {out / 'synthetic_methods.csv'}")


if __name__ == "__main__":
    main()
