"""Combine every training seed: how much do the results move between seeds?

One trained model is one draw from the training randomness (initial head
weights, batch order, augmentation). This script collects each seed's P2
results and reruns P4 (leave-one-site-out) and P5/P6 (detection and repair)
on that seed's score cache, then reports the mean and standard deviation
across seeds. Data splits are identical for every seed, so only training
randomness varies.

The corner-mask question is answered PAIRED: within each seed, the masked
and unmasked models differ only in the mask, so we look at the difference
per seed and count in how many seeds the mask helped.

Run:  python -m tbshift.experiments.multi_seed [--repeats 1000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tbshift.experiments import loso, real_shift

BASE_RUN = "p2_densenet"


def find_runs(results_root: Path | str = "results", scores_root: Path | str = "artifacts/scores") -> dict[int, str]:
    """{seed: run_name} for every P2 run that has both a summary and a score cache."""
    runs = {}
    for summary in Path(results_root).glob(f"{BASE_RUN}*/summary.csv"):
        name = summary.parent.name
        if name == BASE_RUN:
            seed = 0
        elif name.startswith(f"{BASE_RUN}_s") and name.removeprefix(f"{BASE_RUN}_s").isdigit():
            seed = int(name.removeprefix(f"{BASE_RUN}_s"))
        else:
            continue
        if (Path(scores_root) / name).is_dir():
            runs[seed] = name
    return dict(sorted(runs.items()))


def mean_sd(df: pd.DataFrame, keys: list[str], values: list[str]) -> pd.DataFrame:
    """Mean, standard deviation, min and max of ``values`` across seeds, per group."""
    g = df.groupby(keys, sort=False)[values]
    out = g.agg(["mean", "std", "min", "max"])
    out.columns = [f"{v}_{stat}" for v, stat in out.columns]
    return out.reset_index().assign(seeds=g.size().to_numpy())


def paired_mask_effect(p2: pd.DataFrame, value: str) -> pd.DataFrame:
    """Per fold: masked minus unmasked ``value`` within each seed, then summarised."""
    wide = p2.pivot_table(index=["fold", "seed"], columns="mask", values=value).reset_index()
    wide["diff"] = wide["corners"] - wide["none"]
    g = wide.groupby("fold")["diff"]
    return pd.DataFrame({
        "metric": value,
        "mean_diff": g.mean(), "sd_diff": g.std(), "min_diff": g.min(), "max_diff": g.max(),
        "seeds_mask_better": g.apply(lambda d: int((d > 0).sum())), "seeds": g.size(),
    }).reset_index()


def collect(runs: dict[int, str], repeats: int = 1000) -> dict[str, pd.DataFrame]:
    p2, p4, p56_meth, p56_det = [], [], [], []
    for seed, run in runs.items():
        p2.append(pd.read_csv(Path("results") / run / "summary.csv").assign(seed=seed))
        p4.append(loso.summarize(loso.run(run, repeats=repeats, seed=0)).assign(seed=seed))
        methods, detection, _ = real_shift.run(run, repeats=repeats, seed=0)
        meth, det = real_shift.summarize(methods, detection)
        p56_meth.append(meth.assign(seed=seed))
        p56_det.append(det.assign(seed=seed))
        print(f"seed {seed}: collected ({run})", flush=True)
    return {
        "p2": pd.concat(p2, ignore_index=True),
        "p4": pd.concat(p4, ignore_index=True),
        "p56_methods": pd.concat(p56_meth, ignore_index=True),
        "p56_detection": pd.concat(p56_det, ignore_index=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate results across training seeds")
    parser.add_argument("--repeats", type=int, default=1000)
    args = parser.parse_args()

    runs = find_runs()
    print(f"found seeds: {list(runs)}")
    data = collect(runs, args.repeats)
    out = Path("results/multi_seed")
    out.mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        df.to_csv(out / f"{name}_per_seed.csv", index=False)

    auc_cols = ["frozen_auc_source_test", "frozen_auc_target", "ft_auc_source_test", "ft_auc_target"]
    p2 = mean_sd(data["p2"], ["fold", "mask"], auc_cols)
    mask = pd.concat([paired_mask_effect(data["p2"], c) for c in ("ft_auc_target", "ft_auc_source_test")])
    p4 = mean_sd(data["p4"], ["source", "mask", "evaluated_on", "method"],
                 ["violation_rate", "mean_sensitivity", "mean_specificity", "trivial_rate"])
    p56 = mean_sd(data["p56_methods"], ["source", "mask", "method", "budget"],
                  ["violation_rate", "mean_sensitivity", "mean_specificity", "flags_everyone_rate"])
    det = mean_sd(data["p56_detection"], ["source", "mask", "batch_size"], ["power", "false_alarm"])
    for name, df in {"p2": p2, "mask_effect": mask, "p4": p4, "p56_methods": p56, "p56_detection": det}.items():
        df.to_csv(out / f"{name}_summary.csv", index=False)

    fmt = {"index": False, "float_format": lambda v: f"{v:.3f}"}
    print("\nP2 AUC across seeds (mean, sd)")
    print(p2[["fold", "mask", "seeds"] + [c for c in p2.columns if c.endswith(("_mean", "_std"))]].to_string(**fmt))
    print("\nCorner mask, paired within seed (masked minus unmasked)")
    print(mask.to_string(**fmt))
    key = p4[(p4.source == "shenzhen") & (p4.method.isin(["marginal", "certified", "fixed"]))]
    print("\nP4, trained on Shenzhen (mean, sd across seeds)")
    print(key[["mask", "evaluated_on", "method", "violation_rate_mean", "violation_rate_std",
               "mean_specificity_mean", "mean_specificity_std"]].to_string(**fmt))
    key = p56[(p56.source == "shenzhen") & (p56.method.isin(["source_certified", "label_free", "detect_defer"])
                                            | ((p56.method == "few_label_certified") & (p56.budget >= 50)))]
    print("\nP5/P6 repair, trained on Shenzhen (mean, sd across seeds)")
    print(key[["mask", "method", "budget", "violation_rate_mean", "violation_rate_std",
               "mean_specificity_mean", "mean_specificity_std"]].to_string(**fmt))
    print("\nP5 detection power, trained on Shenzhen")
    print(det[det.source == "shenzhen"][["mask", "batch_size", "power_mean", "power_std",
                                         "false_alarm_mean"]].to_string(**fmt))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
