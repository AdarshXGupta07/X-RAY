"""Phase P9: regenerate every figure and table in the paper with one command.

    python -m tbshift.experiments.paper_assets            # from results/ CSVs (seconds)
    python -m tbshift.experiments.paper_assets --rerun    # first recompute those CSVs
                                                          # from the cached scores (~6 min CPU)

Writes figures to ``paper/figures/`` (SVG for GitHub, PDF for LaTeX) and
rewrites every block in ``paper/paper.md`` that sits between
``<!-- table:NAME -->`` and ``<!-- /table:NAME -->``, so the numbers in the
paper's tables can never drift from the results. Only aggregate numbers are
written: no images, no per-patient values.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SOURCE, TARGET, MASK = "shenzhen", "montgomery", "none"   # the main fold
TARGET_SENS = 0.90
ALPHA = 0.10

# Categorical slots 1-3 of the dataviz reference palette (validated for CVD as a
# set of three), plus neutral ink for reference lines. Light mode only: print.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.6, "lines.markersize": 5,
    "svg.hashsalt": "tb-recertify", "svg.fonttype": "path",
})


# --------------------------------------------------------------------------- data

def load(results: Path) -> dict[str, pd.DataFrame]:
    ms = results / "multi_seed"
    return {
        "p2": pd.read_csv(ms / "p2_per_seed.csv"),
        "p4": pd.read_csv(ms / "p4_per_seed.csv"),
        "det": pd.read_csv(ms / "p56_detection_summary.csv"),
        "det_seed": pd.read_csv(ms / "p56_detection_per_seed.csv"),
        "methods": pd.read_csv(ms / "p56_methods_summary.csv"),
        "methods_seed": pd.read_csv(ms / "p56_methods_per_seed.csv"),
        "mask": pd.read_csv(ms / "mask_effect_summary.csv"),
        "p7": pd.read_csv(results / "p7_degradations" / f"train_{SOURCE}__{MASK}.csv"),
        "validity": pd.read_csv(results / "validity_simulation.csv"),
    }


def _main_fold(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["source"] == SOURCE) & (df["mask"] == MASK)]


# --------------------------------------------------------------------------- figures

def _save(fig, out: Path, name: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("svg", "pdf"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", metadata={"Date": None} if ext == "svg" else
                    {"CreationDate": None, "ModDate": None})
        paths.append(p)
    plt.close(fig)
    return paths


def _target_line(ax, y, label, x_text):
    ax.axhline(y, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)
    ax.text(x_text, y, f" {label}", color=MUTED, fontsize=7.5, va="bottom")


def fig_validity(d, out: Path):
    """S1: in-distribution check of the two thresholds (synthetic scores, no shift)."""
    v = d["validity"]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for method, color, label in [("marginal", ORANGE, "Marginal conformal"),
                                 ("certified", BLUE, "Certified (PAC)")]:
        s = v[(v.method == method) & (v.trivial_rate < 1)].sort_values("n_pos")
        ax.plot(s.n_pos, s.violation_rate, marker="o", color=color, label=label)
    _target_line(ax, ALPHA, "allowed: 10%", 160)
    ax.set_xscale("log")
    ax.set_xticks([9, 22, 50, 100, 200], ["9", "22", "50", "100", "200"])
    ax.set_xlabel("TB-positive calibration cases")
    ax.set_ylabel("Violation rate\n(sensitivity < 90%)")
    ax.set_ylim(0, 0.55)
    ax.legend(loc="center right")
    return _save(fig, out, "s1_validity")


def fig_sensitivity(d, out: Path):
    """F2: achieved sensitivity of the certified threshold per training seed."""
    p4 = _main_fold(d["p4"])
    p4 = p4[p4.method == "certified"]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    offsets = {"source_test": -0.12, "target": 0.12}
    names = {"source_test": f"Source ({SOURCE.title()}) control",
             "target": f"New site ({TARGET.title()})"}
    for where, color in [("source_test", MUTED), ("target", BLUE)]:
        s = p4[p4.evaluated_on == where].sort_values("seed")
        x = s.seed + offsets[where]
        ax.vlines(x, s.sens_p05, s.mean_sensitivity, color=color, linewidth=1.4, alpha=0.7)
        ax.plot(x, s.mean_sensitivity, "o", color=color, label=names[where], markersize=6)
        ax.plot(x, s.sens_p05, "_", color=color, markersize=8, markeredgewidth=1.4)
    _target_line(ax, TARGET_SENS, "promise: 90%", -0.48)
    bad = p4[(p4.evaluated_on == "target") & (p4.violation_rate > 0.5)]
    for _, r in bad.iterrows():
        ax.annotate(f"below 90% in\n{r.violation_rate:.0%} of draws",
                    (r.seed + 0.12, r.mean_sensitivity), xytext=(8, 0),
                    textcoords="offset points", fontsize=7.5, color=INK, va="center")
    ax.set_xticks(range(5), [f"seed {i}" for i in range(5)])
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylabel("Sensitivity\n(dot: mean, tick: 5th percentile)")
    ax.set_ylim(min(0.6, p4.sens_p05.min() - 0.03), 1.02)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    return _save(fig, out, "f2_sensitivity_by_seed")


def fig_detection(d, out: Path):
    """F3: KS detector firing rate against unlabelled batch size."""
    det = _main_fold(d["det"]).sort_values("batch_size")
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.fill_between(det.batch_size, det.power_min, det.power_max, color=BLUE, alpha=0.15, lw=0)
    ax.plot(det.batch_size, det.power_mean, marker="o", color=BLUE,
            label=f"{TARGET.title()} batch (power)")
    ax.fill_between(det.batch_size, det.false_alarm_min, det.false_alarm_max, color=ORANGE,
                    alpha=0.15, lw=0)
    ax.plot(det.batch_size, det.false_alarm_mean, marker="s", color=ORANGE,
            label=f"{SOURCE.title()} batch (false alarm)")
    _target_line(ax, 0.05, "test level 5%", 52)
    ax.set_xlabel("Unlabelled images in the batch")
    ax.set_ylabel("Share of draws where\nthe detector fires")
    ax.set_xticks(det.batch_size.tolist())
    ax.set_ylim(0, 1.05)
    ax.legend(loc="center right")
    return _save(fig, out, "f3_detection_power")


def fig_label_budget(d, out: Path):
    """F4: violation rate and specificity against the number of new-site labels."""
    m = _main_fold(d["methods"])
    seed_rows = _main_fold(d["methods_seed"])
    pos = (seed_rows[seed_rows.method == "few_label_certified"]
           .groupby("budget").mean_positives_used.mean())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    series = [("few_label_certified", BLUE, "Few-label certified", "o"),
              ("few_label_marginal", ORANGE, "Few-label marginal", "s")]
    for method, color, label, marker in series:
        s = m[m.method == method].sort_values("budget")
        for ax, col in [(a1, "violation_rate"), (a2, "mean_specificity")]:
            ax.fill_between(s.budget, s[f"{col}_min"], s[f"{col}_max"], color=color, alpha=0.15,
                            lw=0)
            ax.plot(s.budget, s[f"{col}_mean"], marker=marker, color=color, label=label)
    lf = m[m.method == "label_free"].iloc[0]
    a1.axhline(lf.violation_rate_mean, color=AQUA, linewidth=1.4, linestyle=(0, (1, 2)),
               label="Label-free (0 labels)")
    a2.axhline(lf.mean_specificity_mean, color=AQUA, linewidth=1.4, linestyle=(0, (1, 2)),
               label="Label-free (0 labels)")
    _target_line(a1, ALPHA, "allowed: 10%", 24)
    a2.annotate("fewer than 22 TB labels:\nthreshold flags everyone", (15, 0.0), xytext=(14, 0.62),
                fontsize=7.5, color=INK, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    budgets = sorted(m[m.method == "few_label_certified"].budget.unique())
    for ax in (a1, a2):
        ax.set_xticks(budgets, [f"{b}\n({pos[b]:.0f} TB)" for b in budgets], fontsize=7.5)
        ax.set_xlabel("Labelled new-site images (TB cases)")
        ax.set_ylim(0, 1.0)
    a1.set_ylabel("Violation rate")
    a2.set_ylabel("Specificity at the new site")
    a1.legend(loc="upper left")
    return _save(fig, out, "f4_label_budget")


def fig_degradations(d, out: Path):
    """F5: controlled degradations of in-distribution images."""
    p7 = d["p7"]
    clean = p7[p7.degradation == "clean"]
    kinds = ["blur", "noise", "jpeg", "contrast"]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.8), sharex=True, sharey="row")
    series = [("certified", BLUE, "Source certified", "o"),
              ("label_free", ORANGE, "Label-free", "s"),
              ("few_label_certified", AQUA, "Few-label certified", "^")]
    for j, kind in enumerate(kinds):
        k = pd.concat([clean, p7[p7.degradation == kind]])
        for method, color, label, marker in series:
            s = k[k.method == method].sort_values("severity")
            axes[0, j].plot(s.severity, s.violation_rate, marker=marker, color=color, label=label,
                            markersize=4)
            axes[1, j].plot(s.severity, s.mean_specificity, marker=marker, color=color,
                            markersize=4)
        det = k[k.method == "certified"].sort_values("severity")
        axes[0, j].plot(det.severity, det.detector_fires, color=MUTED, linestyle=(0, (4, 3)),
                        linewidth=1.0, label="Detector fires")
        axes[0, j].axhline(ALPHA, color=MUTED, linewidth=0.8, linestyle=":")
        axes[0, j].set_title(kind.upper() if kind == "jpeg" else kind.title())
        axes[1, j].set_xlabel("Severity")
        axes[1, j].set_xticks(range(6))
    axes[0, 0].set_ylabel("Violation rate\n(dashed: detector)")
    axes[1, 0].set_ylabel("Specificity")
    for ax in axes.flat:
        ax.set_ylim(-0.03, 1.03)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, out, "f5_degradations")


# --------------------------------------------------------------------------- tables

def _md(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def table_datasets(manifests: Path) -> str | None:
    """T1. Built from the local manifests (git-ignored); only counts and averages are written."""
    rows = []
    for site, place in [("shenzhen", "Shenzhen No. 3 People's Hospital, China"),
                        ("montgomery", "Montgomery County TB programme, USA")]:
        f = manifests / f"{site}.csv"
        if not f.exists():
            return None
        m = pd.read_csv(f)
        tb, nm = m[m.label == 1], m[m.label == 0]
        rows.append({
            "Site": site.title(), "Origin": place, "Images": len(m), "TB": len(tb),
            "TB share": f"{len(tb) / len(m):.0%}",
            "Mean age (TB / normal)": f"{tb.age.mean():.1f} / {nm.age.mean():.1f}",
            "Male (TB / normal)": f"{(tb.sex == 'M').mean():.0%} / {(nm.sex == 'M').mean():.0%}",
            "Split train/val/cal/test": "/".join(str((m.split == s).sum())
                                                 for s in ("train", "val", "cal", "test")),
        })
    return _md(pd.DataFrame(rows))


def table_main(d) -> str:
    """T2. Per training seed, trained at the source, tested at the new site."""
    p4 = _main_fold(d["p4"])
    cert = p4[p4.method == "certified"]
    ctrl = cert[cert.evaluated_on == "source_test"].set_index("seed")
    tgt = cert[cert.evaluated_on == "target"].set_index("seed")
    p2 = _main_fold(d["p2"]).set_index("seed")
    ms = _main_fold(d["methods_seed"])
    lf = ms[ms.method == "label_free"].set_index("seed")
    full_budget = ms[ms.method == "few_label_certified"].budget.max()
    fl = ms[(ms.method == "few_label_certified") & (ms.budget == full_budget)].set_index("seed")
    rows = []
    for seed in sorted(tgt.index):
        rows.append({
            "Seed": seed,
            "Source AUC": f"{p2.loc[seed, 'ft_auc_source_test']:.3f}",
            "New-site AUC": f"{p2.loc[seed, 'ft_auc_target']:.3f}",
            "Control viol.": f"{ctrl.loc[seed, 'violation_rate']:.3f}",
            "New-site viol.": f"**{tgt.loc[seed, 'violation_rate']:.3f}**",
            "Sens.": f"{tgt.loc[seed, 'mean_sensitivity']:.3f}",
            "Spec.": f"{tgt.loc[seed, 'mean_specificity']:.3f}",
            "Label-free viol.": f"{lf.loc[seed, 'violation_rate']:.3f}",
            f"Few-label viol. ({full_budget} labels)": f"{fl.loc[seed, 'violation_rate']:.3f}",
            "Few-label spec.": f"{fl.loc[seed, 'mean_specificity']:.3f}",
        })
    return _md(pd.DataFrame(rows))


def table_reverse(d) -> str:
    """T3. The reverse direction and the corner-mask comparison, mean (min-max) over seeds."""
    p2 = d["p2"]
    rows = []
    for (source, mask), g in p2.groupby(["source", "mask"], sort=False):
        rows.append({
            "Trained on": source.title(), "Corner mask": "yes" if mask == "corners" else "no",
            "Source test AUC": _mm(g.ft_auc_source_test), "New-site AUC": _mm(g.ft_auc_target),
            "Frozen-feature baseline, new-site AUC": f"{g.frozen_auc_target.iloc[0]:.3f}",
        })
    return _md(pd.DataFrame(rows).sort_values(["Trained on", "Corner mask"], ascending=[False, True]))


def _mm(s: pd.Series) -> str:
    return f"{s.mean():.3f} ({s.min():.3f}–{s.max():.3f})"


def table_degradations(d) -> str:
    """T4. Selected P7 conditions."""
    p7 = d["p7"].set_index(["degradation", "severity", "method"])
    picks = [("clean", 0), ("blur", 3), ("blur", 5), ("noise", 3), ("jpeg", 5), ("contrast", 4),
             ("contrast", 5)]
    rows = []
    for kind, sev in picks:
        def g(method, col, kind=kind, sev=sev):
            return p7.loc[(kind, sev, method), col]
        rows.append({
            "Condition": "clean" if kind == "clean" else f"{kind}, severity {sev}",
            "Certified viol.": f"{g('certified', 'violation_rate'):.2f}",
            "Certified spec.": f"{g('certified', 'mean_specificity'):.2f}",
            "Label-free viol.": f"{g('label_free', 'violation_rate'):.2f}",
            "Few-label viol.": f"{g('few_label_certified', 'violation_rate'):.2f}",
            "Few-label spec.": f"{g('few_label_certified', 'mean_specificity'):.2f}",
            "Detector fires": f"{g('certified', 'detector_fires'):.2f}",
        })
    return _md(pd.DataFrame(rows))


def fill_tables(paper: Path, tables: dict[str, str | None]) -> list[str]:
    """Replace each marked block in the paper; return the names that were filled."""
    if not paper.exists():
        return []
    text = paper.read_text(encoding="utf-8")
    filled = []
    for name, body in tables.items():
        if body is None:
            continue
        pattern = re.compile(rf"(<!-- table:{name} -->\r?\n)(?:.*?\r?\n)?(<!-- /table:{name} -->)",
                             re.DOTALL)
        text, n = pattern.subn(lambda m, body=body: m.group(1) + body + "\n" + m.group(2), text)
        if n:
            filled.append(name)
    paper.write_text(text, encoding="utf-8")
    return filled


# --------------------------------------------------------------------------- main

RERUN = [
    ["tbshift.experiments.multi_seed"],
    ["tbshift.experiments.p7_degradations", "analyze"],
]


def build(results: Path = Path("results"), out: Path = Path("paper/figures"),
          paper: Path = Path("paper/paper.md"), manifests: Path = Path("data/manifests")) -> dict:
    d = load(results)
    figures = []
    for fn in (fig_validity, fig_sensitivity, fig_detection, fig_label_budget, fig_degradations):
        figures += fn(d, out)
    tables = {
        "t1": table_datasets(manifests),
        "t2": table_main(d),
        "t3": table_reverse(d),
        "t4": table_degradations(d),
    }
    return {"figures": figures, "tables": tables, "filled": fill_tables(paper, tables)}


def main() -> None:
    parser = argparse.ArgumentParser(description="P9: regenerate the paper's figures and tables")
    parser.add_argument("--rerun", action="store_true",
                        help="first recompute the result CSVs from the cached model scores")
    args = parser.parse_args()
    if args.rerun:
        for cmd in RERUN:
            print("running", " ".join(cmd), flush=True)
            subprocess.run([sys.executable, "-m", *cmd], check=True)
    result = build()
    for p in result["figures"]:
        print("wrote", p)
    missing = [k for k, v in result["tables"].items() if v is None]
    print("tables filled in paper:", ", ".join(result["filled"]) or "none")
    if missing:
        print("not built (inputs missing):", ", ".join(missing))


if __name__ == "__main__":
    main()
