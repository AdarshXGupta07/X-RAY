"""Phases P5 and P6 on real data: detect the site shift, then try to repair it.

Reads only the P2 score cache. For each model (fold x corner mask) and each of
many repeats:

Source side (as in P4)
    A class-stratified half of the source site's held-out pool calibrates
    temperature and the certified 90% threshold. The other half is used,
    WITHOUT labels, as the source reference sample for detection and
    quantile matching.

Target side
    The unseen site is split, stratified by class, into an ADAPTATION half and
    an EVALUATION half. Methods may only look at the adaptation half (labels
    only where the method pays for them). Every method is scored on the
    evaluation half, which no method has seen.

RQ2, detection: does a two-sample KS test on scores fire when comparing the
source reference with an unlabelled target batch of size B? The false-alarm
rate uses a same-size batch from the source calibration half instead. A
domain classifier on the 1024-d embeddings gives a second, label-free view.

RQ3, repair (all aim at 90% sensitivity):
    source_certified    no repair (the P4 baseline)
    label_free          quantile-match the whole unlabelled adaptation half
    detect_defer        flag everyone if the KS test fires on the adaptation half
    few_label_*_n       recompute the threshold from n labelled adaptation images

Run:  python -m tbshift.experiments.real_shift [--repeats 1000]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tbshift.conformal import certified_threshold
from tbshift.experiments.loso import stratified_half
from tbshift.metrics import sensitivity, specificity, violation_rate
from tbshift.recalibration import few_label_threshold, label_free_threshold
from tbshift.shift import detect_score_shift, domain_classifier_auc

ALPHA = 0.10
DELTA = 0.10
BATCH_SIZES = (10, 20, 50, 100, 200)
BUDGETS = (10, 20, 50, 100, 200)


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in ("logits", "labels", "splits", "embeddings")}


def _sizes(grid, limit: int) -> list[int]:
    """Grid values that fit, plus the limit itself (use everything available)."""
    sizes = [s for s in grid if s < limit]
    return sizes + [limit]


def one_model(source, target, repeats: int, rng: np.random.Generator):
    pool = np.isin(source["splits"], ("cal", "test"))
    z_src, y_src = source["logits"][pool], source["labels"][pool].astype(int)
    z_tgt, y_tgt = target["logits"], target["labels"].astype(int)

    n_adapt = int(np.sum(~stratified_half(y_tgt, np.random.default_rng(0))))  # size check only
    batch_sizes = _sizes(BATCH_SIZES, n_adapt)
    budgets = _sizes(BUDGETS, n_adapt)

    method_rows, detect_rows = [], []
    for r in range(repeats):
        cal = stratified_half(y_src, rng)
        z_cal, y_cal = z_src[cal], y_src[cal]
        z_ref = z_src[~cal]  # unlabelled source reference
        # No temperature scaling here: every threshold below depends only on the
        # ORDER of the scores, and dividing by T > 0 never changes the order.
        tau = certified_threshold(z_cal[y_cal == 1], ALPHA, DELTA)

        adapt = ~stratified_half(y_tgt, rng)  # True = adaptation half
        z_ad = z_tgt[adapt]  # its labels are only read by the few-label methods
        z_ev, y_ev = z_tgt[~adapt], y_tgt[~adapt]

        # RQ2: detection power on target batches and false alarms on source batches.
        for b in batch_sizes:
            tb = rng.choice(z_ad, size=b, replace=False)
            sb = rng.choice(z_cal, size=min(b, z_cal.size), replace=False)
            detect_rows.append({
                "repeat": r, "batch_size": b,
                "fires_on_target": detect_score_shift(z_ref, tb).shifted,
                "fires_on_source": detect_score_shift(z_ref, sb).shifted,
            })

        # RQ3: repair methods, all scored on the untouched evaluation half.
        fired = detect_score_shift(z_ref, z_ad).shifted
        thresholds = [
            ("source_certified", None, tau, math.nan),
            ("label_free", None, label_free_threshold(tau, z_ref, z_ad), math.nan),
            ("detect_defer", None, -math.inf if fired else tau, math.nan),
        ]
        for n in budgets:
            pick = rng.choice(np.flatnonzero(adapt), size=n, replace=False)
            zs, ys = z_tgt[pick], y_tgt[pick]
            thresholds += [
                ("few_label_marginal", n, few_label_threshold(zs, ys, ALPHA), int(ys.sum())),
                ("few_label_certified", n, few_label_threshold(zs, ys, ALPHA, DELTA), int(ys.sum())),
            ]
        for method, budget, t, n_pos in thresholds:
            method_rows.append({
                "repeat": r, "method": method, "budget": budget, "threshold": t,
                "flags_everyone": t == -math.inf, "positives_used": n_pos,
                "sensitivity": sensitivity(y_ev, z_ev, t), "specificity": specificity(y_ev, z_ev, t),
            })
    return pd.DataFrame(method_rows), pd.DataFrame(detect_rows)


def domain_aucs(source, target, seed: int = 0) -> dict[str, float]:
    """Embedding-space detector: source vs target, and source vs source (null)."""
    pool = np.isin(source["splits"], ("cal", "test"))
    e_src = source["embeddings"][pool].astype(np.float32)
    e_tgt = target["embeddings"].astype(np.float32)
    half = stratified_half(source["labels"][pool].astype(int), np.random.default_rng(seed))
    return {
        "domain_auc_source_vs_target": domain_classifier_auc(e_src, e_tgt, seed=seed),
        "domain_auc_source_vs_source": domain_classifier_auc(e_src[half], e_src[~half], seed=seed),
    }


def run(run_name="p2_densenet", repeats=1000, seed=0, scores_root="artifacts/scores"):
    rng = np.random.default_rng(seed)
    methods, detection, domains = [], [], []
    for fold_dir in sorted((Path(scores_root) / run_name).iterdir()):
        fold, mask = fold_dir.name.split("__")
        src_site = fold.removeprefix("train_")
        tgt_site = next(p.stem for p in fold_dir.glob("*.npz") if p.stem != src_site)
        source, target = load(fold_dir / f"{src_site}.npz"), load(fold_dir / f"{tgt_site}.npz")
        m, d = one_model(source, target, repeats, rng)
        tag = {"source": src_site, "target": tgt_site, "mask": mask}
        methods.append(m.assign(**tag))
        detection.append(d.assign(**tag))
        domains.append({**tag, **domain_aucs(source, target, seed)})
    return pd.concat(methods, ignore_index=True), pd.concat(detection, ignore_index=True), pd.DataFrame(domains)


def summarize(methods: pd.DataFrame, detection: pd.DataFrame):
    keys = ["source", "target", "mask", "method", "budget"]
    m = methods.assign(budget=methods["budget"].fillna(0).astype(int))
    g = m.groupby(keys, sort=False)
    meth = g.agg(
        mean_sensitivity=("sensitivity", "mean"),
        mean_specificity=("specificity", "mean"),
        flags_everyone_rate=("flags_everyone", "mean"),
        mean_positives_used=("positives_used", "mean"),
    ).reset_index()
    meth.insert(5, "violation_rate", g["sensitivity"].apply(lambda s: violation_rate(s, 1 - ALPHA)).to_numpy())
    det = detection.groupby(["source", "target", "mask", "batch_size"], sort=False).agg(
        power=("fires_on_target", "mean"), false_alarm=("fires_on_source", "mean")
    ).reset_index()
    return meth, det


def main() -> None:
    parser = argparse.ArgumentParser(description="P5/P6 on the real score cache")
    parser.add_argument("--run", default="p2_densenet")
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    methods, detection, domains = run(args.run, args.repeats, args.seed)
    meth, det = summarize(methods, detection)
    out = Path("results") / "p5p6_real"
    out.mkdir(parents=True, exist_ok=True)
    meth.to_csv(out / "methods.csv", index=False)
    det.to_csv(out / "detection.csv", index=False)
    domains.to_csv(out / "domain_classifier.csv", index=False)

    fmt = {"float_format": lambda v: f"{v:.3f}"}
    print(f"{args.repeats} repeats per model, seed {args.seed}; target sensitivity 90%, delta {DELTA}\n")
    print("Detection (KS on scores): power on target batches vs false alarms on source batches")
    print(det.to_string(index=False, **fmt))
    print("\nDomain classifier on embeddings (0.5 = cannot tell apart)")
    print(domains.to_string(index=False, **fmt))
    print("\nRepair methods, scored on the untouched half of the unseen site")
    print(meth.to_string(index=False, **fmt))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
