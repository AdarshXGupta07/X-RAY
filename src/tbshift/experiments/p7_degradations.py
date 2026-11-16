"""Phase P7: controlled "new hospitals" made by degrading images.

Two sites gave a shift that hurt specificity but not sensitivity. To see
what happens when a shift DOES hurt sensitivity, we take images the model
handles well (the held-out Shenzhen pool, which it never trained on), damage
them in one controlled way, and rerun the promise / detection / repair study.
Because only the degradation changes, any effect is caused by it.

Stage 1, ``score`` (GPU, about a minute):
    Load the P2 Shenzhen model, score the held-out pool clean (severity 0) and
    under every degradation and severity, and save each as a score file.
    The clean scores are checked against the P2 cache as a sanity test.

Stage 2, ``analyze`` (CPU):
    For each degradation and severity, repeat many times:
      * a stratified half of the CLEAN pool calibrates the thresholds
        (as a hospital would, before anything changed);
      * the other half, DEGRADED, is split into an adaptation quarter (what
        repair methods may look at) and an evaluation quarter (where every
        method is scored);
      * methods: fixed (logit >= 0), marginal, certified, label-free
        (quantile matching on the unlabelled adaptation quarter),
        detect-and-defer (KS on the adaptation quarter), and few-label
        certified/marginal using all adaptation-quarter labels.

Run:
    python -m tbshift.experiments.p7_degradations score
    python -m tbshift.experiments.p7_degradations analyze [--repeats 1000]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tbshift.conformal import certified_threshold, conformal_threshold
from tbshift.degradations import DEGRADATIONS, SEVERITIES, degrade
from tbshift.experiments.loso import stratified_half
from tbshift.metrics import sensitivity, specificity, violation_rate
from tbshift.recalibration import few_label_threshold, label_free_threshold
from tbshift.shift import detect_score_shift

ALPHA = 0.10
DELTA = 0.10
OUT_ROOT = Path("artifacts/scores/p7_degradations")


class DegradeFn:
    """Picklable per-image degradation with a reproducible per-image seed."""

    def __init__(self, name: str, severity: int, seed: int = 0):
        self.name, self.severity, self.seed = name, severity, seed

    def __call__(self, image: np.ndarray, index: int) -> np.ndarray:
        return degrade(image, self.name, self.severity, seed=self.seed * 1_000_003 + index)


def condition_name(name: str, severity: int) -> str:
    return "clean" if severity == 0 else f"{name}_s{severity}"


# --------------------------------------------------------------------- stage 1


def score(fold: str = "train_shenzhen", mask: str = "none", config: str = "configs/p2_densenet.yaml",
          seed: int = 0) -> None:
    import torch
    import yaml
    from torch.utils.data import DataLoader

    from tbshift.data.dataset import CXRDataset
    from tbshift.models.densenet import build_densenet121, score_and_embed

    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    corner = cfg["corner_masks"][mask]
    corner = tuple(corner) if corner else None
    site = fold.removeprefix("train_")
    run = cfg["run_name"]

    model = build_densenet121(pretrained=False)
    state = torch.load(Path("artifacts/checkpoints") / run / f"{fold}__{mask}.pt", map_location="cpu")
    model.load_state_dict(state)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    manifest = pd.read_csv(f"data/manifests/{site}.csv")
    pool = manifest[manifest["split"].isin(["cal", "test"])].reset_index(drop=True)
    out = OUT_ROOT / f"{fold}__{mask}"
    out.mkdir(parents=True, exist_ok=True)

    conditions = [("clean", 0)] + [(d, s) for d in DEGRADATIONS for s in SEVERITIES if s > 0]
    for name, sev in conditions:
        fn = None if sev == 0 else DegradeFn(name, sev, seed)
        ds = CXRDataset(pool, image_size=cfg["image_size"], corner_mask=corner, image_fn=fn)
        loader = DataLoader(ds, batch_size=cfg["batch_size"], num_workers=0)
        logits, emb = score_and_embed(model, loader, device)
        np.savez_compressed(
            out / f"{condition_name(name, sev)}.npz",
            logits=logits.astype(np.float32), labels=pool["label"].to_numpy().astype(np.int8),
            image_ids=pool["image_id"].to_numpy().astype(str), splits=pool["split"].to_numpy().astype(str),
            embeddings=emb.astype(np.float16),
        )
        print(f"scored {condition_name(name, sev):12s} mean logit TB {logits[pool.label == 1].mean():+.2f}, "
              f"normal {logits[pool.label == 0].mean():+.2f}", flush=True)

    # Sanity: the clean rescoring must reproduce the P2 score cache.
    with np.load(Path("artifacts/scores") / run / f"{fold}__{mask}" / f"{site}.npz") as p2:
        ref = dict(zip(p2["image_ids"], p2["logits"], strict=True))
    with np.load(out / "clean.npz") as clean:
        diff = max(abs(float(z) - float(ref[i])) for i, z in zip(clean["image_ids"], clean["logits"], strict=True))
    print(f"max |clean - P2 cache| logit difference: {diff:.4f}")


# --------------------------------------------------------------------- stage 2


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        return f["logits"], f["labels"].astype(int)


def analyze_condition(z_clean, z_deg, y, repeats: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for r in range(repeats):
        cal = stratified_half(y, rng)
        rest = np.flatnonzero(~cal)
        adapt_local = stratified_half(y[rest], rng)
        adapt, evl = rest[adapt_local], rest[~adapt_local]

        z_cal, y_cal = z_clean[cal], y[cal]
        tau_cert = certified_threshold(z_cal[y_cal == 1], ALPHA, DELTA)
        fired = detect_score_shift(z_cal, z_deg[adapt]).shifted
        methods = {
            "fixed": 0.0,
            "marginal": conformal_threshold(z_cal[y_cal == 1], ALPHA),
            "certified": tau_cert,
            "label_free": label_free_threshold(tau_cert, z_cal, z_deg[adapt]),
            "detect_defer": -math.inf if fired else tau_cert,
            "few_label_marginal": few_label_threshold(z_deg[adapt], y[adapt], ALPHA),
            "few_label_certified": few_label_threshold(z_deg[adapt], y[adapt], ALPHA, DELTA),
        }
        for method, tau in methods.items():
            rows.append({
                "repeat": r, "method": method, "fired": fired, "flags_everyone": tau == -math.inf,
                "sensitivity": sensitivity(y[evl], z_deg[evl], tau),
                "specificity": specificity(y[evl], z_deg[evl], tau),
                "adapt_tb": int(y[adapt].sum()), "eval_tb": int(y[evl].sum()),
            })
    return pd.DataFrame(rows)


def analyze(fold_dir: Path, repeats: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z_clean, y = _load(fold_dir / "clean.npz")
    summaries = []
    for path in sorted(fold_dir.glob("*.npz")):
        cond = path.stem
        z_deg, y_deg = _load(path)
        if not np.array_equal(y, y_deg):
            raise ValueError(f"{path.name}: labels do not match the clean file")
        per = analyze_condition(z_clean, z_deg, y, repeats, rng)
        name, sev = ("clean", 0) if cond == "clean" else (cond.rsplit("_s", 1)[0], int(cond.rsplit("_s", 1)[1]))
        g = per.groupby("method", sort=False)
        s = g.agg(mean_sensitivity=("sensitivity", "mean"), mean_specificity=("specificity", "mean"),
                  detector_fires=("fired", "mean"), flags_everyone_rate=("flags_everyone", "mean"),
                  adapt_tb=("adapt_tb", "median"), eval_tb=("eval_tb", "median")).reset_index()
        s.insert(1, "violation_rate", g["sensitivity"].apply(lambda v: violation_rate(v, 1 - ALPHA)).to_numpy())
        s.insert(0, "severity", sev)
        s.insert(0, "degradation", name)
        s.insert(2, "mean_logit_shift_tb", float(z_deg[y == 1].mean() - z_clean[y == 1].mean()))
        summaries.append(s)
    return pd.concat(summaries, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="P7 degradation study")
    parser.add_argument("stage", choices=["score", "analyze"])
    parser.add_argument("--fold", default="train_shenzhen")
    parser.add_argument("--mask", default="none")
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.stage == "score":
        score(args.fold, args.mask, seed=args.seed)
        return
    summary = analyze(OUT_ROOT / f"{args.fold}__{args.mask}", args.repeats, args.seed)
    out = Path("results/p7_degradations")
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / f"{args.fold}__{args.mask}.csv", index=False)
    pivot = summary.pivot_table(index=["degradation", "severity"], columns="method", values="violation_rate",
                                sort=False)
    spec = summary.pivot_table(index=["degradation", "severity"], columns="method", values="mean_specificity",
                               sort=False)
    extra = summary[summary.method == "certified"].set_index(["degradation", "severity"])[
        ["mean_logit_shift_tb", "detector_fires"]]
    fmt = {"float_format": lambda v: f"{v:.2f}"}
    print(f"{args.fold}, mask {args.mask}: {args.repeats} repeats; violation = sensitivity below 90%\n")
    print("VIOLATION RATE\n" + pivot.join(extra).to_string(**fmt))
    print("\nSPECIFICITY\n" + spec.to_string(**fmt))
    print(f"\nwrote {out / (args.fold + '__' + args.mask + '.csv')}")


if __name__ == "__main__":
    main()
