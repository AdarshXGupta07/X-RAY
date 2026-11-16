"""Shortcut checks: can the label be predicted from things that are NOT the disease?

If a simple model can tell TB from normal using only image shape, image
corners (where text markers live) or demographics, a deep network can learn
the same cue instead of TB, and its accuracy would not transfer to a new
hospital. Each check reports a cross-validated AUC with a bootstrap interval.

Within each site, TB vs normal from:

    geometry      landscape flag, log width, log height, aspect ratio (raw images)
    corners       four 48x48 corner patches of the cached 512 px image, shrunk to 12x12
    top_left ... bottom_right   each corner on its own (markers sit in the top corners)
    upper_lungs   two 96x96 patches over the upper lung fields (positive control)
    lower_lungs   two 96x96 patches over the lower lung fields (positive control)
    demographics  age and sex (context: the image itself reveals both)

Across sites (both classes pooled), site from:

    site_thumb    the whole cached image shrunk to 16x16

How to read it: about 0.5 means no usable cue. The lung patches should be
clearly above 0.5 because TB is in the lungs. A corner that predicts the label
as well as the lungs is a shortcut the network can learn instead of TB.

Run:  python -m tbshift.experiments.shortcut_checks [--out results/shortcut_checks.csv]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tbshift.metrics import auc_roc, bootstrap_ci

SITES = ("montgomery", "shenzhen")
CORNER = 48
SHRINK = 12


def cv_auc(x: np.ndarray, y: np.ndarray, seed: int = 0, n_boot: int = 1000):
    """5-fold out-of-fold AUC of a standardised, L2-regularised logistic regression."""
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p = cross_val_predict(model, x, y, cv=folds, method="predict_proba")[:, 1]
    return bootstrap_ci(auc_roc, y, p, n_boot=n_boot, seed=seed)


def _patch(a: np.ndarray, top: int, left: int, size: int) -> np.ndarray:
    small = Image.fromarray(a[top : top + size, left : left + size]).resize(
        (SHRINK, SHRINK), Image.Resampling.BOX
    )
    return np.asarray(small, dtype=np.float32).ravel() / 255.0


CORNER_NAMES = ("top_left", "top_right", "bottom_left", "bottom_right")
# (top, left) of the lung-field patches as a fraction of the image side.
UPPER_LUNGS = ((0.18, 0.15), (0.18, 0.62))
LOWER_LUNGS = ((0.45, 0.15), (0.45, 0.62))


def image_features(paths) -> dict[str, np.ndarray]:
    out: dict[str, list] = {k: [] for k in (*CORNER_NAMES, "upper_lungs", "lower_lungs", "thumb")}
    for p in paths:
        with Image.open(p) as img:
            a = np.asarray(img.convert("L"))
        n, c = a.shape[0], CORNER
        for name, (top, left) in zip(CORNER_NAMES, ((0, 0), (0, n - c), (n - c, 0), (n - c, n - c))):
            out[name].append(_patch(a, top, left, c))
        out["upper_lungs"].append(np.concatenate([_patch(a, int(t * n), int(s * n), 96) for t, s in UPPER_LUNGS]))
        out["lower_lungs"].append(np.concatenate([_patch(a, int(t * n), int(s * n), 96) for t, s in LOWER_LUNGS]))
        out["thumb"].append(
            np.asarray(Image.fromarray(a).resize((16, 16), Image.Resampling.BOX), dtype=np.float32).ravel() / 255.0
        )
    feats = {k: np.array(v) for k, v in out.items()}
    feats["corners"] = np.hstack([feats[k] for k in CORNER_NAMES])
    return feats


def run(manifest_dir: str = "data/manifests", seed: int = 0) -> pd.DataFrame:
    rows = []
    frames = []
    for site in SITES:
        df = pd.read_csv(f"{manifest_dir}/{site}.csv")
        frames.append(df)
        y = df["label"].to_numpy()
        feats = image_features(df["cached_path"])
        geometry = np.column_stack(
            [
                (df.width > df.height).astype(float),
                np.log(df.width),
                np.log(df.height),
                df.width / df.height,
            ]
        )
        known = df["age"].notna() & df["sex"].notna()
        demo = np.column_stack([df.loc[known, "age"], df.loc[known, "sex"].eq("M").astype(float)])
        checks = {
            "geometry": (geometry, y),
            "corners": (feats["corners"], y),
            **{name: (feats[name], y) for name in CORNER_NAMES},
            "upper_lungs": (feats["upper_lungs"], y),
            "lower_lungs": (feats["lower_lungs"], y),
            "demographics": (demo, y[known.to_numpy()]),
        }
        for name, (x, yy) in checks.items():
            r = cv_auc(x, yy, seed=seed)
            rows.append({"check": name, "site": site, "n": len(yy), "auc": r.point, "lo": r.lower, "hi": r.upper})

    both = pd.concat(frames, ignore_index=True)
    thumbs = image_features(both["cached_path"])["thumb"]
    site_y = (both["site"] == "shenzhen").astype(int).to_numpy()
    r = cv_auc(thumbs, site_y, seed=seed)
    rows.append({"check": "site_thumb", "site": "both", "n": len(site_y), "auc": r.point, "lo": r.lower, "hi": r.upper})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Shortcut checks on the manifests")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args()
    df = run(seed=args.seed)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
