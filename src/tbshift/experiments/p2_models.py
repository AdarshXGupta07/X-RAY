"""Phase P2: train the fixed model per fold and write the score cache.

For every leave-one-site-out fold (train on one site, the other is unseen)
and every corner-mask setting in the config:

1. Frozen baseline: ImageNet DenseNet-121 features + logistic regression
   (C picked on the source validation split). Fast, and easy to explain.
2. Fine-tune: warm up the new head with the backbone frozen, then train the
   whole network; keep the epoch with the best source validation AUC.
3. Score cache: run the fine-tuned model over EVERY image of EVERY site and
   save logits, labels, ids, splits and 1024-d embeddings. All later phases
   (P3-P7) read only these files.

Output (all git-ignored):
    artifacts/checkpoints/<run>/<fold>__<mask>.pt
    artifacts/scores/<run>/<fold>__<mask>/<site>.npz
    results/<run>/summary.csv, config.yaml, run_info.json, logs

Run:  python -m tbshift.experiments.p2_models --config configs/p2_densenet.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from tbshift.data.dataset import CXRDataset
from tbshift.metrics import auc_roc
from tbshift.models.densenet import build_densenet121, score_and_embed, set_backbone_trainable
from tbshift.training.train import TrainConfig, seed_everything, train_model


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 - not a git checkout
        return "unknown"


def _loader(frame, cfg, mask, train=False, shuffle=False):
    ds = CXRDataset(frame, image_size=cfg["image_size"], train=train, corner_mask=mask)
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle,
                      num_workers=cfg["num_workers"], pin_memory=torch.cuda.is_available())


def frozen_baseline(frames, source, target, cfg, mask, device, log):
    """ImageNet features + logistic regression. Returns AUCs on source test and target."""
    model = build_densenet121(pretrained=True)
    feats = {}
    for site, df in frames.items():
        _, emb = score_and_embed(model, _loader(df, cfg, mask), device)
        feats[site] = emb
    src = frames[source]
    pick = {s: src["split"].to_numpy() == s for s in ("train", "val", "test")}
    y = src["label"].to_numpy()
    best = (-1.0, None)
    for c in cfg["frozen_baseline"]["C_grid"]:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
        clf.fit(feats[source][pick["train"]], y[pick["train"]])
        val_auc = auc_roc(y[pick["val"]], clf.decision_function(feats[source][pick["val"]]))
        if val_auc > best[0]:
            best = (val_auc, clf, c)
    _, clf, c = best
    out = {
        "frozen_C": c,
        "frozen_auc_source_test": auc_roc(y[pick["test"]], clf.decision_function(feats[source][pick["test"]])),
        "frozen_auc_target": auc_roc(frames[target]["label"].to_numpy(), clf.decision_function(feats[target])),
    }
    log(f"  frozen baseline (C={c}): source test AUC {out['frozen_auc_source_test']:.3f}, "
        f"target AUC {out['frozen_auc_target']:.3f}")
    del model
    return out


def fine_tune(frames, source, cfg, mask, device, log):
    ft = cfg["fine_tune"]
    src = frames[source]
    train_loader = _loader(src[src.split == "train"], cfg, mask, train=True, shuffle=True)
    val_loader = _loader(src[src.split == "val"], cfg, mask)

    model = build_densenet121(pretrained=True)
    log("  head warm-up (backbone frozen)")
    set_backbone_trainable(model, False)
    train_model(model, train_loader, val_loader,
                TrainConfig(epochs=ft["head_epochs"], lr=ft["head_lr"], patience=ft["head_epochs"],
                            amp=ft["amp"]), device, log)
    log("  full fine-tune")
    set_backbone_trainable(model, True)
    result = train_model(model, train_loader, val_loader,
                         TrainConfig(epochs=ft["epochs"], lr=ft["lr"], weight_decay=ft["weight_decay"],
                                     patience=ft["patience"], amp=ft["amp"]), device, log)
    return model, result


def write_score_cache(model, frames, cfg, mask, device, out_dir: Path) -> dict[str, np.ndarray]:
    out_dir.mkdir(parents=True, exist_ok=True)
    logits = {}
    for site, df in frames.items():
        z, emb = score_and_embed(model, _loader(df, cfg, mask), device)
        np.savez_compressed(
            out_dir / f"{site}.npz",
            logits=z.astype(np.float32),
            labels=df["label"].to_numpy().astype(np.int8),
            image_ids=df["image_id"].to_numpy().astype(str),
            splits=df["split"].to_numpy().astype(str),
            embeddings=emb.astype(np.float16),
        )
        logits[site] = z
    return logits


def save_summary(path: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Merge rows into summary.csv, replacing any existing row for the same fold and mask."""
    if path.exists():
        old = pd.read_csv(path)
        keys = ["fold", "mask"]
        old = old[~old.set_index(keys).index.isin(new_rows.set_index(keys).index)]
        new_rows = pd.concat([old, new_rows], ignore_index=True)
    new_rows = new_rows.sort_values(["fold", "mask"]).reset_index(drop=True)
    new_rows.to_csv(path, index=False)
    return new_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="P2: train per fold and write the score cache")
    parser.add_argument("--config", default="configs/p2_densenet.yaml")
    parser.add_argument("--only-fold", help="train on this site only (for quick checks)")
    parser.add_argument("--only-mask", help="run only this corner-mask setting")
    parser.add_argument("--seed", type=int, help="override the training seed in the config")
    parser.add_argument("--run-name", help="override run_name (default: config value, plus _s<seed> "
                                           "when --seed differs from the config)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.seed is not None and args.seed != cfg["seed"]:
        # Only training randomness changes (initial head weights, batch order,
        # augmentation). The data splits come from the manifests and stay fixed.
        cfg["run_name"] = f"{cfg['run_name']}_s{args.seed}"
        cfg["seed"] = args.seed
    if args.run_name:
        cfg["run_name"] = args.run_name
    run = cfg["run_name"]
    results_dir = Path("results") / run
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    log_file = open(results_dir / "log.txt", "a", encoding="utf-8")  # noqa: SIM115

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    (results_dir / "run_info.json").write_text(json.dumps({
        "git_commit": _git_commit(), "seed": cfg["seed"], "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")

    frames = {s: pd.read_csv(f"data/manifests/{s}.csv") for s in cfg["sites"]}
    rows = []
    for source in cfg["sites"]:
        if args.only_fold and source != args.only_fold:
            continue
        target = next(s for s in cfg["sites"] if s != source)
        fold = f"train_{source}"
        for mask_name, mask in cfg["corner_masks"].items():
            if args.only_mask and mask_name != args.only_mask:
                continue
            mask = tuple(mask) if mask else None
            log(f"\n=== {fold} | mask {mask_name} | target {target} ===")
            seed_everything(cfg["seed"])
            row = {"fold": fold, "source": source, "target": target, "mask": mask_name}
            row.update(frozen_baseline(frames, source, target, cfg, mask, device, log))

            seed_everything(cfg["seed"])
            model, result = fine_tune(frames, source, cfg, mask, device, log)
            tag = f"{fold}__{mask_name}"
            ckpt = Path("artifacts/checkpoints") / run / f"{tag}.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt)
            logits = write_score_cache(model, frames, cfg, mask, device,
                                       Path("artifacts/scores") / run / tag)
            src, tgt = frames[source], frames[target]
            test = src["split"].to_numpy() == "test"
            row.update({
                "ft_best_epoch": result.best_epoch,
                "ft_val_auc": result.best_val_auc,
                "ft_auc_source_test": auc_roc(src["label"].to_numpy()[test], logits[source][test]),
                "ft_auc_target": auc_roc(tgt["label"].to_numpy(), logits[target]),
                "ft_minutes": result.seconds / 60,
            })
            log(f"  fine-tuned (best epoch {result.best_epoch}): source test AUC "
                f"{row['ft_auc_source_test']:.3f}, target AUC {row['ft_auc_target']:.3f}, "
                f"{row['ft_minutes']:.1f} min")
            rows.append(row)
            # Save after every finished model, so a crash later in the run
            # (e.g. running out of RAM) loses only the model in progress.
            summary = save_summary(results_dir / "summary.csv", pd.DataFrame([row]))
            del model
            torch.cuda.empty_cache()

    if rows:
        log("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    log_file.close()


if __name__ == "__main__":
    main()
