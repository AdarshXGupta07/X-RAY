"""Training loop for a binary TB classifier (plan phase P2).

The loop in one sentence: for each batch, run the model (forward), measure
how wrong it is (loss), compute how each weight should change (backward), and
take a small step (optimizer). After each epoch, check validation AUC and
keep the best weights; stop when it has not improved for ``patience`` epochs.

Mixed precision (``torch.autocast`` + ``GradScaler``) runs most of the maths
in 16-bit floats: about half the GPU memory and faster on this RTX card.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from tbshift.metrics import auc_roc


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so a rerun gives the same splits and order.

    GPU kernels can still differ in the last decimal places between runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


@dataclass
class TrainConfig:
    epochs: int = 25
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 5
    amp: bool = True
    log_every_epoch: bool = True


@dataclass
class TrainResult:
    best_epoch: int
    best_val_auc: float
    history: list[dict] = field(default_factory=list)
    seconds: float = 0.0


@torch.no_grad()
def predict_logits(model: nn.Module, loader, device, amp: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Logits and labels for every item in ``loader``."""
    model.eval()
    use_amp = amp and torch.device(device).type == "cuda"
    zs, ys = [], []
    for x, y in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            z = model(x.to(device, non_blocking=True)).squeeze(1)
        zs.append(z.float().cpu())
        ys.append(y)
    return torch.cat(zs).numpy(), torch.cat(ys).numpy().astype(int)


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg: TrainConfig,
    device: str | torch.device = "cuda",
    log=print,
) -> TrainResult:
    """Train in place; on return ``model`` holds the weights of the best epoch."""
    device = torch.device(device)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_auc, best_epoch, best_state, bad_epochs = -1.0, 0, None, 0
    history = []
    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(y)
            n += len(y)

        z, y_val = predict_logits(model, val_loader, device, cfg.amp)
        val_auc = auc_roc(y_val, z) if 0 < y_val.sum() < len(y_val) else float("nan")
        history.append({"epoch": epoch, "train_loss": total / n, "val_auc": val_auc})
        if cfg.log_every_epoch:
            log(f"  epoch {epoch:2d}  train loss {total / n:.4f}  val AUC {val_auc:.3f}")

        if val_auc > best_auc:
            best_auc, best_epoch, bad_epochs = val_auc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                log(f"  early stop: no val improvement for {cfg.patience} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(best_epoch, best_auc, history, time.time() - start)
