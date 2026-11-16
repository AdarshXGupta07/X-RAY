"""Tests for the training loop, using a tiny model on CPU so they run in seconds."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tbshift.training.train import TrainConfig, predict_logits, seed_everything, train_model


def _tiny_model():
    return nn.Sequential(nn.Flatten(), nn.Linear(2 * 4 * 4, 16), nn.ReLU(), nn.Linear(16, 1))


def _data(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = (torch.rand(n, generator=g) < 0.5).float()
    x = torch.randn(n, 2, 4, 4, generator=g) + 1.5 * y.view(-1, 1, 1, 1)  # learnable signal
    return TensorDataset(x, y)


def test_loop_learns_a_learnable_problem():
    seed_everything(0)
    model = _tiny_model()
    loader = DataLoader(_data(), batch_size=16, shuffle=True)
    result = train_model(model, loader, DataLoader(_data(seed=1), batch_size=32),
                         TrainConfig(epochs=15, lr=1e-2, patience=15, log_every_epoch=False), device="cpu")
    assert result.best_val_auc > 0.95
    assert result.history[-1]["train_loss"] < result.history[0]["train_loss"]


def test_can_overfit_eight_examples():
    # The classic sanity check: if a model cannot memorise 8 examples, the loop is broken.
    seed_everything(1)
    data = _data(n=8, seed=2)
    loader = DataLoader(data, batch_size=8)
    model = _tiny_model()
    result = train_model(model, loader, loader,
                         TrainConfig(epochs=200, lr=1e-2, patience=200, log_every_epoch=False), device="cpu")
    # Check the training loss, not the restored weights: validation AUC hits 1.0
    # within a few epochs, so "best epoch" is early by design.
    assert result.history[-1]["train_loss"] < 0.05


def test_best_epoch_weights_are_restored():
    seed_everything(2)
    model = _tiny_model()
    val = DataLoader(_data(seed=3), batch_size=64)
    result = train_model(model, DataLoader(_data(), batch_size=16), val,
                         TrainConfig(epochs=6, lr=1e-2, patience=6, log_every_epoch=False), device="cpu")
    from tbshift.metrics import auc_roc

    z, y = predict_logits(model, val, "cpu")
    assert np.isclose(auc_roc(y, z), result.best_val_auc)


def test_seeding_makes_runs_repeatable():
    def run():
        seed_everything(5)
        m = _tiny_model()
        train_model(m, DataLoader(_data(), batch_size=16, shuffle=True), DataLoader(_data(seed=1), batch_size=64),
                    TrainConfig(epochs=3, lr=1e-2, log_every_epoch=False), device="cpu")
        return predict_logits(m, DataLoader(_data(seed=4), batch_size=64), "cpu")[0]

    np.testing.assert_allclose(run(), run())
