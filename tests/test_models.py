"""Tests for the DenseNet-121 wrapper (random weights: no download, CPU, small images)."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tbshift.models.densenet import (
    EMBED_DIM,
    build_densenet121,
    embed,
    score_and_embed,
    set_backbone_trainable,
)


def _model():
    torch.manual_seed(0)
    return build_densenet121(pretrained=False, memory_efficient=False).eval()


def test_one_logit_per_image():
    out = _model()(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 1)


def test_embedding_feeds_the_head_exactly():
    model, x = _model(), torch.randn(3, 3, 64, 64)
    with torch.no_grad():
        e = embed(model, x)
        assert e.shape == (3, EMBED_DIM)
        assert torch.allclose(model.classifier(e), model(x), atol=1e-5)


def test_score_and_embed_keeps_order_and_shapes():
    model = _model()
    x = torch.randn(5, 3, 64, 64)
    loader = DataLoader(TensorDataset(x, torch.zeros(5)), batch_size=2)
    logits, emb = score_and_embed(model, loader, device="cpu")
    assert logits.shape == (5,) and emb.shape == (5, EMBED_DIM)
    with torch.no_grad():
        np.testing.assert_allclose(logits, model(x).squeeze(1).numpy(), atol=1e-4)


def test_freeze_backbone_leaves_head_trainable():
    model = _model()
    set_backbone_trainable(model, False)
    assert not any(p.requires_grad for p in model.features.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())


def test_save_summary_merges_and_replaces(tmp_path):
    import pandas as pd

    from tbshift.experiments.p2_models import save_summary

    path = tmp_path / "summary.csv"
    save_summary(path, pd.DataFrame([{"fold": "a", "mask": "none", "auc": 0.7}]))
    save_summary(path, pd.DataFrame([{"fold": "b", "mask": "none", "auc": 0.8}]))
    out = save_summary(path, pd.DataFrame([{"fold": "a", "mask": "none", "auc": 0.9}]))  # rerun of a
    assert out.set_index("fold")["auc"].to_dict() == {"a": 0.9, "b": 0.8}
