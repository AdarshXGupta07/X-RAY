"""DenseNet-121 with a single TB logit (plan phase P2).

One fixed architecture for the whole study, so differences between sites and
methods are not differences between models.

* ``build_densenet121`` replaces ImageNet's 1000-class head with one output:
  the TB logit (log-odds). ``memory_efficient=True`` trades a little speed for
  much less GPU memory, which matters on a 4 GB card.
* ``score_and_embed`` runs the model over a DataLoader once and returns, for
  every image, the logit and the 1024-number feature vector from just before
  the head. Those are what the score cache stores.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121

EMBED_DIM = 1024


def build_densenet121(pretrained: bool = True, memory_efficient: bool = True) -> nn.Module:
    """DenseNet-121 (ImageNet weights if ``pretrained``) with a 1-logit head."""
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = densenet121(weights=weights, memory_efficient=memory_efficient)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model


def embed(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Feature vector before the head (same steps as torchvision's DenseNet.forward)."""
    features = F.relu(model.features(x), inplace=True)
    return torch.flatten(F.adaptive_avg_pool2d(features, (1, 1)), 1)


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze everything except the classifier head."""
    for p in model.features.parameters():
        p.requires_grad = trainable


@torch.no_grad()
def score_and_embed(
    model: nn.Module, loader, device: str | torch.device = "cpu", amp: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits, embeddings) for every image in ``loader``, in loader order."""
    model.eval().to(device)
    use_amp = amp and torch.device(device).type == "cuda"
    logits, embeddings = [], []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            e = embed(model, x)
            z = model.classifier(e)
        embeddings.append(e.float().cpu())
        logits.append(z.float().squeeze(1).cpu())
    return torch.cat(logits).numpy(), torch.cat(embeddings).numpy()
