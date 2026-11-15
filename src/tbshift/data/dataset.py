"""PyTorch Dataset for cached chest X-rays (plan phase P1).

Web-dev analogy: a Dataset is a tiny API with two endpoints. ``len(ds)`` says
how many records there are; ``ds[i]`` returns record i. The DataLoader then
calls ``ds[i]`` in parallel and stacks the results into batches.

Each item is ``(image, label)``:

* image: float32 tensor, shape (3, image_size, image_size), normalised with
  the ImageNet mean and standard deviation, because the pretrained network
  expects inputs that look like its training data. The single gray channel is
  copied three times to fill the RGB slots.
* label: float32 scalar, 1.0 = TB, 0.0 = normal.

Augmentation (training only) is deliberately mild: small rotations, shifts,
zoom and contrast changes. No horizontal flips: the heart is on the left and
the burned-in "R" markers would be mirrored, creating images that never occur.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _gray_to_rgb(x: torch.Tensor) -> torch.Tensor:
    """Copy the single gray channel into three. A named function, not a lambda,
    so DataLoader worker processes on Windows can pickle it."""
    return x.expand(3, -1, -1)


def build_transform(image_size: int = 224, train: bool = False) -> v2.Compose:
    steps = [v2.ToImage(), v2.Resize((image_size, image_size), antialias=True)]
    if train:
        steps += [
            v2.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05)),
            v2.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    steps += [
        v2.ToDtype(torch.float32, scale=True),  # 0..255 -> 0..1
        v2.Lambda(_gray_to_rgb),  # gray -> 3 identical channels
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return v2.Compose(steps)


class CXRDataset(Dataset):
    """Images and labels for the rows of a manifest (usually one split of one or more sites)."""

    def __init__(
        self,
        frame: pd.DataFrame,
        root: Path | str = ".",
        path_column: str = "cached_path",
        image_size: int = 224,
        train: bool = False,
    ):
        if path_column not in frame.columns:
            raise ValueError(f"frame has no {path_column!r} column; run the preprocess step first")
        self.paths = [Path(root) / p for p in frame[path_column]]
        self.labels = torch.tensor(frame["label"].to_numpy(), dtype=torch.float32)
        self.image_ids = frame["image_id"].tolist() if "image_id" in frame else None
        self.transform = build_transform(image_size, train)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.paths[i]) as img:
            image = self.transform(img.convert("L"))
        return image, self.labels[i]
