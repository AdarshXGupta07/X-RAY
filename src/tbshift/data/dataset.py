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

Corner mask (optional): blacks out a rectangle in each corner, where the
burned-in "R", "L" and "PA ERECT" markers sit. The shortcut checks showed
that Montgomery's top-left corner alone predicts TB with AUC 0.895, better
than the lung fields, so the model must be tested with and without it.

Augmentation (training only) is deliberately mild: small rotations, shifts,
zoom and contrast changes. No horizontal flips: the heart is on the left and
the burned-in "R" markers would be mirrored, creating images that never occur.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
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


class CornerMask:
    """Set four corner rectangles to black. A class (not a lambda) so it pickles.

    ``width`` and ``height`` are fractions of the image side, e.g. (0.25, 0.12)
    masks the outer quarter of the width in the top and bottom 12% of rows.
    """

    def __init__(self, width: float = 0.25, height: float = 0.12):
        if not (0 <= width < 0.5 and 0 <= height < 0.5):
            raise ValueError("corner mask fractions must be in [0, 0.5)")
        self.width, self.height = width, height

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        dh, dw = round(self.height * h), round(self.width * w)
        if dh == 0 or dw == 0:
            return x
        x = x.clone()
        x[..., :dh, :dw] = 0
        x[..., :dh, w - dw :] = 0
        x[..., h - dh :, :dw] = 0
        x[..., h - dh :, w - dw :] = 0
        return x

    def __repr__(self) -> str:
        return f"CornerMask(width={self.width}, height={self.height})"


def build_transform(
    image_size: int = 224, train: bool = False, corner_mask: tuple[float, float] | None = None
) -> v2.Compose:
    steps = [v2.ToImage(), v2.Resize((image_size, image_size), antialias=True)]
    if train:
        steps += [
            v2.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05)),
            v2.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    steps += [v2.ToDtype(torch.float32, scale=True)]  # 0..255 -> 0..1
    if corner_mask is not None:
        # After augmentation, so a shifted marker cannot slide out from under the mask.
        steps += [v2.Lambda(CornerMask(*corner_mask))]
    steps += [
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
        corner_mask: tuple[float, float] | None = None,
        image_fn: Callable[[np.ndarray, int], np.ndarray] | None = None,
    ):
        if path_column not in frame.columns:
            raise ValueError(f"frame has no {path_column!r} column; run the preprocess step first")
        self.paths = [Path(root) / p for p in frame[path_column]]
        self.labels = torch.tensor(frame["label"].to_numpy(), dtype=torch.float32)
        self.image_ids = frame["image_id"].tolist() if "image_id" in frame else None
        self.transform = build_transform(image_size, train, corner_mask)
        # Optional hook used by phase P7: receives the gray image as float32 in
        # [0, 1] plus the item index (for a per-image noise seed) and returns
        # the degraded image. Applied before resizing and normalisation.
        self.image_fn = image_fn

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.paths[i]) as img:
            gray = img.convert("L")
        if self.image_fn is not None:
            a = self.image_fn(np.asarray(gray, dtype=np.float32) / 255.0, i)
            gray = Image.fromarray(np.round(np.clip(a, 0.0, 1.0) * 255).astype(np.uint8))
        return self.transform(gray), self.labels[i]
