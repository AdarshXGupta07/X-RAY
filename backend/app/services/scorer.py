"""Turn uploaded X-ray files into model logits.

The model is loaded once, when the server starts (see ``main.lifespan``), not
per request: loading DenseNet-121 takes seconds, scoring one image a fraction
of that. Uploaded images go through exactly the preprocessing used for
training (grayscale, centre-crop to square, 512 px cache size, then the
224 px evaluation transform). Nothing is written to disk.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image, UnidentifiedImageError

from tbshift.data.preprocess import prepare_image

MAX_IMAGE_BYTES = 30 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 60_000_000   # NLM images are ~20 MP; refuse decompression bombs


class ImageError(ValueError):
    """An upload that is not a readable image."""


class Scorer(Protocol):
    def score(self, images: list[Image.Image]) -> np.ndarray: ...


def decode_image(data: bytes, filename: str) -> Image.Image:
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageError(f"{filename}: larger than {MAX_IMAGE_BYTES // 2**20} MB")
    try:
        with Image.open(io.BytesIO(data)) as img:
            return prepare_image(img)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ImageError(f"{filename}: not a readable PNG or JPEG image ({exc})") from exc


class DenseNetScorer:
    """The fine-tuned DenseNet-121 from P2."""

    def __init__(self, checkpoint: Path, corner_mask: tuple[float, float] | None = None,
                 device: str = "cpu", image_size: int = 224, batch_size: int = 16):
        import torch  # imported here so the API can start (and be tested) without a model

        from tbshift.data.dataset import build_transform
        from tbshift.models.densenet import build_densenet121

        self._torch = torch
        self.device = device
        self.batch_size = batch_size
        self.transform = build_transform(image_size, train=False, corner_mask=corner_mask)
        model = build_densenet121(pretrained=False, memory_efficient=False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        self.model = model.to(device).eval()

    def score(self, images: list[Image.Image]) -> np.ndarray:
        torch = self._torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(images), self.batch_size):
                batch = torch.stack([self.transform(img) for img in images[i:i + self.batch_size]])
                out.append(self.model(batch.to(self.device)).squeeze(1).float().cpu().numpy())
        return np.concatenate(out) if out else np.empty(0, dtype=np.float32)
