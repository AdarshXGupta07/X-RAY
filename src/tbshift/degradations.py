"""Simulated image degradations (plan phase P7).

NOT IMPLEMENTED YET. The contracts below are pinned down by
``tests/test_degradations.py``.

Images are 2-D float32 arrays (height x width) with values in [0, 1].
``severity`` is an integer 0..5 where 0 returns the image unchanged.
Every function returns a new array with the same shape, dtype float32, and
values still in [0, 1].
"""

from __future__ import annotations

import numpy as np

SEVERITIES = (0, 1, 2, 3, 4, 5)
DEGRADATIONS = ("noise", "blur", "jpeg", "contrast")


def gaussian_noise(image: np.ndarray, severity: int, seed: int = 0) -> np.ndarray:
    """Add zero-mean Gaussian noise. Same seed gives the same output."""
    raise NotImplementedError("phase P7")


def gaussian_blur(image: np.ndarray, severity: int) -> np.ndarray:
    """Blur with a Gaussian kernel whose width grows with severity."""
    raise NotImplementedError("phase P7")


def jpeg_compress(image: np.ndarray, severity: int) -> np.ndarray:
    """Round-trip through JPEG at a quality that drops with severity."""
    raise NotImplementedError("phase P7")


def reduce_contrast(image: np.ndarray, severity: int) -> np.ndarray:
    """Pull pixel values toward the image mean."""
    raise NotImplementedError("phase P7")


def degrade(image: np.ndarray, name: str, severity: int, seed: int = 0) -> np.ndarray:
    """Dispatch by name (one of ``DEGRADATIONS``). Unknown names raise ValueError."""
    raise NotImplementedError("phase P7")
