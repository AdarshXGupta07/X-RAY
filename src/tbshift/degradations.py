"""Simulated image degradations (plan phase P7).

A controlled way to create "a new hospital": take images the model already
handles well, damage them in one specific way, and see whether the rule-out
promise survives. Each degradation mimics a real acquisition difference:

    noise     a noisier detector or a lower radiation dose
    blur      a softer detector, patient motion, or heavy downsampling
    jpeg      lossy compression in a PACS export or a phone photo of a film
    contrast  different exposure or post-processing (a "flatter" image)

Images are 2-D float32 arrays (height x width) with values in [0, 1].
``severity`` is an integer 0..5 where 0 returns the image unchanged.
Every function returns a NEW float32 array of the same shape, values in [0, 1].

Blur strength is defined in pixels of a 224-pixel image and scaled up for
larger images, so a given severity means the same anatomy-level blur at 224 or
512 px. Smaller images keep the 224-pixel values.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

SEVERITIES = (0, 1, 2, 3, 4, 5)
DEGRADATIONS = ("noise", "blur", "jpeg", "contrast")

NOISE_SIGMA = (0.0, 0.02, 0.04, 0.07, 0.10, 0.15)  # in units of the full 0..1 range
BLUR_SIGMA_224 = (0.0, 0.5, 1.0, 1.5, 2.5, 4.0)  # pixels at 224 px
JPEG_QUALITY = (None, 80, 50, 30, 15, 5)
CONTRAST_FACTOR = (1.0, 0.85, 0.7, 0.55, 0.4, 0.25)


def _check(image: np.ndarray, severity: int) -> np.ndarray:
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}, got {severity}")
    a = np.asarray(image, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError("image must be a 2-D grayscale array")
    return a


def _finish(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0.0, 1.0).astype(np.float32)


def gaussian_noise(image: np.ndarray, severity: int, seed: int = 0) -> np.ndarray:
    """Add zero-mean Gaussian noise. Same seed gives the same output."""
    a = _check(image, severity)
    if severity == 0:
        return a.copy()
    rng = np.random.default_rng(seed)
    return _finish(a + rng.normal(0.0, NOISE_SIGMA[severity], size=a.shape))


def gaussian_blur(image: np.ndarray, severity: int) -> np.ndarray:
    """Blur with a Gaussian kernel whose width grows with severity."""
    a = _check(image, severity)
    if severity == 0:
        return a.copy()
    # Scale up for larger images; never scale down, or tiny images would get a
    # kernel so narrow (< 0.2 px) that it changes nothing.
    sigma = BLUR_SIGMA_224[severity] * max(min(a.shape) / 224, 1.0)
    return _finish(gaussian_filter(a, sigma=sigma, mode="reflect"))


def jpeg_compress(image: np.ndarray, severity: int) -> np.ndarray:
    """Round-trip through 8-bit JPEG at a quality that drops with severity."""
    a = _check(image, severity)
    if severity == 0:
        return a.copy()
    img = Image.fromarray(np.round(np.clip(a, 0, 1) * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY[severity])
    buf.seek(0)
    with Image.open(buf) as decoded:
        return _finish(np.asarray(decoded, dtype=np.float32) / 255.0)


def reduce_contrast(image: np.ndarray, severity: int) -> np.ndarray:
    """Pull pixel values toward the image mean (a convex mix, so values stay in [0, 1])."""
    a = _check(image, severity)
    if severity == 0:
        return a.copy()
    mean = float(a.mean())
    return _finish(mean + CONTRAST_FACTOR[severity] * (a - mean))


def degrade(image: np.ndarray, name: str, severity: int, seed: int = 0) -> np.ndarray:
    """Dispatch by name (one of ``DEGRADATIONS``). Unknown names raise ValueError."""
    if name == "noise":
        return gaussian_noise(image, severity, seed)
    if name == "blur":
        return gaussian_blur(image, severity)
    if name == "jpeg":
        return jpeg_compress(image, severity)
    if name == "contrast":
        return reduce_contrast(image, severity)
    raise ValueError(f"unknown degradation {name!r}; expected one of {DEGRADATIONS}")
