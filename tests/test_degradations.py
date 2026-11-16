"""Specification for tbshift.degradations (phase P7)."""

import numpy as np
import pytest

from tbshift.degradations import DEGRADATIONS, degrade, gaussian_blur, gaussian_noise


@pytest.fixture
def image():
    """A 64x64 test pattern: smooth gradient plus a sharp bright square."""
    img = np.tile(np.linspace(0.2, 0.8, 64, dtype=np.float32), (64, 1))
    img[24:40, 24:40] = 1.0
    return img


@pytest.mark.parametrize("name", DEGRADATIONS)
def test_severity_zero_is_identity(image, name):
    assert np.array_equal(degrade(image, name, severity=0), image)


@pytest.mark.parametrize("name", DEGRADATIONS)
@pytest.mark.parametrize("severity", [1, 3, 5])
def test_output_keeps_shape_dtype_and_range(image, name, severity):
    out = degrade(image, name, severity)
    assert out.shape == image.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.parametrize("name", DEGRADATIONS)
def test_input_is_not_modified(image, name):
    before = image.copy()
    degrade(image, name, severity=3)
    assert np.array_equal(image, before)


@pytest.mark.parametrize("name", DEGRADATIONS)
def test_higher_severity_changes_image_more(image, name):
    mild = np.abs(degrade(image, name, 1) - image).mean()
    strong = np.abs(degrade(image, name, 5) - image).mean()
    assert strong > mild > 0


def test_noise_is_reproducible_with_seed(image):
    assert np.array_equal(gaussian_noise(image, 3, seed=7), gaussian_noise(image, 3, seed=7))
    assert not np.array_equal(gaussian_noise(image, 3, seed=7), gaussian_noise(image, 3, seed=8))


def test_blur_softens_edges(image):
    edge_before = np.abs(np.diff(image, axis=1)).max()
    edge_after = np.abs(np.diff(gaussian_blur(image, 3), axis=1)).max()
    assert edge_after < edge_before


def test_unknown_degradation_raises_value_error(image):
    # This test must fail with NotImplementedError today and pass once written.
    with pytest.raises(ValueError):
        degrade(image, "sepia", 1)


def test_blur_scales_with_image_size():
    # The same severity should blur the same fraction of the anatomy at 224 and 448 px.
    small = np.zeros((224, 224), np.float32); small[:, 112:] = 1.0
    big = np.zeros((448, 448), np.float32); big[:, 224:] = 1.0
    width = lambda a: int(((a > 0.05) & (a < 0.95)).sum(axis=1)[0])
    assert width(gaussian_blur(big, 4)) == pytest.approx(2 * width(gaussian_blur(small, 4)), abs=2)


@pytest.mark.parametrize("bad", [6, -1])
def test_bad_severity_raises(image, bad):
    with pytest.raises(ValueError):
        degrade(image, "noise", bad)
