"""Specification for tbshift.degradations (phase P7). Expected to fail until implemented."""

import numpy as np
import pytest

from tbshift.degradations import DEGRADATIONS, degrade, gaussian_blur, gaussian_noise

pytestmark = pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="phase P7 not written")


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
