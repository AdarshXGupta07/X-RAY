"""Tests for splits, the resize cache and the PyTorch dataset."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from tbshift.data.dataset import IMAGENET_MEAN, IMAGENET_STD, CXRDataset
from tbshift.data.preprocess import build_cache, preprocess_image, to_square
from tbshift.data.splits import SPLITS, allocate, assign_splits, summarize


def _manifest(n_tb=58, n_normal=80, images_per_patient=1):
    rows = []
    for label, n in ((1, n_tb), (0, n_normal)):
        for i in range(n):
            for k in range(images_per_patient):
                rows.append({"image_id": f"{label}_{i}_{k}", "patient_id": f"p{label}_{i}", "label": label})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- splits


def test_allocate_sums_and_follows_fractions():
    counts = allocate(58, {"train": 0.6, "val": 0.1, "cal": 0.15, "test": 0.15})
    assert sum(counts.values()) == 58
    assert counts["train"] == 35 and counts["val"] == 6


def test_every_row_gets_a_known_split():
    split = assign_splits(_manifest())
    assert split.notna().all() and set(split) <= set(SPLITS)


def test_splits_are_stratified():
    df = _manifest()
    table = summarize(df, assign_splits(df))
    assert table.loc["cal", "TB"] == 9  # round(58 * 0.15) by largest remainder
    shares = table["TB"] / table.sum(axis=1)
    assert (shares.sub(58 / 138).abs() < 0.08).all()


def test_same_seed_same_split_any_row_order():
    df = _manifest()
    a = assign_splits(df, seed=3)
    shuffled = df.sample(frac=1, random_state=1)
    b = assign_splits(shuffled, seed=3).reindex(df.index)
    assert a.equals(b)
    assert not a.equals(assign_splits(df, seed=4))


def test_patient_images_stay_together():
    df = _manifest(n_tb=20, n_normal=20, images_per_patient=3)
    df["split"] = assign_splits(df)
    assert (df.groupby("patient_id")["split"].nunique() == 1).all()


@pytest.mark.parametrize(
    "fractions",
    [{"train": 0.5, "test": 0.4}, {"train": 0.5, "holdout": 0.5}, {"train": 1.2, "test": -0.2}],
)
def test_bad_fractions_raise(fractions):
    with pytest.raises(ValueError):
        assign_splits(_manifest(), fractions)


# ---------------------------------------------------------------- squaring


def test_crop_keeps_the_centre():
    img = Image.new("L", (40, 30), 0)
    img.paste(255, (5, 0, 35, 30))  # bright block in the central 30 columns
    out = to_square(img, "crop")
    assert out.size == (30, 30)
    assert np.asarray(out).min() == 255  # the dark edges were cut away


def test_pad_adds_black_bars():
    img = Image.new("L", (40, 30), 200)
    out = np.asarray(to_square(img, "pad"))
    assert out.shape == (40, 40)
    assert out[0].max() == 0 and out[-1].max() == 0  # top and bottom bars
    assert out[20, 20] == 200


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        to_square(Image.new("L", (4, 3)), "stretch")


def test_preprocess_and_cache(tmp_path):
    (tmp_path / "raw").mkdir()
    for i in range(3):
        Image.new("I;16", (60, 80), 1000 * i).save(tmp_path / "raw" / f"x{i}.png")
    df = pd.DataFrame({"image_id": ["x0", "x1", "x2"], "path": [f"raw/x{i}.png" for i in range(3)]})

    assert preprocess_image(tmp_path / "raw/x1.png", size=32).size == (32, 32)
    cached = build_cache(df, "cache", project_root=tmp_path, size=32)
    assert cached.tolist() == ["cache/x0.png", "cache/x1.png", "cache/x2.png"]
    for p in cached:
        with Image.open(tmp_path / p) as im:
            assert im.size == (32, 32) and im.mode == "L"

    # A second run must not rewrite existing files.
    stamp = (tmp_path / "cache/x0.png").stat().st_mtime_ns
    build_cache(df, "cache", project_root=tmp_path, size=32)
    assert (tmp_path / "cache/x0.png").stat().st_mtime_ns == stamp


# ------------------------------------------------------------------ dataset


@pytest.fixture
def cached_frame(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(4):
        Image.fromarray(rng.integers(0, 256, (64, 64), dtype=np.uint8)).save(tmp_path / f"c{i}.png")
    return tmp_path, pd.DataFrame(
        {"image_id": [f"c{i}" for i in range(4)], "cached_path": [f"c{i}.png" for i in range(4)],
         "label": [0, 1, 1, 0]}
    )


def test_dataset_item_shape_type_and_label(cached_frame):
    root, frame = cached_frame
    ds = CXRDataset(frame, root=root, image_size=32)
    x, y = ds[1]
    assert len(ds) == 4
    assert x.shape == (3, 32, 32) and x.dtype == torch.float32
    # Undo the per-channel normalisation: all three channels must be the same gray image.
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    raw = x * std + mean
    assert torch.allclose(raw[0], raw[1], atol=1e-5) and torch.allclose(raw[0], raw[2], atol=1e-5)
    assert 0.0 <= raw.min() and raw.max() <= 1.0 + 1e-5
    assert y.item() == 1.0


def test_dataset_eval_mode_is_deterministic(cached_frame):
    root, frame = cached_frame
    ds = CXRDataset(frame, root=root, image_size=32, train=False)
    assert torch.equal(ds[0][0], ds[0][0])


def test_dataset_train_mode_augments(cached_frame):
    root, frame = cached_frame
    ds = CXRDataset(frame, root=root, image_size=32, train=True)
    torch.manual_seed(0)
    assert not torch.equal(ds[0][0], ds[0][0])


def test_dataset_needs_cached_paths(cached_frame):
    _, frame = cached_frame
    with pytest.raises(ValueError):
        CXRDataset(frame.drop(columns="cached_path"))


def test_dataloader_batches(cached_frame):
    root, frame = cached_frame
    loader = torch.utils.data.DataLoader(CXRDataset(frame, root=root, image_size=32), batch_size=2)
    x, y = next(iter(loader))
    assert x.shape == (2, 3, 32, 32) and y.shape == (2,)


def test_dataset_works_with_worker_processes(cached_frame):
    # On Windows, workers are new processes and every transform must be picklable.
    root, frame = cached_frame
    loader = torch.utils.data.DataLoader(
        CXRDataset(frame, root=root, image_size=32), batch_size=2, num_workers=1
    )
    assert next(iter(loader))[0].shape == (2, 3, 32, 32)
