"""Tests for tbshift.data.manifest, using tiny fake datasets on disk."""

import pandas as pd
import pytest
from PIL import Image

from tbshift.data.manifest import (
    build_manifest,
    check_manifest,
    parse_clinical_reading,
    parse_filename,
)


def _make_fake_set(root, prefix, labels, with_masks=True):
    """Write a miniature NLM-style folder: CXR_png, ClinicalReadings, ManualMask."""
    (root / "CXR_png").mkdir(parents=True)
    (root / "ClinicalReadings").mkdir()
    for i, label in enumerate(labels, start=1):
        stem = f"{prefix}_{i:04d}_{label}"
        Image.new("L", (40, 30)).save(root / "CXR_png" / f"{stem}.png")
        (root / "ClinicalReadings" / f"{stem}.txt").write_text(
            f"Patient's Sex: {'F' if i % 2 else 'M'}\nPatient's Age: 0{30 + i}Y\n"
            + ("normal\n" if label == 0 else "right upper lobe infiltrate\n"),
            encoding="utf-8",
        )
        if with_masks:  # masks share file names with the X-rays and must be ignored
            (root / "ManualMask" / "leftMask").mkdir(parents=True, exist_ok=True)
            Image.new("1", (40, 30)).save(root / "ManualMask" / "leftMask" / f"{stem}.png")


def test_parse_filename():
    assert parse_filename("MCUCXR_0001_0.png") == ("MCUCXR", 1, 0)
    assert parse_filename("CHNCXR_0662_1.png") == ("CHNCXR", 662, 1)
    assert parse_filename("MCUCXR_0001_2.png") is None  # label must be 0 or 1
    assert parse_filename("notes.png") is None


@pytest.mark.parametrize(
    "text, sex, age",
    [
        ("Patient's Sex: M\nPatient's Age: 035Y\nnormal\n", "M", 35),
        ("Patient's Sex: F\nPatient's Age: 061Y\nbilateral TB\n", "F", 61),
        ("male 35yrs\nnormal\n", "M", 35),
        ("female 8yrs\nPTB in the right upper field\n", "F", 8),
    ],
)
def test_parse_clinical_reading(text, sex, age):
    r = parse_clinical_reading(text)
    assert (r["sex"], r["age"]) == (sex, age)
    assert r["finding"]


def test_build_manifest_reads_labels_and_ignores_masks(tmp_path):
    _make_fake_set(tmp_path / "raw", "MCUCXR", [0, 1, 1, 0])
    df = build_manifest(tmp_path / "raw", "montgomery", project_root=tmp_path)
    assert len(df) == 4  # masks not counted
    assert df["label"].tolist() == [0, 1, 1, 0]
    assert df["path"].str.startswith("raw/CXR_png/").all()
    assert (df[["width", "height"]].to_numpy() == [40, 30]).all()
    assert df["sex"].tolist() == ["F", "M", "F", "M"]
    assert df["age"].tolist() == [31, 32, 33, 34]


def test_build_manifest_skips_other_sites(tmp_path):
    _make_fake_set(tmp_path / "raw", "CHNCXR", [0, 1], with_masks=False)
    assert build_manifest(tmp_path / "raw", "montgomery", project_root=tmp_path).empty


def test_check_manifest_flags_count_mismatch(tmp_path):
    _make_fake_set(tmp_path / "raw", "MCUCXR", [0, 1])
    df = build_manifest(tmp_path / "raw", "montgomery", project_root=tmp_path)
    assert any("counts" in p for p in check_manifest(df, "montgomery"))


def test_check_manifest_flags_duplicates():
    df = pd.DataFrame({"image_id": ["a", "a"], "label": [0, 1]})
    assert any("duplicate" in p for p in check_manifest(df, "other"))


def test_unknown_site_raises(tmp_path):
    with pytest.raises(ValueError):
        build_manifest(tmp_path, "atlantis")
