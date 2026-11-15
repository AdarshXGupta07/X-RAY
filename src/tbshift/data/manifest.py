"""Build a manifest (one CSV row per image) for an NLM TB chest X-ray set.

A manifest is the project's single source of truth about the data: which
images exist, their label, their site, and basic image facts. Every later
step (splits, training, scoring) reads the manifest, never the folder.

Supported sets (file names are documented in the NLM ReadMe files):

    montgomery   MCUCXR_<number>_<0|1>.png
    shenzhen     CHNCXR_<number>_<0|1>.png

The trailing 0/1 is the label: 0 = normal, 1 = TB manifestation.
Clinical readings, when present, sit in ClinicalReadings/<same stem>.txt.

Licence note: NLM asks that the data not be shared outside your research
group. Manifests contain per-patient age and sex, so they are git-ignored
along with the images. Regenerate them with:

    python -m tbshift.data.manifest --site montgomery --raw data/raw/montgomery
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from PIL import Image

SITE_PREFIX = {"montgomery": "MCUCXR", "shenzhen": "CHNCXR"}
EXPECTED_COUNTS = {  # (TB, normal) from the NLM ReadMe files
    "montgomery": (58, 80),
    "shenzhen": (336, 326),
}
NAME_RE = re.compile(r"^(?P<prefix>[A-Z]+)_(?P<number>\d+)_(?P<label>[01])\.png$", re.IGNORECASE)
COLUMNS = [
    "image_id", "site", "label", "path", "patient_id",
    "width", "height", "mode", "sex", "age", "finding",
]


def parse_filename(name: str) -> tuple[str, int, int] | None:
    """'MCUCXR_0001_0.png' -> ('MCUCXR', 1, 0). Returns None if the name does not match."""
    m = NAME_RE.match(name)
    if m is None:
        return None
    return m["prefix"].upper(), int(m["number"]), int(m["label"])


# Montgomery: "Patient's Sex: M" / "Patient's Age: 035Y" (DICOM style: Y, M, W, D).
_SEX_FIELD = re.compile(r"sex\s*:\s*(?P<v>[a-z])", re.IGNORECASE)
_AGE_FIELD = re.compile(r"age\s*:\s*(?P<num>\d{1,3})\s*(?P<unit>[a-z]*)", re.IGNORECASE)
# Shenzhen: "male 35yrs", "female24yrs", "femal 32yrs" (sic), "Female, 28yrs", "Male , 50yrs",
# "male 16month", "female 64days".
_SEX_AGE_LINE = re.compile(
    r"^(?P<sex>f(?:emale?)?|m(?:ale)?)[\s,]*(?P<num>\d{1,3})\s*(?P<unit>[a-z]*)", re.IGNORECASE
)


def _age_in_years(num: str, unit: str) -> float:
    """Convert '16' + 'month' (or 'M', 'days', 'W', 'yrs', '') to years."""
    u = unit.lower()
    n = int(num)
    if u.startswith("m"):  # month(s), or DICOM 'M'
        return round(n / 12, 2)
    if u.startswith("w"):
        return round(n / 52, 2)
    if u.startswith("d"):
        return round(n / 365, 2)
    return float(n)  # y, yr, yrs, year(s) or no unit


def parse_clinical_reading(text: str) -> dict:
    """Pull sex, age (in years) and the free-text finding out of a readings file.

    Lines that state sex or age are parsed; every other non-empty line is
    treated as the radiologist's finding. Anything not found is None, and
    "O" (other/unknown) for sex is also None.
    """
    sex = age = None
    finding_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if m := _SEX_AGE_LINE.match(line):
            sex = "F" if m["sex"].lower().startswith("f") else "M"
            age = _age_in_years(m["num"], m["unit"])
        elif m := _SEX_FIELD.search(line):
            sex = {"f": "F", "m": "M"}.get(m["v"].lower())
        elif m := _AGE_FIELD.search(line):
            age = _age_in_years(m["num"], m["unit"])
        else:
            finding_lines.append(line)
    finding = " ".join(finding_lines) or None
    return {"sex": sex, "age": age, "finding": finding}


def build_manifest(raw_root: Path, site: str, project_root: Path | None = None) -> pd.DataFrame:
    """Scan ``raw_root`` for the site's chest X-rays and return the manifest.

    Only PNGs inside a folder named ``CXR_png`` are used, so lung masks
    (which share the X-ray file names) are never mistaken for X-rays.
    Paths are stored relative to ``project_root`` with forward slashes.
    """
    if site not in SITE_PREFIX:
        raise ValueError(f"unknown site {site!r}; expected one of {sorted(SITE_PREFIX)}")
    raw_root = Path(raw_root)
    project_root = Path(project_root) if project_root else Path.cwd()
    prefix = SITE_PREFIX[site]

    rows = []
    for png in sorted(raw_root.rglob("*.png")):
        if png.parent.name.lower() != "cxr_png":
            continue
        parsed = parse_filename(png.name)
        if parsed is None or parsed[0] != prefix:
            continue
        _, number, label = parsed

        with Image.open(png) as img:  # reads the header only, not the pixels
            width, height = img.size
            mode = img.mode

        reading_file = png.parent.parent / "ClinicalReadings" / f"{png.stem}.txt"
        reading = (
            parse_clinical_reading(reading_file.read_text(encoding="utf-8", errors="replace"))
            if reading_file.exists()
            else {"sex": None, "age": None, "finding": None}
        )
        rows.append(
            {
                "image_id": png.stem,
                "site": site,
                "label": label,
                "path": png.resolve().relative_to(project_root.resolve()).as_posix(),
                # The sets give no patient IDs; each image is treated as one patient.
                "patient_id": f"{site}_{number:05d}",
                "width": width,
                "height": height,
                "mode": mode,
                **reading,
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def check_manifest(df: pd.DataFrame, site: str) -> list[str]:
    """Return a list of problems (empty list = all good)."""
    problems = []
    if df.empty:
        return ["manifest is empty: wrong folder, or extraction failed"]
    if df["image_id"].duplicated().any():
        problems.append(f"{int(df['image_id'].duplicated().sum())} duplicate image ids")
    if not set(df["label"]).issubset({0, 1}):
        problems.append("labels other than 0/1")
    expected = EXPECTED_COUNTS.get(site)
    if expected is not None:
        found = (int((df["label"] == 1).sum()), int((df["label"] == 0).sum()))
        if found != expected:
            problems.append(f"(TB, normal) counts {found} differ from the ReadMe's {expected}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a dataset manifest CSV")
    parser.add_argument("--site", required=True, choices=sorted(SITE_PREFIX))
    parser.add_argument("--raw", required=True, help="folder the dataset was extracted into")
    parser.add_argument("--out", help="CSV path (default data/manifests/<site>.csv)")
    args = parser.parse_args()

    df = build_manifest(Path(args.raw), args.site)
    out = Path(args.out or f"data/manifests/{args.site}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"{args.site}: {len(df)} images "
          f"({int((df['label'] == 1).sum())} TB, {int((df['label'] == 0).sum())} normal) -> {out}")
    problems = check_manifest(df, args.site)
    print("checks: OK" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))


if __name__ == "__main__":
    main()
