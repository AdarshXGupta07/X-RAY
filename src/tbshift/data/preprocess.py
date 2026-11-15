"""Resize every X-ray once and cache it (plan phase P1).

The NLM images are 3,000-5,000 pixels on a side. Decoding them every epoch
would make training on a laptop painfully slow, so each image is made square,
shrunk once to ``size`` pixels, and saved as an 8-bit PNG. Training reads the
small copies; the originals are never modified.

Making the image square
-----------------------
``crop`` (default)  cut the long side down to the short side, keeping the centre.
``pad``             add black bars to the short side.

Montgomery has portrait and landscape images in different proportions for TB
and normal cases. Padding would put the black bars in a different place for
each orientation, handing the model a free "which class" hint; stretching
would distort anatomy differently. Centre-cropping avoids both at the cost of
a few centimetres at the image edges. Whether any orientation signal survives
is checked in phase P2.

Run:  python -m tbshift.data.preprocess --site montgomery
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

MODES = ("crop", "pad")


def to_square(img: Image.Image, mode: str = "crop") -> Image.Image:
    """Return a square version of ``img`` by centre-cropping or black padding."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    w, h = img.size
    if w == h:
        return img.copy()
    if mode == "crop":
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        return img.crop((left, top, left + side, top + side))
    side = max(w, h)
    canvas = Image.new(img.mode, (side, side), 0)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def preprocess_image(path: Path, size: int = 512, mode: str = "crop") -> Image.Image:
    """Load one X-ray as 8-bit grayscale, make it square, resize to size x size."""
    with Image.open(path) as img:
        gray = img.convert("L")  # any bit depth -> 8-bit grayscale
    # LANCZOS gives a sharp, alias-free downsample; it is slow but runs only once.
    return to_square(gray, mode).resize((size, size), Image.Resampling.LANCZOS)


def build_cache(
    manifest: pd.DataFrame,
    out_dir: Path,
    project_root: Path | None = None,
    size: int = 512,
    mode: str = "crop",
) -> pd.Series:
    """Write the small copies and return their paths (relative to project_root).

    Files that already exist are skipped, so an interrupted run can be resumed.
    """
    project_root = Path(project_root) if project_root else Path.cwd()
    out_dir = Path(out_dir)
    (project_root / out_dir).mkdir(parents=True, exist_ok=True)
    cached = []
    for row in manifest.itertuples():
        rel = (out_dir / f"{row.image_id}.png").as_posix()
        target = project_root / rel
        if not target.exists():
            preprocess_image(project_root / row.path, size, mode).save(target)
        cached.append(rel)
    return pd.Series(cached, index=manifest.index, name="cached_path")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache square, downsized copies of a site's X-rays")
    parser.add_argument("--site", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--mode", choices=MODES, default="crop")
    parser.add_argument("--manifest-dir", default="data/manifests")
    args = parser.parse_args()

    path = Path(args.manifest_dir) / f"{args.site}.csv"
    df = pd.read_csv(path)
    out_dir = Path("data/processed") / f"{args.site}_{args.mode}{args.size}"
    df["cached_path"] = build_cache(df, out_dir, size=args.size, mode=args.mode)
    df.to_csv(path, index=False)
    print(f"{args.site}: {len(df)} images cached in {out_dir}; manifest updated")


if __name__ == "__main__":
    main()
