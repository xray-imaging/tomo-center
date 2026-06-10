"""Read a folder of pre-reconstructed TIFF slices and pair each with a candidate center.

Each TIFF in the folder is expected to be a single 2D reconstructed slice produced
at one candidate center-of-rotation. The inference pipeline scores them and picks
the best.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tifffile


# Match a float (with optional sign and decimal point) anywhere in the filename stem.
# Examples it pulls a center out of:
#   recon_1234.50.tif  -> 1234.50
#   slice_0987.tif     -> 987.0
#   cor-1024p25.tif    -> (no match — use --centers-file instead)
_CENTER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def list_tiffs(folder: Path) -> List[Path]:
    """Return TIFF paths under `folder`, sorted lexicographically."""
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")
    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in (".tif", ".tiff")
    )
    if not files:
        raise FileNotFoundError(f"No .tif/.tiff files found in {folder}")
    return files


def centers_from_filenames(paths: List[Path]) -> List[float]:
    """Extract a float center from each filename stem (last numeric token)."""
    centers = []
    for p in paths:
        matches = _CENTER_RE.findall(p.stem)
        if not matches:
            raise ValueError(
                f"No numeric center found in filename '{p.name}'. "
                "Use --centers-file to provide centers explicitly."
            )
        # Prefer the LAST number in the stem — names like recon_0001_1234.50 are common.
        centers.append(float(matches[-1]))
    return centers


def centers_from_file(centers_file: Path, expected_n: int) -> List[float]:
    """One float per non-empty, non-comment line; count must match TIFF count."""
    centers_file = Path(centers_file)
    lines = [
        ln.strip() for ln in centers_file.read_text().splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if len(lines) != expected_n:
        raise ValueError(
            f"{centers_file} has {len(lines)} centers but folder has {expected_n} TIFFs."
        )
    return [float(ln) for ln in lines]


def load_stack(paths: List[Path]) -> np.ndarray:
    """Load all TIFFs as a (N, H, W) float32 array. All slices must share H, W."""
    imgs = []
    shape0 = None
    for p in paths:
        a = tifffile.imread(str(p))
        if a.ndim != 2:
            raise ValueError(f"{p.name}: expected 2D image, got shape {a.shape}")
        if shape0 is None:
            shape0 = a.shape
        elif a.shape != shape0:
            raise ValueError(
                f"{p.name}: shape {a.shape} differs from first slice {shape0}"
            )
        imgs.append(a.astype(np.float32, copy=False))
    return np.stack(imgs, axis=0)


def load_folder(
    folder: Path,
    centers_file: Path | None = None,
) -> Tuple[np.ndarray, List[float], List[Path]]:
    """High-level: sorted TIFF stack + matching center list + the paths."""
    paths = list_tiffs(Path(folder))
    if centers_file is not None:
        centers = centers_from_file(Path(centers_file), len(paths))
    else:
        centers = centers_from_filenames(paths)
    stack = load_stack(paths)
    return stack, centers, paths
