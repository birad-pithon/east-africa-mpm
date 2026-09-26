"""Full-grid inference: trained bundle -> probability GeoTIFF.

Reads the model bundle written by :func:`src.models.train.train_group`,
recomputes the identical feature columns over every pixel of the grid
using the raster stack recorded in ``feature_names`` (``<stem>:<band>``),
and writes P(deposit) as a single-band float32 GeoTIFF aligned to the
first input raster.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import rasterio

logger = logging.getLogger(__name__)

__all__ = ["predict_raster"]


def _build_band_index(paths: list[Path]) -> dict[str, tuple[Path, int]]:
    """Map 'stem:description' / 'stem:bN' -> (path, band number)."""
    index: dict[str, tuple[Path, int]] = {}
    for p in paths:
        with rasterio.open(p) as src:
            for b in range(1, src.count + 1):
                desc = src.descriptions[b - 1]
                key = f"{p.stem}:{desc}" if desc else f"{p.stem}:b{b}"
                index.setdefault(key, (p, b))
    return index


def predict_raster(
    bundle_path: Path | str,
    feature_rasters: list[Path | str],
    out_path: Path | str | None = None,
    block: int = 512,
) -> Path:
    """Predict deposit probability for every pixel; write GeoTIFF."""
    bundle = joblib.load(bundle_path)
    clf = bundle["model"]
    wanted: list[str] = list(bundle["feature_names"])

    paths = [Path(p) for p in feature_rasters]
    bands = _build_band_index(paths)
    missing = [w for w in wanted if w not in bands]
    if missing:
        raise KeyError(f"rasters do not provide features: {missing}")

    ref_path = paths[0]
    out_path = Path(out_path) if out_path else \
        Path(bundle_path).with_name(
            Path(bundle_path).stem.replace("model_", "proba_") + ".tif"
        )

    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
        # strip tiling hints if incomplete (GDAL rejects BLOCKXSIZE
        # without TILED=YES); plain striped output is fine for maps
        if not profile.get("tiled"):
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
        profile.update(count=1, dtype="float32", nodata=np.nan)
        readers = {p: rasterio.open(p) for p in paths}

        def read_window(win):
            cols = []
            for name in wanted:
                p, b = bands[name]
                arr = readers[p].read(b, window=win).astype("float64")
                nd = readers[p].nodatavals[b - 1]
                if nd is not None:
                    arr[arr == nd] = np.nan
                cols.append(arr.ravel())
            return np.column_stack(cols)

        proba = np.full(ref.height * ref.width, np.nan, dtype="float64")
        for row0 in range(0, ref.height, block):
            for col0 in range(0, ref.width, block):
                win = rasterio.windows.Window(
                    col0, row0,
                    min(block, ref.width - col0),
                    min(block, ref.height - row0),
                )
                mat = read_window(win)
                good = ~np.isnan(mat).any(axis=1)
                if good.any():
                    proba_mat = clf.predict_proba(mat[good])[:, 1]
                    idx = (row0 * ref.width) + col0
                    flat_rows, flat_cols = np.where(good.reshape(win.height, win.width))
                    pos = idx + flat_rows * ref.width + flat_cols
                    proba[pos] = proba_mat

        proba_grid = proba.reshape(ref.height, ref.width)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(proba_grid.astype("float32"), 1)
            dst.set_band_description(1, "p_deposit")

    for r in readers.values():
        r.close()
    logger.info("probability raster -> %s", out_path)
    return out_path
