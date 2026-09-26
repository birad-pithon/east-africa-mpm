"""Multi-band feature stack orchestration.

Aggregates every available feature raster (terrain, spectral,
distances, magnetic, lithology) into a single multi-band GeoTIFF and
writes a JSON manifest describing band order — consumed downstream by
``src.models`` for training and ``src.predict`` for inference.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import rasterio

from src.preprocess.grid import GridSpec

logger = logging.getLogger(__name__)


def build_feature_stack(
    grid: GridSpec,
    feature_bands: dict[str, np.ndarray],
    out_path: Path | str,
    manifest_path: Path | str | None = None,
) -> tuple[Path, list[str]]:
    """Write aligned feature rasters to a single multi-band GeoTIFF.

    Parameters
    ----------
    grid : GridSpec
        Canonical analysis grid (defines CRS, transform, dimensions).
    feature_bands : ordered dict
        {band_name: 2D array} — must all match ``grid.shape``.
    out_path : destination GeoTIFF path.
    manifest_path : optional, defaults to ``<out_path>.json``.

    Returns
    -------
    (out_path, band_names)
        """
    names = list(feature_bands.keys())
    arrays = list(feature_bands.values())
    h, w = grid.shape

    for name, arr in zip(names, arrays, strict=False):
        if arr.shape != (h, w):
            raise ValueError(
                f"Band '{name}' shape {arr.shape} != grid {grid.shape}"
            )

    count = len(names)
    profile = grid.profile(dtype="float32", count=count)
    profile.update(blockxsize=256, blockysize=256, tiled=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **profile) as dst:
        for i, (name, arr) in enumerate(zip(names, arrays, strict=False), start=1):
            dst.write(arr.astype(np.float32), i)
            dst.set_band_description(i, name)

    if manifest_path is None:
        manifest_path = out_path.with_suffix(".json")
    manifest = {
        "bands": [{"name": n, "index": i + 1} for i, n in enumerate(names)],
        "crs": grid.crs,
        "transform": list(grid.transform)[:6],
        "width": grid.width,
        "height": grid.height,
        "dtype": "float32",
    }
    Path(manifest_path).write_text(json.dumps(manifest, indent=2),
                                   encoding="utf-8")

    logger.info("Wrote %d-band stack -> %s (%d bands)",
                count, out_path, count)
    return out_path, names
