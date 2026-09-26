"""Geology feature engineering onto the canonical analysis grid.

Turns the ingested USGS/GSC world-geology vectors
(``src.ingest.worldgeol``) into grid-aligned model features:

* ``lithology_code``   — integer rock-type code (uint8, 0 = nodata) with
  a JSON class map persisted alongside;
* ``geol_is_<class>``  — one-hot masks per rock-type class (float32 0/1);
* ``dist_to_<class>_m``— Euclidean distance to the nearest pixel of each
  rock-type class (metres);
* ``dist_to_contact_m``— distance to the nearest mapped geological
  contact (line feature, proxy for structural/alteration zones).

All bands share ``grid.shape`` and are written as one multi-band GeoTIFF
plus manifest via :func:`src.features.stack.build_feature_stack`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

from src.features.stack import build_feature_stack
from src.ingest.worldgeol import ROCK_TYPE_LABELS
from src.preprocess.grid import GridSpec

logger = logging.getLogger(__name__)

__all__ = ["build_geology_features", "default_geology_paths"]


def default_geology_paths(cache_dir: str | Path | None = None) -> dict[str, Path]:
    """Convention paths of the ingested geology GeoPackages."""
    root = Path(cache_dir) if cache_dir else Path("data/raw/world_geology")
    return {
        "geology": root / "geology_belt.gpkg",
        "contacts": root / "contacts_belt.gpkg",
    }


def _rasterize_mask(
    gdf: gpd.GeoDataFrame,
    grid: GridSpec,
    value: int = 1,
    all_touched: bool = True,
) -> np.ndarray:
    """Rasterize geometries as a binary mask on *grid*."""
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS")
    gdf = gdf.to_crs(grid.crs)
    return rasterize(
        ((geom, value) for geom in gdf.geometry if geom is not None),
        out_shape=grid.shape,
        transform=grid.transform,
        fill=0,
        dtype="uint8",
        all_touched=all_touched,
    )


def _distance_m(mask: np.ndarray, res_m: float) -> np.ndarray:
    """Euclidean distance (metres) to the nearest True cell of *mask*."""
    dt = distance_transform_edt(1 - mask, sampling=[res_m, res_m])
    return dt.astype(np.float32)


def build_geology_features(
    grid: GridSpec,
    geology: gpd.GeoDataFrame,
    contacts: gpd.GeoDataFrame | None = None,
    field: str = "rxtp",
    out_dir: str | Path | None = None,
) -> tuple[Path, dict[str, int]]:
    """Rasterize lithology + contact vectors into grid-aligned features.

    Parameters
    ----------
    grid : canonical analysis grid.
    geology : lithology polygons (any CRS) with categorical *field*.
    contacts : optional line contacts for the distance band.
    field : column carrying the rock-type code (``rxtp``).
    out_dir : output folder; default ``data/interim/features``.

    Returns
    -------
    (stack_path, class_map)
        Path of the written multi-band GeoTIFF and the
        ``{class_name: integer_code}`` mapping for ``lithology_code``.
    """
    if field not in geology.columns:
        raise ValueError(
            f"field '{field}' not in geology columns {list(geology.columns)}"
        )

    # ── categorical lithology code (0 = nodata, 1..N classes) ──────────
    # fill missing codes so classes stay pure strings (pandas 3 string
    # dtype keeps NaN as NA even after astype(str))
    field_vals = geology[field].astype("object").where(
        geology[field].notna(), "UNKNOWN"
    )
    classes = sorted({str(v) for v in field_vals.unique()})
    class_to_code = {c: i + 1 for i, c in enumerate(classes)}
    code_raster = rasterize(
        ((geom, class_to_code[str(val)])
         for geom, val in zip(geology.to_crs(grid.crs).geometry,
                              field_vals, strict=False)),
        out_shape=grid.shape,
        transform=grid.transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )

    # ── one-hot + per-class distance rasters ───────────────────────────
    bands: dict[str, np.ndarray] = {"lithology_code":
                                    code_raster.astype(np.float32)}
    labels: dict[str, str] = {}
    for cls in classes:
        label = ROCK_TYPE_LABELS.get(cls, cls.lower())
        labels[cls] = label
        mask = (code_raster == class_to_code[cls])
        bands[f"geol_is_{label}"] = mask.astype(np.float32)
        bands[f"dist_to_{label}_m"] = _distance_m(mask, grid.resolution_m)

    # ── distance to mapped contacts (structure proxy) ──────────────────
    if contacts is not None and len(contacts):
        contact_mask = _rasterize_mask(contacts, grid)
        bands["dist_to_contact_m"] = _distance_m(
            contact_mask > 0, grid.resolution_m
        )

    out_dir = Path(out_dir) if out_dir else Path("data/interim/features")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "geology_features.tif"
    path, names = build_feature_stack(grid, bands, out_path)

    # class map JSON: raw rxtp code -> {code_int, human_label}
    class_map = {
        cls: {"code": class_to_code[cls], "label": labels[cls]}
        for cls in classes
    }
    (out_dir / "geology_class_map.json").write_text(
        json.dumps(class_map, indent=2), encoding="utf-8"
    )
    logger.info("Geology features: %d bands -> %s", len(names), path)
    return path, {cls: class_to_code[cls] for cls in classes}
