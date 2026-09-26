"""Rasterize label points onto the common grid; sample stacks at points."""
from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize

from src.preprocess.grid import GridSpec

logger = logging.getLogger(__name__)

__all__ = ["rasterize_points", "label_rasters", "extract_at_points"]


def rasterize_points(
    gdf: gpd.GeoDataFrame,
    grid: GridSpec,
    value_col: str | None = None,
    default_value: float = 1.0,
    buffer_m: float = 0.0,
    dtype: str = "float32",
) -> np.ndarray:
    """Burn point geometries onto the grid.

    Parameters
    ----------
    gdf : points in any CRS (reprojected to ``grid.crs`` internally).
    value_col : column providing per-point burn values; else default.
    buffer_m : optionally dilate each point to a disc of this radius
        (metres) before rasterizing - useful for thinning sparse labels.
    """
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS")
    pts = gdf.to_crs(grid.crs)
    if buffer_m > 0:
        shapes = [(g.buffer(buffer_m), v) for g, v in _values(pts,
                                                             value_col,
                                                             default_value)]
    else:
        shapes = list(_values(pts, value_col, default_value))
    return rasterize(
        shapes,
        out_shape=grid.shape,
        transform=grid.transform,
        fill=0,
        dtype=dtype,
    )


def _values(gdf, value_col, default):
    if value_col is not None and value_col in gdf.columns:
        for geom, val in zip(gdf.geometry, gdf[value_col], strict=False):
            yield geom, val
    else:
        for geom in gdf.geometry:
            yield geom, default


def label_rasters(
    positives: gpd.GeoDataFrame,
    background: gpd.GeoDataFrame | None,
    grid: GridSpec,
    buffer_m: float = 60.0,
) -> dict[str, np.ndarray]:
    """Build training rasters from positive + background point sets.

    Returns
    -------
    dict with keys
        ``positive`` : 1 where a (buffered) positive cell exists, else 0
        ``background`` : 1 where a background cell exists (if given)
        ``y`` : combined training mask (1 positive, 0 background where
            available, nodata elsewhere encoded as 255 uint8 when both
            provided).
    """
    pos = rasterize_points(positives, grid, buffer_m=buffer_m)
    out = {"positive": (pos > 0).astype("uint8")}
    if background is not None and len(background):
        out["background"] = (
            rasterize_points(background, grid) > 0
        ).astype("uint8")
        y = np.full(grid.shape, 255, dtype="uint8")   # 255 = unlabelled
        y[out["background"] > 0] = 0
        y[out["positive"] > 0] = 1                    # wins over background
        out["y"] = y
    return out


def extract_at_points(
    stack: np.ndarray,
    band_names: list[str],
    grid: GridSpec,
    points: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Sample every stack band at point locations.

    Parameters
    ----------
    stack : (bands, height, width) array aligned to *grid*.
    band_names : names per band (become DataFrame columns).
    points : GeoDataFrame; reprojected to the grid CRS; rows outside the
      grid are dropped.

    Returns
    -------
    DataFrame with one row per in-grid point and one column per band
    (plus all original attribute columns except geometry duplicates).
    """
    if stack.ndim != 3 or stack.shape[0] != len(band_names):
        raise ValueError(
            f"stack {stack.shape} inconsistent with {len(band_names)} bands"
        )
    pts = points.to_crs(grid.crs)
    cols, rows = grid.world_to_pixel(pts.geometry.x, pts.geometry.y)
    ci = np.round(cols).astype(int)
    ri = np.round(rows).astype(int)
    keep = (
        (ci >= 0) & (ci < grid.width) & (ri >= 0) & (ri < grid.height)
    )
    data = {
        name: stack[b, ri[keep], ci[keep]]
        for b, name in enumerate(band_names)
    }
    attrs = pts.drop(columns=pts.geometry.name, errors="ignore").loc[keep]
    out = pd.concat([attrs.reset_index(drop=True),
                     pd.DataFrame(data)], axis=1)
    logger.info("Extracted %d point samples x %d bands",
                len(out), len(band_names))
    return out
