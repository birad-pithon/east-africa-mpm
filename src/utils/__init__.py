"""Common geospatial utilities shared across the MPM pipeline.

This module provides factory functions and helpers for:
  * Loading YAML configuration files
  * Defining a common analysis grid (CRS + affine transform)
  * Reprojecting and resampling raster layers to that grid
  * Snapping geometries to the grid origin
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import affine
import geopandas as gpd
import pyproj
import rasterio
import rioxarray  # noqa: F401  (registers .rio accessor)
import yaml
from rasterio.warp import Resampling, reproject

# ── Project-root anchoring ─────────────────────────────────────────────
# All data/config paths are anchored here so the pipeline behaves the same
# no matter which directory it is invoked from. This file lives at
# ``<root>/src/utils/__init__.py`` so the repo root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str) -> Path:
    """Resolve *parts* relative to the repository root."""
    return PROJECT_ROOT.joinpath(*parts)


# ── Config loading ─────────────────────────────────────────────────────


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return it as a dict.

    Config files may contain non-ASCII characters (e.g. arrows, em-dashes
    in comments) so they are always read as UTF-8 regardless of the
    platform default encoding. Relative paths are tried against the
    current working directory first, then against the project root, so
    the pipeline works identically whichever directory it is launched
    from.
    """
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        candidate = PROJECT_ROOT / p
        if candidate.exists():
            p = candidate
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Grid factory ──────────────────────────────────────────────────────

# Global origin lattice (metres). All analysis grids snap their top-left
# corner to a multiple of this, guaranteeing that rasters produced at
# different resolutions (30 m production, coarse test grids) share the
# same anchor and align pixel-for-pixel after resampling.
GRID_LATTICE_M = 300.0


def make_grid_transform(
    crs: str, resolution_m: float, aoi_bounds_utm: tuple[float, float, float, float]
) -> affine.Affine:
    """Build an ``Affine`` transform for the common analysis grid.

    Parameters
    ----------
    crs : str
        Target CRS, e.g. ``"EPSG:32736"``.
    resolution_m : float
        Pixel size in metres (square pixels assumed).
    aoi_bounds_utm : tuple
        ``(min_x, min_y, max_x, max_y)`` in metres (target CRS units).
    """
    min_x, min_y, max_x, max_y = aoi_bounds_utm
    # Snap origins onto ONE global lattice so every analysis grid --
    # regardless of resolution -- shares identical anchors and stays
    # pixel-aligned across sources. 300 m is an exact multiple of the
    # 30 m base resolution, so per-resolution alignment assertions hold.
    lattice = GRID_LATTICE_M
    origin_x = (min_x // lattice) * lattice
    origin_y = (max_y // lattice) * lattice  # top-left Y
    return affine.Affine(resolution_m, 0, origin_x, 0, -resolution_m, origin_y)


def wgs84_to_utm(
    bbox_wgs84: dict[str, float], crs: str = "EPSG:32736"
) -> tuple[float, float, float, float]:
    """Convert a WGS-84 bounding box to the target UTM CRS.

    Parameters
    ----------
    bbox_wgs84 : dict with keys ``min_lon, max_lon, min_lat, max_lat``.
    crs : target CRS string.

    Returns
    -------
    tuple of (min_x, min_y, max_x, max_y) in target CRS metres.
    """
    project = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    corners = [
        (bbox_wgs84["min_lon"], bbox_wgs84["min_lat"]),
        (bbox_wgs84["max_lon"], bbox_wgs84["min_lat"]),
        (bbox_wgs84["max_lon"], bbox_wgs84["max_lat"]),
        (bbox_wgs84["min_lon"], bbox_wgs84["max_lat"]),
    ]
    xs, ys = zip(*[project(lon, lat) for lon, lat in corners], strict=False)
    return min(xs), min(ys), max(xs), max(ys)


def reproject_raster_to_grid(
    src_path: str | Path,
    dst_path: str | Path,
    dst_crs: str,
    dst_transform: affine.Affine,
    dst_width: int,
    dst_height: int,
    resampling: Resampling = Resampling.bilinear,
) -> str:
    """Reproject and resample a raster to the common analysis grid."""
    with rasterio.open(src_path) as src:
        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": dst_crs,
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height,
                "dtype": src.dtypes[0],
            }
        )
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    dst_width=dst_width,
                    dst_height=dst_height,
                    resampling=resampling,
                )
    return str(dst_path)


def reproject_vector(gdf: gpd.GeoDataFrame, dst_crs: str) -> gpd.GeoDataFrame:
    """Reproject a GeoDataFrame to *dst_crs*."""
    return gdf.to_crs(dst_crs)


def grid_dimensions(
    aoi_bounds_utm: tuple[float, float, float, float],
    transform: affine.Affine,
) -> tuple[int, int]:
    """Return (width, height) of the raster grid for the given AOI bounds."""
    min_x, min_y, max_x, max_y = aoi_bounds_utm
    origin_x = transform.c
    origin_y = transform.f
    res = transform.a
    width = int(round((max_x - origin_x) / res))
    height = int(round((origin_y - min_y) / res))
    return width, height
