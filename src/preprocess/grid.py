"""Canonical analysis-grid specification shared across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import affine
import numpy as np

from src.utils import grid_dimensions, make_grid_transform, wgs84_to_utm


@dataclass(frozen=True)
class GridSpec:
    """Immutable description of the common analysis grid.

    Attributes
    ----------
    crs : target projected CRS, e.g. ``"EPSG:32736"``.
    resolution_m : pixel size (square).
    transform : Affine mapping pixel -> world (top-left anchored).
    width, height : raster dimensions in pixels.
    bounds : ``(min_x, min_y, max_x, max_y)`` in target CRS metres.
    """

    crs: str
    resolution_m: float
    transform: affine.Affine
    width: int
    height: int
    bounds: tuple[float, float, float, float]

    # ── construction ────────────────────────────────────────────────────
    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        resolution_m: float | None = None,
    ) -> GridSpec:
        """Build from a pipeline YAML config (``grid:`` section).

        Parameters
        ----------
        cfg : loaded config dict containing ``grid.aoi_wgs84`` etc.
        resolution_m : override the configured resolution (used by tests
            to build small fast grids without touching the YAML).
        """
        g = cfg["grid"]
        res = float(resolution_m if resolution_m is not None
                    else g["resolution_m"])
        crs = g["crs"]
        bounds = wgs84_to_utm(g["aoi_wgs84"], crs)
        transform = make_grid_transform(crs, res, bounds)
        w, h = grid_dimensions(bounds, transform)
        return cls(crs=crs, resolution_m=res, transform=transform,
                   width=w, height=h, bounds=bounds)

    # ── coordinate conversions ──────────────────────────────────────────
    def world_to_pixel(self, xs, ys) -> tuple[np.ndarray, np.ndarray]:
        """World coords -> fractional (col, row) arrays."""
        inv = ~self.transform
        cols, rows = [], []
        for x, y in zip(np.atleast_1d(xs), np.atleast_1d(ys), strict=False):
            c, r = inv * (float(x), float(y))
            cols.append(c)
            rows.append(r)
        return np.asarray(cols), np.asarray(rows)

    def pixel_to_world(self, cols, rows) -> tuple[np.ndarray, np.ndarray]:
        """Pixel centres -> world coords (vectorised via the transform)."""
        cols = np.asarray(cols, dtype=float) + 0.5
        rows = np.asarray(rows, dtype=float) + 0.5
        xs = self.transform.c + self.transform.a * cols + self.transform.b * rows
        ys = self.transform.f + self.transform.d * cols + self.transform.e * rows
        return xs, ys

    def contains(self, xs, ys) -> np.ndarray:
        """Boolean mask of points falling inside the grid extent."""
        cols, rows = self.world_to_pixel(xs, ys)
        return (
            (cols >= 0) & (cols < self.width)
            & (rows >= 0) & (rows < self.height)
        )

    # ── misc ────────────────────────────────────────────────────────────
    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def profile(self, dtype: str = "float32", count: int = 1) -> dict:
        """rasterio profile for writing a grid-aligned raster."""
        return {
            "driver": "GTiff",
            "height": self.height,
            "width": self.width,
            "count": count,
            "dtype": dtype,
            "crs": self.crs,
            "transform": self.transform,
        }
