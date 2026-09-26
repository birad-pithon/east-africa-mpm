"""Terrain derivatives from a DEM on the common analysis grid.

Pure numpy/scipy implementations (no extra GDAL tools needed):

* slope & aspect (Horn-style 3x3 gradient)
* topographic position index (TPI = elevation minus neighbourhood mean)
* plateau-flatness index - the key bauxite predictor: smooth, low-slope,
  high-standing cells score high
* relative relief (elevation minus local minimum)

All functions accept and return 2-D float32 arrays of identical shape.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["slope_aspect", "tpi", "plateau_index", "relative_relief",
           "dem_features"]


def _gradients(dem: np.ndarray, res: float) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference dz/dx, dz/dy (edge-replicated)."""
    dem = dem.astype("float64", copy=False)
    dz_dy, dz_dx = np.gradient(dem, res, edge_order=2)
    return dz_dx, dz_dy


def slope_aspect(dem: np.ndarray, resolution_m: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Slope in degrees (0-90) and aspect in degrees clockwise from north."""
    dz_dx, dz_dy = _gradients(dem, resolution_m)
    slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
    slope = np.degrees(slope_rad).astype("float32")

    # Aspect points DOWNHILL (direction of steepest descent), measured
    # clockwise from north. Row index grows southward on a north-up grid
    # (negative-y affine), so world dy = -dz/drow.
    aspect = np.degrees(np.arctan2(-dz_dx, -dz_dy))
    aspect = np.mod(aspect, 360.0)                          # -> [0, 360)
    flat = slope < 0.1                                      # undefined on flats
    aspect[flat] = -1.0
    return slope, aspect.astype("float32")


def tpi(dem: np.ndarray, window_cells: int = 11) -> np.ndarray:
    """Topographic position index: elev minus mean over ``window`` box.

    Uses a separable uniform filter so large windows do not require a dense
    ``window x window`` kernel. ``nearest`` edge handling matches the prior
    correlation behavior while keeping memory use linear in the raster size.
    """
    win = max(int(window_cells) | 1, 3)                     # force odd
    dem64 = dem.astype("float64")
    mean = ndimage.uniform_filter(dem64, size=win, mode="nearest")
    return (dem64 - mean).astype("float32")


def relative_relief(dem: np.ndarray, window_cells: int = 15) -> np.ndarray:
    """Elevation minus local minimum within the window (valley depth)."""
    win = max(int(window_cells) | 1, 3)
    lo = ndimage.minimum_filter(dem, size=win, mode="nearest")
    return (dem.astype("float64") - lo.astype("float64")).astype("float32")


def plateau_index(
    dem: np.ndarray,
    resolution_m: float,
    slope_threshold_deg: float = 5.0,
    tpi_window: int = 21,
) -> np.ndarray:
    """Bauxite lateritic-plateau favourability in [0, 1].

    High where terrain is flat AND topographically high relative to its
    surroundings (lateritic caps on plateaus):
        score = sigmoid(slope term) * sigmoid(TPI term)
    """
    slope, _ = slope_aspect(dem, resolution_m)
    position = tpi(dem, tpi_window)

    def _sig(x, centre, scale):
        # High (->1) when x BELOW centre, e.g. flat cells (slope ~ 0)
        # against a threshold: u>0 <=> x < centre.
        u = (centre - x) / scale
        return 1.0 / (1.0 + np.exp(-u))

    slope_term = _sig(slope, slope_threshold_deg, 2.0)       # flat -> ~1
    # Zero-centred TPI term: cells standing high relative to their own
    # neighbourhood score higher. Fixed centring keeps scores comparable
    # across tiles (a per-tile global mean would shift with scene content).
    tpi_term = 1.0 / (1.0 + np.exp(-position / 5.0))         # high -> ~1
    score = (slope_term * tpi_term).astype("float32")
    return np.clip(score, 0.0, 1.0)


def dem_features(
    dem: np.ndarray,
    resolution_m: float,
    include_raw: bool = True,
) -> dict[str, np.ndarray]:
    """Compute the full terrain feature dictionary for a DEM tile."""
    slope, aspect = slope_aspect(dem, resolution_m)
    feats = {
        "slope_deg": slope,
        "aspect_deg": aspect,
        "tpi_630m": tpi(dem, window_cells=max(3, round(630 / resolution_m))),
        "relief_900m": relative_relief(
            dem, window_cells=max(3, round(900 / resolution_m))
        ),
        "plateau_index": plateau_index(
            dem, resolution_m,
            tpi_window=max(3, round(630 / resolution_m)),
        ),
    }
    if include_raw:
        feats = {"elevation": dem.astype("float32"), **feats}
    return {k: v.astype("float32") for k, v in feats.items()}
