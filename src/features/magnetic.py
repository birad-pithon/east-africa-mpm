"""Magnetic anomaly gradient features.

Total-gradient amplitude (TGA) and horizontal gradient magnitude from a
gridded magnetic anomaly raster. Intrusion detection for Sn-W-Ta veins
and Cu-Zn traps.

Input: 1-band magnetic grid (nT) aligned to the common UTM/30 m grid.
Output: float32 rasters of the same shape.
"""
from __future__ import annotations

import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter, sobel


def _read_grid(path_or_arr) -> np.ndarray:
    if isinstance(path_or_arr, np.ndarray):
        return path_or_arr.astype(np.float64)
    with rasterio.open(path_or_arr) as src:
        return src.read(1).astype(np.float64)


def magnetic_gradient_features(
    mag_grid,
    grid_res_m: float = 30.0,
    smooth_sigma: float = 1.0,
    detrend: bool = True,
) -> dict[str, np.ndarray]:
    """Compute magnetic anomaly gradient features.

    Parameters
    ----------
    mag_grid : str path or 2-D ndarray
        Magnetic anomaly raster (nT).
    grid_res_m : pixel size (metres).
    smooth_sigma : Gaussian smoothing for noise suppression (pixels).
    detrend : subtract a planar background from the anomaly.

    Returns
    -------
    dict with keys:
        'tga'       -- total-gradient amplitude
        'gx', 'gy'  -- horizontal gradient components
        'mod_grad'  -- sqrt(gx^2 + gy^2) (same as tga if detrend=False)
    """
    mag = _read_grid(mag_grid)
    if mag.ndim != 2:
        raise ValueError(f"Expected 2-D grid, got shape {mag.shape}")

    if smooth_sigma and smooth_sigma > 0:
        mag = gaussian_filter(mag, sigma=smooth_sigma)

    if detrend:
        h, w = mag.shape
        yy, xx = np.mgrid[0:h, 0:w]
        # least-squares planar trend: z = a*x + b*y + c
        A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(h * w)])
        coef, *_ = np.linalg.lstsq(A, mag.ravel(), rcond=None)
        trend = (coef[0] * xx + coef[1] * yy + coef[2])
        residual = mag - trend
    else:
        residual = mag

    # Sobel gradient in physical units (nT / metre)
    gx = sobel(residual, axis=1, mode="reflect") / grid_res_m
    gy = sobel(residual, axis=0, mode="reflect") / grid_res_m

    tga = np.sqrt(gx ** 2 + gy ** 2)
    return {
        "tga": tga.astype(np.float32),
        "gx": gx.astype(np.float32),
        "gy": gy.astype(np.float32),
        "mod_grad": tga.astype(np.float32),
    }
