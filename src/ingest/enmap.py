"""EnMAP / PRISMA hyperspectral ingest onto the common analysis grid.

P1 T6: two-tier sensor strategy support.  Sentinel-2 provides the
regional sweep; its broad SWIR band cannot separate the 2.20 um Al-OH
uptake (phyllic / gibbsite) from the 2.33 um Mg-OH/carbonate uptake
(propylitic / chlorite).  Flagged anomalies are inspected with
hyperspectral cubes (EnMAP ~224 bands, 420-2450 nm, 30 m; PRISMA
similar) using band-depth and spectral-angle analysis
(see T7 - src.features.hyperspectral).

This module provides:
* :func:`parse_wavelengths`        - band centre wavelengths from
  GeoTIFF band descriptions (e.g. ``"B042 (1234.56nm)"`` or ``"1234.56"``).
* :func:`load_enmap_cube`          - read a (bands, h, w) reflectance cube
  plus its wavelength vector.
* :func:`cloud_mask`               - brightness-based cloud proxy mask.
* :func:`align_to_grid`            - reproject the cube onto the belt's
  ``GridSpec`` (nearest neighbour, preserving wavelength band
  descriptions) and write a multi-band GeoTIFF; cloudy / nodata pixels
  become NaN so downstream training rows drop.

CLI:
    python -m src.ingest.enmap --input data/raw/enmap/enmap_tile.tif \
        --config configs/karagwe.yml \
        --out data/interim/hyperspectral/enmap_kabar_aligned.tif
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from src.preprocess.grid import GridSpec
from src.utils import load_config

logger = logging.getLogger(__name__)

__all__ = [
    "parse_wavelengths",
    "load_enmap_cube",
    "cloud_mask",
    "align_to_grid",
]

_NM_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*nm", re.IGNORECASE)
_NUM_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*$")


def parse_wavelengths(descriptions: list[str | None]) -> np.ndarray | None:
    """Extract band centre wavelengths (nm) from band descriptions.

    Accepts per band: ``"B042 (1234.56nm)"``, ``"1234.56nm"``, or a bare
    ``"1234.56"``.  Returns None if fewer than half the bands yield a
    parseable wavelength (descriptions are then assumed absent/unusable).
    """
    wl: list[float | None] = []
    for d in descriptions:
        if not d:
            wl.append(None)
            continue
        m = _NM_RE.search(d) or _NUM_RE.match(d)
        wl.append(float(m.group(1)) if m else None)
    got = [w for w in wl if w is not None]
    if len(got) < max(1, len(wl) // 2):
        return None
    return np.asarray([w if w is not None else np.nan for w in wl],
                      dtype="float64")


def load_enmap_cube(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read a hyperspectral GeoTIFF -> (cube (bands, h, w), wavelengths_nm).

    Integer DN cubes (common for L2A GeoTIFFs, scale ~1e4) are
    auto-scaled to reflectance when the 95th percentile exceeds 1.5.
    """
    with rasterio.open(path) as src:
        cube = src.read().astype("float64")
        wl = parse_wavelengths(list(src.descriptions))
    if np.nanpercentile(cube, 95) > 1.5:
        logger.info("auto-scaling DN cube by 1/10000")
        cube /= 10_000.0
    cube[~np.isfinite(cube)] = np.nan
    return cube, wl


def cloud_mask(
    cube: np.ndarray,
    wavelengths_nm: np.ndarray | None = None,
    brightness_thr: float = 0.45,
) -> np.ndarray:
    """Boolean cloud proxy mask (True = cloudy) from a reflectance cube.

    Simple brightness screen: clouds are bright across the whole VNIR.
    When wavelengths are available, only bands below 1000 nm are used so
    bright SWIR alteration targets are not misflagged.
    """
    vnir = cube
    if wavelengths_nm is not None:
        sel = np.where(wavelengths_nm < 1000.0)[0]
        if len(sel):
            vnir = cube[sel]
    with np.errstate(invalid="ignore"):
        bright = np.nanmean(vnir, axis=0)
    return bright > brightness_thr


# ── Grid alignment ────────────────────────────────────────────────────


def align_to_grid(
    cube_path: str | Path,
    grid: GridSpec,
    out_path: str | Path,
    brightness_thr: float = 0.45,
) -> Path:
    """Reproject a hyperspectral cube onto the belt's common grid.

    Nearest-neighbour resampling (spectral values must not be smoothed
    across material boundaries).  Cloudy and outside-coverage pixels
    become NaN so that training/inference rows drop instead of
    contaminating the model.

    Returns the written GeoTIFF path; band descriptions are the band
    centre wavelengths in nm.
    """
    cube, wl = load_enmap_cube(cube_path)
    n_bands, h_src, w_src = cube.shape

    with rasterio.open(cube_path) as src:
        src_transform, src_crs = src.transform, src.crs
        src_nodata = src.nodata
    if src_nodata is not None:
        cube[cube == src_nodata] = np.nan

    cloudy = cloud_mask(cube, wl, brightness_thr)

    profile = grid.profile(dtype="float32", count=n_bands)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dst = np.full((n_bands, grid.height, grid.width), np.nan,
                  dtype="float64")
    for b in range(n_bands):
        reproject(
            source=cube[b],
            destination=dst[b],
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=np.nan,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )

    # Warp the cloud indicator to the destination grid (nearest), then
    # mask - the boolean mask lives on the SOURCE grid, not the output.
    cloud_band = np.where(cloudy, 1.0, 0.0)
    cloud_dst = np.zeros((grid.height, grid.width), dtype="float64")
    reproject(
        source=cloud_band,
        destination=cloud_dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        src_nodata=np.nan,
        dst_nodata=0.0,
        resampling=Resampling.nearest,
    )
    dst[:, cloud_dst > 0.5] = np.nan
    if wl is not None:
        descriptions = [f"{w:.2f}nm" if np.isfinite(w) else f"b{i + 1}"
                        for i, w in enumerate(wl)]
    else:
        descriptions = [f"b{i + 1}" for i in range(n_bands)]

    with rasterio.open(out_path, "w", **profile) as handle:
        for b in range(n_bands):
            handle.write(dst[b].astype("float32"), b + 1)
            handle.set_band_description(b + 1, descriptions[b])

    n_valid = int(np.isfinite(dst[0]).sum())
    logger.info("aligned %d-band cube -> %s (%.1f%% valid coverage)",
                n_bands, out_path, 100.0 * n_valid / dst[0].size)
    return out_path


def main() -> None:
    """CLI entry point for hyperspectral ingest."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Align an EnMAP/PRISMA cube onto the common grid")
    parser.add_argument("--input", required=True,
                        help="hyperspectral GeoTIFF (bands = wavelengths)")
    parser.add_argument("--config", required=True,
                        help="belt YAML config defining the GridSpec")
    parser.add_argument("--out", required=True,
                        help="output aligned multi-band GeoTIFF")
    parser.add_argument("--brightness-thr", type=float, default=0.45,
                        help="VNIR mean-reflectance cloud threshold")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")

    cfg = load_config(args.config)
    grid = GridSpec.from_config(cfg)
    align_to_grid(args.input, grid, args.out,
                  brightness_thr=args.brightness_thr)


if __name__ == "__main__":
    main()
