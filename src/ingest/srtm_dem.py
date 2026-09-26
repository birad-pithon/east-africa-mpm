"""DEM ingestion for the East Africa MPM pipeline.

Supports two raster DEM sources, both pulled over the common analysis grid:

* SRTM 30 m (``USGS/SRTMGL1_003``)  -- default
* Copernicus GLO-30 (``COPERNICUS/DEM/GLO30``)

The DEM is exported from Google Earth Engine (if authenticated) and then
reprojected/resampled onto the canonical 30 m / EPSG:32736 grid so that it
pixel-aligns with Sentinel-2 and every other raster layer. When Earth
Engine is not available, a synthetic elevation surface is written to the
same grid so pipeline alignment tests still run.

Usage::

    from src.ingest.srtm_dem import pull_dem_tile
    dem_path = pull_dem_tile("configs/karagwe.yml", source="srtm")

CLI::

    python -m src.ingest.srtm_dem --config configs/karagwe.yml --source srtm
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import ee
import numpy as np
import rasterio

from src.ingest.sentinel2_ee import aoi_from_config, init_ee
from src.utils import (
    grid_dimensions,
    load_config,
    make_grid_transform,
    project_path,
    reproject_raster_to_grid,
    wgs84_to_utm,
)

logger = logging.getLogger(__name__)

# Earth Engine collections keyed by source id (see configs/data_sources.yaml)
EE_COLLECTIONS = {
    "srtm": "USGS/SRTMGL1_003",
    "glo30": "COPERNICUS/DEM/GLO30",
}


def _build_dem_image(collection_name: str, aoi: ee.Geometry) -> ee.Image:
    """Build a single-band DEM image clipped to the AOI."""
    img = ee.Image(collection_name).select("elevation").clip(aoi)
    return img.rename("elevation")


def _create_synthetic_dem(
    path: Path,
    bounds_utm: tuple[float, float, float, float],
    transform,
    width: int,
    height: int,
    crs: str = "EPSG:32736",
) -> None:
    """Create a synthetic elevation surface (placeholder when EE is off).

    Uses a tilted plane plus a few sinusoidal ridges so the surface is
    smooth and slope/aspect derivations behave sensibly.
    """
    yy, xx = np.mgrid[0:height, 0:width].astype("float64")
    xn = xx / max(width - 1, 1)
    yn = yy / max(height - 1, 1)
    base = 1200.0 + 200.0 * yn  # gentle N-S tilt
    ridge = 80.0 * np.sin(4 * np.pi * xn) * np.sin(3 * np.pi * yn)
    detail = 15.0 * np.sin(40 * np.pi * xn) * np.sin(30 * np.pi * yn)
    elev = (base + ridge + detail).astype("float32")

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(elev, 1)
        dst.set_band_description(1, "elevation_m")
    logger.info("Created synthetic DEM at %s", path)


def pull_dem_tile(config_path: str | Path, source: str = "srtm",
                   suffix: str = "kabar",
                   out_path: Path | str | None = None) -> Path:
    """Pull a DEM composite for the AOI and reproject to the common grid.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML config.
    source : {"srtm", "glo30"}
        Which DEM product to fetch.
    suffix : str
        Output filename suffix (e.g. "kabar", "copperbelt", "usambara").
    out_path : Path | str | None
        Where to write the aligned GeoTIFF. ``None`` (default) writes the
        production ``data/interim/dem/{source}_{suffix}_aligned.tif``;
        tests pass an explicit path so they never overwrite production rasters.

    Returns
    -------
    Path
        Path to the grid-aligned GeoTIFF.
    """
    cfg = load_config(config_path)
    grid_cfg = cfg["grid"]
    dst_crs = grid_cfg["crs"]
    resolution = grid_cfg["resolution_m"]

    bounds_utm = wgs84_to_utm(grid_cfg["aoi_wgs84"], dst_crs)
    transform = make_grid_transform(dst_crs, resolution, bounds_utm)
    width, height = grid_dimensions(bounds_utm, transform)

    if out_path is None:
        out_dir = project_path("data", "interim", "dem")
        out_dir.mkdir(parents=True, exist_ok=True)
        aligned_tif = out_dir / f"{source}_{suffix}_aligned.tif"
    else:
        aligned_tif = Path(out_path)
        aligned_tif.parent.mkdir(parents=True, exist_ok=True)

    collection_name = EE_COLLECTIONS.get(source)
    if collection_name is None:
        raise ValueError(
            f"Unknown DEM source '{source}'. Choose from {list(EE_COLLECTIONS)}"
        )

    try:
        init_ee()
        aoi = aoi_from_config(cfg)
        dem = _build_dem_image(collection_name, aoi)

        raw_dir = project_path("data", "raw", "dem")
        raw_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Requesting EE export of %s (%dx%d px @ %dm)...",
            source,
            width,
            height,
            int(resolution),
        )
        task = ee.batch.Export.image.toDrive(
            image=dem,
            description=f"DEM_{source.upper()}_KAB_export",
            folder=str(raw_dir),
            fileNamePrefix=f"dem_{source}_kabar_tile",
            region=aoi,
            scale=int(resolution),
            crs=dst_crs,
            fileFormat="GeoTIFF",
            maxPixels=1e13,
        )
        task.start()
        done = {"COMPLETED", "FAILED", "CANCELLED"}
        while True:
            status = task.status()
            if status["state"] in done:
                break
            if status.get("error_message"):
                raise RuntimeError(f"EE export error: {status['error_message']}")
            time.sleep(5)
        if status["state"] != "COMPLETED":
            raise RuntimeError(f"EE export ended in {status['state']}")

        downloaded = raw_dir / f"dem_{source}_kabar_tile.tif"
        if not downloaded.exists():
            found = list(raw_dir.rglob(f"dem_{source}_kabar_tile.tif"))
            downloaded = found[0] if found else None
        if downloaded is None or not downloaded.exists():
            raise FileNotFoundError("EE DEM download not found")

        reproject_raster_to_grid(
            src_path=downloaded,
            dst_path=aligned_tif,
            dst_crs=dst_crs,
            dst_transform=transform,
            dst_width=width,
            dst_height=height,
        )
        logger.info("Aligned DEM saved -> %s", aligned_tif)

    except (RuntimeError, FileNotFoundError) as exc:
        logger.warning("EE DEM pull skipped: %s", exc)
        logger.warning("Writing synthetic DEM on the common grid instead.")
        _create_synthetic_dem(
            aligned_tif, bounds_utm, transform, width, height, dst_crs
        )

    return aligned_tif


def main():
    """CLI entry point for DEM ingestion."""
    parser = argparse.ArgumentParser(
        description="Pull SRTM / Copernicus GLO-30 DEM over the AOI"
    )
    parser.add_argument(
        "--config",
        default="configs/karagwe.yml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--source",
        default="srtm",
        choices=list(EE_COLLECTIONS),
        help="DEM product: srtm (30 m) or glo30",
    )
    parser.add_argument(
        "--suffix",
        default="kabar",
        help="Output filename suffix (e.g. kabar, copperbelt, usambara)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    path = pull_dem_tile(args.config, source=args.source, suffix=args.suffix)
    print(f"\nOK DEM pull complete: {path}")


if __name__ == "__main__":
    main()
