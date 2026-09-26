#!/usr/bin/env python3
"""Generate terrain features for a belt area from its DEM.

Usage:
    python scripts/gen_features_for_belt.py --config configs/copperbelt.yml --suffix copperbelt
    python scripts/gen_features_for_belt.py --config configs/usambara.yml --suffix usambara
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio

from src.preprocess.dem_features import dem_features
from src.preprocess.grid import GridSpec
from src.utils import load_config, project_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate terrain features for a belt area"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--suffix",
        required=True,
        help="Suffix for output filename (e.g. copperbelt, usambara)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    grid = GridSpec.from_config(cfg)

    # Find the DEM file
    dem_dir = project_path("data", "interim", "dem")
    dem_files = list(dem_dir.glob(f"*_{args.suffix}_aligned.tif"))
    if not dem_files:
        logger.error(f"No DEM found with suffix '{args.suffix}' in {dem_dir}")
        return

    dem_path = dem_files[0]
    logger.info(f"Reading DEM: {dem_path}")

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float32")
        logger.info(f"DEM shape: {dem.shape}, CRS: {src.crs}")

    # Generate terrain features
    logger.info("Generating terrain features...")
    terrain = dem_features(dem, grid.resolution_m)

    # Write terrain features to multi-band GeoTIFF
    out_dir = project_path("data", "interim", "features")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"terrain_features_{args.suffix}.tif"

    profile = {
        "driver": "GTiff",
        "height": grid.height,
        "width": grid.width,
        "count": len(terrain),
        "dtype": "float32",
        "crs": grid.crs,
        "transform": grid.transform,
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        for i, (name, data) in enumerate(terrain.items(), 1):
            dst.write(data.astype("float32"), i)
            dst.set_band_description(i, name)

    logger.info(f"Terrain features written to {out_path}")
    logger.info(f"Bands: {list(terrain.keys())}")

    # Also generate geology features if world geology data is available
    geology_dir = project_path("data", "raw", "world_geology")
    geology_file = geology_dir / "geology_belt.gpkg"
    contacts_file = geology_dir / "contacts_belt.gpkg"

    if geology_file.exists():
        logger.info("Generating geology features...")
        from src.features.geology_features import build_geology_features
        import geopandas as gpd

        geology = gpd.read_file(geology_file)
        contacts = gpd.read_file(contacts_file) if contacts_file.exists() else None

        geo_path = out_dir / f"geology_features_{args.suffix}.tif"
        temp_dir = out_dir / f".geology_{args.suffix}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        default_geo, _ = build_geology_features(
            grid, geology, contacts, out_dir=temp_dir
        )
        if geo_path.exists():
            geo_path.unlink()
        default_geo.replace(geo_path)
        logger.info(f"Geology features written to {geo_path}")
    else:
        logger.warning(f"World geology data not found at {geology_dir}")
        logger.warning("Using existing geology features (may not align with this area)")


if __name__ == "__main__":
    main()
