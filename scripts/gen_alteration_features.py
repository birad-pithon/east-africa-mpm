#!/usr/bin/env python3
"""Generate alteration-spectral + deposit-proximity features for a belt.

P0 additions to the feature store:

* ``alteration_features_<suffix>.tif`` — hydrothermal alteration /
  weathering band ratios from the belt's aligned Sentinel-2 raster
  (Al-OH, Mg-OH/carbonate proxy, ferrous iron, iron-oxide/gossan), see
  ``src.features.spectral``.
* ``deposit_distance_<group>.tif`` — leakage-safe leave-one-out distance
  to known deposits, see ``src.features.distances``.

Usage:
    python scripts/gen_alteration_features.py \
        --config configs/karagwe.yml --suffix kabar \
        --group tin_tungsten_tantalum
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd

from src.features.distances import distance_to_known_deposits_loo
from src.features.spectral import spectral_indices
from src.features.stack import build_feature_stack
from src.preprocess.grid import GridSpec
from src.utils import load_config, project_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate alteration + deposit-proximity features"
    )
    parser.add_argument("--config", required=True,
                        help="Path to YAML configuration file")
    parser.add_argument("--suffix", required=True,
                        help="Belt suffix, e.g. kabar / copperbelt / usambara")
    parser.add_argument("--group", required=True,
                        help="Commodity group key (for labels_<group>.gpkg)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    grid = GridSpec.from_config(cfg)

    out_dir = project_path("data", "interim", "features")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Alteration spectral indices from aligned Sentinel-2 ──────────
    s2_path = (project_path("data", "interim", "sentinel2")
               / f"sentinel2_{args.suffix}_aligned.tif")
    if not s2_path.exists():
        raise FileNotFoundError(f"aligned Sentinel-2 raster missing: {s2_path}")

    logger.info("Computing alteration indices from %s", s2_path)
    indices = spectral_indices(str(s2_path))
    alt_path, bands = build_feature_stack(
        grid, indices, out_dir / f"alteration_features_{args.suffix}.tif")
    logger.info("Alteration features: %d bands -> %s", len(bands), alt_path)

    # ── 2. Leakage-safe distance-to-known-deposits prior ────────────────
    labels_path = project_path("data", "processed", f"labels_{args.group}.gpkg")
    if not labels_path.exists():
        raise FileNotFoundError(f"label GeoPackage missing: {labels_path}")

    deposits = gpd.read_file(labels_path)
    logger.info("Computing LOO deposit-distance prior from %s (%d points)",
                labels_path, len(deposits))
    prior = distance_to_known_deposits_loo(grid, deposits,
                                           res_m=grid.resolution_m)
    dep_path, dep_bands = build_feature_stack(
        grid, prior, out_dir / f"deposit_distance_{args.group}.tif")
    logger.info("Deposit-distance prior: %d band -> %s", len(dep_bands),
                dep_path)


if __name__ == "__main__":
    main()
