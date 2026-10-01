"""Pytest configuration: fixtures shared across test modules."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `src` importable no matter where pytest is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "configs" / "karagwe.yml"


@pytest.fixture(scope="session")
def config() -> dict:
    """Load the Karagwe-Ankole Belt YAML config once per session."""
    from src.utils import load_config

    return load_config(CONFIG_PATH)


@pytest.fixture(scope="session")
def grid_params(config: dict) -> dict:
    """Extract common grid parameters from the config."""
    g = config["grid"]
    return {
        "crs": g["crs"],
        "resolution_m": g["resolution_m"],
        "aoi_wgs84": g["aoi_wgs84"],
    }


@pytest.fixture(scope="session")
def usgs_data(config: dict, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run USGS MRDS ingestion once per session and return the GeoPackage path."""
    from src.utils import project_path

    out = project_path("data", "raw", "usgs_africa_minerals.gpkg")
    if not out.exists():
        from src.ingest.usgs_africa import ingest_usgs_africa

        ingest_usgs_africa(str(CONFIG_PATH))
    return out


@pytest.fixture(scope="session")
def common_grid(config: dict) -> dict:
    """Compute the common analysis grid (CRS, transform, dims)."""
    from src.utils import (
        grid_dimensions,
        make_grid_transform,
        wgs84_to_utm,
    )

    g = config["grid"]
    crs = g["crs"]
    res = g["resolution_m"]
    bounds_utm = wgs84_to_utm(g["aoi_wgs84"], crs)
    transform = make_grid_transform(crs, res, bounds_utm)
    width, height = grid_dimensions(bounds_utm, transform)
    return {
        "crs": crs,
        "resolution_m": res,
        "transform": transform,
        "width": width,
        "height": height,
        "bounds_utm": bounds_utm,
    }
