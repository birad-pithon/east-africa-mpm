"""Ingest the USGS *General geologic map of the world* (GSC Open File
2915d, ~1:35M) as the lithology / structure base layer.

Served by USGS as OGC web services:

* WFS 1.1.0 — ``https://mrdata.usgs.gov/services/wfs/worldgeol``
  layers ``geology`` (polygons: ``rxtp`` rock-type code, ``terrane``,
  ``agera``) and ``contacts`` (line contacts).

Fetched with GDAL's WFS driver, clipped to a WGS-84 bbox and cached as
GeoPackages under ``data/raw/world_geology/``.  This is the production
stand-in for the BRGM 1:10M Africa map (``src.ingest.brgm_geology``),
whose hosts are unreachable; swap by pointing ``data_sources.yaml`` at a
new source and re-running this module.

Usage::

    from src.ingest.worldgeol import ingest_world_geology
    geol, contacts = ingest_world_geology()

CLI::

    python -m src.ingest.worldgeol [--min-lon 24 --max-lon 34 ...]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd

logger = logging.getLogger(__name__)

WFS_BASE = "WFS:https://mrdata.usgs.gov/services/wfs/worldgeol"

# Belt-scale default AOI (Karagwe-Ankole + Copperbelt edge), WGS-84
BELT_BBOX = (24.0, -6.5, 34.0, -0.5)  # (min_lon, min_lat, max_lon, max_lat)

# Human-readable descriptions for the GENEOL rock-type codes seen in
# central/east Africa (unknown codes fall back to the raw code).
ROCK_TYPE_LABELS: dict[str, str] = {
    "SED": "sedimentary",
    "PLX": "plutonic_or_metamorphic",
    "PLA": "plutonic_granitoid",
    "VOL": "volcanic",
    "SXV": "mixed_sedimentary_volcanic",
    "MET": "metamorphic",
    "PLV": "plutonic_volcanic_mixed",
    "SEDV": "sedimentary_volcanic_mixed",
    "ICE": "ice_or_water",
}


def _cache_paths(cache_dir: Path, layer: str) -> Path:
    return cache_dir / f"{layer}_belt.gpkg"


def _fetch_layer(layer: str, bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """Read one WFS layer clipped to *bbox* (min_lon, min_lat, max_lon, max_lat)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    gdf = gpd.read_file(WFS_BASE, layer=layer, bbox=(min_lon, min_lat, max_lon, max_lat))
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    logger.info("WFS %s: %d features in AOI", layer, len(gdf))
    return gdf


def ingest_world_geology(
    bbox: tuple[float, float, float, float] | None = None,
    cache_dir: str | Path | None = None,
    layers: tuple[str, ...] = ("geology", "contacts"),
    force: bool = False,
) -> dict[str, gpd.GeoDataFrame]:
    """Download (or load cached) USGS world-geology layers for *bbox*.

    Parameters
    ----------
    bbox : (min_lon, min_lat, max_lon, max_lat) in WGS-84; default belt AOI.
    cache_dir : destination folder, default ``data/raw/world_geology``.
    layers : WFS feature types to pull (``geology``, ``contacts``).
    force : re-download even when the cached GeoPackage exists.

    Returns
    -------
    dict mapping layer name -> clipped GeoDataFrame (EPSG:4326).
    """
    bbox = tuple(bbox) if bbox is not None else BELT_BBOX
    cache_dir = Path(cache_dir) if cache_dir else Path("data/raw/world_geology")
    cache_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, gpd.GeoDataFrame] = {}
    for layer in layers:
        path = _cache_paths(cache_dir, layer)
        if path.exists() and not force:
            logger.info("Load cached %s", path)
            out[layer] = gpd.read_file(path)
            continue
        gdf = _fetch_layer(layer, bbox)
        gdf.to_file(path, driver="GPKG")
        logger.info("Cached %d %s features -> %s", len(gdf), layer, path)
        out[layer] = gdf
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest USGS/GSC world geology via WFS"
    )
    parser.add_argument("--min-lon", type=float, default=BELT_BBOX[0])
    parser.add_argument("--min-lat", type=float, default=BELT_BBOX[1])
    parser.add_argument("--max-lon", type=float, default=BELT_BBOX[2])
    parser.add_argument("--max-lat", type=float, default=BELT_BBOX[3])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    layers = ingest_world_geology(
        bbox=(args.min_lon, args.min_lat, args.max_lon, args.max_lat),
        force=args.force,
    )
    print("\nOK world geology ingested:")
    for name, gdf in layers.items():
        print(f"  {name}: {len(gdf)} features")


if __name__ == "__main__":
    main()
