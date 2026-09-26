"""Ingest the BRGM 1:10M geological map of Africa as the continental
lithology / structure base layer.

The map is clipped to the study area (Karabge-Ankole Belt + Copperbelt
extent) and written to ``data/raw/brgm_africa/geology.gpkg``. The
download endpoint is configured in ``configs/data_sources.yaml``
(``sources.brgm_africa_geology``). If a local archive is supplied it is
used instead of a network download, which keeps the pipeline useable when
the endpoint is not yet pinned.

Usage::

    from src.ingest.brgm_geology import ingest_brgm_geology
    gdf = ingest_brgm_geology(local_archive="data/raw/brgm_africa.brgm_download.zip")

CLI::

    python -m src.ingest.brgm_geology --config configs/karagwe.yml
"""

from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path

import geopandas as gpd
import requests
from tqdm import tqdm

from src.utils import load_config, project_path

logger = logging.getLogger(__name__)

# Study-country AOI (East Africa MPM region), WGS-84
# Karagwe-Ankole Belt (Rwanda) + edge of the Central African Copperbelt
EAST_AFRICA_BBOX = {
    "min_lon": 28.0,
    "max_lon": 32.0,
    "min_lat": -6.0,
    "max_lat": -0.5,
}


def _download_archive(url: str, dest: Path) -> Path:
    """Download the BRGM GIS package to *dest*."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(
        url, stream=True, timeout=120, headers={"User-Agent": "Mozilla/5.0"}
    )
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="Downloading BRGM Africa geology"
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    return dest


def _locate_layer(extract_dir: Path) -> Path:
    """Find the first shapefile or GeoPackage inside *extract_dir*."""
    for pat in ("*.shp", "*.gpkg", "*.geojson"):
        hits = list(extract_dir.rglob(pat))
        if hits:
            return hits[0]
    # Esri File Geodatabase directories (folders ending in .gdb)
    for gdb in extract_dir.rglob("*.gdb"):
        for pat in ("*.shp", "*.gpkg"):
            hits = list(gdb.rglob(pat))
            if hits:
                return hits[0]
    raise FileNotFoundError(f"No usable vector layer under {extract_dir}")


def ingest_brgm_geology(
    config_path: str | Path = "configs/karagwe.yml",
    local_archive: str | Path | None = None,
    bbox: dict[str, float] | None = None,
    force: bool = False,
) -> gpd.GeoDataFrame:
    """Load the BRGM Africa geology, clip to the study area, save GeoPackage.

    Parameters
    ----------
    config_path : str | Path
        Path to the pipeline YAML config.
    local_archive : str | Path | None
        Optional local archive (zip) with the BRGM data. When provided the
        network download is skipped.
    bbox : dict | None
        Clip bbox in WGS-84 (keys ``min_lon, max_lon, min_lat, max_lat``).
        Defaults to :data:`EAST_AFRICA_BBOX`.
    force : bool
        Re-download / re-process even if the output exists.

    Returns
    -------
    gpd.GeoDataFrame
        Clipped lithology polygons in EPSG:4326.
    """
    bbox = bbox or EAST_AFRICA_BBOX
    cfg = load_config(config_path)
    src_cfg = cfg["sources"].get("brgm_africa_geology", {})

    raw_dir = project_path("data", "raw", "brgm_africa")
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / "geology_east_africa.gpkg"

    if out_path.exists() and not force:
        logger.info("Load cached %s", out_path)
        return gpd.read_file(out_path)

    # 1. Obtain archive
    cache_zip = raw_dir / "brgm_africa.zip"
    if local_archive is not None:
        cache_zip = Path(local_archive)
    elif src_cfg.get("url"):
        if not cache_zip.exists() or force:
            _download_archive(src_cfg["url"], cache_zip)

    extract_dir = raw_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    if cache_zip.exists() and cache_zip.suffix.lower() == ".zip":
        with zipfile.ZipFile(cache_zip) as zf:
            zf.extractall(extract_dir)

    # 2. Locate + read layer
    layer_path = _locate_layer(extract_dir)
    logger.info("Reading %s", layer_path)
    gdf = gpd.read_file(layer_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    # 3. Clip to study region
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    west, east = bbox["min_lon"], bbox["max_lon"]
    south, north = bbox["min_lat"], bbox["max_lat"]
    clipped = gdf[
        gdf.intersects(
            gpd.GeoSeries([gpd.box(west, south, east, north)], crs="EPSG:4326").iloc[0]
        )
    ].copy()

    # 4. Save
    clipped.to_file(out_path, driver="GPKG")
    logger.info("Saved %d polygons -> %s", len(clipped), out_path)
    return clipped


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest BRGM 1:10M geological map of Africa"
    )
    parser.add_argument("--config", default="configs/karagwe.yml")
    parser.add_argument(
        "--local",
        default=None,
        help="Local BRGM archive (zip) to use instead of download",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    gdf = ingest_brgm_geology(args.config, local_archive=args.local, force=args.force)
    print(f"\nOK BRGM geology ingested: {len(gdf)} polygons")


if __name__ == "__main__":
    main()
