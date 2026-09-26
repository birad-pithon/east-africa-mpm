"""Ingest USGS Mineral Resources Data System (MRDS) for Africa.

Downloads the USGS MRDS flattened shapefile from ``mrdata.usgs.gov``,
extracts it to ``data/raw/usgs_mrds/``, filters records to the African
continent and target commodities (Sn, W, Ta, Cu, Zn, Al/bauxite),
and writes the result to ``data/raw/usgs_africa_minerals.gpkg``.
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

DEFAULT_USGS_URL = "https://mrdata.usgs.gov/mrds/mrds-trim.zip"

COMMODITY_MAP: dict[str, str] = {
    "Tin": "SN",
    "Tungsten": "W",
    "Tantalum": "TA",
    "Copper": "CU",
    "Zinc": "ZN",
    "Aluminum": "AL",
}

AFRICA_BBOX = (10.0, -35.0, 50.0, 37.0)


def download_zip(url: str, dest: Path) -> Path:
    """Download a ZIP file from *url* to *dest* with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(
        url, stream=True, timeout=120, headers={"User-Agent": "Mozilla/5.0"}
    )
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="Downloading USGS MRDS"
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    return dest


def _filter_africa(
    gdf: gpd.GeoDataFrame, bbox: tuple = AFRICA_BBOX
) -> gpd.GeoDataFrame:
    """Spatial filter: keep points inside the Africa bbox."""
    min_lon, min_lat, max_lon, max_lat = bbox
    mask = (
        (gdf.geometry.x >= min_lon)
        & (gdf.geometry.x <= max_lon)
        & (gdf.geometry.y >= min_lat)
        & (gdf.geometry.y <= max_lat)
    )
    return gdf[mask].copy()


def _filter_commodities(gdf: gpd.GeoDataFrame, codes: set[str]) -> gpd.GeoDataFrame:
    """Keep records whose commodity column contains any of *codes*.

    MRDS CODE_LIST values are space-separated commodity codes
    (e.g. ``"CU AU AG"``).
    """
    col = None
    for c in gdf.columns:
        if c.lower() in ("commodity", "commodities", "comm_code", "code_list", "code"):
            col = c
            break
    if col is None:
        logger.warning("No commodity column found — returning all.")
        return gdf.copy()

    def _has(val) -> bool:
        if val is None:
            return False
        tokens = set(str(val).upper().split())
        return bool(tokens & codes)

    return gdf[gdf[col].apply(_has)].copy()


def _load_and_clean(zip_path: Path, extract_dir: Path) -> gpd.GeoDataFrame:
    """Extract ZIP and read the shapefile."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    shp_files = list(extract_dir.rglob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp found in {extract_dir}")
    gdf = gpd.read_file(shp_files[0])
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


def ingest_usgs_africa(config_path: str | Path) -> gpd.GeoDataFrame:
    """Full pipeline: download → extract → filter Africa → filter commodities."""
    cfg = load_config(config_path)
    sources = cfg["sources"]
    raw_dir = project_path("data", "raw", "usgs_mrds")
    zip_path = raw_dir / "mrds-trim.zip"
    out_path = project_path("data", "raw", "usgs_africa_minerals.gpkg")

    if not zip_path.exists():
        url = sources.get("usgs_mrds", {}).get("shapefile_url", DEFAULT_USGS_URL)
        download_zip(url, zip_path)

    gdf = _load_and_clean(zip_path, raw_dir)
    logger.info("Total MRDS records loaded: %d", len(gdf))

    cfg_bbox = sources.get("usgs_mrds", {}).get("africa_bbox")
    africa_bbox = (
        (
            cfg_bbox["min_lon"],
            cfg_bbox["min_lat"],
            cfg_bbox["max_lon"],
            cfg_bbox["max_lat"],
        )
        if cfg_bbox
        else AFRICA_BBOX
    )

    gdf_africa = _filter_africa(gdf, africa_bbox)
    logger.info("Records in Africa: %d", len(gdf_africa))

    target_codes = {
        COMMODITY_MAP[c]
        for c in sources.get("usgs_mrds", {}).get("commodities", [])
        if c in COMMODITY_MAP
    }
    if not target_codes:
        target_codes = set(COMMODITY_MAP.values())

    gdf_filtered = _filter_commodities(gdf_africa, target_codes)
    logger.info("Records after commodity filter: %d", len(gdf_filtered))

    # Keep key columns (case-insensitive match against MRDS uppercase names)
    preferred = [
        "site_name",
        "code_list",
        "dev_stat",
        "url",
        "dep_id",
        "country",
        "state",
        "lat_deg",
        "lon_deg",
    ]
    cols_lower = {c.lower(): c for c in gdf_filtered.columns}
    keep = ["geometry"]
    for p in preferred:
        if p in cols_lower and cols_lower[p] not in keep:
            keep.append(cols_lower[p])
    gdf_out = gdf_filtered[keep].copy().to_crs("EPSG:4326")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf_out.to_file(out_path, driver="GPKG")
    logger.info("Saved %d records → %s", len(gdf_out), out_path)
    return gdf_out


# ── CLI ────────────────────────────────────────────────────────────────


def main():
    """CLI entry point for USGS MRDS ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest USGS MRDS mineral data for Africa"
    )
    parser.add_argument(
        "--config",
        default="configs/karagwe.yml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if local file exists"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.force:
        for p in [
            Path("data/raw/usgs_mrds/mrds-trim.zip"),
            Path("data/raw/usgs_africa_minerals.gpkg"),
        ]:
            if p.exists():
                p.unlink()

    gdf = ingest_usgs_africa(args.config)
    print(f"\n✓ USGS MRDS Africa ingestion complete — {len(gdf)} records")
    print("  Saved to: data/raw/usgs_africa_minerals.gpkg")
    print(f"  CRS: {gdf.crs}")


if __name__ == "__main__":
    main()
