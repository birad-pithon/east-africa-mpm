"""DLR EOC OGC-WMS reconnaissance layers (verified table + GetMap fetch).

The DLR Earth Observation Center (EOC) GeoService publishes two OGC WMS
1.3.0 endpoints that are free of charge and carry no access constraints:

* **imagery** - ``https://geoservice.dlr.de/eoc/imagery/wms``  (28 layers)
* **land**    - ``https://geoservice.dlr.de/eoc/land/wms``     (141 layers)

Both were evaluated against the three belt AOIs on 2026-09-22; the
measured tile opacity per belt is recorded in :data:`VERIFIED_LAYERS`.
Outcome of that evaluation:

* ``imagery`` is **not** usable as a data source here.  19 of its 28
  layers are Germany / Europe only (Sentinel-2 MAJA + WASP mosaics,
  MODIS-EU >= 27.7 deg N, RapidEye Germany 2015, 3K Brunswick aerial).
  Its only global layers are the ``ENMAP_HSI_L0_QL_*`` quicklooks and
  ``S2_TILE_GRID`` - and every EnMAP quicklook layer renders an empty
  (0 % opaque) tile for every window tested, *including a Germany
  control*, so the layer currently carries no data.
* ``land`` **is** useful over East Africa as a *context / screening*
  layer set: the SoilSuite bare-surface reflectance composite (Africa),
  the TanDEM-X forest/non-forest mask, World Settlement Footprint,
  ESA-CCI land-ocean and the TimeScan Landsat composite.

WMS returns an **8-bit colour-stretched rendering**, not digital numbers,
so these layers are for visual reconnaissance, QA of probability maps and
mask screening - they are *not* radiometric inputs for feature mining.
Radiometric hyperspectral ingest stays on the EnMAP L2A route
(:mod:`src.ingest.enmap`).

Usage::

    from src.ingest.dlr_eoc_wms import fetch_wms_map, VERIFIED_LAYERS
    png = fetch_wms_map("soilsuite_src_africa", (37.9, -5.6, 39.1, -4.1))

CLI::

    python -m src.ingest.dlr_eoc_wms --list
    python -m src.ingest.dlr_eoc_wms --layer soilsuite_src_africa \\
        --bbox 37.9,-5.6,39.1,-4.1 --out data/raw/dlr/soilsuite_usambara.png
"""
from __future__ import annotations

import io
import logging
import math
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from src.utils import project_path

logger = logging.getLogger(__name__)

__all__ = [
    "IMAGERY_WMS_URL",
    "LAND_WMS_URL",
    "EAST_AFRICA_BBOX",
    "BELT_BBOX",
    "VERIFIED_LAYERS",
    "DASHBOARD_LAYERS",
    "wgs84_to_mercator",
    "bbox_to_mercator",
    "build_getmap_url",
    "fetch_wms_map",
    "save_wms_map",
    "opaque_fraction",
    "is_blank_tile",
    "layer_summary",
    "fetch_belt_context",
]

IMAGERY_WMS_URL = "https://geoservice.dlr.de/eoc/imagery/wms"
LAND_WMS_URL = "https://geoservice.dlr.de/eoc/land/wms"

SERVICE_URLS = {"imagery": IMAGERY_WMS_URL, "land": LAND_WMS_URL}

#: Continental envelope used for coverage screening (WGS-84 degrees).
EAST_AFRICA_BBOX = {
    "min_lon": 25.0,
    "min_lat": -14.0,
    "max_lon": 42.0,
    "max_lat": 3.0,
}

#: Production belt envelopes (WGS-84 degrees), matching the belt configs.
BELT_BBOX = {
    "tin_tungsten_tantalum": (29.0, -3.2, 31.4, -0.9),    # Karagwe-Ankole
    "copper_zinc": (25.9, -12.0, 28.6, -9.4),             # Central African Copperbelt
    "bauxite": (37.9, -5.6, 39.1, -4.1),                  # Usambara laterite plateau
}

#: Layers whose WMS rendering was verified on 2026-09-22 over the belt
#: AOIs.  ``opaque`` is the measured fraction of non-transparent pixels in
#: a 512 x 512 PNG render over that belt (``None`` = empty / no coverage);
#: ``None`` for every belt means the layer renders nothing at all.
VERIFIED_LAYERS: dict[str, dict[str, Any]] = {
    "soilsuite_src_africa": {
        "service": "land",
        "layer": "SOILSUITE_SRC_AFR_P4Y",
        "style": "soilsuite-src-afr-p4y-fin",
        "title": "SoilSuite - Bare Surface Reflectance Composite - Mean, Africa",
        "coverage": "Africa 28.75E-50.31E / 7.22S-16.06N (4-year composite)",
        "use": "Regional bare-ground reflectance context for laterite / bauxite "
               "and soil-regolith screening; no Copperbelt coverage",
        "opaque": {"tin_tungsten_tantalum": 0.632, "copper_zinc": None,
                   "bauxite": 0.278},
        "status": "usable",
    },
    "soilsuite_src_ci95_africa": {
        "service": "land",
        "layer": "SOILSUITE_SRC-CI95_AFR_P4Y",
        "style": "soilsuite-src-ci95-afr-p4y-fin",
        "title": "SoilSuite - Bare Surface Reflectance Composite - 95% "
                 "Confidence, Africa",
        "coverage": "Africa 28.75E-50.31E / 7.22S-16.06N",
        "use": "Confidence companion to the SoilSuite mean composite",
        "opaque": {"tin_tungsten_tantalum": 0.632, "copper_zinc": None,
                   "bauxite": 0.278},
        "status": "usable",
    },
    "soilsuite_src_std_africa": {
        "service": "land",
        "layer": "SOILSUITE_SRC-STD_AFR_P4Y",
        "style": "soilsuite-src-std-afr-p4y-fin",
        "title": "SoilSuite - Bare Surface Reflectance Composite - Standard "
                 "deviation, Africa",
        "coverage": "Africa 28.75E-50.31E / 7.22S-16.06N",
        "use": "Per-pixel temporal variability of the bare-surface composite "
               "(instability / land-cover change indicator)",
        "opaque": {"tin_tungsten_tantalum": 0.632, "copper_zinc": None,
                   "bauxite": 0.277},
        "status": "usable",
    },
    "soilsuite_bare_frequency_africa": {
        "service": "land",
        "layer": "SOILSUITE_SFREQ-BSF_AFR_P4Y",
        "style": "soilsuite-sfreq-bsf-afr-p4y-fin",
        "title": "SoilSuite - Bare surface frequency, Africa",
        "coverage": "Africa 28.75E-50.31E / 7.22S-16.06N",
        "use": "Exposure mask: how often a pixel is bare ground - screening "
               "weight for alteration / laterite detection",
        "opaque": {"tin_tungsten_tantalum": 1.0, "copper_zinc": None,
                   "bauxite": 1.0},
        "status": "usable",
    },
    "tandemx_forest_nonforest": {
        "service": "land",
        "layer": "TDM_FNF50",
        "style": "fnf",
        "title": "TanDEM-X Forest/Non-Forest (FNF), 50 m",
        "coverage": "global 58S-79N (categorical, 4 classes rendered)",
        "use": "Vegetation mask for every belt incl. the Copperbelt - supports "
               "the existing vegetation / canopy features",
        "opaque": {"tin_tungsten_tantalum": 1.0, "copper_zinc": 1.0,
                   "bauxite": 1.0},
        "status": "usable",
    },
    "world_settlement_footprint_2019": {
        "service": "land",
        "layer": "WSF_2019",
        "style": "wsf2019",
        "title": "World Settlement Footprint (WSF) 2019",
        "coverage": "global 60S-78N",
        "use": "Anthropogenic-disturbance mask (settlements, mining towns, "
               "infrastructure) to separate artefacts from geological anomalies",
        "opaque": {"tin_tungsten_tantalum": 0.348, "copper_zinc": 0.020,
                   "bauxite": 0.082},
        "status": "usable",
    },
    "world_settlement_evolution": {
        "service": "land",
        "layer": "WSF_Evolution",
        "style": "wsfevolution",
        "title": "World Settlement Footprint (WSF) Evolution",
        "coverage": "global 60S-78N",
        "use": "Settlement change over time - flags growing artisanal / "
               "small-scale mining (relevant to the Sn-W-Ta and Cu-Zn belts)",
        "opaque": {"tin_tungsten_tantalum": 0.004, "copper_zinc": None,
                   "bauxite": None},
        "status": "usable",
    },
    "guf28_urban_footprint": {
        "service": "land",
        "layer": "GUF28_DLR_v1_Mosaic",
        "style": "guf_8bit",
        "title": "Global Urban Footprint (GUF) 28 m - DLR v1 mosaic",
        "coverage": "global 65S-85N (non-commercial use only)",
        "use": "Coarse urban / built-up context; sparse over the rural belts",
        "opaque": {"tin_tungsten_tantalum": 0.019, "copper_zinc": 0.009,
                   "bauxite": 0.009},
        "status": "usable",
    },
    "esa_land_ocean_map": {
        "service": "land",
        "layer": "ESA_LAND_OCEAN_MAP",
        "style": "oceanmask",
        "title": "ESA-CCI Land-Ocean Map",
        "coverage": "global",
        "use": "Land / water mask (Lake Victoria, Tanganyika, Kivu, Mweru) for "
               "trimming the analysis grid",
        "opaque": {"tin_tungsten_tantalum": 0.089, "copper_zinc": None,
                   "bauxite": None},
        "status": "usable",
    },
    "timescan_landsat_2015": {
        "service": "land",
        "layer": "TS_LANDSAT_2015",
        "style": "timescan-standard",
        "title": "TimeScan - Landsat - Global, 2015",
        "coverage": "global 90S-90N",
        "use": "Independent Landsat median composite for visual QA of the "
               "probability surface (different sensor to the model inputs)",
        "opaque": {"tin_tungsten_tantalum": 1.0, "copper_zinc": 1.0,
                   "bauxite": 1.0},
        "status": "usable",
    },
    "countries": {
        "service": "land",
        "layer": "ne_countries",
        "style": "",
        "title": "Countries (Natural Earth)",
        "coverage": "global",
        "use": "National boundaries for AOI masks and map labelling",
        "opaque": {"tin_tungsten_tantalum": 0.012, "copper_zinc": None,
                   "bauxite": None},
        "status": "usable",
    },
    "s2_tile_grid": {
        "service": "imagery",
        "layer": "S2_TILE_GRID",
        "style": "s2-tile-grid",
        "title": "Sentinel-2 Tile Grid System",
        "coverage": "global",
        "use": "Reference grid only - enumerating Sentinel-2 tiles when "
               "planning an ingest run (the imagery service's only working "
               "global layer)",
        "opaque": {"tin_tungsten_tantalum": 1.0, "copper_zinc": 1.0,
                   "bauxite": 1.0},
        "status": "usable",
    },
    "enmap_l0_quicklook_footprints": {
        "service": "imagery",
        "layer": "ENMAP_HSI_L0_QL_FOOTPRINTS",
        "style": "enmap-l0-ql-footprints",
        "title": "EnMAP HSI L0 Quicklook Footprints",
        "coverage": "declared global, but the layer renders EMPTY (0% opaque; "
                    "GetFeatureInfo numberReturned=0 everywhere incl. Europe)",
        "use": "NOT usable - registered for traceability only; EnMAP scene "
               "discovery stays with the DLR EOWEB portal / STAC search",
        "opaque": {"tin_tungsten_tantalum": None, "copper_zinc": None,
                   "bauxite": None},
        "status": "empty",
    },
    "enmap_l0_quicklook_swir": {
        "service": "imagery",
        "layer": "ENMAP_HSI_L0_QL_SWIR",
        "style": "enmap-l0-ql-swir",
        "title": "EnMAP HSI L0 Quicklook SWIR (1050 / 1650 / 2200 nm false colour)",
        "coverage": "declared global, but the layer renders EMPTY (0% opaque) - "
                    "no EnMAP quicklook over the belts",
        "use": "NOT usable - hyperspectral ingest remains EnMAP L2A via "
               "src.ingest.enmap; registered for traceability only",
        "opaque": {"tin_tungsten_tantalum": None, "copper_zinc": None,
                   "bauxite": None},
        "status": "empty",
    },
}

#: Verified non-empty layers offered as toggles in the dashboard.
DASHBOARD_LAYERS = (
    "soilsuite_src_africa",
    "soilsuite_src_ci95_africa",
    "soilsuite_bare_frequency_africa",
    "tandemx_forest_nonforest",
    "world_settlement_footprint_2019",
    "esa_land_ocean_map",
    "timescan_landsat_2015",
    "countries",
    "s2_tile_grid",
)

def wgs84_to_mercator(lon: float, lat: float) -> tuple[float, float]:
    """Convert WGS-84 degrees to Web Mercator (EPSG:3857) metres.

    Lets callers pass human-friendly ``min_lon,min_lat,max_lon,max_lat``
    boxes while talking to the service in EPSG:3857, the CRS ``folium``
    renders in.
    """
    lat = max(min(lat, 89.99), -89.99)  # guard the tan() singularity
    x = lon * 20037508.342789244 / 180.0
    y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    return x, y * 20037508.342789244 / 180.0


def bbox_to_mercator(
    bbox: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Convert ``(min_lon, min_lat, max_lon, max_lat)`` to EPSG:3857 bounds."""
    min_lon, min_lat, max_lon, max_lat = bbox
    x0, y0 = wgs84_to_mercator(min_lon, min_lat)
    x1, y1 = wgs84_to_mercator(max_lon, max_lat)
    return x0, y0, x1, y1


def _resolve(layer: str) -> dict[str, Any]:
    """Look up a verified layer entry (accepts the key or the WMS name)."""
    if layer in VERIFIED_LAYERS:
        return VERIFIED_LAYERS[layer]
    for spec in VERIFIED_LAYERS.values():
        if spec["layer"].lower() == layer.lower():
            return spec
    raise KeyError(
        f"unknown WMS layer {layer!r}; known keys: {sorted(VERIFIED_LAYERS)}"
    )


def build_getmap_url(
    layer: str,
    bbox: tuple[float, float, float, float],
    width: int = 512,
    height: int = 512,
    crs: str = "EPSG:3857",
) -> str:
    """Build a WMS 1.3.0 GetMap URL for a verified layer.

    ``bbox`` is ``(min_lon, min_lat, max_lon, max_lat)`` in WGS-84 and is
    re-projected to *crs* (``EPSG:3857`` by default; ``CRS:84`` also works
    and is passed through unchanged).
    """
    spec = _resolve(layer)
    base = SERVICE_URLS[spec["service"]]
    if crs.upper() == "CRS:84":
        min_x, min_y, max_x, max_y = bbox
    else:
        min_x, min_y, max_x, max_y = bbox_to_mercator(bbox)
    params = {
        "service": "WMS",
        "version": "1.3.0",
        "request": "GetMap",
        "layers": spec["layer"],
        "styles": spec["style"],
        "crs": crs,
        "bbox": f"{min_x:.3f},{min_y:.3f},{max_x:.3f},{max_y:.3f}",
        "width": str(width),
        "height": str(height),
        "format": "image/png",
        "transparent": "TRUE",
    }
    return f"{base}?{urllib.parse.urlencode(params)}"


def _raise_on_service_exception(payload: bytes) -> None:
    """Turn a WMS ``ServiceExceptionReport`` body into a Python error."""
    if b"ServiceException" in payload[:2000]:
        text = payload[:400].decode("utf-8", "replace")
        raise RuntimeError(f"WMS ServiceException: {text}")


def fetch_wms_map(
    layer: str,
    bbox: tuple[float, float, float, float],
    width: int = 512,
    height: int = 512,
    timeout: int = 120,
) -> bytes:
    """Download a rendered GetMap PNG for a verified layer / belt window.

    Raises
    ------
    RuntimeError
        If the service answers with a ``ServiceExceptionReport``.
    urllib.error.URLError
        On network failure.
    """
    url = build_getmap_url(layer, bbox, width=width, height=height)
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        payload = resp.read()
    _raise_on_service_exception(payload)
    return payload


def save_wms_map(
    layer: str,
    bbox: tuple[float, float, float, float],
    out_path: str | Path,
    width: int = 512,
    height: int = 512,
    timeout: int = 120,
) -> Path:
    """Fetch a WMS render and write it to *out_path* (PNG)."""
    payload = fetch_wms_map(layer, bbox, width=width, height=height, timeout=timeout)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    logger.info("wrote %s (%d bytes, layer=%s)", out, len(payload), layer)
    return out


def opaque_fraction(payload: bytes) -> float:
    """Fraction of non-transparent pixels in a rendered PNG.

    A WMS layer with no data over the requested window returns a fully
    transparent tile, so this is the coverage test used in the evaluation
    recorded in :data:`VERIFIED_LAYERS`.
    """
    from PIL import Image

    image = Image.open(io.BytesIO(payload)).convert("RGBA")
    hist = image.getchannel("A").histogram()
    total = sum(hist)
    return float(sum(hist[1:]) / total) if total else 0.0


def is_blank_tile(payload: bytes, threshold: float = 0.001) -> bool:
    """True when a rendered tile is (near-)empty, i.e. no data coverage."""
    return opaque_fraction(payload) <= threshold


def layer_summary() -> str:
    """Plain-text inventory of the verified layers (no network access)."""
    lines = [
        "DLR EOC WMS - verified layers (evaluated 2026-09-22)",
        f"  imagery service: {IMAGERY_WMS_URL}",
        f"  land service   : {LAND_WMS_URL}",
        "",
    ]
    for key, spec in VERIFIED_LAYERS.items():
        lines.append(f"{key}  [{spec['status']}]")
        lines.append(f"    wms layer : {spec['layer']}")
        lines.append(f"    style     : {spec['style']!r}")
        lines.append(f"    title     : {spec['title']}")
        lines.append(f"    coverage  : {spec['coverage']}")
        lines.append(f"    use       : {spec['use']}")
        lines.append(f"    opaque    : {spec['opaque']}")
        lines.append("")
    return "\n".join(lines)

def fetch_belt_context(
    belt: str | tuple[float, float, float, float],
    keys: list[str] | None = None,
    out_dir: str | Path = "data/raw/dlr",
    width: int = 1024,
    height: int = 1024,
) -> list[Path]:
    """Download the context layers for one belt into *out_dir*.

    Parameters
    ----------
    belt : str or tuple
        Belt key from :data:`BELT_BBOX` (``tin_tungsten_tantalum``,
        ``copper_zinc``, ``bauxite``) or a ``(w, s, e, n)`` WGS-84 box.
    keys : list[str], optional
        Subset of :data:`DASHBOARD_LAYERS`; defaults to all of them.
    out_dir : str or Path
        Destination directory (created if missing); a relative path is
        resolved against the project root.

    Returns
    -------
    list[Path]
        Written PNG paths, in request order.  Layers that fail or that are
        flagged ``empty`` are skipped (and logged) rather than aborting.
    """
    if isinstance(belt, tuple):
        bbox = belt
    elif belt in BELT_BBOX:
        bbox = BELT_BBOX[belt]
    else:
        raise KeyError(f"unknown belt {belt!r}; known: {sorted(BELT_BBOX)}")

    keys = list(keys or DASHBOARD_LAYERS)
    base = Path(out_dir)
    if not base.is_absolute():
        base = project_path(*base.parts)

    written: list[Path] = []
    for key in keys:
        spec = _resolve(key)
        if spec["status"] != "usable":
            logger.warning("skipping empty WMS layer %s", key)
            continue
        out = base / f"{belt}_{key}.png"
        try:
            written.append(save_wms_map(key, bbox, out, width=width, height=height))
        except Exception as exc:  # noqa: BLE001 - keep going across layers
            logger.error("failed to fetch %s for %s: %s", key, belt, exc)
    return written


def main() -> None:
    """CLI entry point: list the layer table or download belt contexts."""
    import argparse

    parser = argparse.ArgumentParser(
        description="DLR EOC WMS context layers for the East Africa belts")
    parser.add_argument("--list", action="store_true",
                        help="print the verified layer inventory and exit")
    parser.add_argument("--layer", help="verified layer key or WMS layer name")
    parser.add_argument("--belt", choices=sorted(BELT_BBOX),
                        help="download every dashboard layer for this belt")
    parser.add_argument("--bbox", help="min_lon,min_lat,max_lon,max_lat (WGS-84)")
    parser.add_argument("--out", help="output PNG path (single layer) or "
                                      "directory (--belt)")
    parser.add_argument("--size", type=int, default=1024,
                        help="raster edge length in pixels (default 1024)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.list or (not args.layer and not args.belt):
        print(layer_summary())
        return

    if args.belt:
        out_dir = args.out or "data/raw/dlr"
        for path in fetch_belt_context(args.belt, out_dir=out_dir,
                                       width=args.size, height=args.size):
            print(path)
        return

    if not args.bbox:
        parser.error("--bbox is required with --layer")
    bbox = tuple(float(v) for v in args.bbox.split(","))
    if len(bbox) != 4:
        parser.error("--bbox must have four comma-separated values")
    out = args.out or f"data/raw/dlr/{args.layer}.png"
    print(save_wms_map(args.layer, bbox, out, width=args.size, height=args.size))


if __name__ == "__main__":
    main()
