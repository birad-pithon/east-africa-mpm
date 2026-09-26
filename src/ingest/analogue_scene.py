"""Analogue target scenes: registry loader + ingest-ready scene configs.

Bridges ``configs/analogue_targets.yaml`` (17 permissive windows) to the
ingest CLIs (``src.ingest.srtm_dem``, ``src.ingest.sentinel2_ee``).  A
scene config is the same schema as the belt configs (``grid`` +
``sources``) but its AOI comes from a target window and its Sentinel-2
date range comes from the target's dry-season window.

Guardrail (see the registry header): a scene config defines a
*permissive search window*, never a label source.  Running ingest for a
window produces imagery on the common grid only; labels still come from
``src.labels.build_labels``.

CLI::

    python -m src.ingest.analogue_scene --list
    python -m src.ingest.analogue_scene --write --id kab_burundi_north
    python -m src.ingest.analogue_scene --write --confidence high
    python -m src.ingest.analogue_scene --aoi --confidence high
    python -m src.ingest.analogue_scene --ingest --id kab_burundi_north
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from src.utils import load_config, project_path

logger = logging.getLogger(__name__)

__all__ = [
    "load_targets",
    "scene_config",
    "write_scene_config",
    "export_aoi",
    "ingest_scene",
]

_REGISTRY = "configs/analogue_targets.yaml"
_DRY_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\s*to\s*(\d{4}-\d{2}-\d{2})")

# Same defaults as configs/karagwe.yml (production S2 band set)
_S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]


def load_targets(
    group: str | None = None,
    confidence: str | None = None,
    target_id: str | None = None,
) -> list[dict[str, Any]]:
    """Load target windows from the analogue registry (optionally filtered)."""
    data = load_config(_REGISTRY)
    targets = data["targets"]
    if group:
        targets = [t for t in targets if t["group"] == group]
    if confidence:
        targets = [t for t in targets if t["confidence"] == confidence]
    if target_id:
        targets = [t for t in targets if t["id"] == target_id]
        if not targets:
            raise KeyError(f"unknown target id: {target_id!r}")
    return targets


def _dry_season_range(dry_season: str) -> tuple[str, str]:
    """Extract the primary (start, end) dates from a dry_season string."""
    m = _DRY_RE.search(dry_season)
    if not m:
        raise ValueError(f"cannot parse dry_season: {dry_season!r}")
    return m.group(1), m.group(2)


def scene_config(target: dict[str, Any]) -> dict[str, Any]:
    """Build an ingest-ready belt-schema config for one target window."""
    start, end = _dry_season_range(target["dry_season"])
    return {
        "_meta": {
            "source": _REGISTRY,
            "target_id": target["id"],
            "group": target["group"],
            "country": target["country"],
            "analogue": target["analogue"],
            "deposit_model": target["deposit_model"],
            "confidence": target["confidence"],
            "enmap_probes_um": target["signatures"]["enmap_probes_um"],
            "dry_season": target["dry_season"],
            "guardrail": "permissive scene-selection window; NOT a label source",
        },
        "grid": {
            "crs": target["crs"],
            "resolution_m": target["resolution_m"],
            "aoi_wgs84": dict(target["bbox_wgs84"]),
            "aoi_utm": None,
        },
        "sources": {
            "sentinel2": {
                "collection": "COPERNICUS/S2_SR_HARMONIZED",
                "bands": list(_S2_BANDS),
                "date_range": {"start": start, "end": end},
                "cloud_cover_threshold": 0.20,
                "scale_m": target["resolution_m"],
            },
            "srtm_dem": {
                "collection": "USGS/SRTMGL1_003",
                "resolution_m": target["resolution_m"],
            },
        },
    }


def write_scene_config(
    target: dict[str, Any],
    out_dir: str | Path | None = None,
) -> Path:
    """Materialise the scene config at ``configs/scenes/<id>.yml``."""
    out = Path(out_dir) if out_dir else project_path("configs", "scenes")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{target['id']}.yml"
    cfg = scene_config(target)
    header = (
        f"# Auto-generated from {_REGISTRY} (target: {target['id']})\n"
        f"# Permissive window — imagery ingest only, never a label source.\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    logger.info("scene config written -> %s", path)
    return path


def export_aoi(
    target: dict[str, Any],
    out_dir: str | Path | None = None,
) -> Path:
    """Write the window bbox as a GeoJSON polygon for EOWEB AOI import.

    The file is a plain WGS84 rectangle FeatureCollection named after the
    target — drag it into the EOWEB GeoPortal map (or paste the bbox
    coordinates from the registry) when searching the EnMAP archive.
    Works fully offline / without credentials.
    """
    b = target["bbox_wgs84"]
    ring = [
        [b["min_lon"], b["min_lat"]],
        [b["max_lon"], b["min_lat"]],
        [b["max_lon"], b["max_lat"]],
        [b["min_lon"], b["max_lat"]],
        [b["min_lon"], b["min_lat"]],
    ]
    fc = {
        "type": "FeatureCollection",
        "name": f"enmap_aoi_{target['id']}",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": target["id"],
                    "group": target["group"],
                    "country": target["country"],
                    "confidence": target["confidence"],
                    "dry_season": target["dry_season"],
                    "crs": target["crs"],
                    "purpose": (
                        "EnMAP archive-search AOI; permissive window, "
                        "NOT a label source"
                    ),
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        ],
    }
    out = (
        Path(out_dir) if out_dir else project_path("data", "raw", "enmap", "aois")
    )
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{target['id']}.geojson"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2)
    logger.info("AOI written -> %s", path)
    return path


def ingest_scene(
    target_id: str,
    dem: bool = True,
    sentinel2: bool = True,
) -> dict[str, Path]:
    """Run the SRTM + Sentinel-2 pulls for one target window.

    Writes the scene config first (always regenerated so the registry
    stays the single source of truth), then calls the existing ingest
    functions.  Without Earth Engine credentials the ingests fall back
    to their synthetic/placeholder rasters on the correct grid — the
    intended dry-run behaviour.
    """
    from src.ingest.sentinel2_ee import pull_sentinel2_tile
    from src.ingest.srtm_dem import pull_dem_tile

    target = load_targets(target_id=target_id)[0]
    cfg_path = write_scene_config(target)
    outputs: dict[str, Path] = {}
    if dem:
        outputs["dem"] = pull_dem_tile(cfg_path, source="srtm", suffix=target_id)
    if sentinel2:
        outputs["s2"] = pull_sentinel2_tile(cfg_path, suffix=target_id)
    for kind, path in outputs.items():
        logger.info("%s -> %s", kind, path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analogue target windows -> scene configs / dry-run ingest"
    )
    parser.add_argument("--list", action="store_true", help="list target windows")
    parser.add_argument("--id", help="target id (e.g. kab_burundi_north)")
    parser.add_argument("--group", help="filter: commodity group")
    parser.add_argument(
        "--confidence", help="filter: confidence tier (high/moderate/low)"
    )
    parser.add_argument(
        "--write", action="store_true", help="write configs/scenes/<id>.yml"
    )
    parser.add_argument(
        "--aoi",
        action="store_true",
        help="write data/raw/enmap/aois/<id>.geojson (no credentials needed)",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="run SRTM + Sentinel-2 pulls (dry-run without EE auth)",
    )
    parser.add_argument("--no-dem", action="store_true", help="skip DEM pull")
    parser.add_argument("--no-s2", action="store_true", help="skip Sentinel-2 pull")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    targets = load_targets(group=args.group, confidence=args.confidence,
                           target_id=args.id)
    if args.list or not (args.write or args.ingest or args.aoi):
        for t in targets:
            print(
                f"{t['id']:32s} {t['group']:22s} {t['country']:11s} "
                f"{t['confidence']:8s} {t['crs']}"
            )
        return
    if args.write and not args.ingest:
        for t in targets:
            print(write_scene_config(t))
    if args.aoi and not args.ingest:
        for t in targets:
            print(export_aoi(t))
    if args.ingest:
        if not args.id:
            parser.error("--ingest requires --id")
        outputs = ingest_scene(args.id, dem=not args.no_dem,
                               sentinel2=not args.no_s2)
        for kind, path in outputs.items():
            print(f"OK {kind}: {path}")


if __name__ == "__main__":
    main()