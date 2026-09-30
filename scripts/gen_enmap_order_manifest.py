#!/usr/bin/env python3
"""Turn model targets into a deduplicated, scene-level EnMAP order manifest.

`outputs/maps/tasking_<group>.geojson` holds one 2 km circle per ranked grid
cell, so adjacent cells inside one anomaly become near-identical requests:
bauxite's 10 exploit cells span roughly 6 x 8 km, i.e. a single EnMAP scene,
yet the tasking file presents them as 10 separate polygons. Per the EOWEB
GeoPortal user guide an order cannot be changed after submission and cannot be
cancelled through the portal at all (cancellation is an e-mail to the DLR
order desk), so redundant order items are expensive to undo.

This script collapses the tasking targets into scene-sized AOI boxes
(single-linkage clustering at the ~30 km EnMAP swath), pairs each scene with
the order parameters it needs, and writes:

  * ``outputs/enmap/enmap_order_manifest.md``  - order runbook + tracker
  * ``data/raw/enmap/aois_model/*.geojson``    - drag-and-drop AOI per scene

Placing the order stays manual: EOWEB GeoPortal exposes no documented API
(the EGP 2.2.0 user guide documents mouse-driven ordering only), so this
script deliberately stops at the manifest.

Usage:
    python scripts/gen_enmap_order_manifest.py
    python scripts/gen_enmap_order_manifest.py --swath-km 30 --pad-km 2
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from functools import cache
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config, project_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TASKING = {
    "tin_tungsten_tantalum": "outputs/maps/tasking_tin_tungsten_tantalum.geojson",
    "copper_zinc": "outputs/maps/tasking_copper_zinc.geojson",
    "bauxite": "outputs/maps/tasking_bauxite.geojson",
}
TIERS = ("top_prob", "margin")
TIER_LABEL = {"top_prob": "exploit", "margin": "explore"}


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in km.

    Matches the convention used by ``src.models.holdout._haversine_km`` so a
    cluster radius here means the same thing as a holdout buffer radius there.
    """
    rlon1, rlat1 = math.radians(lon1), math.radians(lat1)
    rlon2, rlat2 = math.radians(lon2), math.radians(lat2)
    dlon, dlat = rlon2 - rlon1, rlat2 - rlat1
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 2.0 * 6371.0088 * math.asin(math.sqrt(min(1.0, a)))


def cluster_indices(points: list[tuple[float, float]],
                    thresh_km: float) -> list[list[int]]:
    """Single-linkage clusters: indices of points closer than *thresh_km*.

    Union-find rather than a full distance matrix, so 50 targets stay cheap.
    Returns a list of index lists, largest cluster first.
    """
    parent = list(range(len(points)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if haversine_km(*points[i], *points[j]) <= thresh_km:
                root_i, root_j = find(i), find(j)
                if root_i != root_j:
                    parent[root_i] = root_j

    buckets: dict[int, list[int]] = {}
    for i in range(len(points)):
        buckets.setdefault(find(i), []).append(i)
    return sorted(buckets.values(), key=len, reverse=True)


def bbox_of(points: list[tuple[float, float]], pad_km: float) -> dict[str, float]:
    """WGS84 bbox of *points*, grown by *pad_km* on every side."""
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    lat_pad = pad_km / 110.574
    mid_lat = (min_lat + max_lat) / 2.0
    lon_pad = pad_km / (111.320 * max(0.1, math.cos(math.radians(mid_lat))))
    return {
        "min_lon": round(min_lon - lon_pad, 4),
        "max_lon": round(max_lon + lon_pad, 4),
        "min_lat": round(min_lat - lat_pad, 4),
        "max_lat": round(max_lat + lat_pad, 4),
    }


def load_tier_targets(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Read a tasking GeoJSON and split its features by strategy tier."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, list[dict[str, Any]]] = {t: [] for t in TIERS}
    for feat in data.get("features", []):
        props = feat.get("properties") or {}
        tier = props.get("strategy")
        if tier in out:
            out[tier].append(props)
    for tier in out:
        out[tier].sort(key=lambda p: p.get("rank", 0))
    return out


@cache
def _windows_for_group(group: str) -> tuple[dict[str, Any], ...]:
    """Registered analogue windows for *group*, from the target registry."""
    registry = load_config(project_path("configs", "analogue_targets.yaml"))
    return tuple(t for t in registry.get("targets", []) if t.get("group") == group)


def _inside(bbox: dict[str, float], lon: float, lat: float) -> bool:
    return (bbox["min_lon"] <= lon <= bbox["max_lon"]
            and bbox["min_lat"] <= lat <= bbox["max_lat"])


def season_context(group: str, lon: float, lat: float) -> str:
    """Dry-season guidance for a scene centre, from the analogue registry.

    Preference order: the window that contains the centre, else the nearest
    window of the same group (flagged, because the season may not transfer),
    else no guidance at all.
    """
    windows = _windows_for_group(group)
    if not windows:
        return "no registered window - pick driest local window"
    for win in windows:
        if _inside(win["bbox_wgs84"], lon, lat):
            return f"{win['dry_season']} (inside `{win['id']}`)"
    nearest = min(windows, key=lambda w: haversine_km(
        lon, lat,
        (w["bbox_wgs84"]["min_lon"] + w["bbox_wgs84"]["max_lon"]) / 2,
        (w["bbox_wgs84"]["min_lat"] + w["bbox_wgs84"]["max_lat"]) / 2))
    dist = haversine_km(
        lon, lat,
        (nearest["bbox_wgs84"]["min_lon"] + nearest["bbox_wgs84"]["max_lon"]) / 2,
        (nearest["bbox_wgs84"]["min_lat"] + nearest["bbox_wgs84"]["max_lat"]) / 2)
    return (f"outside all registered windows (nearest `{nearest['id']}` "
            f"{dist:.0f} km) - verify season before ordering")


@cache
def _probe_wavelengths(group: str) -> tuple[float, ...]:
    """Cached probe wavelengths; returned as a tuple so callers cannot mutate."""
    fps = load_config(project_path("configs", "spectral_fingerprints.yaml"))
    entry = (fps.get("fingerprints") or {}).get(group) or {}
    return tuple((entry.get("hyperspectral") or {}).get("probe_wavelengths_um") or [])


def probes_for_group(group: str) -> list[float]:
    """EnMAP probe wavelengths for *group* from the fingerprint registry."""
    return list(_probe_wavelengths(group))


def build_scenes(group: str, tier: str, targets: list[dict[str, Any]],
                 swath_km: float, pad_km: float) -> list[dict[str, Any]]:
    """Collapse one tier's targets into scene-sized AOI boxes."""
    if not targets:
        return []
    points = [(t["lon"], t["lat"]) for t in targets]
    scenes: list[dict[str, Any]] = []
    for idxs in cluster_indices(points, swath_km):
        members = [targets[i] for i in idxs]
        member_pts = [points[i] for i in idxs]
        bbox = bbox_of(member_pts, pad_km)
        centre_lon = round(sum(p[0] for p in member_pts) / len(member_pts), 4)
        centre_lat = round(sum(p[1] for p in member_pts) / len(member_pts), 4)
        width = haversine_km(bbox["min_lon"], centre_lat, bbox["max_lon"], centre_lat)
        height = haversine_km(centre_lon, bbox["min_lat"], centre_lon, bbox["max_lat"])
        scenes.append({
            "scene_id": "",
            "group": group,
            "tier": tier,
            "n_cells": len(members),
            "max_prob": max(m.get("prob", 0.0) for m in members),
            "best_rank": min(m.get("rank", 0) for m in members),
            "centre_lon": centre_lon,
            "centre_lat": centre_lat,
            "bbox": bbox,
            "width_km": round(width, 1),
            "height_km": round(height, 1),
            "tiles": max(1, math.ceil(width / swath_km)
                         * math.ceil(height / swath_km)),
            "dry_season": season_context(group, centre_lon, centre_lat),
        })
    scenes.sort(key=lambda s: s["best_rank"])
    for n, scene in enumerate(scenes, start=1):
        scene["scene_id"] = f"{group}_{TIER_LABEL[tier]}_{n:02d}"
    return scenes


def write_aoi(scene: dict[str, Any], out_dir: str | Path | None = None) -> Path:
    """Write one scene bbox as a plain WGS84 GeoJSON rectangle.

    Same shape as ``src.ingest.analogue_scene.export_aoi`` so both kinds of
    AOI drop into the EOWEB map the same way.
    """
    b = scene["bbox"]
    ring = [
        [b["min_lon"], b["min_lat"]],
        [b["max_lon"], b["min_lat"]],
        [b["max_lon"], b["max_lat"]],
        [b["min_lon"], b["max_lat"]],
        [b["min_lon"], b["min_lat"]],
    ]
    fc = {
        "type": "FeatureCollection",
        "name": f"enmap_aoi_{scene['scene_id']}",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": scene["scene_id"],
                    "group": scene["group"],
                    "tier": TIER_LABEL[scene["tier"]],
                    "n_cells": scene["n_cells"],
                    "max_prob": scene["max_prob"],
                    "dry_season": scene["dry_season"],
                    "purpose": (
                        "EnMAP order AOI from model target clusters; "
                        "permissive window, NOT a label source"
                    ),
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        ],
    }
    out = (Path(out_dir) if out_dir
           else project_path("data", "raw", "enmap", "aois_model"))
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{scene['scene_id']}.geojson"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2)
    return path


def prune_stale_aois(
    scenes_by_group: dict[str, list[dict[str, Any]]],
    out_dir: str | Path | None = None,
) -> list[Path]:
    """Delete AOI files from an earlier run whose scene no longer exists.

    :func:`write_aoi` only ever overwrites the files it still produces, so when
    the target set shrinks the leftover rectangles keep describing ground that
    is no longer targeted. An EnMAP order cannot be cancelled through EOWEB,
    so a stale drag-and-drop AOI is a genuine hazard rather than untidy:
    remove it.

    Only files that look like this script's own output are considered
    (``<group>_<tier>_<NN>.geojson`` for the groups in *scenes_by_group*);
    anything else in the directory is left alone, as are the AOIs belonging to
    groups that were skipped this run.
    """
    out = (Path(out_dir) if out_dir
           else project_path("data", "raw", "enmap", "aois_model"))
    if not out.is_dir():
        return []
    keep = {scene["scene_id"]
            for scenes in scenes_by_group.values() for scene in scenes}
    ours = tuple(f"{group}_{TIER_LABEL[tier]}_"
                 for group in scenes_by_group for tier in TIERS)
    removed: list[Path] = []
    for path in sorted(out.glob("*.geojson")):
        if path.stem in keep or not path.stem.startswith(ours):
            continue
        path.unlink()
        removed.append(path)
    return removed


def render_manifest(scenes_by_group: dict[str, list[dict[str, Any]]],
                    pad_km: float) -> str:
    """Render the order runbook + tracker as markdown."""
    lines: list[str] = [
        "# EnMAP order manifest — model anomaly targets",
        "",
        "**Generated by:** `scripts/gen_enmap_order_manifest.py`",
        "**Scope:** the exploit/explore cells in `outputs/maps/tasking_*.geojson`,",
        "collapsed to EnMAP scene-sized AOIs.",
        "",
        "## Why this file exists",
        "",
        "`tasking_*.geojson` holds one buffered circle per ranked grid cell, so",
        "cells inside a single anomaly become near-identical requests. Ordering",
        "them one by one would place dozens of orders for a handful of distinct",
        "pieces of ground.",
        "",
        "**Ordering stays manual.** The EOWEB GeoPortal 2.2.0 user guide",
        "documents mouse-driven ordering only (Products → AOI filter → Order →",
        "Cart → checkout → licence → Submit); no public API, token flow or",
        "scripted-order endpoint is documented, so no script can place these.",
        "This manifest is the step before that: one row per order item.",
        "",
        "> Per the same user guide an order **cannot be changed after",
        "> submission** and **cannot be cancelled through EOWEB** — cancellation",
        "> is an e-mail to `DFD-Orderdesk@dlr.de`. Check every row before you",
        "> submit.",
        "",
        "## Before you order anything",
        "",
        "1. Run an **archive-first** search for each AOI (EOWEB → Products tab →",
        "   import the GeoJSON below → collection `EnMAP` → filter **L2A** →",
        "   date range from the `dry season` column → sort by ascending cloud).",
        "   EnMAP has been acquiring since 2022, so much of this may already be",
        "   on the shelf and need no new acquisition.",
        "2. Only if the archive is empty, request on-demand L1B/L1C/L2A",
        "   regeneration (same menu; the processor runs within ~6 days of",
        "   downlink).",
        "3. Only if the area was never acquired, submit a targeted-acquisition",
        "   proposal at `planning.enmap.org` (a separate account and a written",
        "   purpose statement are required).",
        "4. Download the **GeoTIFF** variant — `src.ingest.enmap` relies on band",
        "   descriptions carrying wavelengths.",
        "",
        "## Credential note",
        "",
        "Nothing here needs a password. Account registration is described in",
        "`ENMAP_ORDERING_CHECKLIST.md` §1; enter credentials in the browser",
        "yourself. Never commit a portal password to this repository.",
        "",
    ]

    for group in sorted(scenes_by_group):
        scenes = scenes_by_group[group]
        probes = probes_for_group(group)
        probe_txt = ", ".join(f"{p:.2f}" for p in probes) or "n/a"
        lines += [
            f"## {group}",
            "",
            f"Probe wavelengths: **{probe_txt} µm** "
            "(from `configs/spectral_fingerprints.yaml`).",
            "",
            "| scene id | tier | cells merged | max prob | "
            "bbox (min_lon, min_lat, max_lon, max_lat) | size (km) | tiles | dry season |",
            "|---|---|---:|---:|---|---:|---:|---|",
        ]
        for s in scenes:
            b = s["bbox"]
            bbox_txt = (f"{b['min_lon']}, {b['min_lat']}, "
                        f"{b['max_lon']}, {b['max_lat']}")
            lines.append(
                f"| `{s['scene_id']}` | {TIER_LABEL[s['tier']]} | {s['n_cells']} | "
                f"{s['max_prob']:.3f} | {bbox_txt} | "
                f"{s['width_km']} × {s['height_km']} | {s['tiles']} | "
                f"{s['dry_season']} |"
            )
        lines.append("")

    return _render_tracker(lines, scenes_by_group, pad_km)


def _render_tracker(lines: list[str], scenes_by_group: dict[str, list[dict[str, Any]]],
                    pad_km: float) -> str:
    """Append the order tracker, per-group totals and ingest notes."""
    lines += [
        "## Order tracker",
        "",
        "Fill in as items are submitted. `product id` is what EOWEB returns for",
        "the scene; `order id` comes from the confirmation page.",
        "",
        "| scene id | product level | product id | order id | "
        "ordered on | delivered | cloud % | ingest path |",
        "|---|---|---|---|---|---|---:|---|",
    ]
    for group in sorted(scenes_by_group):
        for s in scenes_by_group[group]:
            lines.append(
                f"| `{s['scene_id']}` | L2A | | | | | | `data/raw/enmap/` |"
            )
    lines += ["", "## Totals", ""]
    for group in sorted(scenes_by_group):
        scenes = scenes_by_group[group]
        n_exploit = sum(1 for s in scenes if s["tier"] == "top_prob")
        n_explore = len(scenes) - n_exploit
        tiles = sum(s["tiles"] for s in scenes)
        lines.append(
            f"* **{group}**: {n_exploit} exploit scene(s) + {n_explore} explore "
            f"scene(s) = **{len(scenes)} AOIs**, ~{tiles} EnMAP tile(s) at 30 km."
        )
    lines += [
        "",
        f"* Scene AOIs assume a {pad_km:.0f} km pad around the clustered cells and",
        "  the ~30 km EnMAP swath; treat `tiles` as an estimate — EOWEB reports",
        "  the scenes that actually intersect.",
        "",
        "## After download",
        "",
        "```bash",
        "python -m src.ingest.enmap \\",
        "  --input data/raw/enmap/enmap_scene.tif \\",
        "  --config configs/scenes/<target>.yml \\",
        "  --out data/interim/hyperspectral/enmap_aligned.tif",
        "```",
        "",
        "**Guardrail:** ordered scenes feed *feature extraction only*. EnMAP",
        "coverage of a window never creates a training label — labels stay",
        "MRDS / national-survey / field-validated via `src.labels.build_labels`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collapse model tasking targets into scene-level EnMAP "
        "order AOIs + manifest")
    parser.add_argument("--swath-km", type=float, default=30.0,
                        help="EnMAP swath; cells closer than this merge into "
                        "one AOI (default 30)")
    parser.add_argument("--pad-km", type=float, default=2.0,
                        help="pad each scene bbox by this much (default 2)")
    parser.add_argument("--out", default=None,
                        help="manifest path (default outputs/enmap/"
                        "enmap_order_manifest.md)")
    parser.add_argument("--aoi-dir", default=None,
                        help="AOI output dir (default "
                        "data/raw/enmap/aois_model)")
    args = parser.parse_args()

    scenes_by_group: dict[str, list[dict[str, Any]]] = {}
    for group, rel in sorted(TASKING.items()):
        path = project_path(rel)
        if not path.exists():
            logger.error("[%s] tasking file missing: %s - skipped", group, path)
            continue
        tiers = load_tier_targets(path)
        scenes: list[dict[str, Any]] = []
        for tier in TIERS:
            scenes += build_scenes(group, tier, tiers[tier],
                                   args.swath_km, args.pad_km)
        if not scenes:
            logger.warning("[%s] no targets found", group)
            continue
        for scene in scenes:
            write_aoi(scene, args.aoi_dir)
        scenes_by_group[group] = scenes
        n_cells = sum(s["n_cells"] for s in scenes)
        n_tiles = sum(s["tiles"] for s in scenes)
        logger.info("[%s] %d cells -> %d AOI(s), ~%d tile(s)",
                    group, n_cells, len(scenes), n_tiles)

    if not scenes_by_group:
        logger.error("nothing to write - are the tasking files present?")
        return

    out = (Path(args.out) if args.out
           else project_path("outputs", "enmap", "enmap_order_manifest.md"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_manifest(scenes_by_group, args.pad_km),
                   encoding="utf-8")
    logger.info("manifest -> %s", out)

    for path in prune_stale_aois(scenes_by_group, args.aoi_dir):
        logger.info("pruned stale AOI: %s", path.name)


if __name__ == "__main__":
    main()
