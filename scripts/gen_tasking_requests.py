#!/usr/bin/env python3
"""Generate hyperspectral acquisition-request polygons from model targets.

P2 T9: two-tier sensor triage. Sentinel-2 sweeps flag anomalies; this
tool converts exploit (high-probability) and explore (high-uncertainty)
targets into buffered acquisition-request polygons for EnMAP/PRISMA
tasking, one set per commodity group.

Each group's probability raster defaults to the one its *shipped* model
produced (``outputs/models/metrics_<group>.json`` -> ``best_algo``), so a
re-train that changes the winning algorithm cannot leave acquisition polygons
- and therefore EnMAP AOIs, which cannot be cancelled once ordered - pointing
at an outdated raster. ``--proba`` still overrides.

Usage:
    python scripts/gen_tasking_requests.py --all --min-dist-to-label-km 5
    python scripts/gen_tasking_requests.py --group copper_zinc \
        --config configs/copperbelt.yml --proba outputs/models/proba_copper_zinc_xgb.tif
    python scripts/gen_tasking_requests.py --all
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
from shapely.geometry import Point

from src.models.catalog import GROUP_CONFIGS
from src.predict.active import rank_candidates_active
from src.preprocess.grid import GridSpec
from src.utils import load_config, project_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# configs come from the trained catalogue (src.models.catalog), not a copy
CATALOG = GROUP_CONFIGS


def shipped_proba(group: str) -> str | None:
    """Relative path of the probability raster for the group's *shipped* model.

    Naming one algorithm per group here was an ordering hazard: after a
    re-train that changes ``best_algo`` the tasking polygons (and therefore the
    EnMAP AOIs) would be built from the previous raster, and an EOWEB order
    can neither be changed nor cancelled through the portal. ``metrics_<group>.
    json`` is what ``train_group`` just wrote, so it cannot go stale.
    """
    metrics = project_path("outputs", "models", f"metrics_{group}.json")
    if not metrics.exists():
        logger.error("[%s] no metrics file %s - cannot resolve the shipped "
                     "model; pass --proba explicitly", group, metrics)
        return None
    algo = json.loads(metrics.read_text(encoding="utf-8")).get("best_algo")
    if not algo:
        logger.error("[%s] %s has no best_algo - pass --proba explicitly",
                     group, metrics)
        return None
    return f"outputs/models/proba_{group}_{algo}.tif"


def requests_for_group(group: str, proba: str | Path, config: str | Path,
                       n_exploit: int, n_explore: int, radius_km: float,
                       labels_path: str | Path | None = None,
                       min_dist_to_label_km: float = 0.0,
                       ) -> gpd.GeoDataFrame:
    df = rank_candidates_active(proba, n_exploit=n_exploit,
                                n_explore=n_explore,
                                explore_strategy="margin",
                                spacing_cells=2,
                                labels_path=labels_path,
                                min_dist_to_label_km=min_dist_to_label_km)
    if not len(df):
        logger.warning("[%s] no targets selected", group)
        return gpd.GeoDataFrame()

    cfg = load_config(config)
    grid = GridSpec.from_config(cfg)

    pts = gpd.GeoDataFrame(
        df.copy(),
        geometry=[Point(lon, lat)
                  for lon, lat in zip(df["lon"], df["lat"], strict=True)],
        crs="EPSG:4326").to_crs(grid.crs)

    # Buffer in projected metres, then return to WGS84 for GeoJSON.
    buffered = pts.copy()
    buffered["geometry"] = pts.buffer(radius_km * 1000.0).to_crs("EPSG:4326")
    buffered["group"] = group
    keep = ["group", "rank", "prob", "strategy", "score", "lon", "lat",
            "geometry"]
    return buffered[[c for c in keep if c in buffered.columns]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EnMAP/PRISMA acquisition-request polygons")
    parser.add_argument("--group", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--proba", default=None)
    parser.add_argument("--all", action="store_true",
                        help="process every group in the catalog")
    parser.add_argument("--n-exploit", type=int, default=10)
    parser.add_argument("--n-explore", type=int, default=15)
    parser.add_argument("--radius-km", type=float, default=2.0,
                        help="request polygon radius around each target")
    parser.add_argument("--min-dist-to-label-km", type=float, default=0.0,
                        help="novel-ground filter: drop candidate cells within "
                             "this many km of the group's own training labels, "
                             "so high-probability cells that merely re-find a "
                             "known deposit are never requested. 0 (default) "
                             "ranks on probability alone")
    args = parser.parse_args()

    maps = project_path("outputs", "maps")
    maps.mkdir(parents=True, exist_ok=True)

    if not args.all and not args.group:
        parser.error("pass --group <key> or --all")
    groups = list(CATALOG) if args.all else [args.group]

    jobs = []
    for group in groups:
        config = args.config or CATALOG.get(group)
        proba = args.proba or shipped_proba(group)
        if config is None or proba is None:
            logger.error("[%s] unknown group or unresolved raster - skipped "
                         "(known groups: %s)", group, ", ".join(sorted(CATALOG)))
            continue
        jobs.append((group, config, proba))

    for group, config, proba in jobs:
        if not project_path(proba).exists():
            logger.error("[%s] proba raster missing: %s - skipped",
                         group, project_path(proba))
            continue

        labels_path = None
        if args.min_dist_to_label_km > 0:
            labels_path = project_path("data", "processed",
                                       f"labels_{group}.gpkg")
            if not labels_path.exists():
                logger.error("[%s] labels missing: %s - skipped", group,
                             labels_path)
                continue
            logger.info("[%s] novel-ground filter: keeping cells > %.1f km "
                        "from any label", group, args.min_dist_to_label_km)

        gdf = requests_for_group(group, project_path(proba),
                                 project_path(config),
                                 args.n_exploit, args.n_explore,
                                 args.radius_km,
                                 labels_path=labels_path,
                                 min_dist_to_label_km=args.min_dist_to_label_km)
        if not len(gdf):
            continue
        out = maps / f"tasking_{group}.geojson"
        gdf.to_file(out, driver="GeoJSON")
        n_exploit = int((gdf["strategy"] == "top_prob").sum())
        print(f"OK [{group}] {len(gdf)} request polygons "
              f"({n_exploit} exploit / {len(gdf) - n_exploit} explore) "
              f"-> {out}")


if __name__ == "__main__":
    main()
