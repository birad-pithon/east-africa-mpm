"""Build commodity-group positive labels and background samples.

Merges two sources of positive labels per commodity group:

1. Curated seed deposits (:mod:`src.labels.known_deposits`) - the named
   deposits from the project brief, always included.
2. USGS MRDS occurrences (``data/raw/usgs_africa_minerals.gpkg``) that
   fall inside the group's belt polygon(s) AND carry at least one of the
   group's commodity codes.

MRDS points within ``seed_merge_m`` of a seed are treated as the same
deposit and dropped (the curated seed wins). Outputs are written to
``data/processed/labels_{group}.gpkg``.

A background sampler draws uniform negatives inside the belt while
enforcing ``min_dist_m`` from every positive - guarding against the
sampling-bias failure mode called out in the project brief.

Usage::

    from src.labels.build_labels import build_labels, sample_background
    pos = build_labels("configs/karagwe.yml", "copper_zinc")
    neg = sample_background(pos, n=5000)

CLI::

    python -m src.labels.build_labels --group copper_zinc --background 5000
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, box

from src.utils import load_config, project_path

logger = logging.getLogger(__name__)

BELTS_CFG = load_config(project_path("configs", "belts.yaml"))


def belt_polygon(belt_id: str) -> Polygon:
    """Return the shapely polygon for *belt_id* from configs/belts.yaml."""
    ring = BELTS_CFG["belts"][belt_id]["polygon"]
    return Polygon(ring)


def utm_epsg(lon: float) -> str:
    """UTM zone (south) EPSG for a WGS-84 longitude."""
    zone = int((lon + 180.0) // 6.0) + 1
    return f"EPSG:327{zone:02d}"


# ── Seeds ──────────────────────────────────────────────────────────────

def seed_points(group: str | None = None) -> gpd.GeoDataFrame:
    """Curated seed deposits as a GeoDataFrame (optionally by group)."""
    from src.labels.known_deposits import SEED_DEPOSITS

    rows = [d for d in SEED_DEPOSITS
            if group is None or d["belt"] in BELTS_CFG["groups"][group]["belts"]]
    gdf = gpd.GeoDataFrame(
        rows,
        geometry=gpd.points_from_xy(
            [r["lon"] for r in rows], [r["lat"] for r in rows]
        ),
        crs="EPSG:4326",
    )
    gdf["source"] = "seed"
    gdf["label"] = 1
    return gdf


# ── MRDS filtering ─────────────────────────────────────────────────────

def mrds_in_belts(
    group: str,
    belts_gpkg: Path | None = None,
) -> tuple[gpd.GeoDataFrame, str]:
    """USGS MRDS occurrences inside the group's belts, commodity-matched.

    Returns
    -------
    (gdf, belt_name)
        Filtered occurrences (EPSG:4326) and a comma-joined belt label.
    """
    gpkg = belts_gpkg or project_path("data", "raw",
                                      "usgs_africa_minerals.gpkg")
    if not Path(gpkg).exists():
        raise FileNotFoundError(
            f"{gpkg} not found - run src.ingest.usgs_africa first."
        )

    spec = BELTS_CFG["groups"][group]
    codes = {c.upper() for c in spec["mrds_codes"]}
    polygons = [belt_polygon(b) for b in spec["belts"]]
    union = polygons[0]
    for p in polygons[1:]:
        union = union.union(p)

    gdf = gpd.read_file(gpkg)
    inside = gdf[gdf.geometry.within(union)].copy()

    code_col = "CODE_LIST" if "CODE_LIST" in inside.columns else None
    if code_col:
        tokens = (
            inside[code_col].fillna("").astype(str).str.upper().str.split()
        )
        keep = tokens.apply(lambda t: bool(set(t) & codes))
        inside = inside[keep]

    inside["source"] = "mrds"
    inside["label"] = 1
    inside["belt_name"] = ", ".join(spec["belts"])
    logger.info("%s: %d MRDS records inside %s matching %s",
                group, len(inside), spec["belts"], sorted(codes))
    return inside, ", ".join(spec["belts"])
def build_labels(
    group: str,
    seed_merge_m: float = 2_000.0,
    out_path: Path | str | None = None,
) -> tuple[gpd.GeoDataFrame, Path]:
    """Build the positive-label set for *group* (seeds + MRDS, de-duped).

    Parameters
    ----------
    group : str
        Key of ``groups`` in configs/belts.yaml.
    seed_merge_m : MRDS occurrences closer than this to any curated seed
        are considered the same deposit and dropped (seed wins).
    out_path : Path | str | None
        Where to write the GPKG. ``None`` (default) writes the
        production ``data/processed/labels_{group}.gpkg``; tests and
        alternative AOIs pass an explicit path so they never overwrite
        production labels.

    Returns
    -------
    (positives, out_path)
        Combined positive-label GeoDataFrame and the written GPKG path.
    """
    seeds = seed_points(group)

    try:
        mrds, belt_name = mrds_in_belts(group)
    except FileNotFoundError as exc:
        logger.warning("MRDS layer unavailable (%s) - seeds only.", exc)
        mrds = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

    # Drop MRDS duplicates sitting on top of a curated seed.
    if len(mrds) and len(seeds):
        epsg = utm_epsg(float(seeds.geometry.x.mean()))
        s_utm = seeds.to_crs(epsg)
        m_utm = mrds.to_crs(epsg)
        tree = cKDTree(
            np.column_stack([s_utm.geometry.x, s_utm.geometry.y])
        )
        dist, _ = tree.query(
            np.column_stack([m_utm.geometry.x, m_utm.geometry.y]), k=1
        )
        mrds = mrds[dist > seed_merge_m].copy()
        logger.info("Dropped %d MRDS duplicates near seeds",
                    int((dist <= seed_merge_m).sum()))

    def _norm(df: gpd.GeoDataFrame, src: str) -> gpd.GeoDataFrame:
        out = gpd.GeoDataFrame(geometry=df.geometry, crs=df.crs)
        out["label"] = 1
        out["source"] = src
        out["name"] = df.get("name", pd.Series(dtype=object))
        out["name"] = out["name"].fillna(df.get("SITE_NAME", ""))
        out["commodities"] = (
            df.get("commodities", pd.Series(dtype=object))
            if "commodities" in df.columns
            else df.get("CODE_LIST", "")
        )
        out["commodities"] = out["commodities"].fillna("")
        out["belt"] = (
            df.get("belt", pd.Series(dtype=object))
            if "belt" in df.columns
            else df.get("belt_name", "")
        )
        out["belt"] = out["belt"].fillna("")
        out["reference"] = (
            df.get("reference", pd.Series(dtype=object))
            if "reference" in df.columns
            else ""
        )
        return out

    positives = pd.concat(
        [_norm(seeds, "seed"), _norm(mrds, "mrds")], ignore_index=True
    )
    positives = gpd.GeoDataFrame(positives, geometry="geometry",
                                 crs="EPSG:4326")

    out_path = Path(out_path) if out_path is not None else project_path(
        "data", "processed", f"labels_{group}.gpkg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    positives.to_file(out_path, driver="GPKG")
    logger.info("%d positive labels [%s] -> %s",
                len(positives), group, out_path)
    return positives, out_path
def sample_background(
    positives: gpd.GeoDataFrame,
    belts: list[str],
    n: int = 5_000,
    min_dist_m: float = 5_000.0,
    seed: int | None = None,
    halo_radius_m: float = 25_000.0,
    halo_fraction: float = 0.5,
    extent: Polygon | None = None,
) -> gpd.GeoDataFrame:
    """Two-stratum background samples correcting exploration bias.

    SAFEGUARD 3 (sampling bias toward known-infrastructure ground).
    A single uniform-negative pool over-represents remote greenfield
    terrain: geological surveys cluster around roads, mines and towns,
    so "barren near a known deposit" is massively under-sampled and the
    model learns distance-to-camp instead of mineralisation. This
    sampler draws TWO strata and tags each point:

    * ``barren_halo`` — ``halo_fraction`` of *n*, drawn from the ring
      ``(min_dist_m, halo_radius_m]`` around positives: ground that is
      geologically comparable (same belts) yet field-explored and
      known barren. This is the hardest, most informative negative.
    * ``greenfield`` — the remainder, farther than ``halo_radius_m``
      from every positive: genuinely unexplored terrain.

    Parameters
    ----------
    positives : GeoDataFrame of positive deposits (any CRS).
    belts : belt keys defining the sampling domain.
    n : total number of background points.
    min_dist_m : hard exclusion radius around every positive.
    halo_radius_m : outer radius of the barren-halo ring.
    halo_fraction : share of *n* drawn from the halo stratum (0-1).
    seed : RNG seed for reproducibility.
    extent : optional WGS-84 polygon restricting sampling (e.g. the
        raster AOI). SAFEGUARD: background outside the feature-raster
        footprint becomes NaN and is dropped at training time, wasting
        samples and distorting the class balance — clip here instead.

    Returns
    -------
    GeoDataFrame with ``label=0, source='background', stratum=...``.
    """
    import shapely

    rng = np.random.default_rng(seed)
    polygons = [belt_polygon(b) for b in belts]
    union = polygons[0]
    for p in polygons[1:]:
        union = union.union(p)
    if extent is not None:
        union = union.intersection(extent)
        if union.is_empty:
            raise ValueError("belt union ∩ extent is empty - check AOI")
        logger.info("Background domain clipped to extent: %s", union.bounds)
    minx, miny, maxx, maxy = union.bounds

    epsg = utm_epsg(0.5 * (minx + maxx))
    pos_utm = positives.to_crs(epsg)
    tree = cKDTree(np.column_stack([pos_utm.geometry.x, pos_utm.geometry.y]))

    n_halo = int(round(n * float(np.clip(halo_fraction, 0.0, 1.0))))
    n_green = n - n_halo

    def _draw(target: int, lo: float, hi: float, tag: str) -> np.ndarray:
        """Rejection-sample the distance band ``lo < d <= hi``."""
        out: list[np.ndarray] = []
        got = 0
        attempts = 0
        while got < target and attempts < 400:
            batch = max(2_000, 2 * (target - got))
            xs = rng.uniform(minx, maxx, batch)
            ys = rng.uniform(miny, maxy, batch)
            inside = shapely.contains_xy(union, xs, ys)
            xs, ys = xs[inside], ys[inside]
            if len(xs):
                ll = gpd.GeoSeries(
                    gpd.points_from_xy(xs, ys), crs="EPSG:4326"
                ).to_crs(epsg)
                cand = np.column_stack([ll.geometry.x, ll.geometry.y])
                d_min, _ = tree.query(cand, k=1)
                keep = (d_min > lo) & (d_min <= hi)
                if keep.any():
                    acc = gpd.GeoSeries(
                        gpd.points_from_xy(xs[keep], ys[keep]),
                        crs="EPSG:4326",
                    )
                    out.append(np.column_stack(
                        [acc.geometry.x, acc.geometry.y]))
                    got += int(keep.sum())
            attempts += 1
        if got < target:
            logger.warning(
                "Background stratum '%s': only %d/%d points after %d "
                "batches (band %.0f-%.0f m) - widening is recommended",
                tag, got, target, attempts, lo, hi)
        return (np.vstack(out)[:target] if out
                else np.empty((0, 2), dtype=float))

    halo_pts = _draw(n_halo, min_dist_m, halo_radius_m, "barren_halo")
    green_pts = _draw(n_green, halo_radius_m, float("inf"), "greenfield")

    arr = np.vstack([halo_pts, green_pts])
    strata = (["barren_halo"] * len(halo_pts)
              + ["greenfield"] * len(green_pts))
    if len(arr) > n:  # trim overflow from rounding
        arr = arr[:n]
        strata = strata[:n]

    out = gpd.GeoDataFrame(
        {
            "label": 0,
            "source": "background",
            "stratum": strata,
            "name": [f"bg_{i:05d}" for i in range(len(arr))],
            "commodities": "",
            "belt": ", ".join(belts),
            "reference": "",
        },
        geometry=gpd.points_from_xy(arr[:, 0], arr[:, 1]),
        crs="EPSG:4326",
    )
    logger.info(
        "Background: %d barren_halo + %d greenfield (halo<=%.0f m, "
        "exclusion>=%.0f m)",
        len(halo_pts), len(green_pts), halo_radius_m, min_dist_m)
    return out


def aoi_extent(config_path: str | Path = "configs/karagwe.yml") -> Polygon:
    """WGS-84 sampling extent from a pipeline config's ``grid.aoi_wgs84``."""
    aoi = load_config(config_path)["grid"]["aoi_wgs84"]
    return box(aoi["min_lon"], aoi["min_lat"], aoi["max_lon"], aoi["max_lat"])


def main():
    """CLI entry point for label building."""
    parser = argparse.ArgumentParser(
        description="Build positive labels + background for a commodity group"
    )
    parser.add_argument("--group", required=True,
                        choices=list(BELTS_CFG["groups"]),
                        help="Commodity group key from configs/belts.yaml")
    parser.add_argument("--background", type=int, default=0,
                        help="Also sample N background negatives")
    parser.add_argument("--seed-merge-m", type=float, default=2_000.0)
    parser.add_argument("--clip-aoi", default=None,
                        help="Clip background to this config's grid.aoi_wgs84 "
                             "(e.g. configs/karagwe.yml)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    positives, path = build_labels(args.group, seed_merge_m=args.seed_merge_m)
    print(f"OK {len(positives)} positives [{args.group}] -> {path}")

    if args.background:
        belts = BELTS_CFG["groups"][args.group]["belts"]
        extent = aoi_extent(args.clip_aoi) if args.clip_aoi else None
        bg = sample_background(positives, belts, n=args.background,
                               seed=42, extent=extent)
        bg_path = project_path("data", "processed",
                               f"background_{args.group}.gpkg")
        bg.to_file(bg_path, driver="GPKG")
        print(f"OK {len(bg)} background -> {bg_path}")


if __name__ == "__main__":
    main()
