"""Leakage-free leave-one-deposit-out holdout harness.

Retrains one commodity group with a single known deposit excluded from
BOTH the positive labels and the deposit-distance prior raster, then
scores whether the model rediscoveres the held-out site from the
probability surface alone.

Why the old ``holdout_validation.py`` was invalid
-------------------------------------------------
It mutated ``SEED_DEPOSITS`` in memory, but ``train_group`` reads the
prebuilt ``data/processed/labels_<group>.gpkg`` — the held-out deposit
stayed in the training table.  Worse, ``deposit_distance_<group>.tif``
is built from ALL labels, so the holdout position was encoded directly
in a feature (distance 0 at the deposit cell).

This harness fixes both leaks:

1. writes an isolated ``outputs/holdout/<group>/labels_<group>.gpkg``
   with the holdout removed — by name AND within ``COLOCATION_KM`` so
   alias rows for the same deposit cannot leak;
2. rebuilds ``deposit_distance_<group>.tif`` from the reduced labels
   into ``outputs/holdout/<group>/features/`` (identical file stem, so
   feature column names and the trained bundle stay consistent) and
   swaps it into the feature list;
3. trains into ``outputs/holdout/<group>/models/`` — production
   artifacts under ``data/processed/`` and ``outputs/models/`` are never
   modified;
4. scores rediscovery as the global probability rank of the best cell
   within ``TOLERANCE_KM`` of the held-out site (sites outside the
   raster extent are reported with NaN scores, never a silent miss).

Accepted limitations (documented, not leaks):

* background points are reused unchanged; background sampling excluded
  a buffer around the ORIGINAL label set, so a hole in background
  density survives near the holdout — this drops rows but encodes no
  holdout information into any feature;
* the deposit-distance prior at remaining deposits still references the
  other deposits (leave-one-out), which is the intended Bayesian prior.

CLI::

    python -m src.models.holdout --group tin_tungsten_tantalum
    python -m src.models.holdout --all [--holdout-name Nyakabingo]
    python -m src.models.holdout --all --report-only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from shapely.geometry import Point

from src.features.distances import distance_to_known_deposits_loo
from src.features.stack import build_feature_stack
from src.models.dataset import default_feature_rasters
from src.models.predict import predict_raster
from src.models.train import train_group
from src.preprocess.grid import GridSpec
from src.utils import load_config, project_path

logger = logging.getLogger(__name__)

__all__ = [
    "GROUP_CONFIGS",
    "DEFAULT_HOLDOUT",
    "HOLDOUT_ROOT",
    "select_holdout",
    "drop_points_near",
    "build_holdout_labels",
    "build_holdout_deposit_distance",
    "substitute_deposit_distance",
    "score_holdout",
    "holdout_report",
    "holdout_sites",
    "write_report",
    "run_holdout_test",
    "main",
]

#: commodity group -> belt config (mirrors production training setup).
GROUP_CONFIGS: dict[str, str] = {
    "tin_tungsten_tantalum": "configs/karagwe.yml",
    "copper_zinc": "configs/copperbelt.yml",
    "bauxite": "configs/usambara.yml",
}

#: Verified in-extent default holdout per group (checked against every
#: allowlisted feature raster before use).
DEFAULT_HOLDOUT: dict[str, str] = {
    "tin_tungsten_tantalum": "Nyakabingo",
    "copper_zinc": "Tenke Fungurume",
    "bauxite": "Lushoto",
}

HOLDOUT_ROOT: Path = project_path("outputs", "holdout")
COLOCATION_KM = 2.0    # alias rows of the same deposit
TOLERANCE_KM = 5.0     # rediscovery tolerance (seed precision ~1-3 km)
TOP_N = 100            # rediscovered := global rank <= TOP_N
TOP50 = 50             # legacy report key kept for compatibility


def _json_default(obj):
    """JSON encoder for numpy scalars/arrays and Paths."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"object of type {type(obj)!r} is not JSON serializable")


def _haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance in km; ``lon2/lat2`` may be scalars or arrays."""
    rlon1, rlat1 = np.radians(lon1), np.radians(lat1)
    rlon2 = np.radians(np.asarray(lon2, dtype=float))
    rlat2 = np.radians(np.asarray(lat2, dtype=float))
    dlon, dlat = rlon2 - rlon1, rlat2 - rlat1
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2.0) ** 2)
    return 2.0 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _load_labels(group: str) -> gpd.GeoDataFrame:
    """Read ``data/processed/labels_<group>.gpkg`` as EPSG:4326."""
    path = project_path("data", "processed", f"labels_{group}.gpkg")
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run src.labels first")
    labels = gpd.read_file(path)
    if labels.crs is None:
        labels = labels.set_crs("EPSG:4326")
    return labels.to_crs("EPSG:4326")


def _extent_mask(labels: gpd.GeoDataFrame, raster_path: Path | str) -> np.ndarray:
    """Boolean mask: each row's geometry inside *raster_path*'s footprint."""
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            return np.ones(len(labels), dtype=bool)
        pts = labels.to_crs(src.crs)
        x = pts.geometry.x.to_numpy()
        y = pts.geometry.y.to_numpy()
        b = src.bounds
        return ((x >= b.left) & (x <= b.right)
                & (y >= b.bottom) & (y <= b.top))


def select_holdout(
    labels: gpd.GeoDataFrame,
    names: list[str] | str | None = None,
    n_holdout: int = 1,
    seed: int = 42,
    in_extent: np.ndarray | None = None,
) -> gpd.GeoDataFrame:
    """Pick the deposit(s) to hold out.

    Parameters
    ----------
    labels
        Label points (EPSG:4326) with a ``name`` column.
    names
        Deposit name(s), matched case-insensitively against ``name``.
        Unknown names raise ``ValueError``. Every row sharing a matched
        name is returned, so alias rows of one deposit are held out
        together instead of leaking through a duplicate row.
    n_holdout, seed
        When *names* is not given, sample ``min(n_holdout, len(pool))``
        sites deterministically from ``numpy.random.default_rng(seed)`` —
        the same seed always yields the same holdout set.
    in_extent
        Optional boolean mask aligned with *labels* rows; random
        selection then draws only from in-extent candidates (falls back
        to all rows with a warning if the mask excludes everything).
    """
    if names is not None:
        wanted_list = [names] if isinstance(names, str) else list(names)
        if "name" not in labels.columns:
            raise ValueError(
                "labels have no 'name' column - cannot select by name")
        key = labels["name"].astype(str).str.strip().str.casefold()
        wanted = {str(w).strip().casefold() for w in wanted_list}
        missing = sorted(wanted - set(key))
        if missing:
            raise ValueError(f"holdout name(s) not in labels: {missing}")
        return labels[key.isin(wanted)].copy()

    if n_holdout < 1:
        raise ValueError(f"n_holdout must be >= 1 (got {n_holdout})")
    pool = labels
    if in_extent is not None:
        ext = labels[np.asarray(in_extent, dtype=bool)]
        if len(ext):
            pool = ext
        else:
            logger.warning(
                "no in-extent candidates - sampling from all labels")
    if len(pool) == 0:
        raise ValueError("no candidate holdout sites available")
    k = min(int(n_holdout), len(pool))  # cap: never exceed the pool
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(pool), size=k, replace=False))
    return pool.iloc[idx].copy()


def drop_points_near(
    gdf: gpd.GeoDataFrame,
    lon: float,
    lat: float,
    radius_km: float,
) -> tuple[gpd.GeoDataFrame, int]:
    """Return ``(gdf_without_points_within_radius_km, n_removed)``.

    ``gdf`` must be EPSG:4326 points. Distance is the haversine great
    circle distance — exact enough at the 2–5 km radii used here.
    """
    d = _haversine_km(
        lon, lat, gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy())
    keep = d > radius_km
    return gdf[keep].copy(), int(np.count_nonzero(~keep))


def build_holdout_labels(
    labels: gpd.GeoDataFrame,
    sites: gpd.GeoDataFrame,
    out_path: Path | str,
    colocation_km: float = COLOCATION_KM,
) -> tuple[gpd.GeoDataFrame, int]:
    """Write a label GeoPackage with the holdout deposit(s) removed.

    Removes every row whose name matches a held-out site (case-insensitive)
    **and** every row within ``colocation_km`` of one — MRDS often carries
    alias rows for the same deposit under a slightly different name, which
    would otherwise keep leaking the site into training and into the
    deposit-distance prior.

    Returns ``(reduced_labels, n_removed)``. Raises when nothing matched
    (a silent no-op would invalidate the whole harness) or when removal
    would empty the label set.
    """
    if len(sites) == 0:
        raise ValueError("no holdout sites supplied")
    drop = np.zeros(len(labels), dtype=bool)
    if "name" in labels.columns:
        keys = labels["name"].astype(str).str.strip().str.casefold()
    else:
        keys = None
    label_x = labels.geometry.x.to_numpy()
    label_y = labels.geometry.y.to_numpy()
    for _, site in sites.iterrows():
        if keys is not None:
            site_name = str(site.get("name", "")).strip().casefold()
            if site_name and site_name != "nan":
                drop |= (keys == site_name).to_numpy()
        geom = site.geometry
        if geom is not None and not geom.is_empty:
            d = _haversine_km(geom.x, geom.y, label_x, label_y)
            drop |= d <= colocation_km
    n_removed = int(np.count_nonzero(drop))
    if n_removed == 0:
        raise ValueError(
            "holdout removal matched no label rows - check site names/coords")
    reduced = labels[~drop].copy()
    if len(reduced) == 0:
        raise ValueError("holdout removal would empty the label set")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reduced.to_file(out_path, driver="GPKG")
    logger.info(
        "holdout labels -> %s (%d removed, %d remain)",
        out_path, n_removed, len(reduced))
    return reduced, n_removed


def build_holdout_deposit_distance(
    group: str,
    labels: gpd.GeoDataFrame,
    out_path: Path | str,
    config_path: str | Path | None = None,
) -> Path:
    """Rebuild ``deposit_distance_<group>.tif`` from the REDUCED labels.

    Uses the same grid construction as ``scripts/gen_alteration_features.py``
    (``GridSpec.from_config``), so the result aligns with the production
    stack, and the output keeps the production file stem so band-derived
    feature column names (``<stem>:<band>``) stay identical.
    """
    cfg = load_config(str(config_path or GROUP_CONFIGS[group]))
    grid = GridSpec.from_config(cfg)
    if len(labels) == 0:
        raise ValueError(
            "cannot build deposit-distance prior from zero labels")
    prior = distance_to_known_deposits_loo(
        grid, labels, res_m=grid.resolution_m)
    path, _bands = build_feature_stack(grid, prior, Path(out_path))
    logger.info(
        "holdout deposit-distance prior -> %s (%d points)",
        path, len(labels))
    return Path(path)


def substitute_deposit_distance(
    rasters: list[Path | str],
    holdout_dd_path: Path | str,
) -> list[Path]:
    """Swap the production deposit-distance raster for the holdout rebuild.

    Matched by file stem so feature column names (``<stem>:<band>``) are
    unchanged — the trained bundle and the prediction stack stay
    consistent. If the allowlist somehow lacks a deposit-distance entry,
    the holdout raster is appended (training and prediction both use the
    returned list, so the run stays self-consistent).
    """
    holdout_dd_path = Path(holdout_dd_path)
    stem = holdout_dd_path.stem
    out: list[Path] = []
    swapped = False
    for p in rasters:
        p = Path(p)
        if p.stem == stem:
            out.append(holdout_dd_path)
            swapped = True
        else:
            out.append(p)
    if not swapped:
        logger.warning(
            "allowlist has no '%s' entry - appending the holdout raster",
            stem)
        out.append(holdout_dd_path)
    return out


def score_holdout(
    proba_path: Path | str,
    holdout_lon: float,
    holdout_lat: float,
    name: str | None = None,
    *,
    tolerance_km: float = TOLERANCE_KM,
    top_n: int = TOP_N,
) -> dict:
    """Rank the held-out site against the full probability surface.

    ``rank`` = 1 + (# of valid cells with a higher probability than the
    best cell within ``tolerance_km`` of the site); ``rediscovered`` is
    true iff ``rank <= top_n``. A site outside the raster extent is
    reported with NaN scores and ``in_extent=False`` — never silently
    coerced into a miss.

    Note: unlike :func:`src.predict.rank.rank_candidates` (which
    declusters for display), the rank here is the raw global rank over
    every valid cell — an explicit, deterministic rediscovery metric.

    Returns a flat, JSON-ready dict (name/lon/lat/scores/counts).
    """
    base: dict = {
        "name": name,
        "lon": float(holdout_lon),
        "lat": float(holdout_lat),
        "tolerance_km": tolerance_km,
        "top_n": int(top_n),
        "in_extent": False,
        "prob_at_holdout": float("nan"),
        "rank": float("nan"),
        "percentile": float("nan"),
        "rediscovered": False,
        "distance_to_top1_km": float("nan"),
        "n_within_tolerance": 0,
        "n_top50_within_tolerance": 0,
        "n_valid_cells": 0,
    }
    with rasterio.open(proba_path) as src:
        if src.crs is None:
            x, y = float(holdout_lon), float(holdout_lat)
        else:
            x, y = Transformer.from_crs(
                "EPSG:4326", src.crs, always_xy=True
            ).transform(float(holdout_lon), float(holdout_lat))
        tf = src.transform
        col_f, row_f = (~tf) * (x, y)
        col, row = int(np.floor(col_f)), int(np.floor(row_f))
        if not (0 <= row < src.height and 0 <= col < src.width):
            return base  # outside the raster footprint: NaN scores stand
        base["in_extent"] = True

        arr = src.read(1).astype(np.float32, copy=False)
        nodata = src.nodata
        if nodata is not None and not np.isnan(np.float64(nodata)):
            arr[arr == nodata] = np.nan
        valid = np.isfinite(arr)
        n_valid = int(np.count_nonzero(valid))
        base["n_valid_cells"] = n_valid
        if n_valid == 0:
            return base
        pv = arr[row, col]
        if np.isfinite(pv):
            base["prob_at_holdout"] = float(pv)
        arr[~valid] = -np.inf  # NaN-safe argmax/partition below

        # cells within tolerance of the site (local window only)
        res_m = float(np.hypot(tf.a, tf.b))
        tol_px = int(np.ceil(tolerance_km * 1000.0 / max(res_m, 1e-6))) + 1
        r0, r1 = max(0, row - tol_px), min(src.height, row + tol_px + 1)
        c0, c1 = max(0, col - tol_px), min(src.width, col + tol_px + 1)
        sub = arr[r0:r1, c0:c1]
        rr, cc = np.mgrid[r0:r1, c0:c1]
        xs = tf.c + tf.a * (cc + 0.5) + tf.b * (rr + 0.5)
        ys = tf.f + tf.d * (cc + 0.5) + tf.e * (rr + 0.5)
        dist_km = np.hypot(xs - x, ys - y) / 1000.0
        within = (dist_km <= tolerance_km) & np.isfinite(sub)
        n_within = int(np.count_nonzero(within))
        base["n_within_tolerance"] = n_within
        if n_within:
            best = float(sub[within].max())
            rank = int(1 + np.count_nonzero(arr > best))
            base["rank"] = rank
            base["percentile"] = float(100.0 * (1.0 - (rank - 1) / n_valid))
            base["rediscovered"] = bool(rank <= top_n)
            k = min(TOP50, n_valid)
            thr = float(np.partition(arr[valid], n_valid - k)[n_valid - k])
            base["n_top50_within_tolerance"] = int(
                np.count_nonzero(within & (sub >= thr)))

        # distance from the site to the single best cell overall
        t1 = int(np.argmax(arr))
        t1r, t1c = divmod(t1, src.width)
        t1x = tf.c + tf.a * (t1c + 0.5) + tf.b * (t1r + 0.5)
        t1y = tf.f + tf.d * (t1c + 0.5) + tf.e * (t1r + 0.5)
        if src.crs is None:
            t1lon, t1lat = float(t1x), float(t1y)
        else:
            t1lon, t1lat = Transformer.from_crs(
                src.crs, "EPSG:4326", always_xy=True).transform(t1x, t1y)
        base["distance_to_top1_km"] = float(
            _haversine_km(float(holdout_lon), float(holdout_lat),
                          t1lon, t1lat))
    return base


def holdout_report(results: list[dict] | dict, out_path: Path | str) -> Path:
    """Write the JSON report: ``{"generated_at", "n_results", "results"}``.

    ``results`` may be a single run result or a list of them.
    """
    results = [results] if isinstance(results, dict) else list(results)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_results": len(results),
        "results": results,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8")
    logger.info("holdout report -> %s", out_path)
    return out_path


def holdout_sites(results: list[dict] | dict, out_path: Path | str) -> Path:
    """Write held-out sites (+ rediscovery scores) as an EPSG:4326 GeoJSON.

    Expects each result to carry a ``sites`` list of flat dicts with
    ``lon``/``lat`` (built by :func:`run_holdout_test`).
    """
    results = [results] if isinstance(results, dict) else list(results)
    rows: list[dict] = []
    for r in results:
        rows.extend(r.get("sites") or [])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text(
            '{"type": "FeatureCollection", "features": []}',
            encoding="utf-8")
        return out_path
    geometry = [Point(float(r["lon"]), float(r["lat"])) for r in rows]
    gdf = gpd.GeoDataFrame(rows, geometry=geometry, crs="EPSG:4326")
    gdf.to_file(out_path, driver="GeoJSON")
    logger.info("holdout sites -> %s (%d features)", out_path, len(rows))
    return out_path


def write_report(
    results: list[dict] | dict,
    out_dir: Path | str,
) -> dict[str, Path]:
    """Write ``holdout_report.json`` + ``holdout_sites.geojson`` into *out_dir*."""
    out_dir = Path(out_dir)
    return {
        "report": holdout_report(results, out_dir / "holdout_report.json"),
        "sites": holdout_sites(results, out_dir / "holdout_sites.geojson"),
    }


def run_holdout_test(
    group: str,
    holdout_name: str | None = None,
    holdout_names: list[str] | None = None,
    holdout_idx: int | None = None,
    random_holdout: bool = False,
    n_holdout: int = 1,
    seed: int = 42,
    algos: list[str] | None = None,
    predict: bool = True,
    config_path: str | Path | None = None,
    pu: bool = False,
    ensemble: bool = False,
    tiny_n: bool | str = "auto",
    tolerance_km: float = TOLERANCE_KM,
    top_n: int = TOP_N,
    out_root: Path | str | None = None,
) -> dict:
    """Retrain *group* without one deposit, then score its rediscovery.

    Selection precedence: ``holdout_names`` > ``holdout_name`` >
    ``holdout_idx`` (Nth in-extent label) > seeded random when
    ``random_holdout`` > ``DEFAULT_HOLDOUT[group]``.

    Everything the run writes lives under ``outputs/holdout/<group>/``
    (reduced labels, deposit-distance prior, models, proba, report) —
    production artifacts under ``data/processed/`` and ``outputs/models/``
    are never modified.

    Returns the result dict (also written to
    ``outputs/holdout/<group>/holdout_report.json``).
    """
    if group not in GROUP_CONFIGS and config_path is None:
        raise ValueError(
            f"unknown group {group!r}; known: {sorted(GROUP_CONFIGS)}")
    config = str(config_path or GROUP_CONFIGS[group])
    load_config(config)  # fail fast on a bad/missing config
    out_dir = Path(out_root or HOLDOUT_ROOT) / group
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = _load_labels(group)
    rasters = default_feature_rasters(group, config_path=config)
    ref = rasters[0]  # dem first (allowlist order) - footprint reference
    extent = _extent_mask(labels, ref)
    in_extent_labels = labels[extent]
    if in_extent_labels.empty:
        raise ValueError(f"no labels inside the raster extent for {group}")

    # -- 1. select the holdout ------------------------------------------
    names = holdout_names or ([holdout_name] if holdout_name else None)
    if names:
        sites = select_holdout(labels, names=names)
    elif holdout_idx is not None:
        if not 0 <= holdout_idx < len(in_extent_labels):
            raise ValueError(
                f"holdout_idx {holdout_idx} out of range "
                f"(0..{len(in_extent_labels) - 1})")
        sites = in_extent_labels.iloc[[holdout_idx]]
    elif random_holdout:
        sites = select_holdout(
            labels, n_holdout=n_holdout, seed=seed, in_extent=extent)
    else:
        default = DEFAULT_HOLDOUT.get(group)
        if default is None:
            raise ValueError(
                f"no DEFAULT_HOLDOUT for {group!r} - pass holdout_name")
        sites = select_holdout(labels, names=[default])  # raises if absent
    site_lon = float(sites.geometry.x.iloc[0])
    site_lat = float(sites.geometry.y.iloc[0])
    if not bool(_extent_mask(sites, ref).all()):
        logger.warning(
            "holdout site(s) %s outside raster footprint %s - "
            "rediscovery score will be NaN",
            list(sites["name"]) if "name" in sites.columns else sites.index,
            Path(ref).name)

    # -- 2. labels without the holdout (leak fix #1) --------------------
    labels_out = out_dir / f"labels_{group}.gpkg"
    reduced, n_removed = build_holdout_labels(labels, sites, labels_out)

    # -- 3. deposit-distance prior without the holdout (leak fix #2) -----
    dd_path = build_holdout_deposit_distance(
        group, reduced,
        out_dir / "features" / f"deposit_distance_{group}.tif",
        config_path=config)
    rasters_h = substitute_deposit_distance(rasters, dd_path)

    # -- 4. retrain into the isolated directory -------------------------
    train_out = train_group(
        group,
        config_path=config,
        algos=algos,
        raster_paths=rasters_h,
        label_gpkg=labels_out,
        output_dir=out_dir / "models",
        pu=pu,
        ensemble=ensemble,
        tiny_n=tiny_n,
    )
    metrics = train_out.get("metrics") or {}
    best_algo = metrics.get("best_algo")

    # -- 5. predict + score rediscovery ---------------------------------
    scores: list[dict] = []
    proba_path: Path | None = None
    if predict:
        if best_algo is None:
            raise ValueError(f"training produced no best_algo for {group}")
        proba_path = out_dir / f"proba_{group}_{best_algo}.tif"
        predict_raster(train_out["bundle_path"], rasters_h, proba_path)
        for _, site in sites.iterrows():
            scores.append(score_holdout(
                proba_path,
                float(site.geometry.x), float(site.geometry.y),
                name=str(site.get("name", "")),
                tolerance_km=tolerance_km, top_n=top_n))

    # -- 6. assemble (flat site records feed the GeoJSON writer) ---------
    if scores:
        site_records = [{"group": group, **s} for s in scores]
    else:
        site_records = [
            {"group": group,
             "name": str(s.get("name", "")),
             "lon": float(s.geometry.x),
             "lat": float(s.geometry.y)}
            for _, s in sites.iterrows()
        ]

    first = scores[0] if scores else None
    if first is None:
        rediscovery: dict = {}
    else:
        rank = first["rank"]
        has_rank = first["in_extent"] and isinstance(rank, (int, np.integer))
        rediscovery = {
            "rank_in_top50": (
                int(rank) if has_rank and rank <= TOP50 else None),
            "distance_to_top1_km": first["distance_to_top1_km"],
            "num_candidates_within_5km": first["n_top50_within_tolerance"],
            "percentile": first["percentile"],
            "rediscovered": first["rediscovered"],
        }

    result = {
        "group": group,
        "config": config,
        "holdout_deposit": site_records[0]["name"],
        "holdout_coords": [site_lon, site_lat],
        "holdout_names": [r["name"] for r in site_records],
        "n_labels_total": int(len(labels)),
        "n_labels_removed": n_removed,
        "n_labels_remaining": int(len(reduced)),
        "n_seeds_in_extent": int(extent.sum()),
        "seeds_in_training": int(len(reduced)),
        "deposit_distance_raster": str(dd_path),
        "feature_rasters": [str(p) for p in rasters_h],
        "proba_raster": str(proba_path) if proba_path else None,
        "model_bundle": str(train_out["bundle_path"]),
        "metrics_path": str(train_out.get("metrics_path", "")),
        "best_algo": best_algo,
        "metrics": metrics,
        "sites": site_records,
        "rediscovery": rediscovery,
    }
    write_report(result, out_dir)
    logger.info(
        "holdout run complete: %s held out %s (rediscovered=%s)",
        group, result["holdout_deposit"], rediscovery.get("rediscovered"))
    return result


def main() -> None:
    """CLI entry (also used by the ``holdout_validation.py`` shim)."""
    parser = argparse.ArgumentParser(
        description="Leakage-free leave-one-deposit-out holdout: retrain "
                    "with one deposit excluded from labels + the "
                    "deposit-distance prior, then score rediscovery")
    parser.add_argument("--group", choices=sorted(GROUP_CONFIGS),
                        help="commodity group to validate")
    parser.add_argument("--all", action="store_true",
                        help="run every commodity group")
    parser.add_argument("--holdout-name", default=None,
                        help="deposit to hold out (default: DEFAULT_HOLDOUT)")
    parser.add_argument("--holdout-names", nargs="+", default=None,
                        help="hold out several named deposits at once")
    parser.add_argument("--holdout-idx", type=int, default=None,
                        help="legacy: hold out the Nth in-extent label")
    parser.add_argument("--random", action="store_true",
                        help="seeded random site instead of the default")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for --random (default: 42)")
    parser.add_argument("--algos", nargs="+", default=None,
                        choices=["rf", "xgb", "lgbm"],
                        help="algorithms to train (default: config list)")
    parser.add_argument("--config", default=None,
                        help="config override (single-group runs only)")
    parser.add_argument("--no-predict", action="store_true",
                        help="train only; skip proba + rediscovery scoring")
    parser.add_argument("--pu", action="store_true",
                        help="also evaluate PU-bagged variants")
    parser.add_argument("--ensemble", action="store_true",
                        help="also evaluate the logistic stack")
    parser.add_argument("--no-tiny-n", action="store_true",
                        help="disable the tiny-n regularization preset")
    parser.add_argument("--report-only", action="store_true",
                        help="rewrite reports from saved per-group results")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.group and not args.all:
        parser.print_help()
        return
    if args.all and args.config:
        logger.warning("--config is ignored with --all "
                       "(each group uses its own belt config)")
        args.config = None
    groups = sorted(GROUP_CONFIGS) if args.all else [args.group]

    results: list[dict] = []
    errors: dict[str, str] = {}
    for group in groups:
        try:
            if args.report_only:
                path = HOLDOUT_ROOT / group / "holdout_report.json"
                if not path.exists():
                    raise FileNotFoundError(f"no saved result: {path}")
                payload = json.loads(path.read_text(encoding="utf-8"))
                results.extend(payload.get("results", []))
                continue
            logger.info("=" * 70)
            results.append(run_holdout_test(
                group,
                holdout_name=args.holdout_name,
                holdout_names=args.holdout_names,
                holdout_idx=args.holdout_idx,
                random_holdout=args.random,
                seed=args.seed,
                algos=args.algos,
                predict=not args.no_predict,
                config_path=args.config,
                pu=args.pu,
                ensemble=args.ensemble,
                tiny_n=False if args.no_tiny_n else "auto",
            ))
        except Exception as exc:  # noqa: BLE001 - per-group isolation
            logger.error("holdout run failed for %s: %s", group, exc,
                         exc_info=True)
            errors[group] = str(exc)

    if results:
        write_report(results, HOLDOUT_ROOT)
        legacy = project_path("outputs", "holdout_validation_results.json")
        legacy.write_text(
            json.dumps(results, indent=2, default=_json_default),
            encoding="utf-8")
        logger.info("aggregate report -> %s",
                    HOLDOUT_ROOT / "holdout_report.json")
        logger.info("=" * 70)
        logger.info("HOLDOUT VALIDATION SUMMARY (tolerance %.1f km, top-%d)",
                    TOLERANCE_KM, TOP_N)
        for r in results:
            rec = r.get("rediscovery") or {}
            rank = rec.get("rank_in_top50")
            dist = rec.get("distance_to_top1_km")
            if rank:
                status = f"REDISCOVERED rank #{rank} (top-1 {dist:.1f} km away)"
            elif isinstance(dist, float) and np.isfinite(dist):
                status = f"not in top 50 (top-1 {dist:.1f} km away)"
            else:
                status = "not scored"
            logger.info("  %-24s held out %-18s %s",
                        r["group"], r.get("holdout_deposit"), status)
    if errors:
        logger.error("failed groups: %s", ", ".join(sorted(errors)))
        sys.exit(1)


if __name__ == "__main__":
    main()


