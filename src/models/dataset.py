"""Training-table construction: features sampled at labelled points.

Reads the positive-label and background GeoPackages produced by
``src.labels.build_labels`` and samples one or more grid-aligned raster
layers at every point location, yielding a model-ready ``(X, y)`` table:

* points are reprojected into each raster's CRS before sampling;
* points falling outside every raster's extent are dropped (counted);
* nodata cells become NaN and rows with any NaN feature are dropped
  (reported, never silently ignored).

Inputs may be passed explicitly (tests / alternate AOIs) or resolved
from convention: ``data/processed/{labels,background}_<group>.gpkg``
plus every ``*.tif`` under ``data/interim/**`` that lies on the grid.
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from src.utils import load_config, project_path

logger = logging.getLogger(__name__)

__all__ = [
    "default_feature_rasters",
    "sample_rasters",
    "build_training_table",
]


def default_feature_rasters(
    group: str | None = None,
    config_path: str | Path = "configs/karagwe.yml",
) -> list[Path]:
    """Raster layers eligible as model features, most specific last.

    SAFEGUARD (feature allowlist): when the config declares an explicit
    per-group ``features.<group>.rasters`` list (paths relative to
    ``data/interim``), ONLY those files are used — a stray test TIFF in
    ``data/interim`` can no longer silently enter training. Without a
    config entry, falls back to the historical convention: every
    ``*.tif`` under ``data/interim`` (a warning is logged).
    """
    root = project_path("data", "interim")
    if not root.exists():
        return []

    allowlist: list[str] | None = None
    if group is not None:
        try:
            cfg = load_config(config_path)
            allowlist = (
                cfg.get("features", {})
                .get("rasters_by_group", {})
                .get(group)
            )
        except Exception:                               # noqa: BLE001
            logger.debug("config allowlist unavailable", exc_info=True)

    if allowlist:
        out: list[Path] = []
        for entry in allowlist:
            matches = sorted(root.glob(entry))
            if not matches:
                logger.warning("allowlisted raster '%s' not found", entry)
            out.extend(matches)
        if not out:
            raise FileNotFoundError(
                f"no allowlisted feature rasters found for '{group}' "
                f"under {root}"
            )
        return out

    logger.warning(
        "no explicit feature allowlist for '%s' in config '%s' "
        "- falling back to every *.tif under data/interim", group, config_path)
    return sorted(root.rglob("*.tif"))


def _band_column(path: Path, src: rasterio.DatasetReader, b: int) -> str:
    desc = src.descriptions[b - 1]
    return f"{path.stem}:{desc}" if desc else f"{path.stem}:b{b}"


def sample_rasters(
    raster_paths: list[Path | str],
    points: gpd.GeoDataFrame,
    nodata_as_nan: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Sample all bands of *raster_paths* at *points* locations.

    Returns
    -------
    (features_df, kept_positions)
        One row per in-extent point; ``kept_positions`` are integer
        positions into *points* of retained rows (original order) --
        callers MUST use these to align labels with sampled features.
    """
    if not raster_paths:
        raise ValueError("no raster paths given")

    # SAFEGUARD 4 (grid alignment): refuse to stack rasters that do not
    # share CRS/origin/resolution/shape - a silent mismatch would sample
    # values from different locations into one feature row.
    from src.preprocess.grid_check import assert_common_grid

    assert_common_grid(raster_paths)
    pts = points.to_crs("EPSG:4326") if points.crs is None else points

    columns: dict[str, np.ndarray] = {}
    keep_mask: np.ndarray | None = None

    for rp in raster_paths:
        rp = Path(rp)
        with rasterio.open(rp) as src:
            if src.crs == pts.crs:
                xy = np.column_stack(
                    [pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy()]
                )
            else:
                reproj = gpd.GeoSeries(pts.geometry, crs=pts.crs).to_crs(
                    src.crs
                )
                xy = np.column_stack(
                    [reproj.geometry.x.to_numpy(),
                     reproj.geometry.y.to_numpy()]
                )
            minx, miny, maxx, maxy = src.bounds
            inside = (
                (xy[:, 0] >= minx) & (xy[:, 0] <= maxx)
                & (xy[:, 1] >= miny) & (xy[:, 1] <= maxy)
            )
            keep_mask = inside if keep_mask is None else (keep_mask & inside)

            samples = np.empty((len(xy), src.count), dtype="float64")
            if inside.any():
                vals = list(src.sample([(x, y) for x, y in xy[inside]]))
                samples[inside] = np.asarray(vals, dtype="float64")
            for b in range(1, src.count + 1):
                col = samples[:, b - 1].copy()
                if nodata_as_nan and src.nodatavals[b - 1] is not None:
                    col[col == src.nodatavals[b - 1]] = np.nan
                columns[_band_column(rp, src, b)] = col

    assert keep_mask is not None
    feats = pd.DataFrame(columns)
    feats = feats.loc[keep_mask].reset_index(drop=True)
    kept_positions = np.flatnonzero(keep_mask)
    logger.info("sampled %d/%d points (%d dropped, out of extent)",
                len(feats), len(points), int((~keep_mask).sum()))
    return feats, kept_positions


def build_training_table(
    group: str,
    raster_paths: list[Path | str] | None = None,
    label_gpkg: Path | str | None = None,
    background_gpkg: Path | str | None = None,
    dropna: bool = True,
    analogs_gdf: gpd.GeoDataFrame | None = None,
    config_path: str | Path = "configs/karagwe.yml",
) -> tuple[pd.DataFrame, np.ndarray, gpd.GeoDataFrame]:
    """Assemble the labelled training table for a commodity group.

    Parameters
    ----------
    analogs_gdf : optional analog (out-of-AOI transfer) deposit points
        from :mod:`src.labels.analog` — appended as positives with
        ``source='analog'``; rows outside the raster extent are dropped
        by the sampler (counted, never silent).
    config_path : path to YAML config for feature allowlist resolution.

    Returns
    -------
    (X, y, points)
        ``X`` — feature matrix (float64 DataFrame),
        ``y`` — int array (1 positive deposit, 0 background),
        ``points`` — the retained label points (EPSG:4326) for CV.
    """
    processed = project_path("data", "processed")
    pos_path = Path(label_gpkg or processed / f"labels_{group}.gpkg")
    bg_path = Path(background_gpkg or processed / f"background_{group}.gpkg")
    if not pos_path.exists():
        raise FileNotFoundError(f"{pos_path} missing - run src.labels first")

    positives = gpd.read_file(pos_path)
    positives["y"] = 1
    parts = [positives]
    if analogs_gdf is not None and len(analogs_gdf):
        analogs = analogs_gdf.copy()
        analogs["y"] = 1
        if "source" not in analogs.columns:
            analogs["source"] = "analog"
        parts.append(analogs)
        logger.info("analog deposits appended: %d", len(analogs))
    if background_gpkg is not None or bg_path.exists():
        background = gpd.read_file(bg_path)
        background["y"] = 0
        parts.append(background)
    else:
        logger.warning("[%s] no background file - training on positives only",
                       group)
    points = pd.concat(parts, ignore_index=True)
    points = gpd.GeoDataFrame(points, geometry="geometry",
                              crs=parts[0].crs)

    rasters = raster_paths or default_feature_rasters(group, config_path=config_path)
    X, kept_positions = sample_rasters(rasters, points)

    # exact label/geometry alignment via ORIGINAL positions from sampler
    y_all = points["y"].to_numpy(dtype=int)
    y_kept = y_all[kept_positions]
    coords_pts = gpd.GeoSeries(points.geometry, crs=points.crs).iloc[
        kept_positions
    ]

    if dropna:
        good = ~X.isna().any(axis=1).to_numpy()
        X = X.loc[good].reset_index(drop=True)
        y_kept = y_kept[good]
        coords_pts = coords_pts.iloc[good].reset_index(drop=True)
        logger.info("dropna: %d/%d rows retained", len(X), len(good))

    # preserve source metadata (source, stratum, ...) aligned to kept rows
    # so downstream PU learning can separate reliable negatives
    # (barren_halo) from genuinely unlabeled greenfield samples.
    meta_cols = [c for c in points.columns
                 if c not in ("geometry", "y")]
    meta_df = points[meta_cols].iloc[kept_positions].reset_index(drop=True)
    if dropna:
        meta_df = meta_df.loc[good].reset_index(drop=True)
    points_out = gpd.GeoDataFrame(
        {"y": y_kept,
         **{c: meta_df[c].to_numpy() for c in meta_cols}},
        geometry=coords_pts.geometry, crs=points.crs,
    )
    logger.info("[%s] table: %d rows x %d features (%d positives)",
                group, len(X), X.shape[1], int(y_kept.sum()))
    return X.reset_index(drop=True), y_kept, points_out
