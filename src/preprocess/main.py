"""Preprocessing orchestrator: grid -> DEM features -> labels -> stack.

Pipeline steps (all outputs land on the canonical analysis grid):

1. Resolve the :class:`~src.preprocess.grid.GridSpec` from config.
2. Locate the DEM tile (``data/interim/dem/``); if absent, pull it via
   ``src.ingest.srtm_dem`` (synthetic fallback keeps this runnable).
3. Compute terrain feature bands (slope/aspect/TPI/relief/plateau).
4. Rasterize positive + background labels for each commodity group.
5. Stack every available feature band into
   ``data/processed/feature_stack_{group}.tif``.
6. Extract band values at labelled points ->
   ``data/processed/samples_{group}.csv`` (model-ready table).

CLI::

    python -m src.preprocess.main --config configs/karagwe.yml \
        --groups tin_tungsten_tantalum copper_zinc bauxite [--res 300]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

from src.preprocess.dem_features import dem_features
from src.preprocess.grid import GridSpec
from src.preprocess.rasterize import extract_at_points, label_rasters
from src.utils import load_config, project_path

logger = logging.getLogger(__name__)

LABEL_BUFFER_M = 60.0        # burn positives as ~2-cell discs


def _find_or_pull_dem(cfg: dict) -> Path:
    """Return a grid-aligned DEM path, pulling one if necessary."""
    dem_dir = project_path("data", "interim", "dem")
    candidates = sorted(dem_dir.glob("*_aligned.tif")) if dem_dir.exists() else []
    if candidates:
        return candidates[0]

    from src.ingest.srtm_dem import pull_dem_tile  # local import: EE optional

    logger.info("No DEM on disk - pulling one (EE or synthetic fallback).")
    return pull_dem_tile(project_path("configs", "karagwe.yml"))
    logger.info("No DEM on disk - pulling one (EE or synthetic fallback).")
    return pull_dem_tile(project_path("configs", "karagwe.yml"))


def run_preprocess(
    config_path: str | Path = "configs/karagwe.yml",
    groups: list[str] | None = None,
    resolution_m: float | None = None,
) -> dict[str, Path]:
    """Run the full preprocessing chain; return written artifact paths."""
    cfg = load_config(config_path)
    grid = GridSpec.from_config(cfg, resolution_m=resolution_m)
    logger.info("Grid: %s @ %gm  (%d x %d px)",
                grid.crs, grid.resolution_m, grid.width, grid.height)

    processed = project_path("data", "processed")
    interim = project_path("data", "interim", "features")
    processed.mkdir(parents=True, exist_ok=True)
    interim.mkdir(parents=True, exist_ok=True)

    # ── 1-3. DEM + terrain features ────────────────────────────────────
    dem_path = _find_or_pull_dem(cfg)
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        if (src.width, src.height) != (grid.width, grid.height):
            from src.utils import reproject_raster_to_grid
            fixed = interim / "dem_grid.tif"
            reproject_raster_to_grid(
                str(dem_path), str(fixed), grid.crs, grid.transform,
                grid.width, grid.height,
            )
            with rasterio.open(fixed) as g:
                dem = g.read(1)

    feats = dem_features(dem, grid.resolution_m)
    band_names = list(feats.keys())
    stack = np.stack([feats[b] for b in band_names])

    feat_tif = interim / "terrain_features.tif"
    profile = grid.profile(count=len(band_names))
    with rasterio.open(feat_tif, "w", **profile) as dst:
        dst.write(stack)
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, name)
    logger.info("Terrain features -> %s (%d bands)", feat_tif, len(band_names))

    # ── 3b. geology features (lithology + contacts) ────────────────────
    from src.features.geology_features import (
        build_geology_features,
        default_geology_paths,
    )

    geol_paths = default_geology_paths()
    if geol_paths["geology"].exists():
        geol = gpd.read_file(geol_paths["geology"])
        contacts = (
            gpd.read_file(geol_paths["contacts"])
            if geol_paths["contacts"].exists() else None
        )
        geo_path, geo_classes = build_geology_features(
            grid, geol, contacts, out_dir=interim,
        )
        with rasterio.open(geo_path) as gsrc:
            for b in range(1, gsrc.count + 1):
                band_names.append(gsrc.descriptions[b - 1])
                stack = np.vstack([stack, gsrc.read(b)[np.newaxis, :, :]])
        logger.info("Geology bands appended: %d classes", len(geo_classes))
    else:
        logger.warning(
            "No geology vectors at %s - run `python -m src.ingest.worldgeol` "
            "to add lithology features; continuing without them.",
            geol_paths["geology"],
        )

    # ── 4-6. per-group labels, stacks, sample tables ───────────────────
    groups = groups or list(
        load_config(project_path("configs", "belts.yaml"))["groups"]
    )
    artifacts: dict[str, Path] = {"grid_features": feat_tif}

    for group in groups:
        pos_path = project_path("data", "processed", f"labels_{group}.gpkg")
        bg_path = project_path("data", "processed",
                               f"background_{group}.gpkg")
        if not pos_path.exists():
            logger.warning("[%s] no label file %s - run src.labels first; "
                           "skipping", group, pos_path.name)
            continue
        positives = gpd.read_file(pos_path)
        background = (
            gpd.read_file(bg_path) if bg_path.exists() else None
        )

        rasters = label_rasters(positives, background, grid,
                                buffer_m=LABEL_BUFFER_M)

        y_path = processed / f"y_{group}.tif"
        prof = grid.profile(dtype="uint8",
                            count=len(rasters))  # pos (+bg, +y)
        with rasterio.open(y_path, "w", **prof) as dst:
            dst.write(rasters["positive"], 1)
            dst.set_band_description(1, "positive")
            if "background" in rasters:
                dst.write(rasters["background"], 2)
                dst.set_band_description(2, "background")
            if "y" in rasters:
                dst.write(rasters["y"], 3)
                dst.set_band_description(3, "y_train")
        artifacts[f"y_{group}"] = y_path

        n_pos_cells = int(rasters["positive"].sum())
        logger.info("[%s] positive cells: %d", group, n_pos_cells)

        table_src = positives if background is None or not len(background) \
            else gpd.GeoDataFrame(
                pd_concat_pos_bg(positives, background),
                geometry="geometry", crs=positives.crs,
            )
        samples = extract_at_points(stack, band_names, grid, table_src)
        samples["label"] = np.concatenate([
            np.ones(len(positives)),
            np.zeros(len(table_src) - len(positives)),
        ])[: len(samples)]
        csv_path = processed / f"samples_{group}.csv"
        samples.to_csv(csv_path, index=False)
        artifacts[f"samples_{group}"] = csv_path
        logger.info("[%s] sample table: %d rows -> %s",
                    group, len(samples), csv_path.name)

    return artifacts


def pd_concat_pos_bg(pos, bg):
    import pandas as pd

    cols = ["geometry", "label", "source"]
    a = pos[cols].copy()
    b = bg[cols].copy()
    b["label"] = 0
    b["source"] = "background"
    return pd.concat([a, b], ignore_index=True)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess layers onto the common analysis grid"
    )
    parser.add_argument("--config", default="configs/karagwe.yml")
    parser.add_argument("--groups", nargs="+", default=None,
                        help="Commodity groups (default: all configured)")
    parser.add_argument("--res", type=float, default=None,
                        help="Override grid resolution in metres (tests)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    arts = run_preprocess(args.config, groups=args.groups,
                          resolution_m=args.res)
    print("\nOK preprocessing artifacts:")
    for k, v in arts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
