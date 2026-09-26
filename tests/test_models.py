"""Tests for src.models: dataset, spatial-CV training, inference.

Fully offline: builds a tiny synthetic 2-band feature raster and
matching label/background GeoPackages in tmp_path, then exercises the
whole chain table -> buffered-blocked CV -> bundle -> probability grid.
"""
from __future__ import annotations

import json

import geopandas as gpd
import joblib
import numpy as np
import pytest
import rasterio
from shapely.geometry import Point

from src.models import (
    build_training_table,
    evaluate_spatial_cv,
    make_model,
    predict_raster,
    train_group,
)
from src.preprocess.grid import GridSpec
from src.utils import (
    grid_dimensions,
    load_config,
    make_grid_transform,
    wgs84_to_utm,
)

RES = 3000          # metres -> ~45 x 45 px KAB test grid


def _test_grid() -> GridSpec:
    cfg = load_config("configs/karagwe.yml")
    g = cfg["grid"]
    bounds = wgs84_to_utm(g["aoi_wgs84"], g["crs"])
    tf = make_grid_transform(g["crs"], RES, bounds)
    w, h = grid_dimensions(bounds, tf)
    return GridSpec(crs=g["crs"], resolution_m=RES, transform=tf,
                    width=w, height=h, bounds=bounds)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Synthetic feature raster + label GPKGs with a learnable signal."""
    tmp = tmp_path_factory.mktemp("models")
    grid = _test_grid()

    # signal field: high near a buried centre inside the grid
    cx = grid.bounds[0] + (grid.bounds[2] - grid.bounds[0]) * 0.35
    cy = grid.bounds[1] + (grid.bounds[3] - grid.bounds[1]) * 0.45

    def signal_at(xs_utm, ys_utm):
        d = np.hypot(np.asarray(xs_utm) - cx, np.asarray(ys_utm) - cy)
        return np.exp(-(d / 12_000.0) ** 2)

    yy, xx = np.mgrid[0:grid.height, 0:grid.width]
    wx, wy = rasterio.transform.xy(grid.transform, yy.ravel(),
                                   xx.ravel(), offset="center")
    base = signal_at(wx, wy).reshape(grid.shape).astype("float32")

    feat_path = tmp / "synth_feats.tif"
    prof = grid.profile(dtype="float32", count=2)
    with rasterio.open(feat_path, "w", **prof) as dst:
        dst.write(base, 1)
        dst.set_band_description(1, "sig_field")
        dst.write((base * 0.5 + 0.25), 2)
        dst.set_band_description(2, "noisy_copy")

    rng = np.random.default_rng(11)
    pts_tf = __import__("pyproj").Transformer.from_crs(
        grid.crs, "EPSG:4326", always_xy=True).transform

    # positives clustered near the signal centre; backgrounds spread wide
    n_pos, n_bg = 24, 120
    px = cx + rng.normal(0, 3_000, n_pos)
    py = cy + rng.normal(0, 3_000, n_pos)
    bx = rng.uniform(grid.bounds[0] + 500, grid.bounds[2] - 500,
                     n_bg)
    by = rng.uniform(grid.bounds[1] + 500, grid.bounds[3] - 500,
                     n_bg)
    lox, loy = pts_tf(px, py)
    bxo, byo = pts_tf(bx, by)

    pos = gpd.GeoDataFrame(
        {"name": [f"pos{i}" for i in range(n_pos)], "y": 1},
        geometry=[Point(a, b) for a, b in zip(lox, loy, strict=False)], crs="EPSG:4326")
    bg = gpd.GeoDataFrame(
        {"name": [f"bg{i}" for i in range(n_bg)], "y": 0},
        geometry=[Point(a, b) for a, b in zip(bxo, byo, strict=False)],
        crs="EPSG:4326")
    return {"dir": tmp, "grid": grid, "feat": feat_path,
            "labels": tmp / "labels_synth.gpkg",
            "background": tmp / "background_synth.gpkg",
            "cx": cx, "cy": cy}, pos, bg


# ── dataset ───────────────────────────────────────────────────────────

class TestDataset:
    def test_build_table_shape_and_labels(self, env):
        e, pos, bg = env
        pos.to_file(e["labels"], driver="GPKG")
        bg.to_file(e["background"], driver="GPKG")
        X, y, pts = build_training_table(
            "synth",
            raster_paths=[e["feat"]],
            label_gpkg=e["labels"],
            background_gpkg=e["background"],
        )
        assert len(X) == len(y) == len(pts)
        assert set(np.unique(y)) == {0, 1}
        assert y.sum() >= 20                  # nearly all positives kept
        assert X.shape[1] == 2                # both bands sampled
        assert X.notna().all().all()

    def test_out_of_extent_points_dropped(self, env):
        e, _, _ = env
        stray = gpd.GeoDataFrame(
            {"y": [1]}, geometry=[Point(35.0, 0.0)], crs="EPSG:4326"
        )
        import pandas as pd
        pos = gpd.GeoDataFrame(
            pd.concat([gpd.read_file(e["labels"]),
                       stray], ignore_index=True),
            crs="EPSG:4326")
        from pathlib import Path
        tmp = Path(e["labels"]).with_name("lab2.gpkg")
        pos.to_file(tmp, driver="GPKG")
        X, y, _ = build_training_table(
            "synth", raster_paths=[e["feat"]],
            label_gpkg=tmp, background_gpkg=e["background"])
        assert (y == 1).sum() == 24           # stray point dropped


# ── CV + training ─────────────────────────────────────────────────────

class TestSpatialCVTraining:
    def test_cv_learns_signal(self, env):
        e, pos, bg = env
        X, y, pts = build_training_table(
            "synth", raster_paths=[e["feat"]],
            label_gpkg=e["labels"], background_gpkg=e["background"])
        utm_tf = __import__("pyproj").Transformer.from_crs(
            "EPSG:4326", e["grid"].crs, always_xy=True).transform
        coords = np.column_stack(
            utm_tf(pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy()))
        res = evaluate_spatial_cv(
            lambda pw: make_model("rf", {"n_estimators": 100}, pw),
            X, y, coords,
            block_size_m=12_000, buffer_m=3_000, n_splits=3)
        assert res["n_folds"] == 3
        assert res["mean_average_precision"] > 0.5   # signal is strong
        # buffering provably shrank some training set vs fold size
        assert all(f["n_train"] < len(y) for f in res["folds"])

    @pytest.mark.parametrize("algo", ["rf", "xgb", "lgbm"])
    def test_make_model_fit_predict(self, algo):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(80, 3))
        y = (X[:, 0] > 0).astype(int)
        m = make_model(algo, {"n_estimators": 40}, 1.5)
        m.fit(X, y)
        p = m.predict_proba(X)
        assert p.shape == (80, 2)

    def test_train_group_end_to_end(self, env):
        e, pos, bg = env
        out = train_group(
            "synth", config_path="configs/karagwe.yml",
            algos=["rf"],
            raster_paths=[e["feat"]],
            block_m=12_000, buffer_km=1.0,
            output_dir=e["dir"],
            label_gpkg=e["labels"],
            background_gpkg=e["background"],
        )
        metrics = json.loads(out["metrics_path"].read_text(encoding="utf-8"))
        assert metrics["best_algo"] in ("rf", "random_forest")
        assert metrics["results"]["rf"]["n_folds"] > 0
        bundle = joblib.load(out["bundle_path"])
        assert bundle["group"] == "synth"
        assert len(bundle["feature_names"]) == 2


# ── inference ─────────────────────────────────────────────────────────

class TestPredictRaster:
    def test_probability_grid_written(self, env):
        e, _, _ = env
        bundle = next(e["dir"].glob("model_synth_*.joblib"))
        proba_tif = predict_raster(bundle, [e["feat"]],
                                   out_path=e["dir"] / "proba.tif")
        with rasterio.open(proba_tif) as src:
            assert src.count == 1
            assert src.dtypes[0] == "float32"
            arr = src.read(1)
            grid = e["grid"]
            assert arr.shape == (grid.height, grid.width)
            valid = arr[~np.isnan(arr)]
            assert valid.size > 0
            assert valid.min() >= 0.0 and valid.max() <= 1.0
        # probability peaks near the buried signal centre (cx, cy),
        # which lives at 35%/45% of the extent -- NOT the raster middle
        col, row = e["grid"].world_to_pixel([e["cx"]], [e["cy"]])
        ci, ri = int(round(col[0])), int(round(row[0]))
        ci = min(max(ci, 2), arr.shape[1] - 3)
        ri = min(max(ri, 2), arr.shape[0] - 3)
        centre_val = float(arr[ri - 2:ri + 3, ci - 2:ci + 3].mean())
        corner_val = float(np.nan_to_num(arr[:3, :3].mean()))
        assert centre_val > corner_val
