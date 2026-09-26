"""Tests for OOF stacking ensemble + Optuna TPE tuning."""
from __future__ import annotations

import geopandas as gpd
import joblib
import numpy as np
import pytest
import rasterio
from shapely.geometry import Point

from src.models.dataset import build_training_table
from src.models.ensemble import (
    StackedEnsemble,
    evaluate_stack,
    fit_stacked_model,
)
from src.models.train import make_model
from src.models.tuning import suggest_params, tune_group
from src.preprocess.grid import GridSpec
from src.utils import (
    grid_dimensions,
    load_config,
    make_grid_transform,
    wgs84_to_utm,
)


def _test_grid(res: float = 3000.0) -> GridSpec:
    cfg = load_config("configs/karagwe.yml")
    g = cfg["grid"]
    bounds = wgs84_to_utm(g["aoi_wgs84"], g["crs"])
    tf = make_grid_transform(g["crs"], res, bounds)
    w, h = grid_dimensions(bounds, tf)
    return GridSpec(crs=g["crs"], resolution_m=res, transform=tf,
                    width=w, height=h, bounds=bounds)


def _make_synth(tmp_path) -> dict:
    """Synthetic 2-band raster + label/background GPKGs with a signal."""
    grid = _test_grid()
    cx = grid.bounds[0] + (grid.bounds[2] - grid.bounds[0]) * 0.35
    cy = grid.bounds[1] + (grid.bounds[3] - grid.bounds[1]) * 0.45

    def signal_at(xs, ys):
        d = np.hypot(np.asarray(xs) - cx, np.asarray(ys) - cy)
        return np.exp(-(d / 12_000.0) ** 2)

    yy, xx = np.mgrid[0:grid.height, 0:grid.width]
    wx, wy = rasterio.transform.xy(grid.transform, yy.ravel(), xx.ravel(),
                                   offset="center")
    base = signal_at(wx, wy).reshape(grid.shape).astype("float32")

    feat = tmp_path / "synth_feats.tif"
    prof = grid.profile(dtype="float32", count=2)
    with rasterio.open(feat, "w", **prof) as dst:
        dst.write(base, 1)
        dst.set_band_description(1, "sig_field")
        dst.write(base * 0.5 + 0.25, 2)
        dst.set_band_description(2, "noisy_copy")

    rng = np.random.default_rng(11)
    tf4326 = __import__("pyproj").Transformer.from_crs(
        grid.crs, "EPSG:4326", always_xy=True).transform
    n_pos, n_bg = 24, 120
    px = cx + rng.normal(0, 3_000, n_pos)
    py = cy + rng.normal(0, 3_000, n_pos)
    bx = rng.uniform(grid.bounds[0] + 500, grid.bounds[2] - 500, n_bg)
    by = rng.uniform(grid.bounds[1] + 500, grid.bounds[3] - 500, n_bg)
    lox, loy = tf4326(px, py)
    bxo, byo = tf4326(bx, by)

    pos = gpd.GeoDataFrame(
        {"source": ["seed"] * n_pos, "stratum": ["na"] * n_pos, "y": [1] * n_pos,
         "geometry": [Point(a, b) for a, b in zip(lox, loy, strict=False)]},
        crs="EPSG:4326")
    bg = gpd.GeoDataFrame(
        {"source": ["background"] * n_bg,
         "stratum": ["barren_halo"] * 80 + ["greenfield"] * 40,
         "y": [0] * n_bg,
         "geometry": [Point(a, b) for a, b in zip(bxo, byo, strict=False)]},
        crs="EPSG:4326")
    labels = tmp_path / "labels_synth.gpkg"
    background = tmp_path / "background_synth.gpkg"
    pos.to_file(labels, driver="GPKG")
    bg.to_file(background, driver="GPKG")
    return {"feat": feat, "labels": labels, "background": background,
            "cfg": load_config("configs/karagwe.yml")}


def _make_clf_set():
    def rf(pw):
        return make_model("rf", {"n_estimators": 60, "random_state": 7}, pw)

    def xgb(pw):
        return make_model("xgb", {"n_estimators": 40, "random_state": 7}, pw)

    def lgbm(pw):
        return make_model("lgbm", {"n_estimators": 40, "random_state": 7}, pw)

    return {"rf": rf, "xgb": xgb, "lgbm": lgbm}


def _coords(pts, cfg):
    utm = __import__("pyproj").Transformer.from_crs(
        "EPSG:4326", cfg["grid"]["crs"], always_xy=True).transform
    return np.column_stack(
        utm(pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy()))


class TestStack:
    def test_evaluate_stack_on_synthetic(self, tmp_path):
        e = _make_synth(tmp_path)
        X, y, pts = build_training_table(
            "synth", raster_paths=[e["feat"]],
            label_gpkg=e["labels"], background_gpkg=e["background"])
        coords = _coords(pts, e["cfg"])

        res = evaluate_stack(_make_clf_set(), X, y, coords,
                             block_size_m=12_000.0, buffer_m=2_000.0,
                             n_splits=3)
        assert res["n_folds"] == 3
        assert res["mean_average_precision"] > 0.3
        assert set(res["stacked_from"]) == {"rf", "xgb", "lgbm"}
        assert all(v > 0.0 for v in res["per_model_oof_ap"].values())
        best_single = max(res["per_model_oof_ap"].values())
        assert res["mean_average_precision"] >= best_single - 0.15

    def test_stacked_ensemble_joblib_roundtrip(self, tmp_path):
        e = _make_synth(tmp_path)
        X, y, _ = build_training_table(
            "synth", raster_paths=[e["feat"]],
            label_gpkg=e["labels"], background_gpkg=e["background"])
        Xn = X.to_numpy(dtype="float64")
        from sklearn.linear_model import LogisticRegression

        base = {name: fac(1.0).fit(Xn, y)
                for name, fac in _make_clf_set().items()}
        meta = LogisticRegression(max_iter=500).fit(
            np.column_stack([m.predict_proba(Xn)[:, 1]
                             for m in base.values()]), y)
        en = StackedEnsemble(base, meta)

        p = en.predict_proba(Xn)
        assert p.shape == (len(y), 2)
        assert p[:, 1].min() >= 0.0 and p[:, 1].max() <= 1.0

        path = tmp_path / "stack.joblib"
        joblib.dump(en, path)
        loaded = joblib.load(path)
        assert np.allclose(loaded.predict_proba(Xn)[:, 1], p[:, 1])

    def test_fit_stacked_model(self, tmp_path):
        e = _make_synth(tmp_path)
        X, y, _ = build_training_table(
            "synth", raster_paths=[e["feat"]],
            label_gpkg=e["labels"], background_gpkg=e["background"])
        from sklearn.linear_model import LogisticRegression

        Xn = X.to_numpy(dtype="float64")
        cls = _make_clf_set()
        oof_base = np.column_stack([
            cls[n](1.0).fit(Xn, y).predict_proba(Xn)[:, 1]
            for n in ("rf", "xgb", "lgbm")])
        meta = LogisticRegression(max_iter=500).fit(oof_base, y)
        en = fit_stacked_model(cls, X, y, oof_base, meta)
        assert en.predict_proba(Xn).shape == (len(y), 2)


class TestTuning:
    def test_suggest_params_shape(self):
        optuna = pytest.importorskip("optuna")

        for algo in ("rf", "xgb", "lgbm"):
            trial = optuna.trial.FixedTrial({
                "n_estimators": 200, "max_depth": 6, "min_samples_leaf": 2,
                "min_samples_split": 2, "max_features": 0.7,
                "learning_rate": 0.05, "subsample": 0.8,
                "colsample_bytree": 0.9, "min_child_weight": 1,
                "num_leaves": 16, "min_child_samples": 10,
            })
            p = suggest_params(trial, algo)
            assert isinstance(p, dict) and len(p) >= 5
        with pytest.raises(ValueError):
            suggest_params(optuna.trial.FixedTrial({}), "knn")

    def test_tune_group_on_synthetic(self, tmp_path):
        pytest.importorskip("optuna")
        e = _make_synth(tmp_path)
        out = tune_group(
            "synth", config_path="configs/karagwe.yml",
            algos=("rf",), n_trials=4,
            raster_paths=[e["feat"]],
            label_gpkg=e["labels"], background_gpkg=e["background"],
            block_m=12_000.0, buffer_km=2.0,
        )
        assert out["best_algo"] == "rf"
        assert "n_estimators" in out["best_params"]
        assert out["n_trials"] == 4
        assert out["best_mean_ap"] >= 0.0
