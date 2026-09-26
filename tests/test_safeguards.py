"""Tests for SOW safeguards #3, #4, #5.

#3 sampling bias    -> two-stratum background (barren_halo / greenfield)
#4 grid mismatch    -> assert_common_grid hard-fails before stacking
#5 feedback loop    -> merge_field_labels with dedup + version manifest
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import Point, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

bl = importlib.import_module("src.labels.build_labels")
from src.labels.field_update import merge_field_labels  # noqa: E402
from src.models.dataset import sample_rasters  # noqa: E402
from src.preprocess.grid_check import (  # noqa: E402
    GridMismatchError,
    assert_common_grid,
)

# ── helpers ───────────────────────────────────────────────────────────


def _test_grid():
    from src.preprocess.grid import GridSpec
    from src.utils import grid_dimensions, load_config, make_grid_transform, wgs84_to_utm

    cfg = load_config("configs/karagwe.yml")
    g = cfg["grid"]
    bounds = wgs84_to_utm(g["aoi_wgs84"], g["crs"])
    tf = make_grid_transform(g["crs"], 3000, bounds)
    w, h = grid_dimensions(bounds, tf)
    return GridSpec(crs=g["crs"], resolution_m=3000, transform=tf,
                    width=w, height=h, bounds=bounds)


def _write_raster(path, grid, value=1.0, **overrides):
    prof = grid.profile(dtype="float32", count=1)
    prof.update(overrides)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(np.full((grid.height, grid.width), value, "float32"), 1)


# ── Safeguard 3: two-stratum background ──────────────────────────────

HALO_BOX = box(32.0, -3.0, 32.25, -2.75)   # ~27 km square, UTM zone 36


@pytest.fixture
def belt_env(monkeypatch):
    """Patch belt_polygon so sampling runs on a fixed synthetic box."""
    monkeypatch.setattr(bl, "belt_polygon", lambda b: HALO_BOX)
    return bl


def _positives():
    return gpd.GeoDataFrame(
        {"name": ["p1", "p2", "p3"]},
        geometry=[Point(32.10, -2.90), Point(32.16, -2.85),
                  Point(32.05, -2.82)],
        crs="EPSG:4326",
    )


def _dist_to_positives(pts_wgs84, positives):
    utm = bl.utm_epsg(32.125)
    p = pts_wgs84.to_crs(utm)
    q = positives.to_crs(utm)
    D = np.hypot(
        p.geometry.x.to_numpy()[:, None] - q.geometry.x.to_numpy()[None, :],
        p.geometry.y.to_numpy()[:, None] - q.geometry.y.to_numpy()[None, :],
    )
    return D.min(axis=1)


def test_background_has_two_strata(belt_env):
    bg = bl.sample_background(
        _positives(), ["kab"], n=60, min_dist_m=3_000,
        halo_radius_m=9_000, halo_fraction=0.5, seed=0)
    assert "stratum" in bg.columns
    counts = bg["stratum"].value_counts()
    assert counts.get("barren_halo", 0) > 0
    assert counts.get("greenfield", 0) > 0


def test_background_stratum_distance_bands(belt_env):
    bg = bl.sample_background(
        _positives(), ["kab"], n=60, min_dist_m=3_000,
        halo_radius_m=9_000, halo_fraction=0.5, seed=1)
    d = _dist_to_positives(bg, _positives())
    halo = (bg["stratum"] == "barren_halo").to_numpy()
    green = (bg["stratum"] == "greenfield").to_numpy()
    assert (d[halo] > 3_000).all()
    assert (d[halo] <= 9_000 + 1e-6).all()
    assert (d[green] > 9_000).all()


def test_background_5050_split_and_deterministic(belt_env):
    a = bl.sample_background(
        _positives(), ["kab"], n=40, min_dist_m=2_000,
        halo_radius_m=8_000, halo_fraction=0.5, seed=42)
    b = bl.sample_background(
        _positives(), ["kab"], n=40, min_dist_m=2_000,
        halo_radius_m=8_000, halo_fraction=0.5, seed=42)
    assert len(a) == 40
    n_halo = int((a["stratum"] == "barren_halo").sum())
    assert 15 <= n_halo <= 25          # 50/50 target with sampling slack
    assert a["stratum"].tolist() == b["stratum"].tolist()
    assert np.allclose(a.geometry.x, b.geometry.x)
    assert np.allclose(a.geometry.y, b.geometry.y)


# ── Safeguard 4: grid alignment ──────────────────────────────────────


def test_grid_check_passes_on_common_grid(tmp_path):
    grid = _test_grid()
    a, b = tmp_path / "a.tif", tmp_path / "b.tif"
    _write_raster(a, grid, 1.0)
    _write_raster(b, grid, 2.0)
    sigs = assert_common_grid([a, b])
    assert len(sigs) == 2


def test_grid_check_fails_on_shifted_origin(tmp_path):
    grid = _test_grid()
    a = tmp_path / "a.tif"
    b = tmp_path / "shifted.tif"
    _write_raster(a, grid)
    t = list(grid.transform)[:6]
    bad = (t[0], t[1], t[2] + 3000, t[3], t[4], t[5])
    _write_raster(b, grid, transform=bad)
    with pytest.raises(GridMismatchError, match="origin/transform"):
        assert_common_grid([a, b])


def test_grid_check_fails_on_crs(tmp_path):
    grid = _test_grid()
    a = tmp_path / "a.tif"
    b = tmp_path / "wgs84.tif"
    _write_raster(a, grid)
    _write_raster(b, grid, crs="EPSG:4326")
    with pytest.raises(GridMismatchError, match="crs"):
        assert_common_grid([a, b])


def test_grid_check_fails_on_shape(tmp_path):
    grid = _test_grid()
    a = tmp_path / "a.tif"
    b = tmp_path / "small.tif"
    _write_raster(a, grid)
    _write_raster(b, grid, width=grid.width - 5, height=grid.height)
    with pytest.raises(GridMismatchError, match="shape"):
        assert_common_grid([a, b])


def test_sample_rasters_rejects_misaligned_stack(tmp_path):
    """Integration: the check fires inside the feature-sampling path."""
    grid = _test_grid()
    a = tmp_path / "a.tif"
    b = tmp_path / "shifted.tif"
    _write_raster(a, grid)
    t = list(grid.transform)[:6]
    bad = (t[0] + 1500, t[1], t[2], t[3], t[4], t[5])
    _write_raster(b, grid, transform=bad)
    pts = gpd.GeoDataFrame(
        {"name": ["x"]}, geometry=[Point(32.1, -2.9)], crs="EPSG:4326")
    with pytest.raises(GridMismatchError):
        sample_rasters([a, b], pts)


# ── Safeguard 5: feedback loop ───────────────────────────────────────


@pytest.fixture
def label_env(tmp_path):
    """Existing labels/background GPKGs + a field-results layer."""
    pos = gpd.GeoDataFrame(
        {"name": ["p1", "p2"], "label": 1},
        geometry=[Point(32.10, -2.90), Point(32.16, -2.85)],
        crs="EPSG:4326")
    bg = gpd.GeoDataFrame(
        {"name": ["b1"], "label": 0},
        geometry=[Point(32.02, -2.95)], crs="EPSG:4326")
    lp, lb = tmp_path / "labels_g.gpkg", tmp_path / "background_g.gpkg"
    pos.to_file(lp, driver="GPKG")
    bg.to_file(lb, driver="GPKG")

    field = gpd.GeoDataFrame(
        {"site": ["f1", "f2", "f3", "f4"],
         "status": ["positive", "barren", "POSITIVE", "positive"]},
        geometry=[
            Point(32.1002, -2.9002),   # ~25 m from p1  -> duplicate
            Point(32.20, -2.80),       # new barren
            Point(32.25, -2.90),       # new positive (case-insensitive)
            Point(32.0201, -2.9501),   # on b1, opposite class -> relabel
        ],
        crs="EPSG:4326")
    fp = tmp_path / "field.gpkg"
    field.to_file(fp, driver="GPKG")
    return {"labels": lp, "background": lb, "field": fp, "dir": tmp_path}


def test_merge_adds_and_dedups(label_env):
    e = label_env
    s = merge_field_labels(
        "g", e["field"], dedup_m=2_000,
        labels_gpkg=e["labels"], background_gpkg=e["background"],
        manifest_path=e["dir"] / "manifest.json")
    assert s["received"] == 4
    assert s["dropped_duplicates"] == 1            # f1 near p1
    assert s["added_positives"] == 2               # f3 + f4
    assert s["added_background"] == 1              # f2
    assert s["relabelled_stale_rows"] == {"0": 1}  # b1 overridden

    pos = gpd.read_file(e["labels"])
    bg = gpd.read_file(e["background"])
    assert len(pos) == 4                           # p1, p2, f3, f4
    assert len(bg) == 1                            # f2 (b1 relabelled away)
    assert (pos["source"] == "field_verified").sum() == 2


def test_merge_writes_versioned_manifest(label_env):
    e = label_env
    mpath = e["dir"] / "manifest.json"
    merge_field_labels("g", e["field"], labels_gpkg=e["labels"],
                       background_gpkg=e["background"],
                       manifest_path=mpath)
    m1 = json.loads(mpath.read_text(encoding="utf-8"))
    assert m1["groups"]["g"]["version"] == 1
    assert m1["groups"]["g"]["history"][0]["added_positives"] == 2

    merge_field_labels("g", e["field"], labels_gpkg=e["labels"],
                       background_gpkg=e["background"],
                       manifest_path=mpath)
    m2 = json.loads(mpath.read_text(encoding="utf-8"))
    assert m2["groups"]["g"]["version"] == 2
    assert len(m2["groups"]["g"]["history"]) == 2


def test_merge_rejects_bad_status_column(label_env, tmp_path):
    e = label_env
    bad = gpd.GeoDataFrame(
        {"site": ["x"], "verdict": ["maybe"]},
        geometry=[Point(32.3, -2.7)], crs="EPSG:4326")
    p = tmp_path / "bad.gpkg"
    bad.to_file(p, driver="GPKG")
    with pytest.raises(ValueError, match="no status column"):
        merge_field_labels("g", p, labels_gpkg=e["labels"],
                           background_gpkg=e["background"],
                           manifest_path=e["dir"] / "m.json")


def test_retrained_table_picks_up_field_labels(label_env):
    """The loop closes: build_training_table reads the enriched GPKGs."""
    import pyproj

    from src.models.dataset import build_training_table

    grid = _test_grid()
    to_wgs = pyproj.Transformer.from_crs(
        grid.crs, "EPSG:4326", always_xy=True).transform

    def in_grid(fx, fy):
        x = grid.bounds[0] + fx * (grid.bounds[2] - grid.bounds[0])
        y = grid.bounds[1] + fy * (grid.bounds[3] - grid.bounds[1])
        return to_wgs(x, y)                      # -> (lon, lat)

    e = label_env
    # rebuild labels with in-extent points (merge tests stay extent-free)
    pos = gpd.GeoDataFrame(
        {"name": ["p1", "p2", "p3"], "label": 1},
        geometry=[Point(*in_grid(0.3, 0.4)), Point(*in_grid(0.6, 0.6)),
                  Point(*in_grid(0.5, 0.3))],
        crs="EPSG:4326")
    bg = gpd.GeoDataFrame(
        {"name": ["b1", "b2"], "label": 0},
        geometry=[Point(*in_grid(0.2, 0.8)), Point(*in_grid(0.8, 0.2))],
        crs="EPSG:4326")
    pos.to_file(e["labels"], driver="GPKG")
    bg.to_file(e["background"], driver="GPKG")

    # field feedback: 1 new positive (~3 km from p1, beyond dedup radius)
    # + 1 exact duplicate of p2
    lon1, lat1 = in_grid(0.32, 0.41)
    lon2, lat2 = in_grid(0.6, 0.6)
    field = gpd.GeoDataFrame(
        {"site": ["f_new", "f_dup"],
         "status": ["positive", "positive"]},
        geometry=[Point(lon1, lat1), Point(lon2, lat2)],
        crs="EPSG:4326")
    fp = e["dir"] / "field2.gpkg"
    field.to_file(fp, driver="GPKG")

    s = merge_field_labels("g", fp, dedup_m=2_000,
                           labels_gpkg=e["labels"],
                           background_gpkg=e["background"],
                           manifest_path=e["dir"] / "manifest.json")
    assert s["added_positives"] == 1 and s["dropped_duplicates"] == 1

    feat = e["dir"] / "feat.tif"
    _write_raster(feat, grid, 0.5)
    X, y, pts = build_training_table(
        "g", raster_paths=[feat],
        label_gpkg=e["labels"], background_gpkg=e["background"])
    assert len(X) == len(y) == 6                 # 3 pos + 2 bg + 1 field
    assert set(np.unique(y)) == {0, 1}
