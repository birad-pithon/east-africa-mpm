"""Tests for src.predict: candidate ranking, licence masking, map export.

Offline: synthetic probability raster + licence polygons in tmp_path.
"""
from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import box

from src.models.dataset import default_feature_rasters  # noqa: F401 (smoke)
from src.predict import (
    candidates_to_geojson,
    rank_candidates,
    render_interactive_map,
)
from src.preprocess.grid import GridSpec
from src.utils import (
    grid_dimensions,
    load_config,
    make_grid_transform,
    wgs84_to_utm,
)


@pytest.fixture(scope="module")
def proba_env(tmp_path_factory):
    """Small grid + probability raster with two hot spots."""
    tmp = tmp_path_factory.mktemp("predict")
    cfg = load_config("configs/karagwe.yml")
    g = cfg["grid"]
    bounds = wgs84_to_utm(g["aoi_wgs84"], g["crs"])
    tf = make_grid_transform(g["crs"], 3000, bounds)
    w, h = grid_dimensions(bounds, tf)
    grid = GridSpec(crs=g["crs"], resolution_m=3000, transform=tf,
                    width=w, height=h, bounds=bounds)

    rng = np.random.default_rng(3)
    arr = rng.uniform(0, 0.2, grid.shape).astype("float32")

    def _hot(fx, fy, sigma=6_000):
        cx = bounds[0] + fx * (bounds[2] - bounds[0])
        cy = bounds[1] + fy * (bounds[3] - bounds[1])
        yy, xx = np.mgrid[0:h, 0:w]
        wx, wy = rasterio.transform.xy(tf, yy.ravel(), xx.ravel(),
                                       offset="center")
        d = np.hypot(wx - cx, wy - cy)
        arr[:] += np.exp(-(d / sigma) ** 2).reshape(h, w).astype("float32")
        return cx, cy

    h1 = _hot(0.30, 0.40)
    h2 = _hot(0.70, 0.65)
    # a NaN hole to verify invalid-cell exclusion
    arr[10:14, 20:24] = np.nan

    p = tmp / "proba.tif"
    with rasterio.open(p, "w", **grid.profile(dtype="float32")) as dst:
        dst.write(arr, 1)
        dst.set_band_description(1, "p_deposit")

    utm_tf = __import__("pyproj").Transformer.from_crs(
        g["crs"], "EPSG:4326", always_xy=True).transform
    lox, loy = utm_tf(*h1)
    lic_box_utm = box(h1[0] - 9000, h1[1] - 9000, h1[0] + 9000,
                      h1[1] + 9000)
    lic_ll = gpd.GeoSeries([lic_box_utm], crs=g["crs"]).to_crs("EPSG:4326")
    licences = gpd.GeoDataFrame(
        {"name": ["demo_licence"]}, geometry=[lic_ll.geometry.iloc[0]],
        crs="EPSG:4326",
    )
    return {"dir": tmp, "grid": grid, "proba": p, "hot2": h2,
            "hot1": h1, "licences": tmp / "lic.geojson"}, licences


class TestRankCandidates:
    def test_basic_ranking_sorted_top_n(self, proba_env):
        e, _ = proba_env
        df = rank_candidates(e["proba"], n=5, spacing_cells=8)
        assert len(df) == 5
        assert df["prob"].is_monotonic_decreasing
        assert list(df["rank"]) == [1, 2, 3, 4, 5]

    def test_picks_sit_on_hot_spots(self, proba_env):
        e, _ = proba_env
        # hot spots are ~18 cells apart; spacing 10 still declusters
        df = rank_candidates(e["proba"], n=2, spacing_cells=10)
        # both picks land on hot anomalies far above the background (0.2)
        assert (df["prob"] > 0.5).all()

    def test_nan_cells_excluded(self, proba_env):
        e, _ = proba_env
        df = rank_candidates(e["proba"], n=200, spacing_cells=1)
        placed = {(int(r), int(c)) for r, c in zip(df["row"], df["col"], strict=False)}
        assert not any(10 <= r < 14 and 20 <= c < 24 for r, c in placed)


class TestLicenceMasking:
    def test_licensed_ground_excluded(self, proba_env, tmp_path):
        e, licences = proba_env
        lp = tmp_path / "lic.geojson"
        licences.to_file(lp, driver="GeoJSON")
        masked = rank_candidates(e["proba"], n=50, licence_path=lp,
                                 spacing_cells=1)
        unmasked = rank_candidates(e["proba"], n=50, spacing_cells=1)
        lic_pts = gpd.GeoSeries(
            gpd.points_from_xy(masked["lon"], masked["lat"]),
            crs="EPSG:4326",
        )
        inside = lic_pts.apply(
            lambda pt: licences.geometry.iloc[0].contains(pt)
        )
        assert not inside.any()              # no candidate on licensed ground
        assert len(unmasked) >= len(masked)  # mask only shrinks the pool
        # the hot spot under the licence lost its crown to the free one:
        assert masked["prob"].iloc[0] <= unmasked["prob"].max()


class TestExports:
    def test_geojson_written_valid(self, proba_env, tmp_path):
        e, _ = proba_env
        df = rank_candidates(e["proba"], n=10, spacing_cells=3)
        out = tmp_path / "cand.geojson"
        candidates_to_geojson(df, out, group="tin_tungsten_tantalum")
        gj = json.loads(out.read_text(encoding="utf-8"))
        assert gj["type"] == "FeatureCollection"
        assert len(gj["features"]) == 10
        feat = gj["features"][0]["properties"]
        assert feat["group"] == "tin_tungsten_tantalum"
        assert 0.0 <= feat["prob"] <= 1.2     # synthetic proba range

    def test_interactive_map_html(self, proba_env, tmp_path):
        e, _ = proba_env
        pytest.importorskip("folium")
        pytest.importorskip("PIL")
        df = rank_candidates(e["proba"], n=5, spacing_cells=3)
        gj = tmp_path / "cand.geojson"
        candidates_to_geojson(df, gj)
        html = render_interactive_map(
            e["proba"], gj, tmp_path / "map.html", group="synth")
        text = html.read_text(encoding="utf-8")
        assert "folium" in text.lower()
        assert "p_deposit" in text.lower() or "image_overlay" in text
