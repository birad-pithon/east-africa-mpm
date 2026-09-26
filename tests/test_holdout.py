"""Offline tests for src.models.holdout: the leak-free holdout harness.

Everything runs without network or retraining: synthetic label points, a
synthetic probability raster on the karagwe grid (coarsened to 3 km), and
tmp_path outputs only — production artifacts under data/processed/ and
outputs/models/ are never touched.
"""
from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import yaml
from pyproj import Transformer
from shapely.geometry import Point

from src.models.holdout import (
    COLOCATION_KM,
    DEFAULT_HOLDOUT,
    build_holdout_deposit_distance,
    build_holdout_labels,
    holdout_report,
    holdout_sites,
    score_holdout,
    select_holdout,
    substitute_deposit_distance,
    write_report,
)
from src.preprocess.grid import GridSpec
from src.utils import (
    grid_dimensions,
    load_config,
    make_grid_transform,
    wgs84_to_utm,
)

#: background-beating cells so a mid-grid background cell necessarily
#: ranks worse than TOP_N (=100): 150 beaters + the hot cell -> rank 152.
N_SPECIAL = 150


def _labels() -> gpd.GeoDataFrame:
    """Synthetic deposit points (EPSG:4326) with one alias row."""
    return gpd.GeoDataFrame(
        {"name": ["Nyakabingo", "Buhweju", "Buhweju North",
                  "Ntungamo", "Gakara"],
         "geometry": [Point(30.067, -1.917), Point(31.0, -1.1),
                      Point(31.001, -1.1), Point(30.5, -1.2),
                      Point(30.9, -1.3)]},
        crs="EPSG:4326")


@pytest.fixture(scope="module")
def proba_env(tmp_path_factory):
    """Karagwe grid @3 km with one hot cell, N_SPECIAL beaters, a NaN hole.

    Layout: beater block in the top rows, hot cell at the centre, NaN hole
    mid-left, miss site at the bottom-right corner — the miss window is
    all zeros, so its rank is exactly 1 + N_SPECIAL + 1 (hot cell).
    """
    tmp = tmp_path_factory.mktemp("holdout")
    cfg = load_config("configs/karagwe.yml")
    g = cfg["grid"]
    bounds = wgs84_to_utm(g["aoi_wgs84"], g["crs"])
    tf = make_grid_transform(g["crs"], 3000, bounds)
    w, h = grid_dimensions(bounds, tf)
    grid = GridSpec(crs=g["crs"], resolution_m=3000, transform=tf,
                    width=w, height=h, bounds=bounds)
    assert h * w >= N_SPECIAL + 10

    arr = np.zeros(grid.shape, dtype="float32")
    arr.ravel()[:N_SPECIAL] = 0.5      # top rows: cells that beat background
    hot_rc, miss_rc = (h // 2, w // 2), (h - 5, w - 5)
    arr[hot_rc] = 0.99                 # the "rediscovered" cell
    arr[10:12, 5:8] = np.nan           # invalid-cell hole

    proba = tmp / "proba.tif"
    with rasterio.open(proba, "w", **grid.profile(dtype="float32")) as dst:
        dst.write(arr, 1)

    to_ll = Transformer.from_crs(g["crs"], "EPSG:4326", always_xy=True)

    def _ll(rc):
        x, y = rasterio.transform.xy(tf, *rc, offset="center")
        return to_ll.transform(x, y)

    return {"dir": tmp, "grid": grid, "proba": proba,
            "hot": _ll(hot_rc), "miss": _ll(miss_rc),
            "outside": (g["aoi_wgs84"]["min_lon"] - 5.0,
                        g["aoi_wgs84"]["min_lat"] - 5.0)}


class TestSelection:
    def test_by_names(self):
        sites = select_holdout(_labels(), names=["buhweju", "GAKARA"])
        assert sorted(sites["name"]) == ["Buhweju", "Gakara"]

    def test_by_names_single_string(self):
        sites = select_holdout(_labels(), names="Nyakabingo")
        assert list(sites["name"]) == ["Nyakabingo"]

    def test_by_names_missing_raises(self):
        with pytest.raises(ValueError, match="not in labels"):
            select_holdout(_labels(), names=["Kabanga"])

    def test_n_holdout_capped(self):
        labels = _labels()
        picked = select_holdout(labels, n_holdout=10, seed=42)
        assert len(picked) == len(labels)  # pool smaller than n

    def test_seeded_sampling_is_deterministic(self):
        labels = _labels()
        a = select_holdout(labels, n_holdout=2, seed=42)
        b = select_holdout(labels, n_holdout=2, seed=42)
        c = select_holdout(labels, n_holdout=2, seed=7)
        assert len(a) == 2
        assert list(a.index) == list(b.index)
        assert list(a.index) != list(c.index)

    def test_random_respects_in_extent_mask(self):
        labels = _labels()
        mask = np.array([True, True, False, False, False])
        picked = select_holdout(labels, n_holdout=2, seed=0, in_extent=mask)
        assert set(picked.index) == {0, 1}

    def test_random_falls_back_when_mask_empty(self):
        labels = _labels()
        mask = np.zeros(len(labels), dtype=bool)
        picked = select_holdout(labels, n_holdout=1, seed=0, in_extent=mask)
        assert len(picked) == 1

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError, match="n_holdout"):
            select_holdout(_labels(), n_holdout=0)


class TestHoldoutLabels:
    def test_alias_and_proximity_removed(self, tmp_path):
        labels = _labels()
        sites = select_holdout(labels, names=["Buhweju"])
        out = tmp_path / "labels.gpkg"
        reduced, n_removed = build_holdout_labels(labels, sites, out)
        # exact-name row + alias row within COLOCATION_KM are gone
        assert n_removed == 2
        assert len(reduced) == len(labels) - 2
        assert not {"Buhweju", "Buhweju North"} & set(reduced["name"])
        assert set(reduced["name"]) == {"Nyakabingo", "Ntungamo", "Gakara"}
        assert COLOCATION_KM > 0
        on_disk = gpd.read_file(out)
        assert len(on_disk) == len(reduced)
        assert on_disk.crs == labels.crs

    def test_far_duplicate_name_removed(self, tmp_path):
        # same name, opposite side of the belt: only the name match can
        # remove it (proximity certainly cannot)
        labels = gpd.GeoDataFrame(
            {"name": ["Target", "Target", "Other"],
             "geometry": [Point(30.0, -1.0), Point(31.5, -2.5),
                          Point(31.0, -2.0)]},
            crs="EPSG:4326")
        sites = labels.iloc[[0]]
        reduced, n_removed = build_holdout_labels(
            labels, sites, tmp_path / "l.gpkg")
        assert n_removed == 2
        assert list(reduced["name"]) == ["Other"]

    def test_nearby_row_without_name_match_still_removed(self, tmp_path):
        labels = gpd.GeoDataFrame(
            {"name": ["Target", "Targey", "Other"],
             "geometry": [Point(30.0, -1.0), Point(30.005, -1.0),
                          Point(31.0, -2.0)]},
            crs="EPSG:4326")  # Targey ~550 m away: inside the 2 km halo
        sites = labels.iloc[[0]]
        reduced, n_removed = build_holdout_labels(
            labels, sites, tmp_path / "l.gpkg")
        assert n_removed == 2
        assert list(reduced["name"]) == ["Other"]

    def test_empty_sites_raises(self, tmp_path):
        labels = _labels()
        with pytest.raises(ValueError, match="no holdout sites"):
            build_holdout_labels(labels, labels.iloc[0:0],
                                 tmp_path / "x.gpkg")

    def test_zero_match_raises(self, tmp_path):
        labels = _labels()
        far = gpd.GeoDataFrame({"name": ["Ghost"]},
                               geometry=[Point(0.0, 0.0)],
                               crs="EPSG:4326")
        with pytest.raises(ValueError, match="matched no label"):
            build_holdout_labels(labels, far, tmp_path / "x.gpkg")


class TestDepositDistance:
    def test_prior_excludes_holdout(self, tmp_path):
        """Rebuilt prior must not read 0 at the held-out deposit cell."""
        cfg = load_config("configs/karagwe.yml")
        cfg["grid"]["resolution_m"] = 3000  # coarsen: fast offline run
        cfg_path = tmp_path / "tiny.yml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        labels = gpd.GeoDataFrame(
            {"name": ["A", "B", "C"],
             "geometry": [Point(30.067, -1.917),
                          Point(30.150, -1.950),
                          Point(30.500, -1.600)]},
            crs="EPSG:4326")
        sites = labels.iloc[[0]]  # hold out A (~10 km from B)
        reduced, n_removed = build_holdout_labels(
            labels, sites, tmp_path / "labels.gpkg")
        assert n_removed == 1
        assert list(reduced["name"]) == ["B", "C"]

        out = (tmp_path / "features"
               / "deposit_distance_tin_tungsten_tantalum.tif")
        path = build_holdout_deposit_distance(
            "tin_tungsten_tantalum", reduced, out, config_path=cfg_path)
        assert path == out
        assert out.with_suffix(".json").exists()  # band manifest
        with rasterio.open(path) as src:
            assert src.count == 1
            assert src.descriptions[0] == "dist_to_deposit_loo_m"
            x, y = Transformer.from_crs(
                "EPSG:4326", src.crs, always_xy=True
            ).transform(30.067, -1.917)
            v = float(next(src.sample([(x, y)]))[0])
        # distance from the HELD-OUT cell to the nearest REMAINING
        # deposit — > 0 proves the holdout leaked nothing into the prior
        assert 1000.0 < v < 20000.0

    def test_empty_labels_raises(self, tmp_path):
        cfg = load_config("configs/karagwe.yml")
        cfg["grid"]["resolution_m"] = 3000
        cfg_path = tmp_path / "tiny.yml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        empty = gpd.GeoDataFrame({"name": []},
                                 geometry=[], crs="EPSG:4326")
        with pytest.raises(ValueError, match="zero labels"):
            build_holdout_deposit_distance(
                "tin_tungsten_tantalum", empty,
                tmp_path / "dd.tif", config_path=cfg_path)


class TestSubstitution:
    def test_swap_by_stem(self, tmp_path):
        prod_dir, hold_dir = tmp_path / "prod", tmp_path / "hold"
        prod_dir.mkdir()
        hold_dir.mkdir()
        stem = "deposit_distance_tin_tungsten_tantalum"
        prod = prod_dir / f"{stem}.tif"
        prod.write_bytes(b"prod")
        hold = hold_dir / f"{stem}.tif"
        hold.write_bytes(b"hold")
        rasters = [prod_dir / "dem.tif", prod, prod_dir / "s2.tif"]

        out = substitute_deposit_distance(rasters, hold)
        assert out == [prod_dir / "dem.tif", hold, prod_dir / "s2.tif"]
        assert prod not in out

    def test_appends_when_absent(self, tmp_path):
        hold = tmp_path / "deposit_distance_tin_tungsten_tantalum.tif"
        rasters = [tmp_path / "dem.tif", tmp_path / "s2.tif"]
        out = substitute_deposit_distance(rasters, hold)
        assert len(out) == 3
        assert out[-1] == hold


class TestScoring:
    def test_score_high_cell_flagged(self, proba_env):
        e = proba_env
        lon, lat = e["hot"]
        s = score_holdout(e["proba"], lon, lat, name="hot_cell")
        assert s["name"] == "hot_cell"
        assert s["in_extent"] is True
        assert s["prob_at_holdout"] == pytest.approx(0.99)
        assert s["rank"] == 1
        assert s["percentile"] == pytest.approx(100.0)
        assert s["rediscovered"] is True
        assert s["distance_to_top1_km"] == pytest.approx(0.0, abs=0.1)
        assert s["n_within_tolerance"] >= 1
        assert s["n_top50_within_tolerance"] >= 1

    def test_background_cell_not_rediscovered(self, proba_env):
        e = proba_env
        lon, lat = e["miss"]
        s = score_holdout(e["proba"], lon, lat, name="far_cell")
        assert s["in_extent"] is True
        assert s["n_within_tolerance"] >= 1  # window is all zeros
        assert s["n_top50_within_tolerance"] == 0
        # 150 beaters + the hot cell outrank the local best (0.0)
        assert s["rank"] == 152
        assert s["rediscovered"] is False
        assert s["percentile"] < 100.0
        assert np.isfinite(s["distance_to_top1_km"])

    def test_tolerance_and_top_n_parameters(self, proba_env):
        e = proba_env
        lon, lat = e["hot"]
        s = score_holdout(e["proba"], lon, lat,
                          tolerance_km=1.0, top_n=1)
        assert s["tolerance_km"] == 1.0
        assert s["top_n"] == 1
        assert s["rediscovered"] is True  # rank 1 <= top_n 1

    def test_out_of_extent_marked_nan(self, proba_env):
        e = proba_env
        lon, lat = e["outside"]
        s = score_holdout(e["proba"], lon, lat, name="outside")
        assert s["in_extent"] is False
        assert np.isnan(s["rank"])
        assert np.isnan(s["prob_at_holdout"])
        assert np.isnan(s["percentile"])
        assert np.isnan(s["distance_to_top1_km"])
        assert s["rediscovered"] is False
        assert s["n_valid_cells"] == 0


class TestReportFile:
    def test_report_and_sites_written(self, tmp_path):
        result = {
            "group": "tin_tungsten_tantalum",
            "holdout_deposit": "Nyakabingo",
            "rediscovery": {"rank_in_top50": 1,
                            "distance_to_top1_km": np.float32(1.25),
                            "rediscovered": True},
            "sites": [{"group": "tin_tungsten_tantalum",
                       "name": "Nyakabingo", "lon": 30.067, "lat": -1.917,
                       "rank": 1, "rediscovered": True}],
        }
        paths = write_report(result, tmp_path)
        assert paths["report"].exists()
        assert paths["sites"].exists()

        payload = json.loads(paths["report"].read_text(encoding="utf-8"))
        assert payload["n_results"] == 1
        (rec,) = payload["results"]
        assert rec["holdout_deposit"] == "Nyakabingo"
        # np.float32 exercised the _json_default encoder
        assert rec["rediscovery"]["distance_to_top1_km"] == 1.25

        gj = json.loads(paths["sites"].read_text(encoding="utf-8"))
        assert gj["type"] == "FeatureCollection"
        assert len(gj["features"]) == 1
        feat = gj["features"][0]
        assert feat["geometry"]["type"] == "Point"
        assert feat["geometry"]["coordinates"] == [30.067, -1.917]
        assert feat["properties"]["name"] == "Nyakabingo"

    def test_multi_result_report_and_empty_sites(self, tmp_path):
        r1 = {"group": "a", "sites": []}
        r2 = {"group": "b", "sites": []}
        report = holdout_report([r1, r2], tmp_path / "r.json")
        payload = json.loads(report.read_text(encoding="utf-8"))
        assert payload["n_results"] == 2
        assert payload["generated_at"]

        sites = holdout_sites([r1, r2], tmp_path / "s.geojson")
        gj = json.loads(sites.read_text(encoding="utf-8"))
        assert gj["type"] == "FeatureCollection"
        assert gj["features"] == []

    def test_default_holdouts_are_real_names(self):
        assert set(DEFAULT_HOLDOUT) == {
            "tin_tungsten_tantalum", "copper_zinc", "bauxite"}
        assert all(DEFAULT_HOLDOUT.values())



