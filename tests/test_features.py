"""Tests for the feature-engineering layer.

Run: pytest tests/test_features.py -v

Covers spectral indices, terrain re-exports, distance transforms,
magnetic gradients, lithology encoding, and multi-band stacking.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import rasterio
from shapely.geometry import box

from src.features import (
    build_feature_stack,
    distance_to_features,
    encode_lithology,
    magnetic_gradient_features,
    spectral_indices,
)
from src.preprocess.grid import GridSpec

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def grid300():
    """Small 300 m grid over KAB AOI for fast offline testing."""
    from src.utils import grid_dimensions, load_config, make_grid_transform, wgs84_to_utm
    cfg = load_config("configs/karagwe.yml")
    g = cfg["grid"]
    crs = g["crs"]
    res = 300
    bounds = wgs84_to_utm(g["aoi_wgs84"], crs)
    transform = make_grid_transform(crs, res, bounds)
    w, h = grid_dimensions(bounds, transform)
    return GridSpec(crs=crs, resolution_m=res, transform=transform,
                    width=w, height=h, bounds=bounds)


@pytest.fixture
def s2_array():
    """Synthetic 6-band Sentinel-2-like array (bands, h, w)."""
    rng = np.random.default_rng(42)
    h, w = 64, 64
    bands = np.stack([
        rng.uniform(0.05, 0.20, (h, w)),   # B2 Blue
        rng.uniform(0.08, 0.25, (h, w)),   # B3 Green
        rng.uniform(0.05, 0.30, (h, w)),   # B4 Red
        rng.uniform(0.20, 0.60, (h, w)),   # B8 NIR
        rng.uniform(0.01, 0.15, (h, w)),   # B11 SWIR1
        rng.uniform(0.01, 0.30, (h, w)),   # B12 SWIR2
    ]).astype(np.float32)
    band_names = ["B2", "B3", "B4", "B8", "B11", "B12"]
    return bands, band_names


# ── Spectral indices ─────────────────────────────────────────────────


class TestSpectralIndices:
    def test_returns_all_indices(self, s2_array):
        arr, names = s2_array
        out = spectral_indices(arr, names)
        assert set(out.keys()) == {
            "ndwi", "mbi", "bsi", "evi", "savi", "nirv",
            "ferrous_iron_ratio", "iron_oxide_ratio",
            "aloh_ratio", "mgoh_carbonate_proxy",
        }

    def test_ndwi_range(self, s2_array):
        arr, names = s2_array
        ndwi = spectral_indices(arr, names)["ndwi"]
        assert ndwi.shape == (64, 64)
        assert np.nanmax(ndwi) <= 1.0
        assert np.nanmin(ndwi) >= -1.0

    def test_all_finite_in_valid_region(self, s2_array):
        arr, names = s2_array
        out = spectral_indices(arr, names)
        for name, band in out.items():
            assert not np.any(np.isinf(band)), name

    def test_missing_band_raises(self):
        arr = np.ones((6, 32, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="Missing S2 bands"):
                        spectral_indices(arr, ["B2", "B3", "B4"])


class TestAlterationIndices:
    """Hydrothermal alteration / weathering band ratios (T1, P0)."""

    def test_aloh_known_value(self):
        # B11=0.30, B12=0.10 -> Al-OH depth proxy = 3.0 (strong 2.2 um dip)
        arr = np.full((6, 8, 8), 0.2, dtype=np.float32)
        arr[4] = 0.30   # B11 SWIR1
        arr[5] = 0.10   # B12 SWIR2
        out = spectral_indices(arr, ["B2", "B3", "B4", "B8", "B11", "B12"])
        assert np.allclose(out["aloh_ratio"], 3.0)
        assert np.allclose(out["mgoh_carbonate_proxy"], 1.0 / 3.0)

    def test_iron_oxide_known_value(self):
        arr = np.full((6, 8, 8), 0.2, dtype=np.float32)
        arr[0] = 0.05   # B2 Blue
        arr[2] = 0.40   # B4 Red
        out = spectral_indices(arr, ["B2", "B3", "B4", "B8", "B11", "B12"])
        assert np.allclose(out["iron_oxide_ratio"], 8.0)

    def test_ferrous_iron_known_value(self):
        arr = np.full((6, 8, 8), 0.2, dtype=np.float32)
        arr[3] = 0.40   # B8 NIR
        arr[4] = 0.10   # B11 SWIR1
        out = spectral_indices(arr, ["B2", "B3", "B4", "B8", "B11", "B12"])
        assert np.allclose(out["ferrous_iron_ratio"], 0.25)

    def test_zero_denominator_is_nan_not_zero(self):
        # A ratio of exactly 0 would falsely read as "max absorption";
        # zero-division must yield NaN so training rows drop instead.
        arr = np.full((6, 8, 8), 0.2, dtype=np.float32)
        arr[5] = 0.0    # B12 SWIR2 = 0 -> aloh = B11/0 undefined
        out = spectral_indices(arr, ["B2", "B3", "B4", "B8", "B11", "B12"])
        assert np.all(np.isnan(out["aloh_ratio"]))

    def test_b12_optional(self):
        # 5-band raster (no SWIR2): Al-OH / Mg-OH proxies omitted, the
        # B11-only ratios still computed.
        arr = np.ones((5, 8, 8), dtype=np.float32)
        out = spectral_indices(arr, ["B2", "B3", "B4", "B8", "B11"])
        assert "aloh_ratio" not in out
        assert "mgoh_carbonate_proxy" not in out
        assert "ferrous_iron_ratio" in out
        assert "iron_oxide_ratio" in out

    def test_nodata_propagates_to_ratios(self, s2_array):
        arr, names = s2_array
        arr[:, 0, 0] = 0.0                     # synthetic nodata pixel
        out = spectral_indices(arr, names)
        for name in ("aloh_ratio", "ferrous_iron_ratio",
                     "iron_oxide_ratio", "mgoh_carbonate_proxy"):
            assert np.isnan(out[name][0, 0]), name


# ── Distance transforms ──────────────────────────────────────────────


class TestDistanceFeatures:
    def test_distance_to_fault(self, grid300):
        import geopandas as gpd
        from shapely.geometry import LineString
        fault = gpd.GeoDataFrame(
            {"id": [1]}, geometry=[
                LineString([
                    (grid300.bounds[0], grid300.bounds[1]),
                    (grid300.bounds[2], grid300.bounds[3])
                ])
            ], crs="EPSG:4326"
        )
        out = distance_to_features(grid300, fault_gdf=fault, res_m=300)
        assert "dist_to_fault_m" in out
        assert out["dist_to_fault_m"].shape == grid300.shape
        assert np.nanmin(out["dist_to_fault_m"]) == pytest.approx(0, abs=350)

    def test_empty_gdf_returns_empty(self, grid300):
        import geopandas as gpd
        empty = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
        out = distance_to_features(grid300, fault_gdf=empty)
        assert out == {}


class TestDepositDistanceLOO:
    """Leakage-safe distance-to-known-deposit prior (T2, P0)."""

    def _deposits(self, grid, offsets):
        import geopandas as gpd
        from shapely.geometry import Point
        tf = grid.transform
        # convert grid cell (col,row) offsets to lon/lat points
        pts = []
        for dc, dr in offsets:
            x = tf.c + (dc + 0.5) * tf.a
            y = tf.f + (dr + 0.5) * tf.e
            pts.append(Point(x, y))
        gdf = gpd.GeoDataFrame(geometry=pts, crs=grid.crs)
        return gdf.to_crs("EPSG:4326")   # function must reproject back

    def test_deposit_cell_never_zero_when_two_deposits(self, grid300):
        from src.features import distance_to_known_deposits_loo
        dep = self._deposits(grid300, [(2, 2), (12, 12)])
        out = distance_to_known_deposits_loo(grid300, dep, res_m=300)
        d = out["dist_to_deposit_loo_m"]
        assert d.shape == grid300.shape

        for dc, dr in [(2, 2), (12, 12)]:
            r, c = int(dr), int(dc)
            assert d[r, c] > 0, "LOO must not leak self-distance 0"

    def test_loo_equals_inter_deposit_distance(self, grid300):
        from src.features import distance_to_known_deposits_loo
        dep = self._deposits(grid300, [(2, 2), (12, 2)])  # 10 cells apart
        d = distance_to_known_deposits_loo(grid300, dep, res_m=300)[
            "dist_to_deposit_loo_m"]
        expected = 10 * 300  # 10 cells * 300 m
        assert d[2, 2] == pytest.approx(expected, rel=0.05)

    def test_background_pixel_measures_nearest_deposit(self, grid300):
        from src.features import distance_to_known_deposits_loo
        dep = self._deposits(grid300, [(2, 2), (12, 2)])
        d = distance_to_known_deposits_loo(grid300, dep, res_m=300)[
            "dist_to_deposit_loo_m"]
        # a cell 3 cells east of deposit A (no deposit there)
        assert d[2, 5] == pytest.approx(3 * 300, rel=0.05)

    def test_single_deposit_falls_back(self, grid300):
        from src.features import distance_to_known_deposits_loo
        dep = self._deposits(grid300, [(5, 5)])
        d = distance_to_known_deposits_loo(grid300, dep, res_m=300)[
            "dist_to_deposit_loo_m"]
        assert d[5, 5] == pytest.approx(0, abs=1e-6)   # documented fallback

    def test_empty_deposits_raise(self, grid300):
        import geopandas as gpd

        from src.features import distance_to_known_deposits_loo
        empty = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
        with pytest.raises(ValueError, match="empty"):
            distance_to_known_deposits_loo(grid300, empty)


# ── Magnetic gradients ───────────────────────────────────────────────


class TestMagneticFeatures:
    def test_synthetic_basin(self):
        rng = np.random.default_rng(7)
        h, w = 64, 64
        yy, xx = np.mgrid[0:h, 0:w]
        anomaly = 100.0 + 50 * np.sin(xx / 10) * np.cos(yy / 8)
        noise = rng.normal(0, 2, (h, w))
        mag = anomaly + noise
        out = magnetic_gradient_features(mag, grid_res_m=30, detrend=True)
        assert set(out.keys()) == {"tga", "gx", "gy", "mod_grad"}
        for _k, v in out.items():
            assert v.shape == (64, 64)
            assert np.all(np.isfinite(v))

    def test_detrend_removes_tilt(self):
        h, w = 64, 64  # bigger grid to reduce Sobel edge effects
        yy, xx = np.mgrid[0:h, 0:w]
        plane = 0.1 * xx + 0.2 * yy + 50.0
        out = magnetic_gradient_features(plane, grid_res_m=1.0, detrend=True)
        assert np.nanstd(out["tga"]) < 0.2  # planar trend removed; only Sobel noise


# ── Lithology encoding ────────────────────────────────────────────────


class TestLithologyEncoding:
    def test_encode_lithology(self, grid300):
        import geopandas as gpd
        poly = box(grid300.bounds[0], grid300.bounds[1],
                   grid300.bounds[0] + 5000, grid300.bounds[1] + 5000)
        gdf = gpd.GeoDataFrame(
            {"rock_type": ["granite"]}, geometry=[poly], crs=grid300.crs
        ).to_crs("EPSG:4326")
        out = encode_lithology(gdf, grid300, field="rock_type")
        assert out.shape == grid300.shape
        assert out.dtype == np.uint8
        assert (out > 0).sum() > 0


# ── Multi-band stack ──────────────────────────────────────────────────


class TestFeatureStack:
    def test_build_and_manifest(self, grid300, tmp_path):
        bands = {
            "slope": np.zeros(grid300.shape, dtype=np.float32),
            "ndwi": np.ones(grid300.shape, dtype=np.float32),
            "tga": np.full(grid300.shape, 2.0, dtype=np.float32),
        }
        tif, names = build_feature_stack(grid300, bands, tmp_path / "stack.tif")
        assert tif.exists()
        assert names == ["slope", "ndwi", "tga"]

        with rasterio.open(tif) as src:
            assert src.count == 3
            assert src.dtypes == ("float32",) * 3
            assert src.crs.to_string() == grid300.crs

        manifest = json.loads((tmp_path / "stack.json").read_text())
        assert manifest["bands"][0]["name"] == "slope"
        assert manifest["width"] == grid300.width

    def test_shape_mismatch_raises(self, grid300, tmp_path):
        bad = np.zeros((10, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            build_feature_stack(grid300, {"bad": bad}, tmp_path / "x.tif")


class TestDistanceToPoints:
    """Plain (non-LOO) distance field behind the novel-ground exclusion."""

    def _points(self, grid, offsets):
        import geopandas as gpd
        from shapely.geometry import Point

        tf = grid.transform
        pts = [Point(tf.c + (dc + 0.5) * tf.a, tf.f + (dr + 0.5) * tf.e)
               for dc, dr in offsets]
        return gpd.GeoDataFrame(geometry=pts, crs=grid.crs).to_crs("EPSG:4326")

    def test_own_cell_reads_zero_where_loo_does_not(self, grid300):
        # LOO reports a deposit as the distance to the NEXT deposit, so with a
        # 2 km exclusion ring this deposit (3 km from its neighbour) would pass
        # the filter and be requested as "novel" ground. The plain field must
        # read 0 on the deposit cell itself.
        from src.features.distances import (
            distance_to_known_deposits_loo,
            distance_to_points_m,
        )
        pts = self._points(grid300, [(2, 2), (12, 2)])
        plain = distance_to_points_m(pts, grid300.transform, grid300.shape,
                                     grid300.crs)
        loo = distance_to_known_deposits_loo(grid300, pts, res_m=300)[
            "dist_to_deposit_loo_m"]
        ring_m = 2000.0
        assert plain[2, 2] <= ring_m      # correctly excluded
        assert loo[2, 2] > ring_m         # LOO would have let it through

    def test_background_cell_measures_nearest_point(self, grid300):
        from src.features.distances import distance_to_points_m
        pts = self._points(grid300, [(2, 2)])
        d = distance_to_points_m(pts, grid300.transform, grid300.shape,
                                 grid300.crs)
        assert d[2, 5] == pytest.approx(3 * 300, rel=0.05)   # 3 cells east

    def test_every_point_cell_is_zero(self, grid300):
        from src.features.distances import distance_to_points_m
        pts = self._points(grid300, [(2, 2), (12, 12), (5, 15)])
        d = distance_to_points_m(pts, grid300.transform, grid300.shape,
                                 grid300.crs)
        for dc, dr in [(2, 2), (12, 12), (5, 15)]:
            assert d[dr, dc] == pytest.approx(0.0, abs=1e-6)

    def test_shape_and_dtype(self, grid300):
        from src.features.distances import distance_to_points_m
        d = distance_to_points_m(self._points(grid300, [(1, 1)]),
                                 grid300.transform, grid300.shape,
                                 grid300.crs)
        assert d.shape == grid300.shape
        assert d.dtype == np.float32

    def test_empty_points_raise(self, grid300):
        import geopandas as gpd

        from src.features.distances import distance_to_points_m

        with pytest.raises(ValueError, match="empty"):
            distance_to_points_m(gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
                                 grid300.transform, grid300.shape,
                                 grid300.crs)


class TestCellSizeM:
    def test_projected_grid_reads_the_affine(self, grid300):
        from src.features.distances import cell_size_m
        assert cell_size_m(grid300.transform, grid300.crs,
                           grid300.shape) == (300.0, 300.0)

    def test_string_crs_is_not_mistaken_for_geographic(self, grid300):
        # grid.crs is the literal string "EPSG:32736"; skipping the string
        # branch would leave a UTM grid mistaken for a degree grid.
        from src.features.distances import cell_size_m
        h, w = cell_size_m(grid300.transform, grid300.crs, grid300.shape)
        assert h == pytest.approx(300.0, rel=1e-9)
        assert w == pytest.approx(300.0, rel=1e-9)

    def test_geographic_grid_converts_degrees_to_metres(self):
        from affine import Affine

        from src.features.distances import cell_size_m
        # 0.01 degree cells centred near the equator
        tf = Affine(0.01, 0.0, 29.0, 0.0, -0.01, -1.0)
        h, w = cell_size_m(tf, "EPSG:4326", (20, 20))
        assert h == pytest.approx(0.01 * 110_540, rel=0.01)
        assert w == pytest.approx(0.01 * 111_320, rel=0.01)
