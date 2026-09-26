"""Tests for preprocessing (grid, rasterize, DEM features, orchestrator).

Run with:  pytest tests/test_preprocess.py -v

Uses a 300 m resolution override so every test runs on a small fast grid
(~450 x 450 px) while exercising exactly the production code paths.
"""
from __future__ import annotations

import numpy as np
import pytest
from shapely import geometry as sgeom

from src.labels import sample_background, seed_points
from src.preprocess import (
    GridSpec,
    dem_features,
    extract_at_points,
    label_rasters,
    rasterize_points,
)
from src.utils import load_config


@pytest.fixture(scope="module")
def grid300():
    cfg = load_config("configs/karagwe.yml")
    return GridSpec.from_config(cfg, resolution_m=300)


# ─── GridSpec ───────────────────────────────────────────────────────────

class TestGridSpec:
    def test_from_config_snapped_origin(self, grid300):
        assert grid300.transform.a == 300
        assert grid300.transform.e == -300
        assert grid300.transform.c % 300 == pytest.approx(0)
        assert grid300.height * 300 >= (
            grid300.bounds[3] - grid300.bounds[1]
        ) - 300

    def test_roundtrip_world_pixel(self, grid300):
        xs = np.array([150_000.0, 200_000.0])
        ys = np.array([9_800_000.0, 9_750_000.0])
        c, r = grid300.world_to_pixel(xs, ys)
        bx, by = grid300.pixel_to_world(np.floor(c), np.floor(r))
        # back-projected cell centres within one cell of originals
        assert (np.abs(bx - xs) <= 300).all()
        assert (np.abs(by - ys) <= 300).all()

    def test_contains(self, grid300):
        inside = grid300.contains([180_000], [9_800_000])
        outside = grid300.contains([500_000], [9_000_000])
        assert inside == [True]
        assert outside == [False]

    def test_resolution_override_changes_size(self):
        cfg = load_config("configs/karagwe.yml")
        g30 = GridSpec.from_config(cfg)
        g300 = GridSpec.from_config(cfg, resolution_m=300)
        # Coarse grids quantize the AOI to their own cell size, so the
        # 10x width ratio holds only to within ~one coarse pixel.
        assert g300.width * 10 == pytest.approx(g30.width, rel=1e-3)


# ─── Rasterize / extract ────────────────────────────────────────────────

class TestRasterize:
    def test_point_burns_one_cell(self, grid300):
        pt = sgeom.Point(grid300.bounds[0] + 15_000,
                         grid300.bounds[1] + 15_000)
        gdf = __import__("geopandas").GeoDataFrame(
            {"v": [7.0]}, geometry=[pt], crs=grid300.crs
        )
        arr = rasterize_points(gdf, grid300, value_col="v")
        assert int((arr > 0).sum()) == 1
        assert float(arr.max()) == 7.0

    def test_label_rasters_y_encoding(self, grid300):
        # use the KAB group so seeds fall inside this Rwanda grid
        pos = seed_points("tin_tungsten_tantalum")
        bg = sample_background(pos, ["karagwe_ankole"], n=50, seed=5)
        r = label_rasters(pos, bg, grid300, buffer_m=600)
        y = r.get("y")
        assert y is not None
        assert set(np.unique(y)) <= {0, 1, 255}
        assert (y == 1).sum() > 0          # positives burned
        assert (y == 0).sum() > 0          # backgrounds present

    def test_extract_at_points_shape(self, grid300):
        rng = np.random.default_rng(0)
        stack = rng.uniform(size=(3, grid300.height, grid300.width)).astype(
            "float32"
        )
        pts = sample_background(
            seed_points("bauxite"), ["usambara_bauxite"], n=40, seed=6
        )
        df = extract_at_points(stack, ["a", "b", "c"], grid300, pts)
        assert len(df) <= 40               # some may fall outside grid? no -
        assert {"a", "b", "c"} <= set(df.columns)

    def test_extract_band_count_mismatch_raises(self, grid300):
        stack = np.zeros((2, grid300.height, grid300.width), "float32")
        with pytest.raises(ValueError):
            extract_at_points(stack, ["x"], grid300,
                              sample_background(seed_points("bauxite"),
                                                ["usambara_bauxite"],
                                                n=10, seed=1))


# ─── DEM features ───────────────────────────────────────────────────────

class TestDemFeatures:
    def _plane(self, n=64):
        yy, xx = np.mgrid[0:n, 0:n]
        return (50.0 + 0.20 * yy - 0.10 * xx).astype("float32")

    def test_slope_of_tilted_plane(self):
        """dz/dy=0.20, dz/dx=-0.10 -> slope atan(sqrt(.05)) ~ 12.6 deg."""
        feats = dem_features(self._plane(), resolution_m=1.0,
                             include_raw=False)
        s = feats["slope_deg"]
        expected = np.degrees(np.arctan(np.hypot(0.10, 0.20)))
        assert np.allclose(s[2:-2, 2:-2], expected, atol=0.5)

    def test_aspect_north_facing(self):
        """Elevation increases northward -> aspect points south (~180)."""
        n = 32
        yy, _ = np.mgrid[0:n, 0:n]
        dem = (yy * 0.5).astype("float32")
        _, aspect = __import__(
            "src.preprocess.dem_features", fromlist=["slope_aspect"]
        ).slope_aspect(dem, 1.0)
        inner = aspect[4:-4, 4:-4]
        assert np.allclose(inner, 180.0, atol=2.0)

    def test_plateau_index_high_on_flat_top(self):
        """Mesa: flat top should score higher plateau index than flanks."""
        n = 64
        yy, xx = np.mgrid[0:n, 0:n]
        mesa = np.where((yy > 16) & (yy < 48) & (xx > 16) & (xx < 48),
                        100.0, 100.0 - 8.0 * (
                            np.abs(yy - 32) + np.abs(xx - 32) - 16
                        ).clip(min=0)).astype("float32")
        pi = __import__(
            "src.preprocess.dem_features", fromlist=["plateau_index"]
        ).plateau_index(mesa, resolution_m=30.0)
        top = pi[26:38, 26:38].mean()
        edge = pi[2:8, 2:8].mean()
        assert top > edge

    def test_output_shapes_and_dtypes(self):
        feats = dem_features(self._plane(), resolution_m=30.0)
        for name, band in feats.items():
            assert band.shape == (64, 64), name
            assert band.dtype == np.float32
