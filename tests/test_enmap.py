"""Tests for src.ingest.enmap (T6, P1)."""
from __future__ import annotations

import affine
import numpy as np
import pytest
import rasterio

from src.ingest.enmap import (
    align_to_grid,
    cloud_mask,
    load_enmap_cube,
    parse_wavelengths,
)
from src.preprocess.grid import GridSpec

WLS = [456.3, 553.1, 652.4, 832.7, 1610.5, 2185.2]   # 6-band toy cube


@pytest.fixture
def grid300():
    from src.utils import grid_dimensions, load_config, \
        make_grid_transform, wgs84_to_utm
    cfg = load_config("configs/karagwe.yml")
    g = cfg["grid"]
    res = 300
    bounds = wgs84_to_utm(g["aoi_wgs84"], g["crs"])
    transform = make_grid_transform(g["crs"], res, bounds)
    w, h = grid_dimensions(bounds, transform)
    return GridSpec(crs=g["crs"], resolution_m=res, transform=transform,
                    width=w, height=h, bounds=bounds)


@pytest.fixture
def cube_tif(tmp_path):
    """Synthetic EnMAP-like GeoTIFF: 6 bands, 30 m, inside the KAB grid."""
    rng = np.random.default_rng(3)
    h, w = 40, 40
    cube = rng.uniform(0.05, 0.35, (6, h, w))
    cube[:, 0, 0] = 0.95                     # bright "cloud" pixel
    cube[0, 0, :] = 0.90                     # bright first band row
    desc = [f"B{i + 1:03d} ({wl}nm)" for i, wl in enumerate(WLS)]
    path = tmp_path / "enmap_cube.tif"
    profile = {
        "driver": "GTiff", "height": h, "width": w, "count": 6,
        "dtype": "float32", "crs": "EPSG:32736",
        "transform": affine.Affine(30.0, 0.0, 100_000.0,
                                   0.0, -30.0, 9_860_000.0),
    }
    with rasterio.open(path, "w", **profile) as dst:
        for b in range(6):
            dst.write(cube[b].astype("float32"), b + 1)
            dst.set_band_description(b + 1, desc[b])
    return path, cube


class TestParseWavelengths:
    def test_parenthesised(self):
        wl = parse_wavelengths([f"B{i:03d} ({w}nm)" for i, w in enumerate(WLS)])
        assert np.allclose(wl, WLS)

    def test_bare_numbers(self):
        wl = parse_wavelengths([str(w) for w in WLS])
        assert np.allclose(wl, WLS)

    def test_unusable_returns_none(self):
        assert parse_wavelengths(["band_1", "band_2", "band_3"]) is None


class TestLoadCube:
    def test_reads_cube_and_wavelengths(self, cube_tif):
        path, _ = cube_tif
        cube, wl = load_enmap_cube(path)
        assert cube.shape[0] == 6
        assert np.allclose(wl, WLS)

    def test_auto_scales_dn(self, tmp_path):
        path = tmp_path / "dn.tif"
        desc = [f"{w}nm" for w in WLS]
        profile = {"driver": "GTiff", "height": 4, "width": 4, "count": 6,
                   "dtype": "uint16", "crs": "EPSG:32736",
                   "transform": affine.Affine(30, 0, 0, 0, -30, 0)}
        data = (np.full((6, 4, 4), 3000)).astype("uint16")
        with rasterio.open(path, "w", **profile) as dst:
            for b in range(6):
                dst.write(data[b], b + 1)
                dst.set_band_description(b + 1, desc[b])
        cube, _ = load_enmap_cube(path)
        assert np.nanpercentile(cube, 95) <= 1.5      # scaled to reflectance


class TestCloudMask:
    def test_flags_bright_pixels(self, cube_tif):
        path, _ = cube_tif
        cube, wl = load_enmap_cube(path)
        mask = cloud_mask(cube, wl, brightness_thr=0.45)
        assert mask[0, 0]          # bright pixel flagged
        assert not mask[20, 20]    # normal reflectance kept

    def test_uses_vnir_only(self):
        # bright SWIR-only target must NOT be flagged when wl given
        cube = np.zeros((2, 4, 4))
        cube[0] = 0.1              # 500 nm - dark
        cube[1] = 0.9              # 2200 nm - bright alteration
        wl = np.array([500.0, 2200.0])
        assert not cloud_mask(cube, wl).any()


class TestAlignToGrid:
    def test_aligns_and_masks(self, cube_tif, grid300, tmp_path):
        path, _ = cube_tif
        out = tmp_path / "aligned.tif"
        result = align_to_grid(path, grid300, out)
        assert result.exists()

        with rasterio.open(out) as src:
            assert src.count == 6
            assert src.crs.to_string() == grid300.crs
            assert "2185.20nm" in (src.descriptions[5] or "")
            band = src.read(1)
        assert np.isfinite(band).any()      # some coverage inside AOI
        assert np.isnan(band).any()         # outside-coverage = NaN
