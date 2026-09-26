"""Tests for the geology feature pipeline (ingest + feature rasters).

The WFS ingest itself is network-bound; these tests exercise the
deterministic feature engineering on synthetic vectors plus the saved
in-repo ingest helpers with a stubbed reader.
"""
from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box

from src.features.geology_features import build_geology_features
from src.ingest.worldgeol import ROCK_TYPE_LABELS
from src.preprocess.grid import GridSpec

# ── fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def small_grid(tmp_path):
    """8x8 grid at 100 m in a projected CRS (UTM 36S)."""
    import affine
    transform = affine.Affine(30.0, 0.0, 500000.0,
                              0.0, -30.0, 9800000.0)
    return GridSpec(
        crs="EPSG:32736", resolution_m=30.0, transform=transform,
        width=8, height=8,
        bounds=(500000.0, 9800000.0 - 8 * 30.0,
                500000.0 + 8 * 30.0, 9800000.0),
    )


@pytest.fixture
def geology_gdf(small_grid):
    """Two lithology polygons: PLX covering the left half, SED the right."""
    minx, miny, maxx, maxy = small_grid.bounds
    midx = minx + (maxx - minx) / 2
    plx = Polygon([(minx, miny), (midx, miny), (midx, maxy), (minx, maxy)])
    sed = Polygon([(midx, miny), (maxx, miny), (maxx, maxy), (midx, maxy)])
    return gpd.GeoDataFrame(
        {"rxtp": ["PLX", "SED"], "terrane": ["a", "b"]},
        geometry=[plx, sed], crs=small_grid.crs,
    )


@pytest.fixture
def contacts_gdf(small_grid):
    minx, miny, maxx, maxy = small_grid.bounds
    midx = minx + (maxx - minx) / 2
    line = LineString([(midx, miny), (midx, maxy)])
    return gpd.GeoDataFrame(geometry=[line], crs=small_grid.crs)


# ── feature engineering ────────────────────────────────────────────────

def test_bands_written_and_named(small_grid, geology_gdf, contacts_gdf, tmp_path):
    out_dir = tmp_path / "feats"
    path, class_map = build_geology_features(
        small_grid, geology_gdf, contacts_gdf, out_dir=out_dir
    )
    assert path.exists()
    assert class_map == {"PLX": 1, "SED": 2}
    cmap = json.loads((out_dir / "geology_class_map.json").read_text())
    assert cmap["PLX"]["code"] == 1
    assert cmap["PLX"]["label"] == "plutonic_or_metamorphic"

    import rasterio
    with rasterio.open(path) as src:
        names = list(src.descriptions)
        assert src.count == 6          # code + 2 one-hot + 2 dist + contact
        assert src.shape == small_grid.shape
    assert names[0] == "lithology_code"
    assert "geol_is_plutonic_or_metamorphic" in names
    assert "dist_to_sedimentary_m" in names
    assert "dist_to_contact_m" in names


def test_code_raster_values(small_grid, geology_gdf, tmp_path):
    import rasterio
    path, _ = build_geology_features(
        small_grid, geology_gdf, None, out_dir=tmp_path
    )
    with rasterio.open(path) as src:
        code = src.read(1)
    assert code.max() <= 2 and code.min() >= 0
    assert set(np.unique(code)) <= {0.0, 1.0, 2.0}


def test_onehot_and_distances(small_grid, geology_gdf, contacts_gdf, tmp_path):
    import rasterio
    path, _ = build_geology_features(
        small_grid, geology_gdf, contacts_gdf, out_dir=tmp_path
    )
    with rasterio.open(path) as src:
        names = list(src.descriptions)
        bands = {n: src.read(i + 1) for i, n in enumerate(names)}

    plx = bands["geol_is_plutonic_or_metamorphic"]
    sed = bands["geol_is_sedimentary"]
    # one-hots are mutually exclusive and both populated
    assert not np.logical_and(plx > 0, sed > 0).any()
    assert plx.max() == 1.0 and sed.max() == 1.0

    d_plx = bands["dist_to_plutonic_or_metamorphic_m"]
    d_sed = bands["dist_to_sedimentary_m"]
    # distance is 0 inside the class, strictly positive outside
    assert d_plx[plx > 0].max() == 0.0
    assert d_plx[sed > 0].min() > 0
    assert d_sed[sed > 0].max() == 0.0

    # distances in metres bounded by grid diagonal
    diag_m = small_grid.resolution_m * np.hypot(8, 8)
    assert d_plx.max() <= diag_m + 1e-3

    d_contact = bands["dist_to_contact_m"]
    assert d_contact.max() > 0          # some ground away from the contact
    assert d_contact.min() == 0.0       # contact line itself


def test_missing_field_raises(small_grid, geology_gdf, tmp_path):
    with pytest.raises(ValueError, match="not in geology columns"):
        build_geology_features(small_grid, geology_gdf, field="nope",
                               out_dir=tmp_path)


def test_band_shapes_match_grid(small_grid, geology_gdf, tmp_path):
    import rasterio
    path, _ = build_geology_features(small_grid, geology_gdf, None,
                                     out_dir=tmp_path)
    with rasterio.open(path) as src:
        for b in range(1, src.count + 1):
            assert src.read(b).shape == (8, 8)


# ── ingest module (offline: cache hit path) ────────────────────────────

def test_ingest_cache_hit(tmp_path, monkeypatch):
    """When cached gpkgs exist, ingest must not touch the network."""
    import src.ingest.worldgeol as wg

    gdf = gpd.GeoDataFrame(
        {"rxtp": ["SED"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326"
    )
    for layer in ("geology", "contacts"):
        gdf.to_file(tmp_path / f"{layer}_belt.gpkg", driver="GPKG")

    def boom(*a, **k):  # network call must not happen
        raise AssertionError("network touched on cache hit")

    monkeypatch.setattr(wg, "WFS_BASE", "WFS:http://127.0.0.1:9/none")
    out = wg.ingest_world_geology(cache_dir=tmp_path)
    assert set(out) == {"geology", "contacts"}
    assert len(out["geology"]) == 1


def test_rock_type_labels_cover_belt_codes():
    for code in ("SED", "PLX", "VOL", "SXV"):
        assert code in ROCK_TYPE_LABELS
