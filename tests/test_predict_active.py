"""Tests for the active-learning candidate selector."""
from __future__ import annotations

import numpy as np
import pytest
import rasterio

from src.predict.active import (
    _EXPLOIT,
    rank_candidates_active,
)


def _write_proba(tmp_path, name="proba.tif"):
    """20x20 EPSG:4326 raster with a hotspot + ambiguous cells."""
    path = tmp_path / name
    arr = np.full((20, 20), 0.2, dtype="float32")
    arr[3:5, 3:5] = 0.92                      # clear hotspot
    arr[10:18, 10:16] = 0.50                  # wide ambiguous band
    arr[16, 15] = 0.48
    profile = {
        "driver": "GTiff", "height": 20, "width": 20, "count": 1,
        "dtype": "float32", "crs": "EPSG:4326",
        "transform": rasterio.transform.from_origin(29.0, -1.0, 0.01, 0.01),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
    return path


class TestActive:
    def test_exploit_and_explore_returned(self, tmp_path):
        p = _write_proba(tmp_path)
        df = rank_candidates_active(p, n_exploit=3, n_explore=6,
                                    explore_strategy="margin",
                                    spacing_cells=1)
        assert set(df.columns) >= {"rank", "row", "col", "prob",
                                   "lon", "lat", "strategy", "score"}
        assert df["strategy"].nunique() == 2
        assert (df["strategy"] == _EXPLOIT).sum() == 3
        assert (df["strategy"] == "margin").sum() == 6
        # exploit picks really are the highest-probability cells
        exploit = df.loc[df["strategy"] == _EXPLOIT, "prob"]
        assert exploit.max() >= 0.9
        # exploration picks really are close to the decision boundary
        explore = df.loc[df["strategy"] == "margin", "prob"]
        assert (np.abs(explore - 0.5) < 0.1).all()

    def test_entropy_strategy(self, tmp_path):
        p = _write_proba(tmp_path)
        df = rank_candidates_active(p, n_exploit=2, n_explore=4,
                                    explore_strategy="entropy",
                                    spacing_cells=1)
        assert (df["strategy"] == "entropy").sum() == 4

    def test_no_repeated_cells_and_min_prob(self, tmp_path):
        p = _write_proba(tmp_path)
        df = rank_candidates_active(p, n_exploit=5, n_explore=8,
                                    explore_strategy="margin",
                                    min_prob=0.0, spacing_cells=1)
        assert df["prob"].min() >= 0.0
        pairs = list(zip(df["row"], df["col"], strict=False))
        assert len(set(pairs)) == len(pairs)     # declustered/uniqued

    def test_bad_strategy_raises(self, tmp_path):
        p = _write_proba(tmp_path)
        with pytest.raises(ValueError, match="explore_strategy"):
            rank_candidates_active(p, explore_strategy="fuzzy")

    def test_empty_when_all_below_min_prob(self, tmp_path):
        p = _write_proba(tmp_path)
        df = rank_candidates_active(p, n_exploit=5, n_explore=5,
                                    min_prob=0.99)
        assert len(df) == 0


def _write_labels(tmp_path, points, name="labels.gpkg"):
    """Small WGS84 point GeoPackage used as the exclusion set."""
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        {"site": [f"p{i}" for i in range(len(points))]},
        geometry=[Point(lon, lat) for lon, lat in points],
        crs="EPSG:4326")
    path = tmp_path / name
    gdf.to_file(path, driver="GPKG")
    return path


# centre of the hotspot cells written by _write_proba (rows/cols 3-5)
_HOTSPOT_CENTRE = (29.035, -1.035)


class TestNovelGroundFilter:
    """``min_dist_to_label_km`` keeps selection off already-known ground."""

    def test_requires_a_label_file(self, tmp_path):
        p = _write_proba(tmp_path)
        with pytest.raises(ValueError, match="labels_path"):
            rank_candidates_active(p, min_dist_to_label_km=5.0)

    def test_negative_radius_rejected(self, tmp_path):
        p = _write_proba(tmp_path)
        with pytest.raises(ValueError, match="min_dist_to_label_km"):
            rank_candidates_active(p, min_dist_to_label_km=-1.0)

    def test_without_the_filter_the_hotspot_is_selected(self, tmp_path):
        """Baseline: the labelled cell is the top-probability cell."""
        p = _write_proba(tmp_path)
        df = rank_candidates_active(p, n_exploit=3, n_explore=3,
                                    spacing_cells=1)
        hot = df[df["strategy"] == _EXPLOIT]
        assert ((hot["row"] >= 3) & (hot["row"] <= 4)
                & (hot["col"] >= 3) & (hot["col"] <= 4)).any()

    def test_with_the_filter_the_hotspot_is_skipped(self, tmp_path):
        p = _write_proba(tmp_path)
        labels = _write_labels(tmp_path, [_HOTSPOT_CENTRE])
        df = rank_candidates_active(p, n_exploit=3, n_explore=3,
                                    spacing_cells=1,
                                    labels_path=labels,
                                    min_dist_to_label_km=5.0)
        assert len(df) == 6                      # same picks, different cells
        hot = df[df["strategy"] == _EXPLOIT]
        on_hotspot = ((hot["row"] >= 3) & (hot["row"] <= 4)
                      & (hot["col"] >= 3) & (hot["col"] <= 4))
        assert not on_hotspot.any()

    def test_no_pick_lands_inside_the_exclusion_ring(self, tmp_path):
        p = _write_proba(tmp_path)
        labels = _write_labels(tmp_path, [_HOTSPOT_CENTRE])
        df = rank_candidates_active(p, n_exploit=3, n_explore=3,
                                    spacing_cells=1,
                                    labels_path=labels,
                                    min_dist_to_label_km=5.0)
        assert not df.empty
        lon, lat = _HOTSPOT_CENTRE
        import geopandas as gpd
        from shapely.geometry import Point

        picked = gpd.GeoSeries(gpd.points_from_xy(df["lon"], df["lat"]),
                               crs="EPSG:4326")
        ring = (gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
                .to_crs(32736).buffer(5000).to_crs("EPSG:4326").iloc[0])
        assert not picked.intersects(ring).any()

    def test_radius_that_excludes_nothing_changes_nothing(self, tmp_path):
        """A ring narrower than one cell cannot remove any candidate."""
        p = _write_proba(tmp_path)
        labels = _write_labels(tmp_path, [_HOTSPOT_CENTRE])
        df = rank_candidates_active(p, n_exploit=3, n_explore=3,
                                    spacing_cells=1,
                                    labels_path=labels,
                                    min_dist_to_label_km=0.01)
        hot = df[df["strategy"] == _EXPLOIT]
        assert ((hot["row"] >= 3) & (hot["row"] <= 4)
                & (hot["col"] >= 3) & (hot["col"] <= 4)).any()
