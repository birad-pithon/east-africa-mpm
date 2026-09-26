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
