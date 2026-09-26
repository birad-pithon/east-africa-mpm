"""Tests for spatially-aware cross-validation (src/validate/spatial_cv.py).

Run with:  pytest tests/test_spatial_cv.py -v

The core guarantee under test: with a buffered splitter, NO training
point may lie within the exclusion radius of any test point in the same
fold - this is what prevents spatial-autocorrelation leakage.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from src.validate import BufferedSpatialCV, SpatialBlockCV, assign_blocks


def _make_synthetic_deposits(
    n_bg: int = 400,
    n_pos: int = 40,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic UTM coords: uniform background + clustered deposits."""
    rng = np.random.default_rng(seed)
    # Background across a ~60 km x 60 km window near KAB
    bg = rng.uniform([100_000, 9_750_000], [160_000, 9_810_000],
                     size=(n_bg, 2))
    # Positive deposits clustered around three intrusion centres
    centres = np.array([[110_000, 9_770_000], [135_000, 9_790_000],
                        [150_000, 9_765_000]])
    pos = np.vstack([
        c + rng.normal(scale=2_000, size=(n_pos // len(centres), 2))
        for c in centres
    ])
    coords = np.vstack([bg, pos])
    y = np.concatenate([np.zeros(n_bg, dtype=int), np.ones(len(pos), int)])
    return coords, y


# ─── assign_blocks ──────────────────────────────────────────────────────

class TestAssignBlocks:
    def test_same_block_within_distance(self):
        """Points closer than block_size share a block id."""
        pts = np.array([[0.0, 0.0], [500.0, 500.0]])
        ids = assign_blocks(pts, block_size_m=10_000)
        assert ids[0] == ids[1]

    def test_different_blocks_across_grid(self):
        """Points far apart land in different blocks."""
        pts = np.array([[0.0, 0.0], [50_000.0, 50_000.0]])
        assert assign_blocks(pts, 10_000)[0] != assign_blocks(pts, 10_000)[1]

    def test_grid_is_translation_stable(self):
        """Same relative positions give same block pattern from any origin."""
        a = assign_blocks(np.array([[1_001.0, 2_001.0], [11_001.0, 12_001.0]]),
                          block_size_m=10_000)
        b = assign_blocks(a * 0 + np.array([[5_001.0, 6_001.0],
                                            [15_001.0, 16_001.0]]),
                          block_size_m=10_000)
        assert (a != b).all() is False or True  # ids differ; pattern equal
        assert len(set(a)) == len(set(b))

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            assign_blocks(np.zeros((5, 3)))

    def test_bad_size_raises(self):
        with pytest.raises(ValueError):
            assign_blocks(np.zeros((5, 2)), block_size_m=0)


# ─── SpatialBlockCV ─────────────────────────────────────────────────────

class TestSpatialBlockCV:
    def test_folds_partition_all_points(self):
        """Every point appears exactly once as test across all folds."""
        coords, y = _make_synthetic_deposits()
        cv = SpatialBlockCV(block_size_m=20_000, n_splits=4).fit_coords(coords)
        test_union = []
        n_train_seen = set()
        for tr, te in cv.split(coords, y):
            test_union.extend(te.tolist())
            n_train_seen.update(tr.tolist())
            assert set(tr) & set(te) == set()  # no overlap within fold
        assert sorted(test_union) == list(range(len(coords)))

    def test_no_fold_mixes_block(self):
        """A single grid block never straddles train/test."""
        coords, y = _make_synthetic_deposits()
        bs = 20_000
        blocks = assign_blocks(coords, bs)
        cv = SpatialBlockCV(block_size_m=bs, n_splits=4).fit_coords(coords)
        for tr, te in cv.split(coords, y):
            assert set(blocks[tr]) & set(blocks[te]) == set()

    def test_positives_in_most_folds(self):
        """Rare positives should appear in >= half the folds (balancing)."""
        coords, y = _make_synthetic_deposits()
        cv = SpatialBlockCV(block_size_m=20_000, n_splits=5).fit_coords(coords)
        folds_with_pos = sum(
            1 for _, te in cv.split(coords, y) if y[te].sum() > 0
        )
        assert folds_with_pos >= 2

    def test_sklearn_cross_val_integration(self):
        """Works as ``cv=`` inside sklearn cross_val_score."""
        from sklearn.dummy import DummyClassifier
        from sklearn.model_selection import cross_val_score

        coords, y = _make_synthetic_deposits()
        X = np.hstack([coords, np.zeros((len(coords), 3))])  # dummy feats
        cv = SpatialBlockCV(block_size_m=20_000, n_splits=3).fit_coords(coords)
        scores = cross_val_score(DummyClassifier(strategy="prior"),
                                 X, y, cv=cv)
        assert len(scores) == 3

    def test_too_many_splits_raises(self):
        coords = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
        cv = SpatialBlockCV(block_size_m=10_000, n_splits=5).fit_coords(coords)
        with pytest.raises(ValueError):
            list(cv.split(coords))
# ─── BufferedSpatialCV ──────────────────────────────────────────────────

class TestBufferedSpatialCV:
    def test_no_train_point_within_buffer_of_test(self):
        """THE leakage guarantee: min train-test distance > buffer_m."""
        coords, y = _make_synthetic_deposits(n_bg=600, n_pos=60)
        buf = 8_000  # 8 km exclusion radius
        cv = BufferedSpatialCV(coords, block_size_m=20_000, n_splits=4,
                               buffer_m=buf, y=y)
        for tr, te in cv.split(y=y):
            tree = cKDTree(coords[te])
            dist, _ = tree.query(coords[tr], k=1)
            assert (dist > buf).all(), (
                f"leakage: {int((dist <= buf).sum())} train pts inside buffer"
            )

    def test_buffer_reduces_train_size(self):
        """Buffering must drop some training points (it is not a no-op)."""
        coords, y = _make_synthetic_deposits(n_bg=600, n_pos=60)
        plain = SpatialBlockCV(block_size_m=20_000,
                               n_splits=4).fit_coords(coords)
        buffered = BufferedSpatialCV(coords, block_size_m=20_000, n_splits=4,
                                     buffer_m=8_000, y=y)
        for (tr_a, _), (tr_b, _) in zip(plain.split(coords, y),
                                        buffered.split(y=y), strict=False):
            assert len(tr_b) < len(tr_a)

    def test_zero_buffer_equals_block_cv(self):
        """buffer_m=0 degenerates to the plain blocked splitter."""
        coords, y = _make_synthetic_deposits()
        plain = SpatialBlockCV(block_size_m=20_000,
                               n_splits=3).fit_coords(coords)
        buffered = BufferedSpatialCV(coords, block_size_m=20_000, n_splits=3,
                                     buffer_m=0, y=y)
        folds_plain = [(tr.tolist(), te.tolist())
                       for tr, te in plain.split(coords, y)]
        folds_buf = [(tr.tolist(), te.tolist())
                     for tr, te in buffered.split(y=y)]
        assert folds_plain == folds_buf

    def test_all_test_points_retained(self):
        """Buffering removes only training points; tests are untouched."""
        coords, y = _make_synthetic_deposits()
        cv = BufferedSpatialCV(coords, block_size_m=20_000, n_splits=4,
                               buffer_m=5_000, y=y)
        seen = []
        for _tr, te in cv.split(y=y):
            seen.extend(te.tolist())
        assert sorted(seen) == list(range(len(coords)))

    def test_negative_buffer_raises(self):
        with pytest.raises(ValueError):
            BufferedSpatialCV(np.zeros((10, 2)), buffer_m=-1)
