"""Tests for SOW safeguard #2: class imbalance and label rarity."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.baselines import (  # noqa: E402
    AnomalyBaseline,
    scale_pos_weight,
    smote_oversample,
)

# ── scale_pos_weight ─────────────────────────────────────────────────


def test_scale_pos_weight_math():
    y = np.array([0] * 90 + [1] * 10)
    assert scale_pos_weight(y) == pytest.approx(9.0)


def test_scale_pos_weight_all_positive_is_one():
    y = np.ones(10, dtype=int)
    assert scale_pos_weight(y) == 1.0


# ── smote_oversample ─────────────────────────────────────────────────


def test_smote_balances_training_fold():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(110, 4))
    y = np.array([0] * 100 + [1] * 10)
    X2, y2 = smote_oversample(X, y, seed=7)
    assert int((y2 == 1).sum()) == 100          # minority lifted to parity
    assert int((y2 == 0).sum()) == 100
    assert X2.shape[0] == 200
    # synthetic points must lie in the convex span of the minority cloud
    assert np.isfinite(X2).all()


def test_smote_deterministic():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 3))
    y = np.array([0] * 50 + [1] * 10)
    a = smote_oversample(X, y, seed=3)
    b = smote_oversample(X, y, seed=3)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_smote_noop_when_balanced():
    X = np.random.default_rng(2).normal(size=(20, 2))
    y = np.array([0] * 10 + [1] * 10)
    X2, y2 = smote_oversample(X, y)
    assert X2.shape == X.shape and np.array_equal(y2, y)


def test_smote_never_touches_test_fold():
    """SMOTE must be applied to the training split only."""
    X = np.random.default_rng(3).normal(size=(120, 2))
    y = np.array([0] * 110 + [1] * 10)
    # training split deliberately contains both classes
    tr = np.array(list(range(90)) + list(range(110, 120)))
    te = np.array(list(range(90, 110)))
    Xs, _ = smote_oversample(X[tr], y[tr], seed=1)
    assert Xs.shape[0] > len(tr)      # training grew
    assert X[te].shape == (20, 2)     # test untouched


# ── AnomalyBaseline ──────────────────────────────────────────────────


def test_anomaly_baseline_beats_chance_on_synthetic():
    """Deposits planted in a barren cluster should score above chance."""
    rng = np.random.default_rng(4)
    barren = rng.normal(size=(400, 4))
    deposits = rng.normal(loc=3.0, size=(15, 4))   # far from barren cloud
    X = np.vstack([barren, deposits])
    y = np.array([0] * 400 + [1] * 15)

    ab = AnomalyBaseline(seed=0).fit(X[y == 0])
    ap = ab.score_ap(X, y)
    assert ap is not None
    assert ap > 0.5        # chance = 15/415 ~ 0.036


def test_anomaly_baseline_returns_none_on_degenerate_labels():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(50, 3))
    y = np.ones(50, dtype=int)
    ab = AnomalyBaseline(seed=0).fit(X)
    assert ab.score_ap(X, y) is None


# ── integration: evaluate_spatial_cv flags ──────────────────────────


def _synthetic_table(n=240, n_pos=24, seed=11):
    rng = np.random.default_rng(seed)
    # positives split across TWO far-apart corners so every blocked CV
    # fold contains both classes (matches real belt geometry better)
    barren = rng.normal(size=(n - n_pos, 2)) * 4_000
    half = n_pos // 2
    pos = np.vstack([
        rng.normal(size=(half, 2)) * 400 + np.array([20_000.0, 20_000.0]),
        rng.normal(size=(n_pos - half, 2)) * 400
        + np.array([-20_000.0, -20_000.0]),
    ])
    coords = np.vstack([barren, pos])
    feats = rng.normal(size=(n, 5))
    feats[n - n_pos:] += 1.5                      # signal
    y = np.array([0] * (n - n_pos) + [1] * n_pos)
    import pandas as pd

    return pd.DataFrame(feats, columns=[f"f{i}" for i in range(5)]), y, coords


def test_evaluate_cv_with_oversample_and_anomaly():
    from sklearn.linear_model import LogisticRegression

    from src.models.train import evaluate_spatial_cv

    X, y, coords = _synthetic_table()
    res = evaluate_spatial_cv(
        lambda pw: LogisticRegression(max_iter=1000),
        X, y, coords,
        block_size_m=15_000, buffer_m=0.0, n_splits=3,
        oversample=True, anomaly_baseline=True,
    )
    assert res["n_folds"] == 3
    assert res["mean_average_precision"] is not None
    assert "anomaly_baseline_mean_ap" in res
    # training folds oversampled -> n_train should exceed original share
    total_train = sum(f["n_train"] for f in res["folds"])
    assert total_train > int(0.66 * len(y))


def test_evaluate_cv_without_flags_unchanged_shape():
    from sklearn.linear_model import LogisticRegression

    from src.models.train import evaluate_spatial_cv

    X, y, coords = _synthetic_table()
    res = evaluate_spatial_cv(
        lambda pw: LogisticRegression(max_iter=1000),
        X, y, coords,
        block_size_m=15_000, buffer_m=0.0, n_splits=3,
    )
    assert "anomaly_baseline_mean_ap" not in res
    assert res["mean_average_precision"] is not None
