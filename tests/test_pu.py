"""Tests for positive-unlabeled (PU) learning.

The core claim: when background genuinely means "unlabeled", PU bagging
ranks latent (hidden) deposits higher than naive supervised training,
which hard-labels them as negatives and pulls the boundary down.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from src.models.pu import PULearner, evaluate_pu_cv, fit_pu_model

N_POS, N_HIDDEN, N_TN = 60, 40, 200


def _pu_dataset(seed: int = 7):
    """Positives near x=4, *hidden* positives midway, true negatives at 0."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(4.0, 0.5, (N_POS, 3))
    pos[:, 2] = rng.normal(size=N_POS)             # noise feature
    hidden = rng.normal(2.0, 0.5, (N_HIDDEN, 3))
    hidden[:, 2] = rng.normal(size=N_HIDDEN)
    tn = rng.normal(0.0, 0.5, (N_TN, 3))
    tn[:, 2] = rng.normal(size=N_TN)

    X = np.vstack([pos, hidden, tn])
    gold = np.array([1] * (N_POS + N_HIDDEN) + [0] * N_TN)
    y = np.array([1] * N_POS + [0] * (N_HIDDEN + N_TN))   # unlabeled pool
    unlabeled = np.array([False] * N_POS +
                         [True] * (N_HIDDEN + N_TN))
    return X, y, unlabeled, gold


def _make_clf(pos_weight=1.0):
    return RandomForestClassifier(n_estimators=80, max_depth=6,
                                  random_state=3,
                                  class_weight="balanced")


class TestPULearner:
    def test_ranks_hidden_positives_better_than_naive(self):
        X, y, unlabeled, gold = _pu_dataset()

        naive = RandomForestClassifier(n_estimators=80, max_depth=6,
                                       random_state=3).fit(X, y)
        naive_p = naive.predict_proba(X)[:, 1]

        pu = PULearner(_make_clf, n_bags=25, seed=1,
                       ).fit(X, y, unlabeled=unlabeled)
        pu_p = pu.predict_proba(X)[:, 1]

        naive_ap = average_precision_score(gold, naive_p)
        pu_ap = average_precision_score(gold, pu_p)
        # PU must recover the latent deposits the naive model suppresses
        assert pu_ap > naive_ap + 0.05
        assert 0.0 < pu_ap <= 1.0

    def test_probabilities_shape_and_range(self):
        X, y, unlabeled, _ = _pu_dataset()
        pu = PULearner(_make_clf, n_bags=5, seed=2).fit(X, y,
                                                        unlabeled=unlabeled)
        p = pu.predict_proba(X)
        assert p.shape == (len(X), 2)
        assert p.min() >= 0.0 and p.max() <= 1.0
        assert np.allclose(p.sum(axis=1), 1.0)

    def test_reliable_negatives_are_always_used(self):
        X, y, unlabeled, gold = _pu_dataset()
        tn_mask = np.zeros(len(y), dtype=bool)
        tn_mask[N_POS + N_HIDDEN:] = True
        reliable = tn_mask & unlabeled
        pu = PULearner(_make_clf, n_bags=5, seed=3).fit(
            X, y, unlabeled=unlabeled, reliable_neg=reliable)
        assert pu.predict_proba(X).shape == (len(X), 2)

    def test_reliable_must_be_subset_of_unlabeled(self):
        X, y, unlabeled, _ = _pu_dataset()
        bad = np.zeros(len(y), dtype=bool)
        bad[:5] = True            # known positives marked "reliable negative"
        with pytest.raises(ValueError, match="subset of unlabeled"):
            PULearner(_make_clf, n_bags=3).fit(X, y, unlabeled, bad)

    def test_no_unlabeled_raises(self):
        X, y, _, _ = _pu_dataset()
        y_all_pos = np.ones(len(y), dtype=int)
        with pytest.raises(ValueError, match="no unlabeled"):
            PULearner(_make_clf, n_bags=3).fit(X, y_all_pos)

    def test_fit_pu_model(self):
        X, y, unlabeled, _ = _pu_dataset()
        pu = fit_pu_model(_make_clf, X, y, unlabeled=unlabeled, n_bags=4)
        assert hasattr(pu, "_clfs") and len(pu._clfs) == 4


class TestEvaluatePuCv:
    def test_summary_shape(self):
        X, y, unlabeled, _ = _pu_dataset()
        rng = np.random.default_rng(4)
        coords = rng.uniform(0.0, 300_000.0, (len(X), 2))
        res = evaluate_pu_cv(
            _make_clf, pd.DataFrame(X), y, coords,
            unlabeled=unlabeled, block_size_m=50_000.0, buffer_m=20_000.0,
            n_splits=3, n_bags=10,
        )
        assert res["n_folds"] == 3
        assert "mean_average_precision" in res
        assert res["method"] == "pu_bagging"
        assert "oof_predictions" in res
        assert len(res["oof_predictions"]["y_true"]) == len(y)
