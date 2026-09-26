"""Tests for src.features.priors (T5, P1)."""
from __future__ import annotations

import numpy as np
import pytest

from src.features.priors import (
    apply_logit_prior,
    distance_decay_logit,
)


class TestDistanceDecayLogit:
    def test_neutral_at_half_decay(self):
        d = np.array([2000.0])           # 2 km == half-decay
        out = distance_decay_logit(d, half_decay_km=2.0, weight=1.0)
        assert np.allclose(out, 0.0)

    def test_plus_weight_at_zero_distance(self):
        out = distance_decay_logit(np.array([0.0]), 5.0, weight=1.5)
        assert np.allclose(out, 1.5)

    def test_minus_weight_far_away(self):
        out = distance_decay_logit(np.array([1_000_000.0]), 2.0, weight=2.0)
        assert np.allclose(out, -2.0, atol=1e-3)

    def test_monotonic_decay(self):
        d = np.array([0.0, 1000.0, 5000.0, 20_000.0], dtype=float)
        out = distance_decay_logit(d, half_decay_km=2.0)
        assert np.all(np.diff(out) < 0)

    def test_nan_distance_is_neutral(self):
        out = distance_decay_logit(np.array([np.nan]), 2.0, weight=3.0)
        assert np.allclose(out, 0.0)


class TestApplyLogitPrior:
    def test_output_in_unit_range(self):
        p = np.full((4, 4), 0.5)
        c = [("dep", np.full((4, 4), 2.0))]
        out, diag = apply_logit_prior(p, c)
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert out.mean() > 0.5            # positive logit lifts probability

    def test_preserves_neutral_case(self):
        p = np.linspace(0.1, 0.9, 16).reshape(4, 4)
        c = [("dep", np.zeros((4, 4)))]    # neutral prior
        out, _ = apply_logit_prior(p, c)
        assert np.allclose(out, p)

    def test_nan_propagates(self):
        p = np.array([[0.5, np.nan]])
        c = [("dep", np.zeros((1, 2)))]
        out, _ = apply_logit_prior(p, c)
        assert np.isnan(out[0, 1])
        assert np.isfinite(out[0, 0])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same grid"):
            apply_logit_prior(np.zeros((4, 4)),
                              [("dep", np.zeros((2, 2)))])

    def test_diagnostics_populated(self):
        p = np.full((2, 2), 0.3)
        c = [("geology_contact", np.full((2, 2), 1.0))]
        _, diag = apply_logit_prior(p, c)
        assert diag["priors"][0]["name"] == "geology_contact"
        assert diag["mean_abs_shift"] > 0
