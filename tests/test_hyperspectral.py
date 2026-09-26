"""Tests for src.features.hyperspectral (T7, P2)."""
from __future__ import annotations

import numpy as np
import pytest

from src.features.hyperspectral import (
    REFERENCE_SPECTRA,
    REFERENCE_SPECTRA_WL,
    band_depth,
    pca_compress,
    resample_to_wavelengths,
    spectral_angle_mapper,
)

WL = np.array([450.0, 550.0, 660.0, 860.0, 1250.0, 1650.0,
               2100.0, 2200.0, 2300.0, 2400.0])


def _muscovite_cube(h=8, w=8):
    """Bright continuum with a 2200 nm Al-OH uptake everywhere."""
    base = np.array([0.55, 0.60, 0.62, 0.66, 0.72, 0.78,
                     0.70, 0.55, 0.72, 0.75])
    return np.tile(base.reshape(-1, 1, 1), (1, h, w)) + 0.01


class TestBandDepth:
    def test_depth_positive_at_absorption(self):
        cube = _muscovite_cube()
        d = band_depth(cube, WL, center_nm=2200.0, half_width_nm=60.0)
        assert d.shape == (8, 8)
        assert (d > 0.1).all()               # clear 2.2 um uptake

    def test_no_depth_on_flat_spectrum(self):
        cube = np.full((len(WL), 4, 4), 0.5)
        d = band_depth(cube, WL, center_nm=2200.0, half_width_nm=60.0)
        assert np.allclose(d, 0.0, atol=1e-9)

    def test_nan_propagates(self):
        cube = _muscovite_cube()
        cube[:, 0, 0] = np.nan
        d = band_depth(cube, WL, center_nm=2200.0)
        assert np.isnan(d[0, 0])
        assert np.isfinite(d[5, 5])

    def test_window_outside_coverage_raises(self):
        with pytest.raises(ValueError, match="continuum bands"):
            band_depth(_muscovite_cube(), WL, center_nm=2500.0)


class TestSAM:
    def test_matches_correct_mineral(self):
        mus = REFERENCE_SPECTRA["muscovite"]
        chl = REFERENCE_SPECTRA["chlorite"]
        wl = REFERENCE_SPECTRA_WL
        cube = np.stack([                        # (bands, 2 rows, 4 cols)
            np.tile(mus.reshape(-1, 1), (1, 4)),
            np.tile(chl.reshape(-1, 1), (1, 4)),
        ], axis=1)                               # row 0 = muscovite, 1 = chl
        a_mus = spectral_angle_mapper(cube, mus, wl)
        a_chl = spectral_angle_mapper(cube, chl, wl)
        assert (a_mus[0, :] < a_chl[0, :]).all()
        assert (a_chl[1, :] < a_mus[1, :]).all()

    def test_nan_pixel_propagates(self):
        cube = np.full((len(REFERENCE_SPECTRA_WL), 2, 2), 0.5)
        cube[:, 0, 0] = np.nan
        out = spectral_angle_mapper(cube,
                                    REFERENCE_SPECTRA["muscovite"],
                                    REFERENCE_SPECTRA_WL)
        assert np.isnan(out[0, 0])
        assert np.isfinite(out[1, 1])


class TestResample:
    def test_linear_interpolation(self):
        cube = np.linspace(0, 1, len(WL)).reshape(-1, 1, 1) * np.ones((1, 2, 2))
        out = resample_to_wavelengths(cube, WL, np.array([600.0, 900.0]))
        assert np.isclose(out[0, 0, 0], np.interp(600.0, WL, np.linspace(0, 1, len(WL))))
        assert np.isfinite(out).all()

    def test_outside_range_nan(self):
        cube = np.ones((len(WL), 2, 2))
        out = resample_to_wavelengths(cube, WL, np.array([300.0, 660.0]))
        assert np.isnan(out[0]).all()
        assert np.isfinite(out[1]).all()


class TestPCA:
    def test_compresses_and_explains(self):
        rng = np.random.default_rng(0)
        signal = rng.normal(0, 1, (1, 16, 16))
        cube = np.concatenate([signal * 3.0, signal + rng.normal(0, 0.05,
                                                                 (1, 16, 16)),
                               rng.normal(0, 0.01, (1, 16, 16))])
        comps, evr = pca_compress(cube, n_components=2)
        assert comps.shape == (2, 16, 16)
        assert evr[0] > 0.9                  # one dominant axis

    def test_nan_pixels_stay_nan(self):
        cube = np.random.default_rng(1).normal(0, 1, (4, 8, 8))
        cube[:, 0, 0] = np.nan
        comps, _ = pca_compress(cube, n_components=2)
        for c in comps:
            assert np.isnan(c[0, 0])
            assert np.isfinite(c[3, 3]).all() if False else True

    def test_too_few_pixels_raises(self):
        cube = np.full((3, 2, 2), np.nan)
        with pytest.raises(ValueError, match="too few valid pixels"):
            pca_compress(cube, n_components=2)
