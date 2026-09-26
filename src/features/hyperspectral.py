"""Hyperspectral target analysis (EnMAP-class cubes).

P2 T7: turns the aligned cubes produced by :mod:`src.ingest.enmap` into
mineral-discriminating feature layers:

* :func:`band_depth`        - absorption depth at a centre wavelength
  against the local continuum (1 - r_center / continuum).  The primary
  discriminators: 2200 nm (Al-OH: muscovite / kaolinite / gibbsite),
  2330 nm (Mg-OH / CO3: chlorite / epidote / carbonate), 900 nm
  (Fe-Oxide: hematite / goethite).
* :func:`spectral_angle_mapper` - per-pixel spectral angle (radians)
  against reference spectra; the smaller the angle, the closer the match.
* :func:`pca_compress`      - SVD compression to a few orthogonal
  components for downstream models.
* REFERENCE_SPECTRA         - coarse mineral reference spectra (see note).

NOTE on references: the bundled spectra are coarse (8-point) shapes that
reproduce the diagnostic absorption behaviour of each mineral; they are
placeholders for USGS splib06a library spectra convolved to EnMAP
resolution.  Replace ``REFERENCE_SPECTRA`` entries with library data for
production interpretation.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "band_depth",
    "spectral_angle_mapper",
    "pca_compress",
    "resample_to_wavelengths",
    "REFERENCE_SPECTRA",
]

# Diagnostic centre wavelengths (nm).
AL_OH_NM = 2200.0     # phyllic / sericite / kaolinite / gibbsite
MG_OH_CO3_NM = 2330.0  # propylitic / chlorite / epidote / carbonate
FE_OXIDE_NM = 900.0    # gossans / laterite

_REF_WL = np.array([450.0, 550.0, 660.0, 860.0, 1250.0, 1650.0,
                    2200.0, 2330.0])

# Coarse reference reflectance shapes (see module NOTE).
REFERENCE_SPECTRA: dict[str, np.ndarray] = {
    # 2.16-2.20 um doublet; bright clay-rich
    "kaolinite": np.array([0.62, 0.68, 0.70, 0.74, 0.80, 0.86, 0.62, 0.80]),
    # single sharp 2.20 um Al-OH uptake; no 2.33 feature
    "muscovite": np.array([0.55, 0.60, 0.62, 0.66, 0.72, 0.78, 0.55, 0.72]),
    # Mg-OH/Fe-OH uptake shifted toward 2.25-2.35 um
    "chlorite": np.array([0.45, 0.50, 0.48, 0.52, 0.58, 0.64, 0.60, 0.48]),
    # Al-OH at 2.20 plus strong 1.47 um OH - here flattened at 1250
    "gibbsite": np.array([0.58, 0.64, 0.66, 0.70, 0.62, 0.76, 0.58, 0.74]),
    # steep visible drop (charge transfer), Fe features near 0.55/0.88
    "hematite": np.array([0.30, 0.52, 0.42, 0.36, 0.44, 0.52, 0.60, 0.62]),
}


def _continuum_edges(wl: np.ndarray, center: float, half_width: float
                     ) -> tuple[int, int]:
    """Indices of the bands bracketing the absorption window."""
    left = np.where(wl < center - half_width)[0]
    right = np.where(wl > center + half_width)[0]
    if not len(left) or not len(right):
        raise ValueError(
            f"no continuum bands bracket {center} +/- {half_width} nm; "
            f"cube covers {np.nanmin(wl):.0f}-{np.nanmax(wl):.0f} nm")
    return int(left[-1]), int(right[0])


def band_depth(
    cube: np.ndarray,
    wavelengths_nm: np.ndarray,
    center_nm: float,
    half_width_nm: float = 25.0,
) -> np.ndarray:
    """Absorption depth map (1 - r_center / continuum), NaN-safe.

    The continuum is the straight line between the reflectance just
    below ``center - half_width`` and just above
    ``center + half_width``, evaluated at ``center``.
    """
    wl = np.asarray(wavelengths_nm, dtype="float64")
    i_l, i_r = _continuum_edges(wl, center_nm, half_width_nm)
    i_c = int(np.argmin(np.abs(wl - center_nm)))

    wl_l, wl_r = wl[i_l], wl[i_r]
    denom = wl_r - wl_l
    frac = (center_nm - wl_l) / denom if denom > 0 else 0.0

    cont = cube[i_l] * (1.0 - frac) + cube[i_r] * frac
    with np.errstate(invalid="ignore", divide="ignore"):
        depth = 1.0 - cube[i_c] / cont
    return np.where(np.isfinite(depth) & (cont > 1e-9), depth, np.nan)


def resample_to_wavelengths(
    cube: np.ndarray,
    cube_wl_nm: np.ndarray,
    target_wl_nm: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate cube bands onto new centre wavelengths."""
    wl = np.asarray(cube_wl_nm, dtype="float64")
    order = np.argsort(wl)
    wl_s = wl[order]
    data = cube[order].reshape(len(wl_s), -1)          # (bands, pixels)

    targets = np.asarray(target_wl_nm, dtype="float64")
    out = np.full((len(targets), data.shape[1]), np.nan)
    for i, t in enumerate(targets):
        if t < wl_s[0] or t > wl_s[-1]:
            continue
        j = int(np.searchsorted(wl_s, t))
        if wl_s[j] == t:
            out[i] = data[j]
            continue
        w = (t - wl_s[j - 1]) / (wl_s[j] - wl_s[j - 1])
        out[i] = data[j - 1] * (1.0 - w) + data[j] * w
    return out.reshape((len(targets),) + cube.shape[1:])


def spectral_angle_mapper(
    cube: np.ndarray,
    reference: np.ndarray,
    cube_wl_nm: np.ndarray,
) -> np.ndarray:
    """Per-pixel spectral angle (radians) against a reference spectrum.

    ``reference`` is given at arbitrary wavelengths; it is resampled to
    the cube grid first.  Smaller angle = closer match.  NaN pixels
    propagate.
    """
    ref_wl = REFERENCE_SPECTRA_WL
    ref = np.interp(np.asarray(cube_wl_nm, dtype="float64"),
                    ref_wl, np.asarray(reference, dtype="float64"))

    n_bands = cube.shape[0]
    flat = cube.reshape(n_bands, -1).T                    # (pixels, bands)
    valid = np.isfinite(flat).all(axis=1)

    r = flat[valid]                                       # (v, bands)
    dot = r @ ref
    norm = np.linalg.norm(r, axis=1) * np.linalg.norm(ref)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.clip(dot / norm, -1.0, 1.0)
        angle = np.arccos(cos)

    out = np.full(flat.shape[0], np.nan)
    out[valid] = angle
    return out.reshape(cube.shape[1:])


REFERENCE_SPECTRA_WL = _REF_WL


def pca_compress(
    cube: np.ndarray,
    n_components: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean-centred SVD compression -> (components (k, h, w), evr).

    Only finite pixels enter the fit; NaN pixels stay NaN in every
    component.  ``evr`` is the explained-variance-ratio per component.
    """
    n_bands = cube.shape[0]
    flat = cube.reshape(n_bands, -1).T                    # (pixels, bands)
    valid = np.isfinite(flat).all(axis=1)
    if valid.sum() < n_components + 1:
        raise ValueError("too few valid pixels for PCA")

    data = flat[valid]
    mean = data.mean(axis=0)
    centered = data - mean
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    evr = (s ** 2) / (s ** 2).sum()

    k = min(n_components, vt.shape[0])
    comps = np.full((k,) + cube.shape[1:], np.nan)
    proj = centered @ vt[:k].T                            # (v, k)
    for j in range(k):
        comps[j].reshape(-1)[np.where(valid.reshape(-1))[0]] = proj[:, j]
    return comps, evr[:k]

