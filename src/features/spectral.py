"""Spectral-index feature engineering from Sentinel-2 MSI bands.

All band arithmetic is pure-Numpy (no Earth-Observation library needed)
so it runs inside the 30-m common grid with whatever Sentinel-2 raster
was produced by ``src.ingest.sentinel2_ee``.

Indices implemented (per configs/karagwe.yml) features/bauxite:
* NDWI  - Normalized Difference Water Index   (B3, B11)
* MBI   - Modified Bareness Index             (B11, B8)
* BSI   - Bare Soil Index                     (B2,B4,B8,B11)
* EVI   - Enhanced Vegetation Index           (B2,B4,B8)
* SAVI  - Soil Adjusted Vegetation Index      (B4,B8)
* NIRv  - Near-Infrared Reflectance of Veg    (B8,B4)

Hydrothermal alteration / weathering indices (per commodity spectral
fingerprints; Sentinel-2 gives COARSE proxies only - the 2.20 um Al-OH
and 2.33 um Mg-OH/CO3 absorptions cannot be separated by the broad B12
band, which requires hyperspectral data):
* aloh_ratio           - B11/B12  Al-OH depth proxy (phyllic, sericite,
                         clay, gibbsite/boehmite); >1 indicates a 2.2 um
                         absorption. Requires B12.
* mgoh_carbonate_proxy - B12/B11  relative Mg-OH / carbonate / chlorite
                         emphasis (propylitic, dolomitization). Requires
                         B12. Discrimination from Al-OH needs EnMAP-class
                         data - treat as a regional screen only.
* ferrous_iron_ratio   - B11/B8   SWIR/NIR ferrous iron index.
* iron_oxide_ratio     - B4/B2    ferric iron / gossan (Red/Blue).

Band names are matched case-insensitively. Sentinel-2 SR integer
(0-10 000) is auto-scaled to reflectance. No-data (NaN, inf) propagated.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import rasterio

S2_BANDS: dict[str, str] = {
    "B2": "Blue", "B3": "Green", "B4": "Red",
    "B8": "NIR",  "B11": "SWIR1", "B12": "SWIR2",
}

S2_SCALE = 10000.0  # Sentinel-2 L2A SR stored as UInt16 x 10000


def _resolve_band_names(descriptions, n_actual: int) -> list[str]:
    """Map raster band descriptions to canonical S2 names (B2, B3, ...)."""
    default = [f"B{i}" for i in range(1, n_actual + 1)]
    if descriptions is None:
        return default
    resolved = []
    for d in descriptions:
        match = (d or "").strip().upper().replace(" ", "")
        resolved.append(match if match in S2_BANDS else (d or ""))
    while len(resolved) < n_actual:
        resolved.append(f"B{len(resolved)+1}")
    return resolved[:n_actual]


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise (a-b)/(a+b) with zero-division suppression."""
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    den = a64 + b64
    out = np.full_like(den, np.nan)
    valid = den > 1e-10
    out[valid] = (a64[valid] - b64[valid]) / den[valid]
    return out


def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise a/b with zero-division -> NaN (never a fake 0)."""
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    out = np.full_like(a64, np.nan)
    valid = (b64 > 1e-10) & np.isfinite(a64) & np.isfinite(b64)
    out[valid] = a64[valid] / b64[valid]
    return out


def spectral_indices(
    s2_raster,
    band_names: Sequence[str] | None = None,
    reflectance_max: float = 1.0,
) -> dict[str, np.ndarray]:
    """Compute spectral indices from Sentinel-2 raster.

    Parameters
    ----------
    s2_raster : str path OR (bands,h,w) ndarray
    band_names : required if ndarray; e.g. ['B2','B3','B4','B8','B11','B12']
    reflectance_max : 1.0 for float reflectance; 10000 for integer SR

    Returns dict: index name -> 2-D float32 array
    """
    if isinstance(s2_raster, np.ndarray):
        arr = s2_raster
        if band_names is None:
            raise ValueError("band_names required when passing ndarray")
    else:
        with rasterio.open(s2_raster) as src:
            arr = src.read()
            band_names = _resolve_band_names(src.descriptions, src.count)

    if arr.ndim != 3:
        raise ValueError(f"expected 3-D array, got {arr.shape}")

    idx: dict[str, int] = {}
    for i, n in enumerate(band_names):
        k = n.upper().replace(" ", "") if n else ""
        if k in S2_BANDS:
            idx[k] = i

    required = {"B2", "B3", "B4", "B8", "B11"}
    missing = required - set(idx.keys())
    if missing:
        raise ValueError(f"Missing S2 bands: {missing}. Have: {band_names}")

    def _band(name: str) -> np.ndarray:
        i = idx[name]
        b = arr[i].astype(np.float64)
        if reflectance_max == 1.0 and b.max() > 1.5:
            b = b / S2_SCALE
        return b

    b2, b3, b4, b8, b11 = (_band(n) for n in ("B2", "B3", "B4", "B8", "B11"))
    b12 = _band("B12") if "B12" in idx else None

    ndwi = _safe_div(b3, b11)
    mbi = _safe_div(b11, b8)
    bsi = _safe_div((b11 + b2) - (b8 + b4), (b11 + b2) + (b8 + b4))
    evi = 2.5 * (b8 - b4) / (b8 + 6 * b4 - 7.5 * b2 + 1)
    savi = ((b8 - b4) * 1.5) / (b8 + 0.5 * b4 + 1)
    nirv = b8 * _safe_div(b8, b4)

    out = {
        "ndwi": np.nan_to_num(ndwi.astype(np.float32)),
        "mbi": np.nan_to_num(mbi.astype(np.float32)),
        "bsi": np.nan_to_num(bsi.astype(np.float32)),
        "evi": np.nan_to_num(evi.astype(np.float32)),
        "savi": np.nan_to_num(savi.astype(np.float32)),
        "nirv": np.nan_to_num(nirv.astype(np.float32)),
    }

    # ── alteration / weathering ratios ────────────────────────────────
    # Ratios have no neutral zero (0 would falsely read as "max 2.2 um
    # absorption"), so NaN is propagated and training rows dropna.
    out["ferrous_iron_ratio"] = _safe_ratio(b11, b8).astype(np.float32)
    out["iron_oxide_ratio"] = _safe_ratio(b4, b2).astype(np.float32)
    if b12 is not None:
        out["aloh_ratio"] = _safe_ratio(b11, b12).astype(np.float32)
        out["mgoh_carbonate_proxy"] = _safe_ratio(b12, b11).astype(np.float32)

    # propagate nodata
    nodata = (b2 == 0) & (b4 == 0) & (b8 == 0)
    for _, v in out.items():
        v[nodata] = np.nan
    return out


def compute_spectral_features(s2_raster, band_names=None,
                              reflectance_max=1.0) -> dict[str, np.ndarray]:
    """Alias for spectral_indices."""
    return spectral_indices(s2_raster, band_names, reflectance_max)
