"""Bayesian spatial conditioning: logit-additive geological priors.

Turns distance evidence (to known deposits, granite contacts, ...) into
log-odds contributions that are ADDED to a trained model's logit, then
squashed back to a calibrated probability:

    z_final(x) = logit(P_model(x)) + SUM_i  w_i * (2*prox_i(x) - 1)
    prox_i(x)  = exp(-ln(2) * d_i(x) / h_i)          # 0.5 at half-decay h_i

Why logit-additive rather than naive multiplication?  Multiplying
P_spectral * P_geology * ... is only valid under conditional
independence; correlated evidence (distance-to-contact and
distance-to-deposits co-vary) then gets double-counted and can push
probabilities out of [0, 1].  Adding evidence in log-odds space keeps
the fusion valid for correlated sources, never leaves [0, 1], and each
prior's influence is an interpretable weight in log-odds.

Prior specifications (half-decay distances and weights) live in
``configs/spectral_fingerprints.yaml`` under ``fingerprints.<group>.
geology.decay``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio

from src.utils import load_config, project_path

logger = logging.getLogger(__name__)

__all__ = [
    "distance_decay_logit",
    "apply_logit_prior",
    "resolve_prior_sources",
    "apply_prior_to_raster",
    "lift_report",
]

_EPS = 1e-6

# Belt suffix for the geology feature stacks, by commodity group.  The
# deposit-distance raster is resolvable from the group name alone; the
# geology stack name follows the belt that produced it.
_SUFFIX_BY_GROUP = {
    "tin_tungsten_tantalum": "kabar",
    "copper_zinc": "copperbelt",
    "bauxite": "usambara",
}

# Recognised symbolic raster names in fingerprints geology.decay entries.
# Maps to (file convention, band description or None for single-band).
_SYMBOLIC_RASTERS = {
    "deposit_distance": ("deposit_distance_{group}.tif", None),
    "geology_contact": ("geology_features_{suffix}.tif", "dist_to_contact_m"),
    "geology_plutonic": ("geology_features_{suffix}.tif",
                         "dist_to_plutonic_or_metamorphic_m"),
}


# ── Math ──────────────────────────────────────────────────────────────


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype="float64"), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype="float64"), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def distance_decay_logit(
    distance_m: np.ndarray,
    half_decay_km: float,
    weight: float = 1.0,
) -> np.ndarray:
    """Log-odds contribution of a distance evidence layer.

    Neutral (0) at ``half_decay_km``, ``+weight`` at zero distance,
    asymptotically ``-weight`` far away.  NaN distances contribute 0
    (no information) rather than poisoning the fusion.
    """
    d = np.asarray(distance_m, dtype="float64")
    prox = np.exp(-np.log(2.0) * d / (half_decay_km * 1000.0))
    contrib = weight * (2.0 * prox - 1.0)
    return np.where(np.isfinite(contrib), contrib, 0.0)


def apply_logit_prior(
    proba: np.ndarray,
    contributions: list[tuple[str, np.ndarray]],
) -> tuple[np.ndarray, dict]:
    """Add *contributions* to ``logit(proba)``; return (proba', diagnostics)."""
    z = _logit(proba)
    diag: dict = {"priors": []}
    for name, c in contributions:
        if c.shape != proba.shape:
            raise ValueError(
                f"prior '{name}' shape {c.shape} != probability shape "
                f"{proba.shape} - rasters are not on the same grid"
            )
        z = z + c
        diag["priors"].append({
            "name": name,
            "mean_contribution": float(np.nanmean(c)),
            "max_contribution": float(np.nanmax(c)),
        })
    out = _sigmoid(z)
    out = np.where(np.isfinite(proba), out, np.nan)
    diag["mean_abs_shift"] = float(np.nanmean(np.abs(out - proba)))
    return out, diag


# ── Source resolution ─────────────────────────────────────────────────


def resolve_prior_sources(
    group: str,
    fingerprints_path: str | Path = "configs/spectral_fingerprints.yaml",
    feature_dir: Path | None = None,
) -> list[tuple[str, np.ndarray, float, float]]:
    """Resolve a group's decay entries into named logit contribution layers.

    Returns ``(name, distance_array_m, half_decay_km, weight)`` tuples.
    Entries whose raster cannot be resolved are skipped with a warning -
    priors are auxiliary evidence, never a hard dependency.
    """
    fp = load_config(fingerprints_path)
    spec = (fp.get("fingerprints", {}).get(group, {})
            .get("geology", {}).get("decay", []))
    if not spec:
        logger.warning("no geology.decay entries for '%s' in %s",
                       group, fingerprints_path)
        return []

    fdir = Path(feature_dir) if feature_dir else project_path(
        "data", "interim", "features")
    suffix = _SUFFIX_BY_GROUP.get(group)

    out: list[tuple[str, np.ndarray, float, float]] = []
    for entry in spec:
        symbol = entry.get("raster", "")
        if symbol not in _SYMBOLIC_RASTERS:
            logger.warning("unknown prior raster symbol '%s' - skipped",
                           symbol)
            continue
        fname_tpl, band = _SYMBOLIC_RASTERS[symbol]
        path = fdir / fname_tpl.format(group=group, suffix=suffix)
        if not path.exists():
            logger.warning("prior raster %s not found - skipped", path)
            continue
        dist = _read_distance_band(path, band)
        if dist is None:
            logger.warning("band '%s' not in %s - prior skipped",
                           band, path)
            continue
        out.append((symbol, dist,
                    float(entry.get("half_decay_km", 5.0)),
                    float(entry.get("weight", 1.0))))
    return out


def _read_distance_band(
    path: Path,
    band_description: str | None,
) -> np.ndarray | None:
    """Read a single-band or named-band distance raster (metres)."""
    with rasterio.open(path) as src:
        if band_description is None:
            if src.count != 1:
                logger.warning("%s has %d bands - expected 1", path,
                               src.count)
                return None
            return src.read(1).astype("float64")
        for b in range(1, src.count + 1):
            if (src.descriptions[b - 1] or "") == band_description:
                return src.read(b).astype("float64")
    return None


def apply_prior_to_raster(
    proba_path: str | Path,
    group: str,
    fingerprints_path: str | Path = "configs/spectral_fingerprints.yaml",
    out_path: str | Path | None = None,
) -> tuple[Path, dict]:
    """Write the prior-conditioned probability GeoTIFF for a trained model.

    Returns ``(out_path, diagnostics)``.
    """
    proba_path = Path(proba_path)
    sources = resolve_prior_sources(group, fingerprints_path)

    with rasterio.open(proba_path) as src:
        proba = src.read(1).astype("float64")
        profile = src.profile.copy()

    contributions = [(name, distance_decay_logit(d, h, w))
                     for name, d, h, w in sources]
    adjusted, diag = apply_logit_prior(proba, contributions)
    diag["group"] = group
    diag["sources"] = [
        {"name": n, "half_decay_km": h, "weight": w}
        for n, _d, h, w in sources
    ]

    if out_path is None:
        out_path = proba_path.with_name(
            proba_path.stem.replace("proba_", "proba_prior_") + ".tif")
    out_path = Path(out_path)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(adjusted.astype("float32"), 1)
        dst.set_band_description(1, "p_deposit_prior_conditioned")
    diag["out_path"] = str(out_path)
    logger.info("prior-conditioned probabilities -> %s", out_path)
    return out_path, diag


# ── A/B evaluation ────────────────────────────────────────────────────


def lift_report(
    base_raster: str | Path,
    prior_raster: str | Path,
    group: str,
) -> dict:
    """Average precision of base vs prior-conditioned raster at labels.

    NOTE (in-sample reference): points are the same labelled deposits and
    stratified background used for training, so this is a directional
    sanity check of prior benefit - NOT a substitute for the spatial-CV
    protocol.  Quantify residual leakage with
    ``src.validate.evaluate_loso`` before field targeting.
    """
    import geopandas as gpd
    import pandas as pd
    from sklearn.metrics import average_precision_score

    processed = project_path("data", "processed")
    pos = gpd.read_file(processed / f"labels_{group}.gpkg")
    pos["y"] = 1
    bg_path = processed / f"background_{group}.gpkg"
    parts = [pos]
    if bg_path.exists():
        bg = gpd.read_file(bg_path)
        bg["y"] = 0
        parts.append(bg)
    pts = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True),
        geometry="geometry", crs=parts[0].crs)

    def _sample(raster: str | Path) -> np.ndarray:
        vals: list[float] = []
        with rasterio.open(raster) as src:
            g = pts.to_crs(src.crs)
            for p in g.geometry:
                r, c = src.index(p.x, p.y)
                if 0 <= r < src.height and 0 <= c < src.width:
                    vals.append(float(src.read(1)[r, c]))
                else:
                    vals.append(np.nan)
        return np.asarray(vals)

    base = _sample(base_raster)
    pri = _sample(prior_raster)
    y = pts["y"].to_numpy(dtype=int)
    ok = np.isfinite(base) & np.isfinite(pri)

    ap_base = float(average_precision_score(y[ok], base[ok]))
    ap_prior = float(average_precision_score(y[ok], pri[ok]))
    return {
        "group": group,
        "n_points": int(ok.sum()),
        "ap_base": ap_base,
        "ap_prior": ap_prior,
        "lift": ap_prior - ap_base,
        "note": "in-sample reference lift; validate with spatial CV",
    }
