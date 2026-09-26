#!/usr/bin/env python3
"""Quantify what the deposit-distance feature and the Bayesian prior
really contribute, using out-of-fold probabilities on certified
spatial folds (T10, P2).

For each commodity group, three variants are evaluated on IDENTICAL
certified blocked+buffered folds:

  A  all features                       (production reference)
  B  features minus deposit-distance    (leakage-free feature set)
  C  B + Bayesian logit-additive prior  (configs/spectral_fingerprints.yaml)

Interpretation
--------------
* A - B  : how much of the reported AP flows through the static
           deposit-distance raster (residual spatial label leakage +
           legitimate clustering prior, entangled).
* C - B  : the prior's genuine out-of-fold value when it is NOT already
           learned as a feature.

Usage:
    python scripts/validate_prior.py --group copper_zinc \
        --config configs/copperbelt.yml [--algo xgb]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import rasterio

from src.models.train import evaluate_spatial_cv, make_model, \
    prepare_training_table
from src.preprocess.grid import GridSpec
from src.utils import load_config, project_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def prior_contributions_at_points(cfg, group, pts):
    """Sample fingerprints decay rasters at labelled points -> logit delta."""
    from src.features.priors import resolve_prior_sources

    grid = GridSpec.from_config(cfg)
    contrib = np.zeros(len(pts), dtype="float64")
    details = []
    for name, dist, half_km, weight in resolve_prior_sources(group):
        vals = np.full(len(pts), np.nan)
        g = pts.to_crs(grid.crs)
        tf = grid.transform
        for i, p in enumerate(g.geometry):
            c = int((p.x - tf.c) / tf.a)
            r = int((p.y - tf.f) / tf.e)
            if 0 <= r < grid.height and 0 <= c < grid.width:
                vals[i] = dist[r, c]
        prox = np.exp(-np.log(2.0) * vals / (half_km * 1000.0))
        contrib += np.where(np.isfinite(prox),
                            weight * (2.0 * prox - 1.0), 0.0)
        details.append({"name": name, "half_decay_km": half_km,
                        "weight": weight,
                        "sampled": int(np.isfinite(vals).sum())})
    return contrib, details


def run_group(group: str, config_path: str, algo: str,
              collect_fold_report: bool = True) -> dict:
    """A/B/C comparison on identical certified folds for one group."""
    from sklearn.metrics import average_precision_score

    from src.validate import BufferedSpatialCV, certify_folds

    cfg, params, _algos, X, y, pts, coords, block_m, buffer_km = \
        prepare_training_table(group, config_path=config_path)

    pos_weight = float((y == 0).sum()) / max(1, int((y == 1).sum()))
    dd_cols = [c for c in X.columns if "deposit_distance" in c]
    X_b = X.drop(columns=dd_cols)

    cv = BufferedSpatialCV(coords=coords, block_size_m=block_m,
                           n_splits=5, buffer_m=buffer_km * 1000.0, y=y)
    splits = list(cv.split(None, y))
    certify_folds(coords, splits, min_buffer_m=buffer_km * 1000.0)
    logger.info("[%s] %d certified folds, %d rows (%d positives), "
                "%d deposit-distance columns dropped for B/C",
                group, len(splits), len(y), int((y == 1).sum()),
                len(dd_cols))

    oof_a = np.full(len(y), np.nan)
    oof_b = np.full(len(y), np.nan)
    Xa = X.to_numpy(dtype="float64")          # numpy: no feature names
    Xb = X_b.to_numpy(dtype="float64")        # (lgbm rejects ':' names)
    for tr, te in splits:
        m = make_model(algo, params, pos_weight)
        m.fit(Xa[tr], y[tr])
        oof_a[te] = m.predict_proba(Xa[te])[:, 1]

        m2 = make_model(algo, params, pos_weight)
        m2.fit(Xb[tr], y[tr])
        oof_b[te] = m2.predict_proba(Xb[te])[:, 1]

    contrib, details = prior_contributions_at_points(cfg, group, pts)
    from src.features.priors import _sigmoid, _logit
    oof_c = _sigmoid(_logit(oof_b) + contrib)

    ap = lambda p: float(average_precision_score(y, p))   # noqa: E731
    return {
        "group": group,
        "algo": algo,
        "n_rows": int(len(y)),
        "n_positives": int((y == 1).sum()),
        "n_folds": len(splits),
        "ap_A_full_features": ap(oof_a),
        "ap_B_no_deposit_distance": ap(oof_b),
        "ap_C_no_dd_plus_prior": ap(oof_c),
        "leakage_share_A_minus_B": ap(oof_a) - ap(oof_b),
        "prior_value_C_minus_B": ap(oof_c) - ap(oof_b),
        "prior_sources": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OOF validation: deposit-distance leakage vs prior value")
    parser.add_argument("--group", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--algo", default="rf",
                        help="rf | xgb | lgbm (default rf)")
    parser.add_argument("--out", default=None,
                        help="optional JSON output path")
    args = parser.parse_args()

    result = run_group(args.group, args.config, args.algo)
    print("\nOK prior validation:")
    for k, v in result.items():
        print(f"  {k:28s}: {v}")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2),
                                  encoding="utf-8")
        print(f"  written -> {args.out}")


if __name__ == "__main__":
    main()

