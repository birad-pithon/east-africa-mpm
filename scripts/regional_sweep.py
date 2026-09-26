#!/usr/bin/env python3
"""Regional sweep: batch predict -> rank -> flag anomalies across belts.

P2 T8. Scans every belt in the catalog, runs full-grid inference with
each group's current best model, ranks candidate cells and flags
high-probability anomalies into a combined GeoJSON + summary JSON.

Usage:
    python scripts/regional_sweep.py --n 50 --threshold 0.9
    python scripts/regional_sweep.py --refresh        # re-run inference
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.dataset import default_feature_rasters
from src.models.predict import predict_raster
from src.predict.rank import rank_candidates
from src.utils import load_config, project_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CATALOG = {
    "tin_tungsten_tantalum": "configs/karagwe.yml",
    "copper_zinc": "configs/copperbelt.yml",
    "bauxite": "configs/usambara.yml",
}


def _best_algo(group: str) -> str:
    metrics = json.loads(
        (project_path("outputs", "models",
                      f"metrics_{group}.json")).read_text(encoding="utf-8"))
    return metrics["best_algo"]


def sweep_group(group: str, n: int, threshold: float, refresh: bool
                ) -> tuple[dict, list[dict]]:
    cfg_path = CATALOG[group]
    algo = _best_algo(group)
    models_dir = project_path("outputs", "models")
    bundle = models_dir / f"model_{group}_{algo}.joblib"
    proba_path = models_dir / f"proba_{group}_{algo}.tif"

    if refresh or not proba_path.exists():
        rasters = default_feature_rasters(
            group, config_path=project_path(cfg_path))
        proba_path = predict_raster(bundle, rasters, out_path=proba_path)
        logger.info("[%s] inference -> %s", group, proba_path)
    else:
        logger.info("[%s] reusing %s", group, proba_path)

    df = rank_candidates(proba_path, n=n, spacing_cells=2)
    anomalies = df[df["prob"] >= threshold]
    logger.info("[%s] %d candidates, %d anomalies (prob >= %.2f)",
                group, len(df), len(anomalies), threshold)

    feats = [{
        "type": "Feature",
        "geometry": {"type": "Point",
                     "coordinates": [float(r.lon), float(r.lat)]},
        "properties": {"group": group, "rank": int(r["rank"]),
                       "prob": float(r.prob), "anomaly": True},
    } for _, r in anomalies.iterrows()]

    summary = {
        "group": group,
        "config": cfg_path,
        "best_algo": algo,
        "bundle": str(bundle),
        "proba": str(proba_path),
        "n_candidates": int(len(df)),
        "n_anomalies": int(len(anomalies)),
        "top_prob": float(df["prob"].max()) if len(df) else None,
        "threshold": threshold,
    }
    return summary, feats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch regional sweep across all belts")
    parser.add_argument("--n", type=int, default=50,
                        help="candidates ranked per group")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="P(deposit) flagging threshold")
    parser.add_argument("--refresh", action="store_true",
                        help="force re-inference even if proba exists")
    args = parser.parse_args()

    maps = project_path("outputs", "maps")
    maps.mkdir(parents=True, exist_ok=True)

    all_feats: list[dict] = []
    summaries: dict[str, dict] = {}
    for group in CATALOG:
        try:
            s, f = sweep_group(group, args.n, args.threshold, args.refresh)
        except FileNotFoundError as exc:
            logger.error("[%s] skipped: %s", group, exc)
            continue
        summaries[group] = s
        all_feats.extend(f)

    fc = {"type": "FeatureCollection", "features": all_feats}
    out_gj = maps / "regional_anomalies.geojson"
    out_gj.write_text(json.dumps(fc, indent=2), encoding="utf-8")
    out_summary = maps / "regional_sweep_summary.json"
    out_summary.write_text(json.dumps(summaries, indent=2),
                           encoding="utf-8")
    print(f"\nOK sweep: {len(all_feats)} anomalies across "
          f"{len(summaries)} belts -> {out_gj}")


if __name__ == "__main__":
    main()
