"""CLI: train per-commodity models + export probability rasters.

    python -m src.models.main --group tin_tungsten_tantalum
    python -m src.models.main --group tin_tungsten_tantalum \
        --algos rf xgb lgbm --block-m 8000 --buffer-km 3 --predict

    # the whole catalogue, each group on its own belt config
    python -m src.models.main --group all --algos rf xgb lgbm --predict

``--group all`` expands to every group in ``configs/belts.yaml`` (see
:mod:`src.models.catalog`) and trains them in one process; a group that fails
is reported at the end and does not stop the remaining groups.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.models.catalog import ALL, known_groups, resolve_groups
from src.models.dataset import default_feature_rasters
from src.models.predict import predict_raster
from src.models.train import train_group

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train commodity-group prospectivity models"
    )
    parser.add_argument("--group", required=True,
                        help=f"commodity group key (see configs/belts.yaml) "
                             f"or '{ALL}' for every group in the catalogue")
    parser.add_argument("--config", default=None,
                        help="pipeline config override; defaults to the belt "
                             "config mapped to the group "
                             f"(src.models.catalog). Not allowed with "
                             f"'--group {ALL}'")
    parser.add_argument("--algos", nargs="+", default=None,
                        help="subset of rf/xgb/lgbm")
    parser.add_argument("--block-m", type=float, default=None)
    parser.add_argument("--buffer-km", type=float, default=None)
    parser.add_argument("--predict", action="store_true",
                        help="also write full-grid probability GeoTIFF")
    parser.add_argument("--prior", action="store_true",
                        help="with --predict: additionally apply the "
                             "Bayesian logit-additive geological prior "
                             "(configs/spectral_fingerprints.yaml) and "
                             "report in-sample A/P lift")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--oversample", action="store_true",
                        help="SMOTE-oversample minority class in each "
                             "training fold (never test folds)")
    parser.add_argument("--anomaly-baseline", action="store_true",
                        help="report IsolationForest-on-barren reference AP")
    parser.add_argument("--pu", action="store_true",
                        help="also evaluate PU-bagged variants (background "
                             "treated as unlabeled, barren_halo as reliable "
                             "negatives)")
    parser.add_argument("--ensemble", action="store_true",
                        help="also evaluate an OOF logistic stack of all "
                             "base algorithms")
    parser.add_argument("--tune", action="store_true",
                        help="run a TPE/Bayesian hyperparameter study "
                             "(requires optuna) instead of training")
    parser.add_argument("--tune-trials", type=int, default=30)
    parser.add_argument("--tune-timeout", type=float, default=None)
    parser.add_argument("--tiny-n", dest="tiny_n", nargs="?",
                        const=True, default=False,
                        help="shallow/regularized preset for few positives; "
                             "'--tiny-n auto' activates only below 15 "
                             "positives")
    parser.add_argument("--analogs", default=None,
                        help="path to analogs_<group>.gpkg (see "
                             "src.labels.analog) - appended as positives")
    return parser


def _run_tune(group: str, config: str, args: argparse.Namespace) -> dict:
    """Hyperparameter study for one group; returns a summary row."""
    from src.models.tuning import tune_group

    study_info = tune_group(
        group, config_path=config,
        n_trials=args.tune_trials, timeout=args.tune_timeout,
        algos=(args.algos or None),
        block_m=args.block_m, buffer_km=args.buffer_km,
    )
    print("\nOK Bayesian tuning (TPE):")
    print(f"  best algo : {study_info['best_algo']}")
    print(f"  best mean AP : {study_info['best_mean_ap']:.3f}")
    print(f"  best params : {study_info['best_params']}")
    print(f"  trials : {study_info['n_trials']}")
    return {"group": group, "best_algo": study_info["best_algo"],
            "ap": float(study_info["best_mean_ap"]), "tuned": True}


def _run_group(group: str, config: str, args: argparse.Namespace) -> dict:
    """Train one group (optionally + prediction/prior): one summary row.

    Raises on failure - ``main`` catches per-group so one group's bad labels
    cannot lose the other groups' work.
    """
    result = train_group(
        group,
        config_path=config,
        algos=args.algos,
        block_m=args.block_m,
        buffer_km=args.buffer_km,
        output_dir=args.out_dir,
        oversample=args.oversample,
        anomaly_baseline=args.anomaly_baseline,
        pu=args.pu,
        ensemble=args.ensemble,
        tiny_n=args.tiny_n,
        analogs_path=args.analogs,
    )
    metrics = result["metrics"]
    best = metrics["best_algo"]
    ap = metrics["results"][best]["mean_average_precision"]
    print(f"\nOK [{group}] best={best}  AP={ap:.3f}")
    for line in result["metrics_path"], result["bundle_path"]:
        print(f"  {line}")

    row = {"group": group, "best_algo": best, "ap": ap, "tuned": False}

    if args.predict:
        proba = predict_raster(
            result["bundle_path"],
            default_feature_rasters(group, config_path=config))
        row["proba"] = str(proba)
        print(f"  {proba}")

        if args.prior:
            from src.features.priors import (
                apply_prior_to_raster,
                lift_report,
            )

            prior_path, diag = apply_prior_to_raster(proba, group)
            print(f"  prior sources : {diag['sources']}")
            print(f"  {prior_path}")
            lift = lift_report(proba, prior_path, group)
            print(f"  lift check    : AP base={lift['ap_base']:.3f} "
                  f"prior={lift['ap_prior']:.3f} "
                  f"delta={lift['lift']:+.3f} ({lift['note']})")
            row["prior"] = str(prior_path)
    return row

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")

    if args.group != ALL and args.config is None:
        known = known_groups()
        if known and args.group not in known:
            print(f"ERROR: unknown group '{args.group}'. Known groups "
                  f"(configs/belts.yaml): {', '.join(known)} "
                  f"(or '{ALL}' for all of them)")
            return 2
    if args.group == ALL and args.config is not None:
        print(f"ERROR: '--config' cannot be combined with '--group {ALL}' - "
              "each group trains on its own belt config; override one group "
              "at a time")
        return 2
    if args.prior and not args.predict:
        print("  --prior requires --predict (conditions a probability "
              "raster); ignoring --prior")

    targets = resolve_groups(args.group, args.config)
    runner = _run_tune if args.tune else _run_group
    rows: list[dict] = []
    failures: list[tuple[str, str]] = []

    for index, (group, config) in enumerate(targets, start=1):
        if len(targets) > 1:
            print(f"\n-- {group} | {config} "
                  f"({index}/{len(targets)})")
        try:
            rows.append(runner(group, config, args))
        except Exception as exc:                        # noqa: BLE001
            logger.exception("[%s] failed", group)
            failures.append((group, f"{type(exc).__name__}: {exc}"))

    if len(targets) > 1:
        print(f"\n== catalogue summary ({len(rows)}/{len(targets)} trained) "
              "==")
        for row in rows:
            extra = (f"  proba={Path(row['proba']).name}"
                     if row.get("proba") else "")
            print(f"  {row['group']:<24s} {row['best_algo']:<6s} "
                  f"AP={row['ap']:.3f}{extra}")
        for group, error in failures:
            print(f"  {group:<24s} FAILED: {error}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
