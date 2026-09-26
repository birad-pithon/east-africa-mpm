"""CLI: train per-commodity models + export probability rasters.

    python -m src.models.main --group tin_tungsten_tantalum
    python -m src.models.main --group tin_tungsten_tantalum \
        --algos rf lgbm --block-m 8000 --buffer-km 3 --predict
"""
from __future__ import annotations

import argparse
import logging

from src.models.predict import predict_raster
from src.models.train import train_group


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train commodity-group prospectivity models"
    )
    parser.add_argument("--group", required=True,
                        help="commodity group key (see configs/belts.yaml)")
    parser.add_argument("--config", default="configs/karagwe.yml")
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
    args = parser.parse_args()

    tiny_n_arg: bool | str = args.tiny_n

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")

    if args.tune:
        from src.models.tuning import tune_group

        study_info = tune_group(
            args.group, config_path=args.config,
            n_trials=args.tune_trials, timeout=args.tune_timeout,
            algos=(args.algos or None),
            block_m=args.block_m, buffer_km=args.buffer_km,
        )
        print("\nOK Bayesian tuning (TPE):")
        print(f"  best algo : {study_info['best_algo']}")
        print(f"  best mean AP : {study_info['best_mean_ap']:.3f}")
        print(f"  best params : {study_info['best_params']}")
        print(f"  trials : {study_info['n_trials']}")
        return

    result = train_group(
        args.group,
        config_path=args.config,
        algos=args.algos,
        block_m=args.block_m,
        buffer_km=args.buffer_km,
        output_dir=args.out_dir,
        oversample=args.oversample,
        anomaly_baseline=args.anomaly_baseline,
        pu=args.pu,
        ensemble=args.ensemble,
        tiny_n=tiny_n_arg,
        analogs_path=args.analogs,
    )
    m = result["metrics"]
    print(f"\nOK [{args.group}] best={m['best_algo']}  "
          f"AP={m['results'][m['best_algo']]['mean_average_precision']:.3f}")
    for line in result["metrics_path"], result["bundle_path"]:
        print(f"  {line}")

    if args.predict:
        from src.models.dataset import default_feature_rasters

        proba = predict_raster(result["bundle_path"],
                               default_feature_rasters(args.group,
                                                       config_path=args.config))
        print(f"  {proba}")

    if args.prior:
        if not args.predict:
            print("  --prior requires --predict (conditions a probability "
                  "raster); skipping")
        else:
            from src.features.priors import apply_prior_to_raster, lift_report

            prior_path, diag = apply_prior_to_raster(proba, args.group)
            print(f"  prior sources : {diag['sources']}")
            print(f"  {prior_path}")
            lift = lift_report(proba, prior_path, args.group)
            print(f"  lift check    : AP base={lift['ap_base']:.3f} "
                  f"prior={lift['ap_prior']:.3f} "
                  f"delta={lift['lift']:+.3f} ({lift['note']})")


if __name__ == "__main__":
    main()
