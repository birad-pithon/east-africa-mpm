"""Bayesian hyperparameter tuning (TPE via Optuna) for the model stack.

Recent MPM literature replaces grid/random search with Tree-structured
Parzen Estimator (TPE) optimisation — cheap for our small tabular models
and reliably better in head-to-head ablations. This module defines the
search spaces per algorithm, runs the study with the **same certified
spatial CV** used for evaluation (never random folds), and reports the
best configuration.

Requires ``optuna`` (pip install optuna / ``pip install -e ".[bayes]"``).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = ["tune_group", "suggest_params"]


def suggest_params(trial, algo: str) -> dict:
    """Optuna hyperparameter suggestions for one algorithm."""
    if algo == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 18),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }
    if algo == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        }
    if algo == "lgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64, step=4),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 5, 50, step=5),
        }
    raise ValueError(f"unknown algo '{algo}' (rf|xgb|lgbm)")


def tune_group(
    group: str,
    config_path: str = "configs/karagwe.yml",
    algos: tuple[str, ...] = ("rf", "xgb", "lgbm"),
    n_trials: int = 30,
    timeout: float | None = None,
    block_m: float | None = None,
    buffer_km: float | None = None,
    raster_paths=None,
    label_gpkg=None,
    background_gpkg=None,
    seed: int = 42,
) -> dict:
    """Run a TPE study over the algorithm params under certified CV.

    Returns a JSON-safe summary; the Optuna ``study`` object is attached
    as ``study`` (ignore for JSON persistence).
    """
    try:
        import optuna
    except ImportError as exc:                     # pragma: no cover
        raise ImportError(
            "optuna is required for tuning - run `pip install optuna` "
            "or `pip install -e .[bayes]`"
        ) from exc

    from src.models.train import (
        _alias_algos,
        evaluate_spatial_cv,
        make_model,
        prepare_training_table,
    )

    cfg, params, cfg_algos, X, y, _, coords_utm, block_m, buffer_km = \
        prepare_training_table(
            group, config_path=config_path, raster_paths=raster_paths,
            label_gpkg=label_gpkg, background_gpkg=background_gpkg,
            block_m=block_m, buffer_km=buffer_km,
        )
    if algos is None:
        algos = tuple(cfg_algos)
    else:
        algos = tuple(_alias_algos(list(algos)))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    def objective(trial) -> float:
        algo = trial.suggest_categorical("algo", list(algos))
        trial_params = {**params, **suggest_params(trial, algo)}
        res = evaluate_spatial_cv(
            lambda pw, a=algo: make_model(a, trial_params, pw),
            X, y, coords_utm,
            block_size_m=block_m,
            buffer_m=buffer_km * 1000.0,
        )
        ap = res["mean_average_precision"]
        return ap if ap is not None else -1.0

    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    best = study.best_trial
    best_algo = best.params.pop("algo")
    return {
        "best_algo": best_algo,
        "best_params": dict(best.params),
        "best_mean_ap": float(best.value),
        "n_trials": len(study.trials),
        "study": study,
    }
