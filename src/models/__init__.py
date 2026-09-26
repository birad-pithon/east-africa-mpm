"""Model training, spatial CV, and inference for the MPM pipeline."""
from src.models.baselines import (
    AnomalyBaseline,
    scale_pos_weight,
    smote_oversample,
)
from src.models.dataset import (
    build_training_table,
    default_feature_rasters,
    sample_rasters,
)
from src.models.ensemble import StackedEnsemble, evaluate_stack, fit_stacked_model
from src.models.predict import predict_raster
from src.models.pu import PULearner, evaluate_pu_cv, fit_pu_model
from src.models.train import (
    evaluate_spatial_cv,
    make_model,
    make_model_with_early_stopping,
    prepare_training_table,
    provenance_input_paths,
    train_group,
)
from src.models.tuning import suggest_params, tune_group

__all__ = [
    "build_training_table",
    "default_feature_rasters",
    "sample_rasters",
    "evaluate_spatial_cv",
    "make_model",
    "make_model_with_early_stopping",
    "train_group",
    "prepare_training_table",
    "provenance_input_paths",
    "predict_raster",
    "AnomalyBaseline",
    "smote_oversample",
    "scale_pos_weight",
    # PU learning
    "PULearner",
    "evaluate_pu_cv",
    "fit_pu_model",
    # stacking ensemble
    "StackedEnsemble",
    "evaluate_stack",
    "fit_stacked_model",
    # Bayesian tuning
    "suggest_params",
    "tune_group",
    # hold-out known-deposit validation
    "run_holdout_test",
    # feature importance
    "get_feature_importance",
    # model comparison
    "compare_models",
]


def run_holdout_test(group, **kwargs):
    """Leave-one-deposit-out holdout retraining + rediscovery scoring.

    Delegates to :mod:`src.models.holdout`, which rebuilds the label
    GeoPackage and the deposit-distance prior without the held-out
    deposit (the historical in-memory seed mutation never reached the
    files ``train_group`` actually reads).
    """
    from src.models.holdout import run_holdout_test as _run

    return _run(group, **kwargs)


def get_feature_importance(model, feature_names: list[str]) -> dict:
    """Extract feature importance from a fitted model.

    Parameters
    ----------
    model : fitted classifier with ``feature_importances_`` or ``coef_`` attribute.
    feature_names : list of feature names corresponding to model inputs.

    Returns
    -------
    dict with ``feature_names``, ``importances``, and ``importance_type``.
    """
    import numpy as np

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_type = "gain"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
        importance_type = "coefficient_magnitude"
    else:
        raise ValueError(
            "Model has no feature_importances_ or coef_ attribute"
        )

    return {
        "feature_names": list(feature_names),
        "importances": importances.tolist(),
        "importance_type": importance_type,
    }


def compare_models(metrics_paths: list, output_path: str | None = None) -> dict:
    """Compare multiple model metrics files and generate a comparison report.

    Parameters
    ----------
    metrics_paths : list of paths to metrics_<group>.json files.
    output_path : optional path to write comparison JSON.

    Returns
    -------
    dict with ``groups``, ``best_per_group``, and ``overall_best``.
    """
    import json
    from pathlib import Path

    comparisons = []
    for path in metrics_paths:
        path = Path(path)
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        comparisons.append(metrics)

    best_per_group = {}
    for m in comparisons:
        group = m.get("group", "unknown")
        results = m.get("results", {})
        if results:
            best_algo = max(
                results.keys(),
                key=lambda k: results[k].get("mean_average_precision", 0),
            )
            best_per_group[group] = {
                "best_algo": best_algo,
                "mean_average_precision": results[best_algo].get(
                    "mean_average_precision"
                ),
                "n_positives": m.get("n_positives"),
                "n_features": len(m.get("features", [])),
            }

    overall_best = None
    if best_per_group:
        overall_best = max(
            best_per_group.items(),
            key=lambda x: x[1].get("mean_average_precision") or 0,
        )[0]

    result = {
        "groups": list(best_per_group.keys()),
        "best_per_group": best_per_group,
        "overall_best": overall_best,
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result
