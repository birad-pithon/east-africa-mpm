"""Model training + spatial cross-validation for prospectivity mapping.

Implements SOW items 6-7: blocked/buffered spatial CV (never random
k-fold) and per-commodity baselines (RandomForest first, XGBoost and
LightGBM comparison) sized for small positive-label counts.

Primary metric is **average precision** (PR-AUC): with ~20 positives
against hundreds of backgrounds, ROC-AUC flatters trivial models.

Usage::

    from src.models.train import train_group
    result = train_group("tin_tungsten_tantalum")

CLI::

    python -m src.models.main --group tin_tungsten_tantalum \
        [--algos rf xgb lgbm] [--block-m 10000] [--buffer-km 5]
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.models.baselines import (
    AnomalyBaseline,
    scale_pos_weight,
    smote_oversample,
)
from src.models.dataset import build_training_table
from src.utils import load_config, project_path

logger = logging.getLogger(__name__)


__all__ = [
    "make_model",
    "make_model_with_early_stopping",
    "evaluate_spatial_cv",
    "provenance_input_paths",
    "train_group",
]


# ── model factories ───────────────────────────────────────────────────


def make_model(algo: str, params: dict, pos_weight: float):
    """Instantiate a classifier; ``pos_weight`` counters class imbalance."""
    seed = int(params.get("random_state", 42))
    if algo == "rf":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 500)),
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params.get("min_samples_leaf", 3)),
            min_samples_split=int(params.get("min_samples_split", 2)),
            max_features=params.get("max_features"),
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if algo == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=int(params.get("n_estimators", 400)),
            max_depth=int(params.get("max_depth") or 4),
            learning_rate=float(params.get("learning_rate", 0.05)),
            subsample=float(params.get("subsample", 0.9)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            min_child_weight=float(params.get("min_child_weight", 1)),
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
    if algo == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=int(params.get("n_estimators", 400)),
            max_depth=int(params.get("max_depth") or -1),
            num_leaves=int(params.get("num_leaves", 15)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            subsample=float(params.get("subsample", 0.9)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            min_child_samples=int(params.get("min_child_samples", 20)),
            scale_pos_weight=pos_weight,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"unknown algo '{algo}' (rf|xgb|lgbm)")


def make_model_with_early_stopping(
    algo: str,
    params: dict,
    pos_weight: float,
    eval_set: tuple | None = None,
    early_stopping_rounds: int = 50,
):
    """Instantiate a classifier with early stopping support.

    For XGBoost and LightGBM, early stopping is enabled when ``eval_set``
    is provided. RandomForest does not support early stopping.

    Parameters
    ----------
    algo : one of ``rf``, ``xgb``, ``lgbm``.
    params : hyperparameter dict.
    pos_weight : class imbalance weight.
    eval_set : optional ``(X_val, y_val)`` tuple for early stopping.
    early_stopping_rounds : rounds without improvement before stopping.

    Returns
    -------
    classifier with ``fit`` method that accepts ``eval_set`` for xgb/lgbm.
    """
    seed = int(params.get("random_state", 42))
    if algo == "rf":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 500)),
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params.get("min_samples_leaf", 3)),
            min_samples_split=int(params.get("min_samples_split", 2)),
            max_features=params.get("max_features"),
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if algo == "xgb":
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            n_estimators=int(params.get("n_estimators", 400)),
            max_depth=int(params.get("max_depth") or 4),
            learning_rate=float(params.get("learning_rate", 0.05)),
            subsample=float(params.get("subsample", 0.9)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            min_child_weight=float(params.get("min_child_weight", 1)),
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
        if eval_set is not None:
            clf.fit(
                eval_set[0], eval_set[1],
                eval_set=[eval_set],
                early_stopping_rounds=early_stopping_rounds,
                verbose=False,
            )
        return clf
    if algo == "lgbm":
        from lightgbm import LGBMClassifier

        clf = LGBMClassifier(
            n_estimators=int(params.get("n_estimators", 400)),
            max_depth=int(params.get("max_depth") or -1),
            num_leaves=int(params.get("num_leaves", 15)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            subsample=float(params.get("subsample", 0.9)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            min_child_samples=int(params.get("min_child_samples", 20)),
            scale_pos_weight=pos_weight,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        if eval_set is not None:
            clf.fit(
                eval_set[0], eval_set[1],
                eval_set=[eval_set],
                early_stopping_rounds=early_stopping_rounds,
                verbose=False,
            )
        return clf
    raise ValueError(f"unknown algo '{algo}' (rf|xgb|lgbm)")


# ── tiny-n regularization preset ──────────────────────────────────────

TINY_N_THRESHOLD = 15  # activate below this many positives


def regularize_for_tiny_n(params: dict, n_positives: int) -> dict:
    """Override hyper-parameters for very small positive counts.

    With < ~15 positives the risk is memorizing a boundary, not
    underfitting: shallow trees, big leaves, strong L2, fewer boosts.
    Keys are set for all algorithms; each ``make_model`` branch picks up
    only the ones it reads.
    """
    p = dict(params)
    p.update({
        # shared / rf
        "max_depth": 3,
        "min_samples_leaf": 5,
        "min_samples_split": 10,
        "max_features": "sqrt",
        # xgb
        "min_child_weight": 10.0,
        "reg_lambda": 10.0,
        "reg_alpha": 1.0,
        "n_estimators": 200,
        "learning_rate": 0.05,
        # lgbm
        "num_leaves": 7,
        "min_child_samples": 30,
    })
    logger.info("tiny-n preset active (n_pos=%d < %d): shallow/regularized",
                n_positives, TINY_N_THRESHOLD)
    return p


# ── spatial CV evaluation ─────────────────────────────────────────────


def evaluate_spatial_cv(
    make_clf,
    X: pd.DataFrame,
    y: np.ndarray,
    coords: np.ndarray,
    block_size_m: float = 10_000.0,
    buffer_m: float = 5_000.0,
    n_splits: int = 5,
    oversample: bool = False,
    anomaly_baseline: bool = False,
    collect_predictions: bool = False,
) -> dict:
    """Run blocked (+buffered) spatial CV; never random folds.

    ``make_clf(pos_weight)`` must return a fresh unfitted classifier so
    each fold is independent. Returns per-fold and summary metrics.

    Parameters
    ----------
    oversample : if True, apply deterministic SMOTE to each training
        fold only (never test) before fitting, defending against
        degenerate all-negative training batches.
    anomaly_baseline : if True, additionally fit an IsolationForest
        on barren cells only and report its average-precision as a
        supervised-model lower-bound reference.
    collect_predictions : if True, the result gains an
        ``oof_predictions`` entry with pooled out-of-fold
        ``y_true``/``y_proba`` arrays (for calibration reporting).
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    from src.validate import BufferedSpatialCV, LeakageError, certify_folds

    cv = BufferedSpatialCV(
        coords=coords,
        block_size_m=block_size_m,
        n_splits=n_splits,
        buffer_m=buffer_m,
        y=y,
    )
    splits = list(cv.split(None, y))
    # SAFEGUARD 1 (spatial leakage): actively certify the fold plan -
    # train/test disjointness and buffer separation - BEFORE fitting.
    # Any violation aborts training instead of inflating scores.
    try:
        cert = certify_folds(coords, splits, min_buffer_m=buffer_m)
    except LeakageError as exc:
        raise LeakageError(
            f"fold plan failed spatial certification: {exc}"
        ) from exc
    logger.info(
        "spatial CV certified: %d folds, min train-test separation %.0f m",
        len(splits),
        cert.get("min_observed_separation_m") or 0.0,
    )

    Xv = X.to_numpy(dtype="float64")
    pos = float(y.sum())
    neg = float(len(y) - y.sum())
    pos_weight = (neg / pos) if pos else 1.0

    folds = []
    ap_anom: list[float] = []
    oof_true: list[np.ndarray] = []
    oof_proba: list[np.ndarray] = []
    for k, (tr, te) in enumerate(splits):
        Xt = Xv[tr]
        yt = y[tr]
        Xe = Xv[te]
        ye = y[te]

        if oversample:
            Xt, yt = smote_oversample(Xt, yt, seed=42)
        # Degenerate-split guard: with only a handful of positives the
        # blocked plan (or the buffer around the test deposits) can strip
        # every positive out of the training batch.  Such a split cannot be
        # fit or scored - predict_proba collapses to one column - so keep the
        # fold entry unscored, exactly like a fold with no positive test
        # labels below, instead of crashing the whole run.
        if np.unique(yt).size < 2:
            logger.warning(
                "fold %d skipped: training split holds a single class "
                "(%d positives of %d rows)",
                k, int(yt.sum()), len(yt),
            )
            folds.append(
                {
                    "fold": k,
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "n_test_positives": int(ye.sum()),
                }
            )
            continue
        pw = scale_pos_weight(yt) if oversample else pos_weight
        clf = make_clf(pw)
        clf.fit(Xt, yt)
        proba = clf.predict_proba(Xe)[:, 1]
        oof_true.append(np.asarray(ye, dtype=int))
        oof_proba.append(np.asarray(proba, dtype=float))

        # anomaly baseline: unsupervised lower bound
        if anomaly_baseline:
            ab = AnomalyBaseline(seed=42)
            if yt.sum() < len(yt):  # has negatives
                ab.fit(Xt[yt == 0])
            else:
                ab.fit(Xt)
            ap = ab.score_ap(Xe, ye)
            if ap is not None:
                ap_anom.append(ap)

        fold_pos = int(ye.sum())
        entry = {
            "fold": k,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "n_test_positives": fold_pos,
        }
        if fold_pos:
            entry["average_precision"] = float(
                average_precision_score(ye, proba)
            )
            if ye.sum() < len(ye):          # ROC needs both classes
                entry["roc_auc"] = float(roc_auc_score(ye, proba))
            # precision within the top decile of scored test cells
            n_top = max(1, int(np.ceil(0.10 * len(proba))))
            top = np.argsort(proba)[::-1][:n_top]
            entry["precision_at_top_decile"] = float(ye[top].mean())
        else:
            logger.warning("fold %d has no positive test labels", k)
        folds.append(entry)

    ap = [f["average_precision"] for f in folds
          if "average_precision" in f]
    roc = [f["roc_auc"] for f in folds if "roc_auc" in f]
    prec = [f["precision_at_top_decile"] for f in folds
            if "precision_at_top_decile" in f]
    result = {
        "n_folds": len(folds),
        "folds": folds,
        "mean_average_precision": float(np.mean(ap)) if ap else None,
        "std_average_precision": float(np.std(ap)) if ap else None,
        "mean_roc_auc": float(np.mean(roc)) if roc else None,
        "mean_precision_at_top_decile": (
            float(np.mean(prec)) if prec else None
        ),
        **({"anomaly_baseline_mean_ap": float(np.mean(ap_anom))}
          if ap_anom else {}),
    }
    if collect_predictions and oof_true:
        result["oof_predictions"] = {
            "y_true": np.concatenate(oof_true),
            "y_proba": np.concatenate(oof_proba),
        }
    return result


# ── group orchestrator ────────────────────────────────────────────────


_ALGO_ALIASES = {"random_forest": "rf", "xgboost": "xgb",
                 "lightgbm": "lgbm"}


def _alias_algos(algos: list[str]) -> list[str]:
    """Map configured algorithm names onto internal ids (rf/xgb/lgbm)."""
    return [_ALGO_ALIASES.get(a, a) for a in algos]


def prepare_training_table(
    group: str,
    config_path: str | Path = "configs/karagwe.yml",
    raster_paths: list[Path | str] | None = None,
    label_gpkg: Path | str | None = None,
    background_gpkg: Path | str | None = None,
    block_m: float | None = None,
    buffer_km: float | None = None,
    analogs_path: str | Path | None = None,
) -> tuple:
    """Load config, build the training table, and project CV coordinates.

    Shared by :func:`train_group` and :func:`src.models.tuning.tune_group`
    so both evaluate on identical data. Returns
    ``(cfg, params, algos, X, y, pts, coords_utm, block_m, buffer_km)``.
    """
    cfg = load_config(config_path)
    mcfg = cfg.get("model", {})
    vcfg = cfg.get("validation", {})
    params = mcfg.get("params", {})
    algos = _alias_algos(list(
        mcfg.get("compare_algorithms",
                 ["random_forest", "xgboost", "lightgbm"])))
    block_m = block_m or 10_000.0
    buffer_km = buffer_km or float(vcfg.get("buffer_km", 5))

    analogs_gdf = None
    if analogs_path is not None:
        import geopandas as gpd

        analogs_gdf = gpd.read_file(analogs_path)
        logger.info("analogs loaded: %d from %s",
                    len(analogs_gdf), analogs_path)

    X, y, pts = build_training_table(
        group,
        raster_paths=raster_paths,
        label_gpkg=label_gpkg,
        background_gpkg=background_gpkg,
        analogs_gdf=analogs_gdf,
        config_path=config_path,
    )
    import pyproj

    coords_ll = np.column_stack([pts.geometry.x, pts.geometry.y])
    tf = pyproj.Transformer.from_crs(
        "EPSG:4326", cfg["grid"]["crs"], always_xy=True).transform
    xs, ys = tf(coords_ll[:, 0], coords_ll[:, 1])
    coords_utm = np.column_stack([xs, ys])
    return (cfg, params, algos, X, y, pts, coords_utm, block_m, buffer_km)


def provenance_input_paths(
    group: str,
    config_path: str | Path = "configs/karagwe.yml",
    raster_paths: list[Path | str] | None = None,
    label_gpkg: Path | str | None = None,
    background_gpkg: Path | str | None = None,
) -> list[Path | str]:
    """Input files recorded in the run provenance manifest for *group*.

    Resolves the feature rasters through the SAME ``config_path`` used to
    build the training table, so provenance lists exactly the layers the
    model consumed (the ``features.rasters_by_group`` allowlist of the
    per-belt config) instead of every ``*.tif`` under ``data/interim``.
    """
    from src.models.dataset import default_feature_rasters

    return [
        label_gpkg or project_path("data", "processed", f"labels_{group}.gpkg"),
        background_gpkg or project_path(
            "data", "processed", f"background_{group}.gpkg"),
        *(raster_paths if raster_paths is not None
          else default_feature_rasters(group, config_path=config_path)),
    ]


def train_group(
    group: str,
    config_path: str | Path = "configs/karagwe.yml",
    algos: list[str] | None = None,
    raster_paths: list[Path | str] | None = None,
    block_m: float | None = None,
    buffer_km: float | None = None,
    output_dir: Path | str | None = None,
    label_gpkg: Path | str | None = None,
    background_gpkg: Path | str | None = None,
    oversample: bool = False,
    anomaly_baseline: bool = False,
    pu: bool = False,
    ensemble: bool = False,
    tiny_n: bool | str = False,
    analogs_path: str | Path | None = None,
    ) -> dict:
    """Train + compare models for one commodity group end-to-end.

    Writes ``outputs/models/metrics_<group>.json`` and
    ``model_<group>_<algo>.joblib`` (best by mean average precision,
    refitted on all labelled rows).

    Parameters
    ----------
    oversample : apply deterministic SMOTE to each CV training fold.
    anomaly_baseline : include IsolationForest-on-barren reference AP.
    pu : also evaluate PU-bagged variants (``<algo>_pu``) that treat
        background as unlabeled rather than hard negatives.
    ensemble : also evaluate an OOF logistic stack of all base algos;
        the ``stack`` entry competes for best.
    """
    # Capture the caller's algo selection BEFORE the tuple unpacking
    # overwrites `algos` with the config-derived list (this was a silent
    # bug: caller algos were always discarded and all 3 algos trained).
    caller_algos = _alias_algos(algos) if algos is not None else None
    cfg, params, cfg_algos, X, y, pts, coords_utm, block_m, buffer_km = \
        prepare_training_table(
            group, config_path=config_path, raster_paths=raster_paths,
            label_gpkg=label_gpkg, background_gpkg=background_gpkg,
            block_m=block_m, buffer_km=buffer_km,
            analogs_path=analogs_path,
        )
    algos = caller_algos if caller_algos is not None else cfg_algos

    # tiny-n preset: auto-activates below TINY_N_THRESHOLD positives
    n_positives = int((y == 1).sum())
    if tiny_n is True or tiny_n == "auto":
        if tiny_n is True or n_positives < TINY_N_THRESHOLD:
            params = regularize_for_tiny_n(params, n_positives)

    # resolve the same inputs build_training_table used, for provenance
    provenance_inputs = provenance_input_paths(
        group, config_path=config_path, raster_paths=raster_paths,
        label_gpkg=label_gpkg, background_gpkg=background_gpkg,
    )

    # PU masks: known deposits (y==1); background = unlabeled pool, of
    # which the barren_halo stratum holds reliable negatives.
    pu_unlabeled = pu_reliable = None
    if pu:
        if {"source", "stratum"} <= set(pts.columns):
            is_bg = (pts["source"].astype(str) == "background").to_numpy()
            pu_unlabeled = is_bg
            pu_reliable = (is_bg & (
                pts["stratum"].astype(str) == "barren_halo")).to_numpy()
        else:
            logger.warning("PU: no source/stratum meta on points - treating "
                           "every y==0 row as unlabeled")
            pu_unlabeled = (y == 0)
            pu_reliable = None
        if not pu_unlabeled.any():
            logger.warning("PU: no background rows - PU variants disabled")
            pu = False

    results = {}
    seed = int(params.get("random_state", 42))
    for algo in algos:
        logger.info("[%s] evaluating %s ...", group, algo)
        def make_factory(pw, a=algo):
            return make_model(a, params, pw)
        results[algo] = evaluate_spatial_cv(
            make_factory,
            X, y, coords_utm,
            block_size_m=block_m,
            buffer_m=buffer_km * 1000.0,
            oversample=oversample,
            anomaly_baseline=anomaly_baseline,
            collect_predictions=True,
        )
        logger.info("[%s] %s AP=%.3f", group, algo,
                    (results[algo]["mean_average_precision"] or float("nan")))
        if pu:
            from src.models.pu import evaluate_pu_cv

            key = f"{algo}_pu"
            try:
                results[key] = evaluate_pu_cv(
                    make_factory, X, y, coords_utm,
                    unlabeled=pu_unlabeled, reliable_neg=pu_reliable,
                    block_size_m=block_m, buffer_m=buffer_km * 1000.0,
                    seed=seed)
                logger.info("[%s] %s AP=%.3f", group, key,
                            (results[key]["mean_average_precision"]
                             or float("nan")))
            except Exception as exc:                   # noqa: BLE001
                logger.warning("PU variant failed for %s: %s", algo, exc)

    stack_state = None
    if ensemble and len(algos) >= 2:
        from src.models.ensemble import evaluate_stack

        factories = {a: (lambda pw, aa=a: make_model(aa, params, pw))
                     for a in algos}
        try:
            st = evaluate_stack(
                factories, X, y, coords_utm,
                block_size_m=block_m, buffer_m=buffer_km * 1000.0,
                oversample=oversample, seed=seed)
            objects = st.pop("_stack_objects")
            results["stack"] = {k: v for k, v in st.items()}
            results["stack"]["oof_predictions"] = {
                "y_true": y,
                "y_proba": objects["meta"].predict_proba(
                    objects["oof_base"])[:, 1],
            }
            stack_state = objects
            logger.info("[%s] stack AP=%.3f", group,
                        st["mean_average_precision"])
        except Exception as exc:                       # noqa: BLE001
            logger.warning("ensemble eval failed: %s", exc)

    best = max(
        (a for a, r in results.items()
         if r.get("mean_average_precision") is not None),
        key=lambda a: results[a]["mean_average_precision"],
        default=algos[0],
    )
    pos_weight = (
        float((len(y) - y.sum()) / y.sum()) if y.sum() else 1.0
    )
    Xn = X.to_numpy(dtype="float64")
    if best == "stack" and stack_state is not None:
        from src.models.ensemble import fit_stacked_model

        final_model = fit_stacked_model(
            factories, X, y, stack_state["oof_base"], stack_state["meta"],
            seed=seed)
    elif best.endswith("_pu"):
        from src.models.pu import fit_pu_model

        base_algo = best[:-3]
        final_model = fit_pu_model(
            lambda pw, a=base_algo: make_model(a, params, pw),
            Xn, y, pu_unlabeled, pu_reliable, seed=seed)
    else:
        final_model = make_model(best, params, pos_weight)
        final_model.fit(Xn, y)

    out_dir = Path(output_dir) if output_dir else \
        project_path("outputs", "models")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── calibration report from pooled out-of-fold predictions ────────
    calibration = None
    oof = results[best].pop("oof_predictions", None)
    if oof is not None and oof["y_true"].sum() > 0:
        from src.models.calibration import (
            calibration_report,
            plot_reliability,
            save_report,
        )

        calibration = calibration_report(oof["y_true"], oof["y_proba"])
        plot_reliability(
            oof["y_true"], oof["y_proba"],
            out_dir / f"reliability_{group}.png",
            title=f"{group} ({best}) — out-of-fold reliability",
        )
        save_report(calibration, out_dir / f"calibration_{group}.json")
        logger.info("[%s] calibration: Brier=%.3f ECE=%.3f", group,
                    calibration["brier"], calibration["ece"])
    for res in results.values():
        res.pop("oof_predictions", None)      # numpy arrays: not JSON-safe

    bundle_path = out_dir / f"model_{group}_{best}.joblib"
    from src.utils.provenance import collect_provenance

    provenance = collect_provenance(
        config_path=config_path, input_paths=provenance_inputs,
    )
    joblib.dump(
        {
            "group": group,
            "algo": best,
            "model": final_model,
            "feature_names": list(X.columns),
            "config": {"block_m": block_m, "buffer_km": buffer_km},
            "provenance": provenance,
        },
        bundle_path,
    )

    metrics = {
        "group": group,
        "n_rows": int(len(y)),
        "n_positives": int(y.sum()),
        "features": list(X.columns),
        "buffer_km": buffer_km,
        "block_m": block_m,
        "oversample": bool(oversample),
        "anomaly_baseline": bool(anomaly_baseline),
        "pu": bool(pu),
        "ensemble": bool(ensemble),
        "best_algo": best,
        "results": results,
        "provenance": provenance,
        **({"calibration": calibration} if calibration is not None else {}),
    }
    metrics_path = out_dir / f"metrics_{group}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # model card is generated from the artifacts above — keep in sync
    try:
        from src.models.model_card import build_model_card

        build_model_card(group, metrics_path=metrics_path,
                         bundle_path=bundle_path)
    except Exception:                                   # noqa: BLE001
        logger.warning("model card generation failed", exc_info=True)
    logger.info("[%s] best=%s -> %s", group, best, bundle_path.name)
    return {"metrics": metrics, "metrics_path": metrics_path,
            "bundle_path": bundle_path}
