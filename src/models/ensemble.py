"""Out-of-fold stacking ensemble for prospectivity models.

Under the data scarcity MPM faces, a boosted/forest committee captured
through a meta-learner consistently beats any single model. This module
implements the standard stacking recipe on top of the existing certified
spatial-CV machinery:

1. Run the certified buffered-block fold plan once.
2. Fit every base model per fold; collect **out-of-fold** probabilities.
3. Fit a logistic-regression meta-learner on the OOF probabilities.
4. Refit base models on the full data; predictions combine them through
   the meta-learner.

The resulting :class:`StackedEnsemble` exposes ``predict_proba`` so it
slots straight into :func:`src.models.predict.predict_raster` and the
bundle format used by ``train_group``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models.baselines import smote_oversample
from src.validate import BufferedSpatialCV, LeakageError, certify_folds

logger = logging.getLogger(__name__)

__all__ = ["StackedEnsemble", "evaluate_stack", "fit_stacked_model"]


class StackedEnsemble:
    """A dict of fitted base estimators + a fitted meta-learner."""

    def __init__(self, base_estimators: dict, meta) -> None:
        self.base_estimators = base_estimators
        self.meta = meta
        self.classes_ = np.array([0, 1])

    def _base_matrix(self, X) -> np.ndarray:
        if isinstance(X, (pd.DataFrame, pd.Series)):
            Xa = X.to_numpy(dtype="float64")
        else:
            Xa = np.asarray(X, dtype="float64")
        return np.column_stack([
            clf.predict_proba(Xa)[:, 1]
            for clf in self.base_estimators.values()
        ])

    def predict_proba(self, X) -> np.ndarray:
        return self.meta.predict_proba(self._base_matrix(X))

    def predict(self, X) -> np.ndarray:
        return self.meta.predict(self._base_matrix(X))


def evaluate_stack(
    clf_factories: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    coords: np.ndarray,
    block_size_m: float = 10_000.0,
    buffer_m: float = 5_000.0,
    n_splits: int = 5,
    oversample: bool = False,
    seed: int = 42,
) -> dict:
    """Evaluate OOF stacking under the certified spatial fold plan.

    Parameters
    ----------
    clf_factories : ``{name: callable(pos_weight) -> fresh classifier}``.

    Returns
    -------
    dict mirroring the ``evaluate_spatial_cv`` shape — ``n_folds``,
    ``mean_average_precision`` (of the stack's OOF predictions) — plus
    ``stacked_from`` (base names), and internal ``_stack_objects`` with
    the OOF base matrix + fitted meta for the final model.
    """
    if len(clf_factories) < 2:
        raise ValueError("stacking needs at least two base models")

    splits = list(BufferedSpatialCV(
        coords=coords, block_size_m=block_size_m, n_splits=n_splits,
        buffer_m=buffer_m, y=y,
    ).split(None, y))
    try:
        certify_folds(coords, splits, min_buffer_m=buffer_m)
    except LeakageError as exc:
        raise LeakageError(
            f"stack fold plan failed spatial certification: {exc}"
        ) from exc

    Xv = X.to_numpy(dtype="float64")
    names = list(clf_factories)
    n = len(y)
    oof = {name: np.full(n, np.nan, dtype="float64") for name in names}
    pos_weight = float((len(y) - y.sum()) / max(1.0, float(y.sum())))

    for tr, te in splits:
        Xt, yt = Xv[tr], y[tr]
        if oversample:
            Xt, yt = smote_oversample(Xt, yt, seed=seed)
        for name in names:
            clf = clf_factories[name](pos_weight)
            clf.fit(Xt, yt)
            oof[name][te] = clf.predict_proba(Xv[te])[:, 1]

    oof_df = pd.DataFrame(oof)
    if oof_df.isna().any().any():
        raise RuntimeError("stack OOF contains NaN - fold coverage broken")
    meta = LogisticRegression(max_iter=2_000, random_state=seed).fit(
        oof_df, y)
    stack_proba = meta.predict_proba(oof_df)[:, 1]

    roc = roc_auc_score(y, stack_proba) if y.sum() < len(y) else None
    return {
        "n_folds": len(splits),
        "mean_average_precision": float(
            average_precision_score(y, stack_proba)),
        "mean_roc_auc": float(roc) if roc is not None else None,
        "method": "oof_stack",
        "stacked_from": names,
        "per_model_oof_ap": {
            name: float(average_precision_score(y, oof_df[name]))
            for name in names
        },
        "_stack_objects": {"oof_base": oof_df, "meta": meta},
    }


def fit_stacked_model(
    clf_factories: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    oof_base: pd.DataFrame,
    meta,
    seed: int = 42,
) -> StackedEnsemble:
    """Refit base models on all data; return the deployable ensemble.

    ``oof_base`` and ``meta`` come from :func:`evaluate_stack`
    (``_stack_objects``). The meta-learner is refit on the full OOF
    matrix before deployment.
    """
    Xa = X.to_numpy(dtype="float64")
    pos_weight = float((len(y) - y.sum()) / max(1.0, float(y.sum())))
    base = {
        name: clf_factories[name](pos_weight).fit(Xa, y)
        for name in clf_factories
    }
    meta = LogisticRegression(max_iter=2_000, random_state=seed).fit(
        oof_base, y)
    return StackedEnsemble(base, meta)
