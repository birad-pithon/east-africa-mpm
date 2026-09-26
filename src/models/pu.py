"""Positive-unlabeled (PU) learning for mineral prospectivity mapping.

Motivation: most background samples are *not* proven barren. greenfield
points are genuinely **unlabeled** — "no known deposit" is not the same
as "barren". Training with all of them as hard negatives biases the
model against unexplored prospective ground — precisely the failure
mode identified for imbalanced positive-unlabeled MPM tasks.

This module implements PU bagging (Mordelet & Vert, 2014) with an
optional set of *reliable negatives* (our field-explored
``barren_halo`` stratum):

* labeled positive    : known deposits (``y == 1``);
* unlabeled pool      : everything else (``y == 0``) — **not** hard
  negatives;
* reliable negative   : optional subset of the unlabeled pool known to
  be barren — always used as negatives in every bag.

References
----------
* Mordelet & Vert (2014), "A bagging SVM to learn from positive and
  unlabeled examples", Pattern Recognition Letters 37:201-209.
* Bekker & Davis (2020), "Learning from positive and unlabeled data:
  a survey", Machine Learning 109:719-760.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.validate import BufferedSpatialCV, LeakageError, certify_folds

logger = logging.getLogger(__name__)

__all__ = ["PULearner", "fit_pu_model", "evaluate_pu_cv"]


class PULearner:
    """PU-bagged classifier that averages a committee of base learners.

    Each bag is trained on every labeled positive plus a fresh random
    subsample of the unlabeled pool (or, when supplied, the reliable
    negatives plus random unlabeled samples). Averaging over bags gives
    the classic PU-bagging estimate of ``P(labeled-positive | x)``.

    Parameters
    ----------
    base_estimator_factory : callable(pos_weight) -> fresh classifier
        with ``fit`` and ``predict_proba`` (e.g. ``make_model``).
    n_bags : number of bootstrap bags.
    neg_ratio : negatives per bag as a fraction of the positive count.
    seed : RNG seed for reproducible bag composition.
    """

    def __init__(
        self,
        base_estimator_factory,
        n_bags: int = 20,
        neg_ratio: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.base_estimator_factory = base_estimator_factory
        self.n_bags = int(n_bags)
        self.neg_ratio = float(neg_ratio)
        self.seed = int(seed)
        self._clfs: list = []
        self.classes_ = np.array([0, 1])

    def fit(
        self,
        X,
        y,
        unlabeled: np.ndarray | None = None,
        reliable_neg: np.ndarray | None = None,
    ) -> PULearner:
        """Fit PU bags.

        Parameters
        ----------
        y : 1 = known positive, 0 = unlabeled (NOT hard negative).
        unlabeled : bool mask of the unlabeled pool inside *y*; defaults
            to ``y == 0``.
        reliable_neg : bool mask (subset of *unlabeled*) denoting points
            known to be barren; always used as negatives in every bag.
        """
        Xa = np.asarray(X, dtype="float64")
        ya = np.asarray(y, dtype=int)
        if ya.ndim != 1 or len(ya) != len(Xa):
            raise ValueError("y must be 1-D and match X length")

        unlabeled = (ya == 0) if unlabeled is None \
            else np.asarray(unlabeled, dtype=bool)
        reliable = (np.zeros(len(ya), dtype=bool)
                    if reliable_neg is None
                    else np.asarray(reliable_neg, dtype=bool))
        if not np.all(reliable <= unlabeled):
            raise ValueError("reliable_neg must be a subset of unlabeled")
        if not unlabeled.any():
            raise ValueError("no unlabeled points given - nothing to PU-learn")

        pos_idx = np.flatnonzero(ya == 1)
        unlab_idx = np.flatnonzero(unlabeled & ~reliable)
        rel_idx = np.flatnonzero(reliable)

        n_pos = len(pos_idx)
        n_neg_target = max(1, int(round(n_pos * self.neg_ratio)))
        pos_weight = float(len(ya) - ya.sum()) / max(1.0, float(n_pos))

        rng = np.random.default_rng(self.seed)
        self._clfs = []
        for _ in range(self.n_bags):
            neg_idx = list(rel_idx)                  # reliable: every bag
            if n_neg_target > len(neg_idx) and len(unlab_idx):
                extra = min(int(n_neg_target - len(neg_idx)), len(unlab_idx))
                neg_idx += list(rng.choice(
                    unlab_idx, size=extra, replace=False))
            if not neg_idx:                          # all-negative fallback
                neg_idx = list(rng.choice(unlab_idx, size=n_neg_target,
                                          replace=False))
            bag_idx = np.concatenate([pos_idx, np.asarray(neg_idx)])
            bag_y = np.concatenate([
                np.ones(len(pos_idx), dtype=int),
                np.zeros(len(neg_idx), dtype=int),
            ])
            clf = self.base_estimator_factory(pos_weight)
            clf.fit(Xa[bag_idx], bag_y)
            self._clfs.append(clf)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Mean bag probability of the positive class, shape (n, 2)."""
        Xa = np.asarray(X, dtype="float64")
        if not self._clfs:
            raise RuntimeError("PULearner not fitted")
        p = np.column_stack([
            clf.predict_proba(Xa)[:, 1] for clf in self._clfs
        ]).mean(axis=1)
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def fit_pu_model(
    base_estimator_factory,
    X,
    y,
    unlabeled: np.ndarray | None = None,
    reliable_neg: np.ndarray | None = None,
    n_bags: int = 20,
    seed: int = 42,
) -> PULearner:
    """Fit a PU learner on the full dataset (used for the final bundle)."""
    pu = PULearner(base_estimator_factory, n_bags=n_bags, seed=seed)
    return pu.fit(X, y, unlabeled=unlabeled, reliable_neg=reliable_neg)


def evaluate_pu_cv(
    make_clf,
    X: pd.DataFrame,
    y: np.ndarray,
    coords: np.ndarray,
    unlabeled: np.ndarray | None = None,
    reliable_neg: np.ndarray | None = None,
    block_size_m: float = 10_000.0,
    buffer_m: float = 5_000.0,
    n_splits: int = 5,
    n_bags: int = 20,
    seed: int = 42,
    collect_predictions: bool = True,
) -> dict:
    """Certified spatial CV for PU-bagged learners.

    Mirrors :func:`src.models.train.evaluate_spatial_cv` so the PU
    result is apples-to-apples with the supervised baselines: same
    buffered block fold plan, same certification, same metric object
    (AP/ROC on test rows with the unlabeled pool treated as negatives —
    a conservative proxy, identical to how the baselines are scored).

    Returns the same per-fold / summary dict shape, optionally with an
    ``oof_predictions`` entry.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    if unlabeled is None:
        unlabeled = np.asarray(y == 0, dtype=bool)

    splits = list(BufferedSpatialCV(
        coords=coords, block_size_m=block_size_m, n_splits=n_splits,
        buffer_m=buffer_m, y=y,
    ).split(None, y))
    try:
        certify_folds(coords, splits, min_buffer_m=buffer_m)
    except LeakageError as exc:
        raise LeakageError(
            f"PU fold plan failed spatial certification: {exc}"
        ) from exc

    Xv = X.to_numpy(dtype="float64")
    folds = []
    oof_true: list[np.ndarray] = []
    oof_proba: list[np.ndarray] = []
    for k, (tr, te) in enumerate(splits):
        pu = PULearner(make_clf, n_bags=n_bags, seed=seed)
        pu.fit(
            Xv[tr], y[tr],
            unlabeled=(unlabeled[tr] if unlabeled is not None else None),
            reliable_neg=(reliable_neg[tr] if reliable_neg is not None
                          else None),
        )
        proba = pu.predict_proba(Xv[te])[:, 1]
        ye = y[te]
        oof_true.append(np.asarray(ye, dtype=int))
        oof_proba.append(np.asarray(proba, dtype=float))

        entry = {
            "fold": k,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "n_test_positives": int(ye.sum()),
        }
        if ye.sum():
            entry["average_precision"] = float(
                average_precision_score(ye, proba))
            if ye.sum() < len(ye):
                entry["roc_auc"] = float(roc_auc_score(ye, proba))
        else:
            logger.warning("fold %d has no positive test labels", k)
        folds.append(entry)

    ap = [f["average_precision"] for f in folds
          if "average_precision" in f]
    roc = [f["roc_auc"] for f in folds if "roc_auc" in f]
    result = {
        "n_folds": len(folds),
        "folds": folds,
        "mean_average_precision": float(np.mean(ap)) if ap else None,
        "std_average_precision": float(np.std(ap)) if ap else None,
        "mean_roc_auc": float(np.mean(roc)) if roc else None,
        "method": "pu_bagging",
        "n_bags": int(n_bags),
    }
    if collect_predictions and oof_true:
        result["oof_predictions"] = {
            "y_true": np.concatenate(oof_true),
            "y_proba": np.concatenate(oof_proba),
        }
    return result
