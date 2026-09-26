"""Class-imbalance and anomaly-detection safeguards.

Implements SOW safeguard #2 (Class imbalance and label rarity).

Two complementary defences:

* :class:`AnomalyBaseline` -- an **unsupervised** Isolation-Forest detector
  fit *only on barren* cells. Its average-precision on held-out deposits
  acts as a null reference: if the supervised models cannot beat this,
  the "model" is really just rediscovering that known-mines are
  unusual, not learning geological association.
* :func:`smote_oversample` -- lightweight deterministic minority
  oversampling (no external deps) so XGBoost / LightGBM / RandomForest
  never train on a degenerate all-negative batch.

Primary metric stays **average precision** (see
:func:`src.models.train.evaluate_spatial_cv`); these utilities simply
ensure that metric is computed and reported rather than swamped.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)

__all__ = ["AnomalyBaseline", "smote_oversample", "scale_pos_weight"]


def scale_pos_weight(y: np.ndarray) -> float:
    """Return ``count(neg) / count(pos)`` for imbalanced *y*."""
    y = np.asarray(y, dtype=int)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return 1.0  # degenerate single-class batch
    return float(len(y) - n_pos) / n_pos


def smote_oversample(
    X: np.ndarray,
    y: np.ndarray,
    target_ratio: float = 1.0,
    k: int = 5,
    seed: int | None = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Lightweight deterministic SMOTE for binary *y*.

    Oversamples the minority class up to ``target_ratio`` of the
    majority count by interpolating between random same-class k-NNs.
    Training-only utility (call before ``clf.fit`` on training data;
    never touches test folds).

    Parameters
    ----------
    X : (n, d) feature matrix
    y : (n,) binary labels {0, 1}
    target_ratio : desired minority / majority ratio (1.0 = balanced)
    k : number of neighbours to consider
    seed : deterministic RNG
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)

    n_counts = np.array([(y == c).sum() for c in (0, 1)], dtype=float)
    if n_counts.min() == 0 or len(X) < 2:
        return X, y  # nothing to balance

    minority = 0 if n_counts[0] < n_counts[1] else 1
    majority = 1 - minority
    n_min = int(n_counts[minority])
    n_maj = int(n_counts[majority])
    n_needed = int(n_maj * target_ratio) - n_min
    if n_needed <= 0:
        return X, y

    X_min = X[y == minority]
    k_eff = min(k, max(1, len(X_min) - 1))
    tree = cKDTree(X_min)
    _, k_idx = tree.query(X_min, k=k_eff + 1)

    new = np.empty((n_needed, X.shape[1]), dtype=np.float64)
    for i in range(n_needed):
        src = rng.integers(0, len(X_min))
        nbr = k_idx[src, rng.integers(1, k_eff + 1)]
        gap = rng.uniform(0, 1)
        new[i] = X_min[src] + gap * (X_min[nbr] - X_min[src])

    X_new = np.vstack([X, new])
    y_new = np.concatenate([y, np.full(n_needed, minority, dtype=int)])
    logger.info(
        "SMOTE +%d minority points (ratio=%.2f)", n_needed, target_ratio
    )
    return X_new, y_new


class AnomalyBaseline:
    """Unsupervised anomaly detector trained on barren cells only.

    Fit inside each training fold on X[train, y==0]; scored across the
    whole feature space. Provides a *lower bound* on what "deposit-like"
    means purely from background geology -- if a supervised model cannot
    outperform this, the problem is unresolved.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._if: IsolationForest | None = None
        self._fitted = False

    def fit(self, X_barren: np.ndarray) -> AnomalyBaseline:
        if len(X_barren) < 4:
            logger.warning("barren-only training set too small; skipping")
            self._if = IsolationForest(
                n_estimators=200, random_state=self.seed, contamination="auto"
            )
            self._fitted = False
            return self
        self._if = IsolationForest(
            n_estimators=300,
            max_samples="auto",
            contamination="auto",
            random_state=self.seed,
            n_jobs=-1,
        )
        self._if.fit(X_barren)
        self._fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or self._if is None:
            return np.zeros(len(X))
        return self._if.decision_function(X)

    def score_ap(
        self, X: np.ndarray, y: np.ndarray
    ) -> float | None:
        """Average-precision of anomaly score vs *y* (1=anomalous)."""
        if not self._fitted:
            return None
        scores = -self.decision_function(X)  # higher = more anomalous
        if y.sum() == 0 or y.sum() == len(y):
            return None
        return float(average_precision_score(y, scores))
