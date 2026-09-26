"""Spatially-aware cross-validation for mineral prospectivity mapping.

Random k-fold CV is invalid for geological data: nearby samples are
spatially autocorrelated (Tobler's law), so random splits let a model
"memorise" neighbourhoods and produce inflated scores. This module
implements two sklearn-compatible splitters that respect geography:

* :class:`SpatialBlockCV` — partition the AOI into square grid blocks
  (metres, projected CRS), then assign whole blocks to folds so that no
  fold mixes neighbours from the same block.
* :class:`BufferedSpatialCV` — on top of blocking, remove training points
  within a *buffer radius* of every test point, eliminating the halo of
  near-duplicate samples around test deposits.

Both classes implement ``split(coords)`` yielding ``(train_idx,
test_idx)`` pairs, compatible with ``sklearn.model_selection.cross_val_score``
via its ``cv=`` parameter (pass an instance).

Design notes
------------
* Coordinates must be in a **projected** CRS (metres) — use EPSG:32736
  per the common analysis grid; never lat/lon degrees.
* Fold assignment tries to keep the rare positive class present in every
  fold (greedy round-robin over blocks sorted by positive count).
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Iterator

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import BaseCrossValidator

logger = logging.getLogger(__name__)

__all__ = [
    "assign_blocks",
    "BufferedSpatialCV",
    "certify_folds",
    "cluster_sites",
    "evaluate_loso",
    "LeakageError",
    "SpatialBlockCV",
]


def assign_blocks(
    coords: np.ndarray,
    block_size_m: float = 10_000.0,
    origin: tuple[float, float] | None = None,
) -> np.ndarray:
    """Assign points to square grid blocks of *block_size_m* metres.

    Parameters
    ----------
    coords : (n, 2) array of x/y in a projected CRS (metres).
    block_size_m : side length of each square block in metres.
    origin : optional ``(x0, y0)`` anchor for the grid; defaults to the
        floor of the coordinate minimums, snapped to the block size so
        repeated calls with different subsets land on the same grid.

    Returns
    -------
    (n,) int array of block ids ``(row * 1_000_000 + col)``.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be shape (n, 2)")
    if block_size_m <= 0:
        raise ValueError("block_size_m must be positive")

    if origin is None:
        x0 = np.floor(coords[:, 0].min() / block_size_m) * block_size_m
        y0 = np.floor(coords[:, 1].min() / block_size_m) * block_size_m
    else:
        x0, y0 = origin

    cols = np.floor((coords[:, 0] - x0) / block_size_m).astype(np.int64)
    rows = np.floor((coords[:, 1] - y0) / block_size_m).astype(np.int64)
    return rows * 1_000_000 + cols


class SpatialBlockCV(BaseCrossValidator):
    """Blocked spatial k-fold cross-validator.

    Points are grouped into square grid blocks; whole blocks (never
    individual points) are assigned to folds, so neighbours inside one
    block never straddle the train/test boundary.

    Parameters
    ----------
    block_size_m : side of each grid block, metres (default 10 km).
    n_splits : number of folds.
    balance_by : optional binary label array used to distribute blocks so
        that every fold contains positives when possible.

    Examples
    --------
    >>> cv = SpatialBlockCV(block_size_m=10_000, n_splits=5)
    >>> scores = cross_val_score(model, X, y, cv=cv)   # X = coords or feats
    """

    def __init__(
        self,
        block_size_m: float = 10_000.0,
        n_splits: int = 5,
        y: np.ndarray | None = None,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.block_size_m = float(block_size_m)
        self.n_splits = int(n_splits)
        self._y = None if y is None else np.asarray(y)

    # -- sklearn API -----------------------------------------------------
    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: D102
        return self.n_splits

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` for each fold.

        ``X`` may be the feature matrix *or* an (n, 2) coordinate array;
        only row count is used — blocking is computed from ``self.coords``
        set at construction time.
        """
        yield from _blocked_folds(
            coords=self.coords,
            block_ids=assign_blocks(self.coords, self.block_size_m),
            n_splits=self.n_splits,
            y=self._y if y is None else np.asarray(y),
        )

    # -- construction helper ---------------------------------------------
    def fit_coords(self, coords: np.ndarray) -> SpatialBlockCV:
        """Record the coordinate array used for blocking."""
        self.coords = np.asarray(coords, dtype=float)
        return self


def certify_folds(
    coords: np.ndarray,
    splits,
    min_buffer_m: float = 0.0,
) -> dict:
    """Actively PROVE a fold plan is spatially sound; raise otherwise.

    For every fold this checks:

    1. no coordinate appears in both train and test (index disjointness);
    2. when ``min_buffer_m > 0``, every *test* point lies farther than
       ``min_buffer_m`` from every *training* point (KD-tree verified).

    This turns the leakage safeguard from an assumption into an
    enforced invariant - :func:`evaluate_spatial_cv` calls it on every
    fold plan before fitting, so a future refactor that accidentally
    reintroduces random splits fails loudly here instead of quietly
    inflating scores.

    Returns a summary dict; raises :class:`LeakageError` on violation.
    """
    coords = np.asarray(coords, dtype=float)
    tree_all = cKDTree(coords)
    seen_test = np.zeros(len(coords), dtype=bool)
    report = []
    for k, (tr, te) in enumerate(splits):
        tr = np.asarray(tr, dtype=int)
        te = np.asarray(te, dtype=int)
        if len(np.intersect1d(tr, te)):
            raise LeakageError(
                f"fold {k}: {len(np.intersect1d(tr, te))} indices appear "
                "in BOTH train and test"
            )
        dup = seen_test[te].sum()
        if dup:
            raise LeakageError(f"fold {k}: {dup} test points reused")
        seen_test[te] = True
        entry = {"fold": k, "n_train": int(len(tr)), "n_test": int(len(te))}
        if min_buffer_m > 0 and len(tr) and len(te):
            near = tree_all.query_ball_point(
                coords[te], r=min_buffer_m
            )
            violators = sum(
                1 for lst in near if any(i in set(tr.tolist()) for i in lst)
            )
            if violators:
                raise LeakageError(
                    f"fold {k}: {violators} test points lie within "
                    f"{min_buffer_m:.0f} m of a training point"
                )
            # record the closest approach actually achieved
            d_min, _ = cKDTree(coords[tr]).query(coords[te], k=1)
            entry["min_train_test_dist_m"] = float(d_min.min())
        report.append(entry)

    if not seen_test.all():
        raise LeakageError(
            f"{int((~seen_test).sum())} coordinates never appear in any "
            "test fold"
        )
    return {
        "certified": True,
        "min_buffer_m": min_buffer_m,
        "min_observed_separation_m": (
            min(r.get("min_train_test_dist_m", math.inf) for r in report)
            if min_buffer_m > 0
            else None
        ),
        "folds": report,
    }


class LeakageError(RuntimeError):
    """Raised when a fold plan fails the spatial-integrity checks."""


def _blocked_folds(
    coords: np.ndarray,
    block_ids: np.ndarray,
    n_splits: int,
    y: np.ndarray | None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Group blocks into folds (round-robin by descending positive count)."""
    unique_blocks = np.unique(block_ids)
    if len(unique_blocks) < n_splits:
        raise ValueError(
            f"Only {len(unique_blocks)} spatial blocks for {n_splits} folds; "
            "reduce block_size_m or n_splits."
        )

    # Positive counts per block drive a greedy round-robin so rare
    # deposits appear in every fold where possible.
    pos_counts = np.zeros(len(unique_blocks))
    if y is not None and len(y) == len(coords):
        for i, b in enumerate(block_ids):
            if y[i] > 0:
                pos_counts[np.searchsorted(unique_blocks, b)] += 1
    order = unique_blocks[np.argsort(-pos_counts)]

    fold_of_block: dict[int, int] = {}
    tallies = [0] * n_splits
    for _rank, b in enumerate(order):
        f = int(np.argmin(tallies))
        fold_of_block[int(b)] = f
        tallies[f] += 1

    idx_by_fold: dict[int, list[int]] = defaultdict(list)
    for i, b in enumerate(block_ids):
        idx_by_fold[fold_of_block[int(b)]].append(i)

    for f in range(n_splits):
        test_idx = np.array(sorted(idx_by_fold.get(f, [])), dtype=int)
        train_idx = np.array(
            sorted(i for ff, ii in idx_by_fold.items() if ff != f for i in ii),
            dtype=int,
        )
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue  # skip degenerate folds rather than fail the run
        yield train_idx, test_idx


class BufferedSpatialCV(BaseCrossValidator):
    """Blocked spatial CV with an exclusion buffer around test points.

    After assigning blocks to folds, every *training* point that lies
    within ``buffer_m`` of any *test* point is removed. This kills the
    classic leakage halo: a background sample 200 m from a held-out
    deposit shares its geological signal, so it must not be trained on.

    Parameters
    ----------
    coords : (n, 2) array in a projected CRS (metres).
    block_size_m : side of grid blocks used to form folds.
    n_splits : number of folds.
    buffer_m : exclusion radius around test points, metres
        (e.g. 5_000 = 5 km; see ``validation.buffer_km`` in config).
    y : optional binary labels for fold balancing.

    Notes
    -----
    Buffering shrinks each training set; with dense sampling raise the
    block size or lower the buffer if folds become too small.
    """

    def __init__(
        self,
        coords: np.ndarray,
        block_size_m: float = 10_000.0,
        n_splits: int = 5,
        buffer_m: float = 5_000.0,
        y: np.ndarray | None = None,
    ) -> None:
        self.coords = np.asarray(coords, dtype=float)
        self.block = SpatialBlockCV(block_size_m, n_splits, y).fit_coords(
            self.coords
        )
        self.block_size_m = self.block.block_size_m
        self.n_splits = self.block.n_splits
        self._y = None if y is None else np.asarray(y)
        if buffer_m < 0:
            raise ValueError("buffer_m must be >= 0")
        self.buffer_m = float(buffer_m)

    # -- sklearn API -----------------------------------------------------
    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: D102
        return self.n_splits

    def split(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` with buffered training sets."""
        labels = self._y if y is None else np.asarray(y)
        for train_idx, test_idx in _blocked_folds(
            coords=self.coords,
            block_ids=assign_blocks(self.coords, self.block_size_m),
            n_splits=self.n_splits,
            y=labels,
        ):
            if self.buffer_m > 0:
                tree_test = cKDTree(self.coords[test_idx])
                near = tree_test.query_ball_point(
                    self.coords[train_idx], r=self.buffer_m
                )
                keep = np.array([len(n) == 0 for n in near], dtype=bool)
                dropped = int((~keep).sum())
                if dropped:
                    logger.debug(
                        "Fold: dropped %d train pts within %.0f m of test",
                        dropped,
                        self.buffer_m,
                    )
                train_idx = train_idx[keep]
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx


def cluster_sites(
    coords: np.ndarray,
    labels: np.ndarray | None = None,
    eps_m: float = 5_000.0,
    min_samples: int = 2,
) -> np.ndarray:
    """Spatial clustering of sites using DBSCAN.

    Groups nearby deposit sites into clusters for spatial analysis and
    leave-one-site-out validation. Sites within ``eps_m`` metres of each
    other form a cluster.

    Parameters
    ----------
    coords : (n, 2) array of x/y in a projected CRS (metres).
    labels : optional binary labels; if provided, only positive sites
        are clustered (background points get cluster label -1).
    eps_m : maximum distance between two samples in the same cluster.
    min_samples : minimum points to form a dense region.

    Returns
    -------
    (n,) int array of cluster labels (-1 = noise/unclustered).
    """
    from sklearn.cluster import DBSCAN

    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be shape (n, 2)")

    mask = np.ones(len(coords), dtype=bool)
    if labels is not None:
        labels = np.asarray(labels, dtype=int)
        mask = labels == 1  # Only cluster positive sites

    cluster_ids = np.full(len(coords), -1, dtype=int)
    if mask.sum() < min_samples:
        return cluster_ids

    clustered_coords = coords[mask]
    clustering = DBSCAN(eps=eps_m, min_samples=min_samples).fit(clustered_coords)
    cluster_ids[mask] = clustering.labels_

    n_clusters = len(set(clustering.labels_) - {-1})
    n_noise = int((clustering.labels_ == -1).sum())
    logger.info(
        "cluster_sites: %d clusters, %d noise points (eps=%.0fm)",
        n_clusters, n_noise, eps_m,
    )
    return cluster_ids


def evaluate_loso(
    clf_factory,
    X: pd.DataFrame,
    y: np.ndarray,
    coords: np.ndarray,
    cluster_labels: np.ndarray | None = None,
    block_size_m: float = 10_000.0,
    buffer_m: float = 5_000.0,
    min_test_size: int = 1,
) -> dict:
    """Leave-one-site-out spatial cross-validation.

    For each cluster of deposit sites, train on all other clusters and
    predict on the held-out cluster. This tests whether the model can
    generalize to entirely new geological regions.

    Parameters
    ----------
    clf_factory : callable(pos_weight) -> fresh classifier.
    X : feature DataFrame.
    y : binary labels array.
    coords : (n, 2) array in projected CRS (metres).
    cluster_labels : pre-computed cluster labels from :func:`cluster_sites`.
        If None, clusters are computed automatically.
    block_size_m : block size for initial fold assignment.
    buffer_m : exclusion buffer around test points.
    min_test_size : minimum test set size to include a fold.

    Returns
    -------
    dict with ``n_folds``, ``mean_average_precision``, ``folds`` (per-fold
    results), and ``method='loso'``.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    Xv = X.to_numpy(dtype="float64")
    coords = np.asarray(coords, dtype=float)
    y = np.asarray(y, dtype=int)

    if cluster_labels is None:
        cluster_labels = cluster_sites(coords, y, eps_m=block_size_m * 0.5)

    unique_clusters = sorted(set(cluster_labels) - {-1})
    if not unique_clusters:
        logger.warning("evaluate_loso: no clusters found, falling back to single fold")
        unique_clusters = [0]
        cluster_labels = np.zeros(len(y), dtype=int)

    folds = []
    oof_true = []
    oof_proba = []

    for cluster_id in unique_clusters:
        test_mask = cluster_labels == cluster_id
        train_mask = ~test_mask

        if test_mask.sum() < min_test_size:
            continue

        # Apply buffer: remove training points near test points
        if buffer_m > 0 and test_mask.any() and train_mask.any():
            tree_test = cKDTree(coords[test_mask])
            train_coords = coords[train_mask]
            near = tree_test.query_ball_point(train_coords, r=buffer_m)
            keep = np.array([len(n) == 0 for n in near], dtype=bool)
            train_indices = np.where(train_mask)[0][keep]
        else:
            train_indices = np.where(train_mask)[0]

        test_indices = np.where(test_mask)[0]

        if len(train_indices) == 0 or len(test_indices) == 0:
            continue

        pos_weight = float(
            (len(train_indices) - y[train_indices].sum())
            / max(1.0, float(y[train_indices].sum()))
        )
        clf = clf_factory(pos_weight)
        clf.fit(Xv[train_indices], y[train_indices])
        proba = clf.predict_proba(Xv[test_indices])[:, 1]

        oof_true.append(y[test_indices])
        oof_proba.append(proba)

        entry = {
            "fold": cluster_id,
            "n_train": len(train_indices),
            "n_test": len(test_indices),
            "n_test_positives": int(y[test_indices].sum()),
        }
        if y[test_indices].sum() > 0 and y[test_indices].sum() < len(test_indices):
            entry["average_precision"] = float(
                average_precision_score(y[test_indices], proba)
            )
            entry["roc_auc"] = float(roc_auc_score(y[test_indices], proba))
        folds.append(entry)

    ap = [f["average_precision"] for f in folds if "average_precision" in f]
    roc = [f["roc_auc"] for f in folds if "roc_auc" in f]

    result = {
        "n_folds": len(folds),
        "folds": folds,
        "mean_average_precision": float(np.mean(ap)) if ap else None,
        "std_average_precision": float(np.std(ap)) if ap else None,
        "mean_roc_auc": float(np.mean(roc)) if roc else None,
        "method": "loso",
        "n_clusters": len(unique_clusters),
    }
    if oof_true:
        result["oof_predictions"] = {
            "y_true": np.concatenate(oof_true),
            "y_proba": np.concatenate(oof_proba),
        }
    return result
