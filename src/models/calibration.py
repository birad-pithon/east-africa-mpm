"""Probability-calibration reporting for prospectivity models.

A ranked list can have great average precision yet badly miscalibrated
probabilities — fatal for MPM, where ``P(deposit)`` is read as a
ground-selection prior ("a 0.4 cell is twice as promising as a 0.2
cell"). This module quantifies calibration from **out-of-fold** CV
predictions:

* Brier score (lower is better; 0.25 = uninformative for balanced data);
* Expected Calibration Error (ECE) over equal-mass bins;
* a reliability table (predicted mean vs observed positive fraction per
  bin) and a matplotlib reliability diagram PNG.

Usage::

    from src.models.calibration import calibration_report, plot_reliability
    report = calibration_report(y_true, y_proba)
    plot_reliability(y_true, y_proba, "outputs/models/reliability.png")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["calibration_report", "plot_reliability"]


def _bin_sums(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    edges: np.ndarray,
) -> list[dict]:
    """Per-bin counts / mean prediction / observed positive fraction."""
    idx = np.clip(np.searchsorted(edges, y_proba, side="right") - 1,
                  0, len(edges) - 2)
    bins: list[dict] = []
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        bins.append({
            "bin_lo": float(edges[b]),
            "bin_hi": float(edges[b + 1]),
            "n": n,
            "mean_predicted": float(y_proba[m].mean()) if n else None,
            "fraction_positive": float(y_true[m].mean()) if n else None,
        })
    return bins


def calibration_report(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    min_bin_n: int = 10,
) -> dict:
    """Calibration summary from out-of-fold predictions.

    Parameters
    ----------
    y_true : binary labels (0/1).
    y_proba : predicted P(class=1), same length.
    n_bins : number of equal-mass (quantile) probability bins.
    min_bin_n : bins with fewer points are excluded from the ECE sum
        (their empirical fraction is too noisy to trust).

    Returns
    -------
    dict with ``brier``, ``ece``, ``n``, ``bins`` (reliability table) and
    ``max_abs_gap`` (largest |predicted - observed| among usable bins).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    if y_true.shape != y_proba.shape or y_true.size == 0:
        raise ValueError("y_true and y_proba must be same non-empty shape")
    if not np.isfinite(y_proba).all():
        raise ValueError("y_proba contains NaN/inf")
    if y_proba.min() < 0 or y_proba.max() > 1:
        raise ValueError("y_proba must lie in [0, 1]")

    brier = float(np.mean((y_proba - y_true) ** 2))

    # equal-mass bin edges; de-duplicate for heavily tied probabilities
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(y_proba, quantiles))
    if len(edges) < 2:                       # all probabilities identical
        edges = np.array([edges[0] - 0.5, edges[0] + 0.5])

    bins = _bin_sums(y_true, y_proba, edges)

    ece = 0.0
    max_gap = 0.0
    for b in bins:
        if b["n"] >= min_bin_n:
            gap = abs(b["mean_predicted"] - b["fraction_positive"])
            ece += b["n"] / len(y_true) * gap
            max_gap = max(max_gap, gap)

    return {
        "n": int(len(y_true)),
        "n_positives": int(y_true.sum()),
        "n_bins_requested": int(n_bins),
        "n_bins_usable": int(sum(b["n"] >= min_bin_n for b in bins)),
        "brier": brier,
        "ece": float(ece),
        "max_abs_gap": float(max_gap),
        "bins": bins,
    }


def plot_reliability(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_path: str | Path,
    n_bins: int = 10,
    title: str = "Reliability diagram (out-of-fold)",
) -> Path:
    """Write a reliability diagram PNG; returns the path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = calibration_report(y_true, y_proba, n_bins=n_bins)
    usable = [b for b in report["bins"] if b["n"] > 0]

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    xs = [b["mean_predicted"] for b in usable]
    ys = [b["fraction_positive"] for b in usable]
    sizes = [max(12, 3.0 * b["n"]) for b in usable]
    ax.scatter(xs, ys, s=sizes, color="tab:blue", zorder=3,
               label="observed")
    for b in usable:
        ax.plot([b["mean_predicted"], b["mean_predicted"]],
                [b["mean_predicted"], b["fraction_positive"]],
                color="tab:red", lw=1, alpha=0.6, zorder=2)
    ax.set_xlabel("mean predicted P(deposit)")
    ax.set_ylabel("observed fraction")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.text(0.02, 0.98,
            f"Brier={report['brier']:.3f}  ECE={report['ece']:.3f}",
            transform=ax.transAxes, va="top", fontsize=9)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Reliability diagram -> %s", out_path)
    return out_path


def save_report(report: dict, out_path: str | Path) -> Path:
    """Persist a calibration report as JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_path
