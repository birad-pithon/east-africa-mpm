"""Uncertainty-aware active-learning candidate selection.

Field confirmation is the expensive step. Rather than only ranking the
highest-probability cells (pure exploitation, which re-verifies known
prospective ground), this module mixes in **exploration** acquisitions
targeting the model's most *uncertain* cells:

* ``margin``  — cells closest to ``p = 0.5`` (the decision boundary);
* ``entropy`` — highest prediction entropy (maximised at 0.5).

Those are the cells a single field visit resolves the most information
about, and every verification feeds back through
``src.labels.field_update`` (Safeguard 5), closing the loop. Output rows
carry a ``strategy`` column so the field crew knows whether a target is
exploitation or exploration.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from src.predict.rank import _licence_cell_mask, load_licence_mask

logger = logging.getLogger(__name__)

__all__ = ["rank_candidates_active"]

_EXPLOIT = "top_prob"
_STRATEGIES = (_EXPLOIT, "margin", "entropy")


def _score(arr: np.ndarray, strategy: str) -> np.ndarray:
    """Per-strategy score for the full array; higher = better."""
    if strategy == _EXPLOIT:
        return arr
    if strategy == "margin":
        return -(np.abs(arr - 0.5))          # closest to boundary first
    if strategy == "entropy":
        p = np.clip(arr, 1e-9, 1 - 1e-9)
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))
    raise ValueError(f"unknown strategy '{strategy}' ({_STRATEGIES})")


def _decluster(rows, cols, scores, max_n, spacing_cells):
    """Greedy top-N picks keeping ``spacing_cells`` Chebyshev separation."""
    order = np.argsort(scores)[::-1]
    picked_r = []
    picked_c = []
    picked_i = []
    min_sep = max(int(spacing_cells), 1)
    for i in order:
        r, c = int(rows[i]), int(cols[i])
        if all(max(abs(r - pr), abs(c - pc)) > min_sep
               for pr, pc in zip(picked_r, picked_c, strict=False)):
            picked_r.append(r)
            picked_c.append(c)
            picked_i.append(i)
        if len(picked_i) >= max_n:
            break
    return picked_i


def rank_candidates_active(
    proba_path: Path | str,
    n_exploit: int = 20,
    n_explore: int = 20,
    explore_strategy: str = "margin",
    licence_path: Path | str | None = None,
    min_prob: float = 0.0,
    spacing_cells: int = 2,
) -> pd.DataFrame:
    """Exploit + explore candidate cells from a probability raster.

    Parameters
    ----------
    n_exploit : exploitation picks (highest probability).
    n_explore : exploration picks (most uncertain). The explore picks are
        spatial-declustered from the exploit anchors.
    explore_strategy : ``margin`` or ``entropy``.
    Others as in :func:`src.predict.rank.rank_candidates`.

    Returns
    -------
    DataFrame with columns ``rank, row, col, prob, lon, lat, strategy,
    score`` (strategy in ``{top_prob, margin, entropy}``).
    """
    if explore_strategy not in ("margin", "entropy"):
        raise ValueError(
            f"explore_strategy must be 'margin' or 'entropy', "
            f"got '{explore_strategy}'")

    with rasterio.open(proba_path) as src:
        arr = src.read(1).astype("float64")
        tf = src.transform
        crs = src.crs

    licences = (
        load_licence_mask(licence_path, crs) if licence_path else None
    )
    lic_mask = _licence_cell_mask(licences, tf, src.width, src.height) \
        if licences is not None else None

    valid = np.isfinite(arr) & (arr >= min_prob)
    if lic_mask is not None:
        valid &= ~lic_mask

    rows, cols = np.where(valid)
    if not len(rows):
        logger.warning("no valid cells after masking/prob filter")
        return pd.DataFrame(columns=["rank", "row", "col", "prob", "lon",
                                     "lat", "strategy", "score"])

    picked_i: list[int] = []
    picked_strat: list[str] = []
    picked_score: list[float] = []

    # exploitation first (establishes the anchor neighbourhoods)
    full_scores = _score(arr, _EXPLOIT)
    exp_i = _decluster(rows, cols, full_scores[rows, cols], n_exploit,
                       spacing_cells)
    picked_i += list(exp_i)
    picked_strat += [_EXPLOIT] * len(exp_i)
    picked_score += [float(arr[rows[i], cols[i]]) for i in exp_i]

    # exploration: most uncertain cells, away from exploit anchors
    score = _score(arr, explore_strategy)
    seen = set(exp_i)
    cand = [i for i in range(len(rows)) if i not in seen]
    if cand:
        sub = np.asarray(cand)
        order = np.argsort(score[rows[sub], cols[sub]])[::-1]
        min_sep = max(int(spacing_cells), 1)
        for i in order:
            idx = sub[i]
            r, c = int(rows[idx]), int(cols[idx])
            if all(max(abs(r - pr), abs(c - pc)) > min_sep
                   for pr, pc in zip(rows[picked_i], cols[picked_i], strict=False)):
                picked_i.append(idx)
                picked_strat.append(explore_strategy)
                picked_score.append(float(score[rows[idx], cols[idx]]))
            if sum(s == explore_strategy for s in picked_strat) >= n_explore:
                break

    sel_rows = rows[picked_i]
    sel_cols = cols[picked_i]
    xs, ys = rasterio.transform.xy(tf, sel_rows.tolist(), sel_cols.tolist(),
                                   offset="center")
    pts_ll = gpd.GeoSeries(gpd.points_from_xy(xs, ys),
                           crs=crs).to_crs("EPSG:4326")

    out = pd.DataFrame({
        "rank": np.arange(1, len(picked_i) + 1),
        "row": sel_rows,
        "col": sel_cols,
        "prob": arr[sel_rows, sel_cols],
        "lon": pts_ll.x.to_numpy(),
        "lat": pts_ll.y.to_numpy(),
        "strategy": picked_strat,
        "score": picked_score,
    })
    logger.info("active candidates: %d exploit + %d explore ('%s')",
                sum(s == _EXPLOIT for s in picked_strat),
                sum(s == explore_strategy for s in picked_strat),
                explore_strategy)
    return out


def main() -> None:
    """CLI entry point for active-learning candidate selection.

    Example::

        python -m src.predict.active \\
            --proba outputs/models/proba_tin_tungsten_tantalum_rf.tif \\
            --group tin_tungsten_tantalum --n-exploit 10 --n-explore 15
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Select exploit + explore (uncertainty) field targets")
    parser.add_argument("--proba", required=True)
    parser.add_argument("--group", default="")
    parser.add_argument("--n-exploit", type=int, default=20)
    parser.add_argument("--n-explore", type=int, default=20)
    parser.add_argument("--explore-strategy", default="margin",
                        choices=("margin", "entropy"))
    parser.add_argument("--licences", default=None)
    parser.add_argument("--min-prob", type=float, default=0.0)
    parser.add_argument("--spacing-cells", type=int, default=2)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")

    df = rank_candidates_active(
        args.proba,
        n_exploit=args.n_exploit,
        n_explore=args.n_explore,
        explore_strategy=args.explore_strategy,
        licence_path=args.licences,
        min_prob=args.min_prob,
        spacing_cells=args.spacing_cells,
    )
    if not len(df):
        print("no candidates returned")
        return

    from src.predict.viz import candidates_to_geojson

    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs/maps")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"active_{args.group}" if args.group else "active"
    geojson = candidates_to_geojson(df, out_dir / f"{stem}.geojson",
                                    group=args.group or "")
    print(f"\nOK {len(df)} active-learning candidates -> {geojson}")
    print(df[["rank", "prob", "lon", "lat", "strategy"]].head(15)
          .to_string(index=False))


if __name__ == "__main__":
    main()
