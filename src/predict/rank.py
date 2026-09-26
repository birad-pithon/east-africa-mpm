"""Candidate-cell ranking: probability raster -> top-N open-ground targets.

Implements SOW item 9: score every cell of a commodity probability
raster (from :mod:`src.models.predict`), optionally mask out cells that
fall inside existing exploration/mining licence areas, and return the
top-N highest-probability **unlicensed** cells as both a table and an
exportable GeoJSON.

A cell is treated as "unexplored ground" when:
* its probability is valid (not NaN), and
* it does not intersect any licence polygon (if a licence layer is given).

Usage::

    from src.predict.rank import rank_candidates
    df = rank_candidates("outputs/models/proba_....tif", n=50)

CLI::

    python -m src.predict.main --proba outputs/maps/proba_x.tif \
        --n 50 [--licences data/processed/licences.geojson]
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

logger = logging.getLogger(__name__)

__all__ = ["load_licence_mask", "rank_candidates"]


def load_licence_mask(licence_path: Path | str,
                      proba_crs: str) -> gpd.GeoDataFrame | None:
    """Load licence polygons reprojected to the probability raster CRS."""
    p = Path(licence_path)
    if not p.exists():
        logger.warning("licence file %s not found - no masking applied", p)
        return None
    gdf = gpd.read_file(p)
    if gdf.crs is None:
        raise ValueError(f"{p} has no CRS - cannot mask reliably")
    out = gdf.to_crs(proba_crs)
    logger.info("loaded %d licence polygons -> %s", len(out), proba_crs)
    return out


def _licence_cell_mask(
    licences: gpd.GeoDataFrame | None,
    grid_transform,
    width: int,
    height: int,
) -> np.ndarray | None:
    """Boolean array, True where the cell intersects any licence."""
    if licences is None or not len(licences):
        return None

    from rasterio.features import rasterize

    mask = rasterize(
        ((geom, 1) for geom in licences.geometry if geom is not None),
        out_shape=(height, width),
        transform=grid_transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )
    return mask.astype(bool)


def rank_candidates(
    proba_path: Path | str,
    n: int = 50,
    licence_path: Path | str | None = None,
    min_prob: float = 0.0,
    spacing_cells: int = 2,
) -> pd.DataFrame:
    """Rank the top-N unlicensed candidate cells by deposit probability.

    Parameters
    ----------
    proba_path : P(deposit) GeoTIFF from ``src.models.predict``.
    n : number of candidates to return.
    licence_path : optional vector layer of licence polygons; cells that
        fall inside any polygon are excluded ("exclude licensed ground").
    min_prob : discard cells below this probability before ranking.
    spacing_cells : greedy minimum separation between picked cells
        (enforces one target per neighbourhood instead of N clustered
        pixels over the same anomaly).

    Returns
    -------
    DataFrame with columns
        rank, row, col, prob, lon, lat  (EPSG:4326 cell centres).
    """
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
        removed = int((valid & lic_mask).sum())
        valid &= ~lic_mask
        logger.info("licence mask removed %d valid cells", removed)

    rows, cols = np.where(valid)
    probs = arr[rows, cols]

    order = np.argsort(probs)[::-1]
    picked: list[int] = []
    min_sep = max(int(spacing_cells), 1)
    # spatial-declustering: skip a cell if a higher-ranked pick already
    # sits within ``spacing_cells`` in BOTH axes (Chebyshev distance)
    picked_rc: list[tuple[int, int]] = []
    for i in order:
        r, c = int(rows[i]), int(cols[i])
        if all(max(abs(r - pr), abs(c - pc)) > min_sep
               for pr, pc in picked_rc):
            picked.append(i)
            picked_rc.append((r, c))
        if len(picked) >= n:
            break

    sel_rows = rows[picked]
    sel_cols = cols[picked]
    xs, ys = rasterio.transform.xy(tf, sel_rows.tolist(), sel_cols.tolist(),
                                   offset="center")
    pts_ll = gpd.GeoSeries(
        gpd.points_from_xy(xs, ys), crs=crs
    ).to_crs("EPSG:4326")

    out = pd.DataFrame({
        "rank": np.arange(1, len(picked) + 1),
        "row": sel_rows,
        "col": sel_cols,
        "prob": probs[picked],
        "lon": pts_ll.x.to_numpy(),
        "lat": pts_ll.y.to_numpy(),
    })
    logger.info("ranked %d candidates (spacing=%d cells)",
                len(out), min_sep)
    return out
