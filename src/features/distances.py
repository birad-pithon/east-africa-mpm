from __future__ import annotations

import math
from typing import Any

import geopandas as gpd
import numpy as np
from affine import Affine
from rasterio.crs import CRS
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

from src.preprocess.grid import GridSpec


def distance_to_features(
    grid: GridSpec,
    fault_gdf: gpd.GeoDataFrame | None = None,
    granite_gdf: gpd.GeoDataFrame | None = None,
    res_m: float = 30.0,
) -> dict[str, np.ndarray]:
    """Distance-to structures (metres) as float32 rasters.

    Parameters
    ----------
    grid : GridSpec
        Common analysis grid (EPSG:32736, 30 m).
    fault_gdf, granite_gdf : optional GeoDataFrames in any CRS
        Features will be reprojected and rasterized onto *grid*.

    Returns
    -------
    dict with keys 'dist_to_fault_m', 'dist_to_granite_m'
    """
    feats: dict[str, np.ndarray] = {}

    if fault_gdf is not None and len(fault_gdf):
        gdf = fault_gdf.to_crs(grid.crs)
        mask = rasterize(
            ((geom, 1) for geom in gdf.geometry),
            out_shape=grid.shape,
            transform=grid.transform,
            fill=0, dtype="uint8",
            all_touched=True,
        )
        # distance_transform_edt returns pixel distances; multiply by res
        # to convert to metres. Use 'sampling' param on newer scipy, else scale.
        dt = distance_transform_edt(1 - mask, sampling=[res_m, res_m])
        feats["dist_to_fault_m"] = dt.astype(np.float32)

    if granite_gdf is not None and len(granite_gdf):
        gdf = granite_gdf.to_crs(grid.crs)
        mask = rasterize(
            ((geom, 1) for geom in gdf.geometry),
            out_shape=grid.shape,
            transform=grid.transform,
            fill=0, dtype="uint8",
            all_touched=True,
        )
        dt = distance_transform_edt(1 - mask, sampling=[res_m, res_m])
        feats["dist_to_granite_m"] = dt.astype(np.float32)

    return feats


def distance_to_known_deposits_loo(
    grid: GridSpec,
    deposits_gdf: gpd.GeoDataFrame,
    res_m: float = 30.0,
) -> dict[str, np.ndarray]:
    """Leakage-safe distance-to-known-deposit prior raster (metres).

    A naive distance-to-deposits raster is a **label leak**: every
    positive training row lands exactly on a deposit cell and reads a
    distance of 0, letting the model separate labels trivially and
    inflating spatial-CV scores.  The leave-one-out (LOO) construction
    fixes the leak at its source:

    * non-deposit pixels -> Euclidean distance to the NEAREST deposit
      (identical to the naive raster; this is the legitimate prior
      "how close is this cell to anything already known");
    * each deposit's own cell -> distance to the nearest **OTHER**
      deposit (computed exactly with a cKDTree in projected metres),
      so no positive row ever references itself.

    Caveat (documented, accepted for P0): with spatial block CV the
    static raster is built from all labels, so a test-fold deposit also
    "sees" train-fold deposits.  That is the intended Bayesian prior
    behaviour at inference time and is not self-referential; residual
    spatial label leakage is bounded by the CV buffer and should be
    quantified with :func:`src.validate.evaluate_loso`.

    Parameters
    ----------
    grid : GridSpec
        Common analysis grid for the belt.
    deposits_gdf : GeoDataFrame
        Positive deposit points, any CRS (reprojected to ``grid.crs``).
    res_m : grid resolution in metres.

    Returns
    -------
    dict with key ``'dist_to_deposit_loo_m'`` -> float32 raster.

    Raises
    ------
    ValueError if ``deposits_gdf`` is empty.
    """
    if deposits_gdf is None or len(deposits_gdf) == 0:
        raise ValueError("deposits_gdf is empty - no distance prior possible")

    pts = deposits_gdf.to_crs(grid.crs)
    xy = np.column_stack([pts.geometry.x.to_numpy(),
                          pts.geometry.y.to_numpy()])

    # Rasterize deposit points onto the grid (points -> nearest cell).
    mask = rasterize(
        ((geom, 1) for geom in pts.geometry),
        out_shape=grid.shape,
        transform=grid.transform,
        fill=0, dtype="uint8",
        all_touched=False,
    )

    dist = (distance_transform_edt(1 - mask,
                                   sampling=[res_m, res_m])
            ).astype(np.float32)

    # LOO correction at deposit cells: distance to nearest OTHER deposit.
    if len(xy) >= 2:
        from scipy.spatial import cKDTree

        tree = cKDTree(xy)
        d_loo, _ = tree.query(xy, k=2)           # [self, nearest other]
        nearest_other = d_loo[:, 1].astype(np.float32)

        # point -> cell mapping (affine transform: col=(x-c)/a, row=(y-f)/e;
        # e < 0 so row increases southward, matching raster row order).
        cols_pt = ((xy[:, 0] - grid.transform.c) / grid.transform.a).astype(int)
        rows_pt = ((xy[:, 1] - grid.transform.f) / grid.transform.e).astype(int)
        h, w = grid.shape
        ok = (rows_pt >= 0) & (rows_pt < h) & (cols_pt >= 0) & (cols_pt < w)
        # Co-located records collapsing onto one cell: keep the MAXIMUM
        # LOO distance of the group (most conservative = least leaky).
        for r, c in zip(rows_pt[ok], cols_pt[ok]):
            sel = ok & (rows_pt == r) & (cols_pt == c)
            dist[r, c] = nearest_other[sel].max()
    else:
        import logging

        logging.getLogger(__name__).warning(
            "only 1 deposit - LOO undefined; falling back to naive "
            "distance raster (self-distance 0 will be visible)"
        )

    return {"dist_to_deposit_loo_m": dist}


def cell_size_m(
    transform: Affine,
    crs: Any,
    shape: tuple[int, int],
) -> tuple[float, float]:
    """``(cell height, cell width)`` in metres for a grid in *crs*.

    Projected grids (every belt config is UTM metres) are read straight off
    the affine. For a geographic grid the degrees are scaled by the grid's
    mean latitude, so metric thresholds stay comparable without having to
    reproject an otherwise fine raster.

    *crs* may be a ``rasterio.crs.CRS`` or a string such as ``"EPSG:32736"``.
    """
    cell_w, cell_h = abs(transform.a), abs(transform.e)
    if isinstance(crs, str):
        crs = CRS.from_user_input(crs)
    if crs is None or not getattr(crs, "is_geographic", False):
        return cell_h, cell_w
    top = transform.f
    bottom = transform.f + transform.e * shape[0]
    mid_lat = (top + bottom) / 2.0
    m_per_deg_lon = 111_320.0 * max(1e-6, math.cos(math.radians(mid_lat)))
    return cell_h * 110_540.0, cell_w * m_per_deg_lon


def distance_to_points_m(
    points_gdf: gpd.GeoDataFrame,
    transform: Affine,
    shape: tuple[int, int],
    crs: Any,
) -> np.ndarray:
    """Distance from every cell centre to the nearest point, in metres.

    Deliberately *not* the leave-one-out raster of
    :func:`distance_to_known_deposits_loo`. LOO reports a point's own cell as
    the distance to the **next** point, so with several points spaced apart a
    deposit inside a kilometre exclusion ring would read a large distance,
    pass the filter and be selected as "novel" ground. Here a point's own
    cell reads 0, which is what an exclusion radius needs.

    Parameters
    ----------
    points_gdf : positive points in any CRS (reprojected to *crs*).
    transform, shape, crs : grid definition, straight from an open raster.
        *crs* is used for the reprojection and to decide how cell size is
        converted to metres (:func:`cell_size_m`).

    Returns
    -------
    float32 array of *shape* holding metres.

    Raises
    ------
    ValueError if ``points_gdf`` holds no points.
    """
    if points_gdf is None or len(points_gdf) == 0:
        raise ValueError("points_gdf is empty - no distance field possible")

    if crs is not None and points_gdf.crs is not None:
        pts = points_gdf.to_crs(crs)
    else:
        pts = points_gdf

    seed = rasterize(
        ((geom, 1) for geom in pts.geometry),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )
    cell_h, cell_w = cell_size_m(transform, crs, shape)
    # `seed == 0` is True away from a point and False on it, so the EDT
    # returns 0 on the point cells and metres elsewhere.
    dist = distance_transform_edt(seed == 0, sampling=[cell_h, cell_w])
    return dist.astype(np.float32)
