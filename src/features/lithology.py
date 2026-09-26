"""Lithology encoding for gridded rock-type mapping.

Converts a polygon lithology layer into a numeric code raster aligned
to the common grid (used as a categorical feature for RF/XGBoost/LightGBM
classification). Each unique rock type string → integer code.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize

from src.preprocess.grid import GridSpec


def encode_lithology(
    litho_gdf: gpd.GeoDataFrame,
    grid: GridSpec,
    field: str = "rock_type",
) -> np.ndarray:
    """Rasterize lithology polygons into integer codes.

    Parameters
    ----------
    litho_gdf : GeoDataFrame with a categorical *field* (e.g. 'rock_type')
    grid : destination GridSpec
    field : column name containing the lithology class strings

    Returns
    -------
    uint8 array of same shape as grid; 0 = nodata, 1..N = lithology codes
    (mapping stored in the returned array's metadata via caller).
    """
    if field not in litho_gdf.columns:
        # auto-pick first string column
        str_cols = litho_gdf.select_dtypes(include="object").columns
        field = next(c for c in str_cols if c != "geometry")

    gdf = litho_gdf.to_crs(grid.crs)
    classes = sorted(gdf[field].unique())
    class_to_code = {c: i + 1 for i, c in enumerate(classes)}

    out = np.zeros(grid.shape, dtype=np.uint8)
    shapes = (
        (geom, class_to_code[val])
        for _, row in gdf.iterrows()
        for geom, val in [(row.geometry, row[field])]
    )
    out = rasterize(
        shapes,
        out_shape=grid.shape,
        transform=grid.transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    return out


def lithology_class_map(litho_gdf: gpd.GeoDataFrame,
                       field: str = "rock_type") -> dict[str, int]:
    """Return {class_name: integer_code} mapping."""
    classes = sorted(litho_gdf[field].unique())
    return {c: i + 1 for i, c in enumerate(classes)}
