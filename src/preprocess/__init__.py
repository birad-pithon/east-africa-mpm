"""Preprocessing: grid alignment, rasterization, terrain features."""
from src.preprocess.dem_features import dem_features, plateau_index
from src.preprocess.grid import GridSpec
from src.preprocess.grid_check import (
    GridMismatchError,
    assert_common_grid,
    grid_signature,
)
from src.preprocess.rasterize import (
    extract_at_points,
    label_rasters,
    rasterize_points,
)

__all__ = [
    "GridSpec",
    "dem_features",
    "plateau_index",
    "rasterize_points",
    "label_rasters",
    "extract_at_points",
    "assert_common_grid",
    "grid_signature",
    "GridMismatchError",
]
