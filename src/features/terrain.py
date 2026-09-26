"""Terrain feature re-exports.

DEm-derived features live in ``src.preprocess.dem_features`` (they
pre-date the feature-engineering subpackage). This thin shim re-exposes
them here so callers can import from the features layer:

    from src.features import dem_features, slope_aspect, tpi, ...

"""
from __future__ import annotations

from src.preprocess.dem_features import (
    dem_features,
    plateau_index,
    relative_relief,
    slope_aspect,
    tpi,
)

__all__ = [
    "slope_aspect",
    "tpi",
    "relative_relief",
    "plateau_index",
    "dem_features",
]
