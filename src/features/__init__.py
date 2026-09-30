"""Commodity-specific feature engineering.

Submodules:
* :mod:`src.features.spectral`  -- Sentinel-2 band-ratio indices
* :mod:`src.features.terrain`   -- DEM-derived slope/aspect/TPI/relief/plateau-flatness
* :mod:`src.features.distances` -- Euclidean distance transforms
* :mod:`src.features.magnetic`  -- magnetic gradient features
* :mod:`src.features.lithology` -- categorical lithology encoding
* :mod:`src.features.geology_features` -- lithology + contact rasters on the grid
* :mod:`src.features.stack`     -- multi-band stack orchestration
"""
from __future__ import annotations

import logging

from src.features.distances import (
    distance_to_features,
    distance_to_known_deposits_loo,
)
from src.features.geology_features import build_geology_features
from src.features.lithology import encode_lithology
from src.features.magnetic import magnetic_gradient_features
from src.features.priors import (
    apply_logit_prior,
    apply_prior_to_raster,
    distance_decay_logit,
)
from src.features.spectral import compute_spectral_features, spectral_indices
from src.features.stack import build_feature_stack
from src.features.terrain import (
    dem_features,
    plateau_index,
    relative_relief,
    slope_aspect,
    tpi,
)

logger = logging.getLogger(__name__)

__all__ = [
    "spectral_indices", "compute_spectral_features",
    "dem_features", "slope_aspect", "tpi", "relative_relief", "plateau_index",
    "distance_to_features", "distance_to_known_deposits_loo",
    "distance_decay_logit", "apply_logit_prior", "apply_prior_to_raster",
    "magnetic_gradient_features",
    "encode_lithology",
    "build_geology_features",
    "build_feature_stack",
]
