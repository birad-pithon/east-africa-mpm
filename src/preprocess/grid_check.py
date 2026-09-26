"""Grid-alignment validation: fail loudly on resolution/CRS mismatches.

Implements SOW safeguard #4 (Resolution and alignment mismatches).

Feature layers come from different sensors and providers (Sentinel-2
10-20 m, SRTM 30 m, geology vectors rasterised on the shared grid).
If any layer is silently misaligned -- shifted origin, different CRS,
resampled to a different pixel size -- then ``sample_rasters`` happily
stacks values from *different locations* into one feature row and the
model trains on geologically meaningless combinations.

:func:`assert_common_grid` compares every raster's CRS, affine
transform, width and height against a reference (by default the first
layer) and raises :class:`GridMismatchError` with a per-layer diagnostic
if any property differs beyond floating-point tolerance. It is wired
into ``build_training_table`` and ``predict_raster`` so a misaligned
layer aborts the run before any feature is sampled.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio

logger = logging.getLogger(__name__)

__all__ = ["GridMismatchError", "grid_signature", "assert_common_grid"]

# tolerance for affine-parameter comparison (metres on the grid)
_TOL = 1e-6


class GridMismatchError(ValueError):
    """Raised when feature rasters do not share a common analysis grid."""


def grid_signature(path: Path | str) -> dict:
    """Return the grid-defining properties of a GeoTIFF."""
    with rasterio.open(path) as src:
        return {
            "crs": str(src.crs),
            "transform": tuple(float(v) for v in src.transform[:6]),
            "width": int(src.width),
            "height": int(src.height),
            "res": tuple(float(r) for r in src.res),
        }


def _transform_matches(a: tuple, b: tuple, tol: float = _TOL) -> bool:
    return bool(np.allclose(a, b, rtol=0.0, atol=tol))


def assert_common_grid(
    raster_paths: list[Path | str],
    reference: Path | str | None = None,
) -> dict[str, dict]:
    """Verify all rasters share CRS, transform and dimensions.

    Parameters
    ----------
    raster_paths : paths to check (must be non-empty).
    reference : optional path to compare against; defaults to the first
        raster in *raster_paths*.

    Returns
    -------
    dict mapping path -> signature for all rasters (on success).

    Raises
    ------
    GridMismatchError
        with a human-readable diff for every offending layer.
    """
    paths = [Path(p) for p in raster_paths]
    if not paths:
        raise ValueError("no raster paths given")

    ref_path = Path(reference) if reference else paths[0]
    ref = grid_signature(ref_path)
    signatures = {str(ref_path): ref}

    problems: list[str] = []
    for p in paths:
        sig = grid_signature(p)
        signatures[str(p)] = sig
        diffs = []
        if sig["crs"] != ref["crs"]:
            diffs.append(f"crs {sig['crs']} != {ref['crs']}")
        if not _transform_matches(sig["transform"], ref["transform"]):
            diffs.append(
                f"origin/transform {sig['transform']} != {ref['transform']}"
            )
        if sig["width"] != ref["width"] or sig["height"] != ref["height"]:
            diffs.append(
                f"shape ({sig['width']}x{sig['height']}) != "
                f"({ref['width']}x{ref['height']})"
            )
        if diffs:
            problems.append(f"  {p.name}: " + "; ".join(diffs))

    if problems:
        msg = (
            f"{len(problems)}/{len(paths)} rasters do not match the "
            f"analysis grid (reference: {ref_path.name}):\n"
            + "\n".join(problems)
            + "\nRe-run src.preprocess.main to reproject all layers "
              "onto the shared GridSpec before training."
        )
        logger.error(msg)
        raise GridMismatchError(msg)

    logger.info("grid check OK: %d rasters share the analysis grid",
                len(paths))
    return signatures
