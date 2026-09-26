"""Export ranked candidates + interactive prospectivity maps (SOW item 8).

* :func:`candidates_to_geojson` — top-N table as a point layer.
* :func:`interactive_map` — leafmap/folium HTML with the probability
  raster as a coloured overlay plus candidate markers. Imports lazily
  so the rest of the pipeline never requires leafmap at runtime.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["candidates_to_geojson", "render_interactive_map"]


def candidates_to_geojson(
    candidates: pd.DataFrame,
    out_path: Path | str,
    group: str = "",
) -> Path:
    """Write the ranked-candidate table as WGS-84 point GeoJSON."""
    gdf = gpd.GeoDataFrame(
        candidates.copy(),
        geometry=gpd.points_from_xy(candidates["lon"], candidates["lat"]),
        crs="EPSG:4326",
    )
    if group:
        gdf["group"] = group
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(p, driver="GeoJSON")
    logger.info("candidates -> %s (%d pts)", p, len(gdf))
    return p


def render_interactive_map(
    proba_path: Path | str,
    candidates_geojson: Path | str | None,
    out_html: Path | str,
    group: str = "prospectivity",
) -> Path:
    """Leafmap HTML: probability raster overlay + candidate markers.

    The raster is repainted into an RGBA PNG tile overlay (viridis-like
    ramp on probability, transparent where NaN), which folium can serve
    without a tiled tile-server backend.
    """
    import tempfile

    import folium  # noqa: deferred - item-8 only path
    import rasterio
    from rasterio.warp import transform_bounds

    with rasterio.open(proba_path) as src:
        arr = src.read(1).astype("float64")
        b = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        crs = src.crs

    # reproject probability to EPSG:4326 for the web overlay
    dst_arr = np.full((256, 256), np.nan)
    try:
        from affine import Affine
        from rasterio.warp import Resampling, reproject

        west, south, east, north = b
        dst_tf = Affine((east - west) / 256, 0, west,
                        0, -(north - south) / 256, north)
        reproject(
            source=arr,
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=crs,
            dst_transform=dst_tf,
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
        )
    except Exception as exc:                    # pragma: no cover
        logger.warning("overlay reprojection failed (%s); using extent "
                       "box only", exc)

    # colour ramp -> RGBA PNG in a temp file
    valid = np.isfinite(dst_arr)
    png_path = None
    if valid.any():
        from PIL import Image

        v = dst_arr[valid]
        lo, hi = float(v.min()), float(v.max())
        norm = np.nan_to_num((dst_arr - lo) / max(hi - lo, 1e-9))
        rgba = np.zeros((*dst_arr.shape, 4), dtype=np.uint8)
        # simple viridis-approximation ramp: dark blue -> yellow
        rgba[..., 0] = (255 * np.clip(1.4 * norm - 0.25, 0, 1)).astype(np.uint8)
        rgba[..., 1] = (255 * np.clip(0.2 + 0.9 * norm, 0, 1)).astype(np.uint8)
        rgba[..., 2] = (255 * np.clip(1.0 - 1.2 * norm, 0, 1)).astype(np.uint8)
        rgba[..., 3] = np.where(valid, int(0.75 * 255), 0).astype(np.uint8)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.fromarray(rgba).save(tmp.name)
        png_path = tmp.name

    centre = [(b[1] + b[3]) / 2, (b[0] + b[2]) / 2]
    fmap = folium.Map(location=centre, zoom_start=9, tiles="OpenStreetMap")

    if png_path:
        from PIL import Image as _I

        w_px, h_px = _I.open(png_path).size
        overlay = folium.raster_layers.ImageOverlay(
            image=png_path,
            bounds=[[b[1], b[0]], [b[3], b[2]]],
            opacity=0.75,
            mercator_project=False,
        )
        overlay.add_to(fmap)
        del w_px, h_px

    if candidates_geojson and Path(candidates_geojson).exists():
        gj = json.loads(Path(candidates_geojson).read_text(encoding="utf-8"))
        folium.GeoJson(
            gj,
            marker=folium.CircleMarker(
                radius=5, color="red", fill=True, fill_opacity=0.9),
            tooltip=folium.GeoJsonTooltip(
                fields=["rank", "prob"],
                aliases=["rank", "P(deposit)"]),
        ).add_to(fmap)

    p = Path(out_html)
    p.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(p))
    logger.info("interactive map -> %s", p)
    return p
