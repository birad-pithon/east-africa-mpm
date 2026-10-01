"""Streamlit web UI for the East Africa MPM dashboard.

Run:
    streamlit run app.py

Views statistics, probability maps, candidate tables, and full model
cards for every trained commodity group (tin-tungsten-tantalum,
copper-zinc, bauxite).
"""
from __future__ import annotations

import json

# Make src importable when running from the project root
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import rasterio
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).parent))
from src.utils import project_path  # noqa: E402

# DLR EOC OGC-WMS context layers (verified 2026-09-22, see
# src/ingest/dlr_eoc_wms.py).  Optional: the dashboard must still load if
# the module or the optional ingest dependencies are unavailable.
try:  # pragma: no cover - import guard
    from src.ingest.dlr_eoc_wms import DASHBOARD_LAYERS, VERIFIED_LAYERS
except Exception:  # noqa: BLE001
    DASHBOARD_LAYERS, VERIFIED_LAYERS = (), {}

WMS_SERVICE_URLS = {
    "imagery": "https://geoservice.dlr.de/eoc/imagery/wms",
    "land": "https://geoservice.dlr.de/eoc/land/wms",
}

# ── Constants ────────────────────────────────────────────────────────────

GROUP_CONFIGS = {
    "tin_tungsten_tantalum": {
        "label": "Tin / Tungsten / Tantalum (KAB)",
        "config": "configs/karagwe.yml",
        "color": "🔵",
    },
    "copper_zinc": {
        "label": "Copper / Zinc (CBC)",
        "config": "configs/copperbelt.yml",
        "color": "🔴",
    },
    "bauxite": {
        "label": "Bauxite (Usambara)",
        "config": "configs/usambara.yml",
        "color": "🟡",
    },
}

MODELS_DIR = project_path("outputs", "models")
MAPS_DIR = project_path("outputs", "maps")


# ── Helpers ──────────────────────────────────────────────────────────────

@st.cache_data
def load_metrics(group: str) -> dict:
    """Load the metrics JSON for a commodity group."""
    mp = MODELS_DIR / f"metrics_{group}.json"
    if not mp.exists():
        return {}
    return json.loads(mp.read_text(encoding="utf-8"))


@st.cache_data
def load_model_card(group: str) -> str:
    """Load the model card markdown for a commodity group."""
    mp = MODELS_DIR / f"model_card_{group}.md"
    if not mp.exists():
        return ""
    return mp.read_text(encoding="utf-8")


@st.cache_data
def load_candidates(group: str) -> pd.DataFrame | None:
    """Load the top-N candidate GeoJSON for a group."""
    gj = MAPS_DIR / f"top25_{group}_candidates.geojson"
    if not gj.exists():
        return None
    import geopandas as gpd
    gdf = gpd.read_file(gj)
    return pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))


@st.cache_data
def get_prob_raster_path(group: str) -> Path | None:
    """Find the best probability raster for a group."""
    metrics = load_metrics(group)
    best = metrics.get("best_algo")
    if best:
        p = MODELS_DIR / f"proba_{group}_{best}.tif"
        if p.exists():
            return p
    # try all algos
    for suffix in ["_rf", "_xgb", "_lgbm"]:
        cand = MODELS_DIR / f"proba_{group}{suffix}.tif"
        if cand.exists():
            return cand
    return None


@st.cache_data
def raster_summary(path: Path) -> dict:
    """Return basic summary stats for a probability raster."""
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).astype("float64")
        return {
            "width": src.width,
            "height": src.height,
            "crs": src.crs.to_string() if src.crs else "unknown",
            "count_valid": int(arr.count()),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "mean": round(float(arr.mean()), 4),
            "median": round(float(np.median(arr.compressed())), 4) if arr.count() > 0 else None,
            "p90": (round(float(np.percentile(arr.compressed(), 90)), 4)
                    if arr.count() > 0 else None),
        }


def render_folium_map(
    raster_path: Path, candidates_gdf=None, wms_keys: tuple[str, ...] = ()
) -> folium.Map:
    """Render the probability raster, candidates and DLR EOC context layers.

    ``wms_keys`` selects verified OGC-WMS context layers
    (:data:`src.ingest.dlr_eoc_wms.DASHBOARD_LAYERS`); they are added as
    hidden overlays that the folium layer control switches on.  They are
    rendered tiles for visual cross-checking, not model inputs.
    """

    """Render a folium map with the probability raster overlay."""
    import matplotlib
    matplotlib.use("Agg")
    from affine import Affine
    from rasterio.warp import Resampling, reproject, transform_bounds

    with rasterio.open(raster_path) as src:
        arr = src.read(1).astype("float64")
        transform = src.transform
        crs = src.crs

    bounds_latlon = transform_bounds(crs, "EPSG:4326", *src.bounds)
    west, south, east, north = bounds_latlon
    out_shape = (512, 512)
    dst_arr = np.full(out_shape, np.nan, dtype="float64")
    dst_tf = Affine(
        (east - west) / out_shape[1], 0, west,
        0, -(north - south) / out_shape[0], north,
    )
    reproject(
        source=arr, destination=dst_arr,
        src_transform=transform, src_crs=crs,
        dst_transform=dst_tf, dst_crs="EPSG:4326",
        resampling=Resampling.bilinear,
    )

    valid = np.isfinite(dst_arr)
    rgba = np.zeros((*out_shape, 4), dtype=np.uint8)
    if valid.any():
        from matplotlib.cm import viridis
        v = dst_arr[valid]
        lo, hi = float(v.min()), float(v.max())
        norm = np.nan_to_num((dst_arr - lo) / max(hi - lo, 1e-9))
        cmap = viridis(norm)
        rgba = (cmap[..., :3] * 255).astype(np.uint8)
        alpha = np.where(valid, int(0.7 * 255), 0).astype(np.uint8)
        rgba = np.dstack([rgba[..., :3], alpha])

    center = [(south + north) / 2, (west + east) / 2]
    fmap = folium.Map(location=center, zoom_start=9, tiles="OpenStreetMap")

    # Save RGBA to temp PNG and use path for ImageOverlay
    # (folium ImageOverlay doesn't accept PIL objects directly)
    if valid.any():
        import tempfile

        from PIL import Image as PILImage
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            PNG_PATH = tf.name
        PILImage.fromarray(rgba).save(PNG_PATH, format="PNG")
        overlay = folium.raster_layers.ImageOverlay(
            image=PNG_PATH,
            bounds=[[south, west], [north, east]],
            opacity=0.7,
        )
        overlay.add_to(fmap)

    if candidates_gdf is not None and len(candidates_gdf) > 0:
        if candidates_gdf.crs is None or candidates_gdf.crs.to_string() != "EPSG:4326":
            candidates_gdf = candidates_gdf.to_crs("EPSG:4326")
        for _, row in candidates_gdf.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=4, color="red", fill=True, fill_color="red",
                fill_opacity=0.8,
                popup=f"#{int(row.get('rank', 0))} | P={row.get('prob', 0.0):.4f}",
            ).add_to(fmap)

    # DLR EOC OGC-WMS context layers (verified; hidden until toggled in the
    # layer control).  Rendered tiles - context / QA only, never model input.
    for key in wms_keys:
        spec = VERIFIED_LAYERS.get(key)
        if not spec or spec.get("status") != "usable":
            continue
        folium.WmsTileLayer(
            url=WMS_SERVICE_URLS[spec["service"]],
            name=spec["title"],
            layers=spec["layer"],
            styles=spec["style"],
            fmt="image/png",
            transparent=True,
            version="1.3.0",
            attr="DLR EOC GeoService",
            overlay=True,
            control=True,
            opacity=0.6,
            show=False,
        ).add_to(fmap)

    folium.LayerControl(collapsed=True).add_to(fmap)

    return fmap


def load_candidates_gdf(group: str):
    """Load candidate GeoJSON as a GeoDataFrame."""
    gj = MAPS_DIR / f"top25_{group}_candidates.geojson"
    if not gj.exists():
        return None
    import geopandas as gpd
    return gpd.read_file(gj)


# ── Sidebar ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="East Africa MPM Dashboard",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.title("🗺️ East Africa MPM")
    st.markdown("---")
    group = st.selectbox(
        "Commodity Group",
        options=list(GROUP_CONFIGS.keys()),
        format_func=lambda k: f"{GROUP_CONFIGS[k]['color']} {GROUP_CONFIGS[k]['label']}",
    )
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "Mineral prospectivity dashboard showing trained model metrics, "
        "probability maps, and ranked exploration targets."
    )

cfg = GROUP_CONFIGS[group]
metrics = load_metrics(group)

# ── Main Header ─────────────────────────────────────────────────────────

st.title("East Africa Mineral Prospectivity Mapping")
st.markdown(f"### {cfg['color']} {cfg['label']}")
st.markdown(
    f"**Config:** `{cfg['config']}` | **Best algorithm:** "
    f"`{metrics.get('best_algo', 'N/A')}`"
)

# ── Summary cards ──────────────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)

best_algo = metrics.get("best_algo", "N/A")
results = metrics.get("results", {})

with col1:
    st.metric("Best Algorithm", best_algo.upper() if best_algo else "N/A")
with col2:
    if best_algo in results:
        st.metric("Mean AP", f"{results[best_algo].get('mean_average_precision', 0):.3f}")
    else:
        st.metric("Mean AP", "N/A")
with col3:
    if best_algo in results:
        st.metric("Mean ROC-AUC", f"{results[best_algo].get('mean_roc_auc', 0):.3f}")
    else:
        st.metric("ROC-AUC", "N/A")
with col4:
    st.metric("Positives", f"{metrics.get('n_positives', 0)}")
with col5:
    st.metric("Total Samples", f"{metrics.get('n_rows', 0)}")

# ── Tabs ────────────────────────────────────────────────────────────────

tab_overview, tab_cv, tab_map, tab_candidates, tab_calib, tab_card = st.tabs([
    "📊 Overview", "📈 Cross-Validation", "🗺️ Probability Map",
        "🎯 Candidates", "⚖️ Calibration", "📄 Model Card",
])

# ── Tab 1: Overview ─────────────────────────────────────────────────────

with tab_overview:
    st.subheader("Dataset Overview")

    info_cols = st.columns(2)
    with info_cols[0]:
        st.markdown("**Configuration**")
        st.json({
            "buffer_km": metrics.get("buffer_km"),
            "block_m": metrics.get("block_m"),
            "oversample": metrics.get("oversample"),
            "anomaly_baseline": metrics.get("anomaly_baseline"),
            "pu_learning": metrics.get("pu"),
            "ensemble": metrics.get("ensemble"),
            "n_features": len(metrics.get("features", [])),
            "n_folds": next(
                (r.get("n_folds") for r in results.values()), "N/A"
            ),
        })

    with info_cols[1]:
        st.markdown("**Feature List**")
        features = metrics.get("features", [])
        st.write(f"{len(features)} features used:")
        for f in features:
            st.markdown(f"  - `{f}`")

    st.markdown("---")
    st.subheader("Algorithm Comparison")

    comp_data = []
    for algo, res in results.items():
        comp_data.append({
            "Algorithm": algo.upper(),
            "Mean AP": round(res.get("mean_average_precision", 0), 4),
            "Std AP": round(res.get("std_average_precision", 0), 4),
            "Mean ROC-AUC": round(res.get("mean_roc_auc", 0), 4),
            "Top-Decile Precision": round(
                res.get("mean_precision_at_top_decile", 0), 4
            ),
            "Best": "⭐" if algo == best_algo else "",
        })
    comp_df = pd.DataFrame(comp_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    fig = px.bar(
        comp_df, x="Algorithm", y="Mean AP",
        color="Algorithm",
        title=f"Mean Average Precision by Algorithm ({group})",
        text="Mean AP",
        )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Cross-Validation ─────────────────────────────────────────────

with tab_cv:
    st.subheader("Per-Fold Performance")

    if best_algo in results:
        fold_data = results[best_algo].get("folds", [])
        if fold_data:
            fold_df = pd.DataFrame(fold_data)
            fold_df["fold"] = fold_df["fold"].astype(str)

            fig_ap = px.bar(
                fold_df, x="fold", y="average_precision",
                title=f"AP per Fold — {best_algo.upper()}",
                text="average_precision",
                labels={"average_precision": "Average Precision", "fold": "Fold"},
                color="average_precision",
                color_continuous_scale="RdYlGn",
            )
            fig_ap.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            st.plotly_chart(fig_ap, use_container_width=True)

            fig_auc = px.bar(
                fold_df, x="fold", y="roc_auc",
                title=f"ROC-AUC per Fold — {best_algo.upper()}",
                text="roc_auc",
                labels={"roc_auc": "ROC-AUC", "fold": "Fold"},
                color="roc_auc",
                color_continuous_scale="RdYlGn",
            )
            fig_auc.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            st.plotly_chart(fig_auc, use_container_width=True)

            st.markdown("#### Fold Details")
            st.dataframe(
                fold_df[["fold", "n_train", "n_test", "n_test_positives",
                         "average_precision", "roc_auc", "precision_at_top_decile"]]
                .rename(columns={
                    "n_train": "Train", "n_test": "Test",
                    "n_test_positives": "Test Positives",
                    "average_precision": "AP",
                    "roc_auc": "ROC-AUC",
                    "precision_at_top_decile": "Top-10% Precision",
                }),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No per-fold data available.")
    else:
        st.warning("No results available for the best algorithm.")

    st.markdown("---")
    st.markdown(
        "💡 **Spatial CV Certification:** The spatial cross-validation uses "
        "10 km × 10 km grid blocks with a 5 km buffer between train and test "
        "folds to prevent spatial leakage. Minimum train-test separation is "
                "certified at >5000 m."
    )

# ── Tab 3: Probability Map ──────────────────────────────────────────────

with tab_map:
    st.subheader("Probability Surface")

    proba_path = get_prob_raster_path(group)
    candidates_gdf = load_candidates_gdf(group)

    if proba_path and proba_path.exists():
        info = raster_summary(proba_path)
        st.markdown(f"**Raster:** `{proba_path.name}`")
        st.markdown(f"**CRS:** `{info['crs']}` | **Grid:** {info['width']} × {info['height']}")

        col_stats = st.columns(6)
        with col_stats[0]:
            st.metric("Valid Pixels", f"{info['count_valid']:,}")
        with col_stats[1]:
            st.metric("Min P", f"{info['min']:.4f}")
        with col_stats[2]:
            st.metric("Max P", f"{info['max']:.4f}")
        with col_stats[3]:
            st.metric("Mean P", f"{info['mean']:.4f}")
        with col_stats[4]:
            st.metric("Median P", f"{info['median']:.4f}")
        with col_stats[5]:
            st.metric("P90", f"{info['p90']:.4f}")

        st.markdown("---")
        st.markdown("### Interactive Map — Probability Overlay")

        wms_keys: tuple[str, ...] = ()
        if DASHBOARD_LAYERS:
            st.markdown("#### Context layers (DLR EOC OGC-WMS)")
            st.caption(
                "Verified WMS context layers - 8-bit rendered tiles for visual "
                "cross-checking only (not reflectance values, never model input). "
                "Toggle each one in the map's top-right layer control."
            )
            wms_keys = tuple(st.multiselect(
                "Overlay layers",
                options=list(DASHBOARD_LAYERS),
                default=[k for k in ("soilsuite_src_africa",
                                     "tandemx_forest_nonforest")
                         if k in DASHBOARD_LAYERS],
                format_func=lambda k: VERIFIED_LAYERS[k]["title"],
                key=f"wms_layers_{group}",
            ))

            with st.expander("Per-belt coverage of the DLR EOC layers"):
                st.dataframe(
                    pd.DataFrame([
                        {
                            "Service": VERIFIED_LAYERS[k]["service"],
                            "WMS layer": VERIFIED_LAYERS[k]["layer"],
                            "Title": VERIFIED_LAYERS[k]["title"],
                            "Coverage in this belt": (
                                "none (empty tile)"
                                if VERIFIED_LAYERS[k]["opaque"].get(group) is None
                                else f"{VERIFIED_LAYERS[k]['opaque'][group] * 100:.1f}% of tile"
                            ),
                            "Declared scope": VERIFIED_LAYERS[k]["coverage"],
                        }
                        for k in DASHBOARD_LAYERS
                    ]),
                    use_container_width=True, hide_index=True,
                )
                st.caption(
                    "Renders measured 2026-09-22 against the belt AOIs "
                    "(512 x 512 PNG, non-transparent pixel fraction). "
                    "`python -m src.ingest.dlr_eoc_wms --list` prints the full table."
                )

        with st.spinner("Rendering probability map..."):
            fmap = render_folium_map(proba_path, candidates_gdf, wms_keys=wms_keys)
        st_folium(fmap, use_container_width=True, height=600, key=f"map_{group}")

    else:
                st.warning(f"No probability raster found for group `{group}`.")

# ── Tab 4: Candidates ───────────────────────────────────────────────────

with tab_candidates:
    st.subheader("Top Exploration Targets")

    candidates_df = load_candidates(group)
    candidates_gdf = load_candidates_gdf(group)

    if candidates_df is not None and len(candidates_df) > 0:
        st.markdown(f"Top **{len(candidates_df)}** ranked candidate cells for `{group}`")
        st.markdown("(excludes licensed ground, spatially declustered)")

        display_df = candidates_df.copy()
        display_df["prob"] = display_df["prob"].round(4)
        if "rank" in display_df.columns:
            display_df["rank"] = display_df["rank"].astype(int)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Scatter plot of candidates
        if "lon" in display_df.columns and "lat" in display_df.columns:
            fig = px.scatter_mapbox(
                display_df,
                lat="lat", lon="lon",
                size="prob",
                color="prob",
                hover_name="rank",
                zoom=7,
                height=500,
                title=f"Top Candidates — {cfg['label']}",
                color_continuous_scale="RdYlGn",
            )
            fig.update_layout(map_style="open-street-map")
            st.plotly_chart(fig, use_container_width=True)

        # Download buttons
        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Candidates (CSV)",
            data=csv,
            file_name=f"top25_{group}_candidates.csv",
            mime="text/csv",
        )

        if candidates_gdf is not None:
            geojson_bytes = candidates_gdf.to_json().encode("utf-8")
            st.download_button(
                label="📥 Download Candidates (GeoJSON)",
                data=geojson_bytes,
                file_name=f"top25_{group}_candidates.geojson",
                mime="application/geo+json",
            )
    else:
        st.info(
            f"No candidate ranking available for `{group}`. "
            "Run prediction + ranking with: `python -m src.predict.main "
                        f"--proba outputs/models/proba_{group}_*.tif --group {group} --n 25 --map`"
        )

# ── Tab 5: Calibration ──────────────────────────────────────────────────

with tab_calib:
    st.subheader("Probability Calibration (Out-of-Fold)")

    calib_path = MODELS_DIR / f"calibration_{group}.json"
    if calib_path.exists():
        calib = json.loads(calib_path.read_text(encoding="utf-8"))

        col_cal = st.columns(3)
        with col_cal[0]:
            st.metric("Brier Score", f"{calib.get('brier', 0):.4f}")
        with col_cal[1]:
            st.metric("Expected Calibration Error (ECE)", f"{calib.get('ece', 0):.4f}")
        with col_cal[2]:
            st.metric("Max |Calibration Gap|", f"{calib.get('max_abs_gap', 0):.4f}")

        st.markdown(
            f"**n = {calib.get('n', 0)} samples** "
            f"({calib.get('n_positives', 0)} positives)"
        )

        bins = calib.get("bins", [])
        if bins:
            bins_df = pd.DataFrame(bins)
            bins_df["confidence_gap"] = bins_df["fraction_positive"] - bins_df["mean_predicted"]
            st.markdown("#### Reliability Bin Table")
            st.dataframe(
                bins_df[["bin_lo", "bin_hi", "n", "mean_predicted",
                         "fraction_positive", "confidence_gap"]],
                use_container_width=True, hide_index=True,
                column_config={
                    "bin_lo": st.column_config.NumberColumn("Lower Bound", format="%.2e"),
                    "bin_hi": st.column_config.NumberColumn("Upper Bound", format="%.2e"),
                    "mean_predicted": st.column_config.NumberColumn(
                        "Mean Predicted", format="%.4f"
                    ),
                    "fraction_positive": st.column_config.NumberColumn(
                        "Fraction Positive", format="%.4f"
                    ),
                    "confidence_gap": st.column_config.NumberColumn("Gap", format="%.4f"),
                },
            )

            fig = px.bar(
                bins_df,
                x=list(range(len(bins_df))),
                y=["mean_predicted", "fraction_positive"],
                barmode="group",
                labels={"value": "Probability", "x": "Bin"},
                title="Reliability Diagram (Predicted vs Actual Positive Rate per Bin)",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No calibration data found.")

    rel_png = MODELS_DIR / f"reliability_{group}.png"
    if rel_png.exists():
        st.markdown("---")
        st.markdown("#### Reliability Diagram (PNG)")
        st.image(str(rel_png), caption=f"Out-of-fold reliability — {group}")

# ── Tab 6: Model Card ───────────────────────────────────────────────────

with tab_card:
    st.subheader("Model Card")

    card = load_model_card(group)
    if card:
        st.markdown(card)
    else:
        st.info(f"No model card found for `{group}`.")

# ── Footer ──────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "East Africa MPM — Mineral Prospectivity Mapping Pipeline. "
    "Dashboard built with Streamlit. Metrics computed via spatial cross-validation "
    "(5-fold buffered blocks, 5 km buffer)."
)







