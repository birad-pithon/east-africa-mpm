"""World deposit references for transfer studies, never training labels."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from src.utils import project_path

__all__ = [
    "GROUP_CODES",
    "WORLD_ANALOGUES",
    "african_out_of_belt_analogues",
    "world_class_analogues",
    "write_analogues",
]

GROUP_CODES = {
    "tin_tungsten_tantalum": {"SN", "W", "TA"},
    "copper_zinc": {"CU", "ZN"},
    "bauxite": {"AL"},
}

# Literature-derived geographic and scientific anchors. These inform deposit
# models and priors; they are not direct model-positive training labels.
WORLD_ANALOGUES = [
    ("Manono-Kitotolo", "SN TA LI", "Kibaran LCT pegmatite", "DR Congo", 27.42, -7.30),
    ("Greenbushes", "SN TA LI", "giant zoned LCT pegmatite", "Australia", 116.07, -33.87),
    ("Bikita", "SN TA LI", "LCT pegmatite in craton margin", "Zimbabwe", 31.05, -19.92),
    ("Kidd Creek", "CU ZN", "bimodal-volcanic VMS", "Canada", -81.63, 49.68),
    ("Kupferschiefer", "CU", "reduced black shale redox front", "Poland/Germany", 15.9, 51.7),
    ("Bisha", "CU ZN", "Arabian-Nubian Shield VMS", "Eritrea", 38.32, 15.48),
    ("Weipa", "AL", "coastal-plateau bauxite", "Australia", 141.87, -12.62),
    ("Sangaredi-Boke", "AL", "high-plateau gibbsite/boehmite bauxite", "Guinea", -13.7, 11.1),
    ("Kindia", "AL", "high-plateau gibbsite bauxite", "Guinea", -12.87, 10.05),
    ("Minim-Martap", "AL", "plateau bauxite; closest climate analogue", "Cameroon", 13.0, 6.0),
    ("Trombetas", "AL", "cratonic-cover plateau bauxite", "Brazil", -56.0, -1.48),
    (
        "Erzgebirge greisen (Altenberg/Cinovec)",
        "SN W",
        "muscovite-topaz-quartz greisen above cupolas",
        "Germany/Czechia",
        13.76,
        50.76,
    ),
]


def _standardise(frame: gpd.GeoDataFrame, source: str, reference_class: str) -> gpd.GeoDataFrame:
    """Add common provenance columns without changing source coordinates."""
    out = frame.copy()
    out["source"] = source
    out["reference_class"] = reference_class
    if "name" not in out:
        out["name"] = out["SITE_NAME"]
    if "reference" not in out:
        out["reference"] = "USGS MRDS record; geological model unclassified"
    return gpd.GeoDataFrame(out, geometry=out.geometry, crs="EPSG:4326")


def african_out_of_belt_analogues(
    group: str,
    belts_gpkg: str | Path | None = None,
    include_dev_stats: tuple[str, ...] = (
        "Producer",
        "Past Producer",
        "Prospect",
        "Occurrence",
    ),
) -> gpd.GeoDataFrame:
    """Return African records of *group* outside configured training belts."""
    from src.labels.build_labels import BELTS_CFG, belt_polygon

    if group not in GROUP_CODES:
        raise KeyError(f"unknown group {group!r}")
    path = (
        Path(belts_gpkg) if belts_gpkg else project_path("data", "raw", "usgs_africa_minerals.gpkg")
    )
    frame = gpd.read_file(path)
    tokens = frame["CODE_LIST"].fillna("").astype(str).str.upper().str.split()
    keep_status = {value.casefold() for value in include_dev_stats}
    selected = frame[
        tokens.map(lambda values: bool(set(values) & GROUP_CODES[group]))
        & frame["DEV_STAT"].fillna("").astype(str).str.casefold().isin(keep_status)
    ].copy()
    polygons = [belt_polygon(belt) for belt in BELTS_CFG["groups"][group]["belts"]]
    union = polygons[0]
    for polygon in polygons[1:]:
        union = union.union(polygon)
    selected = selected[~selected.geometry.within(union)].copy().reset_index(drop=True)
    selected["group"] = group
    return _standardise(selected, "analog_africa_oob", "out_of_belt")


def world_class_analogues(group: str | None = None) -> gpd.GeoDataFrame:
    """Return curated world analogues, optionally filtered by commodity group."""
    if group is not None and group not in GROUP_CODES:
        raise KeyError(f"unknown group {group!r}")
    fields = ("name", "commodities", "deposit_model", "country", "lon", "lat")
    rows = [dict(zip(fields, row, strict=True)) for row in WORLD_ANALOGUES]
    if group is not None:
        rows = [row for row in rows if set(row["commodities"].split()) & GROUP_CODES[group]]
    frame = gpd.GeoDataFrame(
        rows, geometry=[Point(row["lon"], row["lat"]) for row in rows], crs="EPSG:4326"
    )
    frame["group"] = group or "all"
    frame["reference"] = "ANALOGUE_DEPOSITS_EVALUATION.md"
    return _standardise(frame, "analog_world", "world_class_reference")


def write_analogues(group: str, out_path: str | Path | None = None) -> tuple[Path, dict]:
    """Write African and curated world references for one commodity group."""
    african = african_out_of_belt_analogues(group)
    world = world_class_analogues(group)
    combined = gpd.GeoDataFrame(
        pd.concat([african, world], ignore_index=True),
        geometry=pd.concat([african.geometry, world.geometry], ignore_index=True),
        crs="EPSG:4326",
    )
    path = (
        Path(out_path)
        if out_path
        else project_path("data", "processed", f"analogs_study_{group}.gpkg")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_file(path, driver="GPKG")
    summary = {
        "group": group,
        "african_out_of_belt": len(african),
        "world_class": len(world),
        "dev_stat": african["DEV_STAT"].value_counts(dropna=False).to_dict(),
        "path": str(path),
        "label_policy": "reference evidence only; never training labels",
    }
    return path, summary
