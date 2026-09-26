"""Auditable reference-deposit evidence layer for mineral-potential transfer.

The verified subset is every target-commodity USGS MRDS record marked ``Producer``
or ``Past Producer``. It is a mining-status evidence class, not a claim about
current operation, grade, tonnage, or deposit model. These records are exported
for scientific/geographic review and are never written to training labels.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod

from src.ingest.analogue_scene import load_targets
from src.labels.analogs import GROUP_CODES, world_class_analogues
from src.utils import project_path

__all__ = [
    "VERIFIED_DEV_STATS",
    "verified_mrds_references",
    "target_relations",
    "build_reference_inventory",
]

VERIFIED_DEV_STATS = frozenset({"Producer", "Past Producer"})
DEFAULT_MRDS = project_path("data", "raw", "usgs_mrds", "mrds-trim.shp")
_GEOD = Geod(ellps="WGS84")
_COLUMNS = ["DEP_ID", "SITE_NAME", "DEV_STAT", "URL", "CODE_LIST", "geometry"]


def _read_target_records(source: str | Path) -> gpd.GeoDataFrame:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"MRDS source not found: {path}")
    frame = gpd.read_file(path, columns=_COLUMNS)
    tokens = frame["CODE_LIST"].fillna("").astype(str).str.upper().str.split()
    keep = tokens.map(lambda v: any(set(v) & codes for codes in GROUP_CODES.values()))
    result = frame[keep].copy().reset_index(drop=True)
    tokens = result["CODE_LIST"].fillna("").astype(str).str.upper().str.split()
    result["groups"] = tokens.map(
        lambda v: ";".join(g for g, c in GROUP_CODES.items() if set(v) & c)
    )
    result["matched_commodities"] = tokens.map(
        lambda v: ";".join(sorted({code for c in GROUP_CODES.values() for code in c if code in v}))
    )
    result["source_file"] = str(path)
    return gpd.GeoDataFrame(result, geometry="geometry", crs=frame.crs)


def _evidence_tier(status: pd.Series) -> pd.Series:
    mapping = {
        "Producer": "verified_mrds_producer",
        "Past Producer": "verified_mrds_past_producer",
        "Prospect": "unverified_prospect",
        "Occurrence": "unverified_occurrence",
        "Unknown": "unclassified_status",
        "Plant": "processing_plant_not_deposit",
    }
    return status.fillna("Unknown").astype(str).map(mapping).fillna("unclassified_status")


def _filter_verified(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    result = frame[frame["DEV_STAT"].isin(VERIFIED_DEV_STATS)].copy().reset_index(drop=True)
    result["evidence_tier"] = _evidence_tier(result["DEV_STAT"])
    result["reference_role"] = "external_mrds_reference"
    result["label_eligible"] = False
    return result


def verified_mrds_references(
    source: str | Path = DEFAULT_MRDS,
) -> gpd.GeoDataFrame:
    """Return all target-commodity Producer/Past Producer records."""
    return _filter_verified(_read_target_records(source))


def _distance_to_bbox_km(lon: np.ndarray, lat: np.ndarray, bbox: dict[str, float]) -> np.ndarray:
    """Great-circle distance to a WGS84 bounding box, zero for points inside."""
    inside = (
        (lon >= bbox["min_lon"])
        & (lon <= bbox["max_lon"])
        & (lat >= bbox["min_lat"])
        & (lat <= bbox["max_lat"])
    )
    edge_lon = np.clip(lon, bbox["min_lon"], bbox["max_lon"])
    edge_lat = np.clip(lat, bbox["min_lat"], bbox["max_lat"])
    _, _, metres = _GEOD.inv(lon, lat, edge_lon, edge_lat)
    return np.where(inside, 0.0, metres / 1000.0)


def _attach_nearest_anchors(references: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    output = references.copy()
    output["nearest_world_anchor"] = ""
    output["nearest_world_anchor_model"] = "unclassified"
    output["nearest_world_anchor_distance_km"] = np.nan
    for group in GROUP_CODES:
        mask = output["groups"].str.split(";").map(
            lambda values, group=group: group in values
        )
        anchors = world_class_analogues(group)
        if not mask.any() or anchors.empty:
            continue
        anchor_lon = anchors.geometry.x.to_numpy()
        anchor_lat = anchors.geometry.y.to_numpy()
        for index in output.index[mask]:
            point = output.loc[index].geometry
            _, _, metres = _GEOD.inv(
                np.full(len(anchors), point.x),
                np.full(len(anchors), point.y),
                anchor_lon,
                anchor_lat,
            )
            nearest = int(np.argmin(metres))
            output.loc[index, "nearest_world_anchor"] = anchors.iloc[nearest]["name"]
            output.loc[index, "nearest_world_anchor_model"] = anchors.iloc[nearest]["deposit_model"]
            output.loc[index, "nearest_world_anchor_distance_km"] = round(metres[nearest] / 1000, 3)
    return output


def target_relations(
    references: gpd.GeoDataFrame, targets: list[dict[str, Any]] | None = None
) -> gpd.GeoDataFrame:
    """Relate each verified reference to every same-group target window."""
    scenes = load_targets() if targets is None else targets
    exploded = references.copy()
    exploded["group"] = exploded["groups"].str.split(";")
    exploded = exploded.explode("group")
    pieces: list[gpd.GeoDataFrame] = []
    for scene in scenes:
        selected = exploded[exploded["group"] == scene["group"]].copy()
        if selected.empty:
            continue
        distance = _distance_to_bbox_km(
            selected.geometry.x.to_numpy(),
            selected.geometry.y.to_numpy(),
            scene["bbox_wgs84"],
        )
        selected["target_id"] = scene["id"]
        selected["target_country"] = scene["country"]
        selected["target_confidence"] = scene["confidence"]
        selected["target_deposit_model"] = scene["deposit_model"]
        selected["target_analogue"] = scene["analogue"]
        selected["target_rationale"] = " ".join(scene["rationale"].split())
        for kind in ("spectral", "terrain", "geophysics"):
            selected[f"target_{kind}_signatures"] = "; ".join(scene["signatures"].get(kind, []))
        selected["target_relation"] = np.where(
            distance == 0,
            "inside_permissive_window",
            np.where(
                distance <= 100, "within_100_km_transfer_context", "global_reference_distance_only"
            ),
        )
        selected["distance_to_target_km"] = distance.round(3)
        selected["label_eligible"] = False
        keep = [
            "DEP_ID",
            "SITE_NAME",
            "DEV_STAT",
            "CODE_LIST",
            "groups",
            "group",
            "matched_commodities",
            "evidence_tier",
            "target_id",
            "target_country",
            "target_confidence",
            "target_deposit_model",
            "target_analogue",
            "target_rationale",
            "target_spectral_signatures",
            "target_terrain_signatures",
            "target_geophysics_signatures",
            "target_relation",
            "distance_to_target_km",
            "label_eligible",
            "geometry",
        ]
        pieces.append(gpd.GeoDataFrame(selected[keep], geometry="geometry", crs=references.crs))
    if not pieces:
        return gpd.GeoDataFrame(
            {"DEP_ID": [], "geometry": []}, geometry="geometry", crs="EPSG:4326"
        )
    return gpd.GeoDataFrame(
        pd.concat(pieces, ignore_index=True), geometry="geometry", crs=references.crs
    )


def _group_counts(frame: gpd.GeoDataFrame) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for group in GROUP_CODES:
        mask = frame["groups"].str.split(";").map(
            lambda values, group=group: group in values
        )
        subset = frame[mask]
        counts[group] = {
            "records": int(len(subset)),
            "unique_sites": int(subset["SITE_NAME"].nunique()),
            "current_producers": int((subset["DEV_STAT"] == "Producer").sum()),
            "past_producers": int((subset["DEV_STAT"] == "Past Producer").sum()),
        }
    return counts


def _relation_counts(frame: gpd.GeoDataFrame) -> dict[str, dict[str, Any]]:
    counts: dict[str, dict[str, Any]] = {}
    for (group, target_id), subset in frame.groupby(["group", "target_id"]):
        counts[target_id] = {
            "group": group,
            "relations": int(len(subset)),
            "inside_window": int((subset["target_relation"] == "inside_permissive_window").sum()),
            "within_100_km": int(
                (subset["target_relation"] == "within_100_km_transfer_context").sum()
            ),
        }
    return counts


def _render_report(
    source: Path,
    all_records: int,
    references: gpd.GeoDataFrame,
    relations: gpd.GeoDataFrame,
    targets: list[dict[str, Any]],
    group_counts: dict[str, dict[str, int]],
    relation_counts: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "# Verified mineral-deposit reference study",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Scope and evidence rule",
        "",
        f"Source: `{source}` (USGS MRDS global extract).",
        "",
        "The verified subset contains every record classified by MRDS as `Producer` or "
        "`Past Producer` and carrying a target commodity code. This is a mining-status "
        "evidence class; it does not independently verify grade, tonnage, ownership, or "
        "current operation. MRDS coverage outside the United States is incomplete.",
        "",
        "Prospects, occurrences, unknown-status records, and processing plants are retained "
        "in the source but are not called verified deposits. Reference records are not "
        "written to training labels and are marked `label_eligible=False`.",
        "",
        "## Scientific transfer framework",
        "",
        "A shared commodity does not establish a shared genesis. Sn-W-Ta is evaluated "
        "against LCT-pegmatite and granite-related greisen models; Cu-Zn is separated into "
        "sediment-hosted, fault-carbonate, and VMS concepts; bauxite is evaluated as a "
        "weathering-profile problem on plateau surfaces. The source extract has no host-rock, "
        "age, deposit-type, alteration, ore-mineral, reserve, or tonnage fields, so individual "
        "records cannot be assigned to a single model from this file alone.",
        "",
        "Each point is related geographically to every same-group permissive target window. "
        "The configured target's deposit model and spectral, terrain, and geophysics signatures "
        "provide the scientific interpretation; distance alone does not establish geological "
        "equivalence.",
        "",
        "## Geographic relation to target regions",
        "",
        "`inside_permissive_window` means only that a point lies inside a configured "
        "search window. `within_100_km_transfer_context` identifies geographic proximity "
        "for local review. Both require deposit-model validation; neither asserts "
        "mineralisation or creates a target.",
        "",
        "| Target | Group | Confidence | Relations | Inside window | Within 100 km |",
        "|---|---|---|---:|---:|---:|",
    ]
    for target in targets:
        count = relation_counts.get(
            target["id"], {"relations": 0, "inside_window": 0, "within_100_km": 0}
        )
        lines.append(
            f"| `{target['id']}` | {target['group']} | {target['confidence']} | "
            f"{count['relations']:,} | {count['inside_window']:,} | {count['within_100_km']:,} |"
        )
    lines += [
        "",
        "## Inventory",
        "",
        f"- Target-commodity records available: **{all_records:,}**",
        f"- Producer/Past Producer reference records: **{len(references):,}**",
        f"- Unique reference site names: **{references['SITE_NAME'].nunique():,}**",
        f"- Target-transfer relations: **{len(relations):,}**",
        "",
        "| Mineral group | MRDS codes | Records | Sites | Current | Past |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for group, codes in GROUP_CODES.items():
        count = group_counts[group]
        lines.append(
            f"| {group} | {' / '.join(sorted(codes))} | {count['records']:,} | "
            f"{count['unique_sites']:,} | {count['current_producers']:,} | "
            f"{count['past_producers']:,} |"
        )
    lines += [
        "",
        "Multi-commodity records are represented in every group they match, so group counts "
        "do not sum to unique MRDS IDs.",
        "",
        "## Use and limitations",
        "",
        "1. Use the inventory to widen geological priors and select external review areas.",
        "2. Build separate strata or features for Cu-Zn VMS, carbonate-replacement, and "
        "sediment-hosted signatures.",
        "3. Validate source status, commodity association, host lithology, alteration, age, "
        "and ore mineralogy before promoting a record to a label or a target.",
        "4. Validate against the target's common feature grid before any transfer-learning "
        "or domain-adaptation experiment.",
        "",
        "The accompanying GeoPackage contains the point records and transfer relations; "
        "the JSON summary contains machine-readable counts and provenance.",
        "The nearest curated world anchor in the point layer is a navigation aid for "
        "literature review, not a classification of the MRDS record's genesis. Host rock, "
        "age, alteration, ore mineralogy, grade, and tonnage require record-level geological "
        "verification.",
    ]
    return "\n".join(lines) + "\n"


def build_reference_inventory(
    source: str | Path = DEFAULT_MRDS, out_dir: str | Path | None = None
) -> dict[str, Any]:
    """Build point and target-relation evidence layers under ``outputs/evidence``."""
    source_path = Path(source)
    all_target_records = _read_target_records(source_path)
    references = verified_mrds_references(source_path)
    references = _attach_nearest_anchors(references)
    targets = load_targets()
    relations = target_relations(references, targets)
    output_dir = Path(out_dir) if out_dir else project_path("outputs", "evidence")
    output_dir.mkdir(parents=True, exist_ok=True)
    points_path = output_dir / "verified_mrds_references.gpkg"
    relations_path = output_dir / "verified_mrds_target_relations.gpkg"
    summary_path = output_dir / "verified_reference_study.json"
    report_path = output_dir / "VERIFIED_REFERENCE_STUDY.md"
    point_columns = [
        "DEP_ID",
        "SITE_NAME",
        "DEV_STAT",
        "URL",
        "CODE_LIST",
        "groups",
        "matched_commodities",
        "evidence_tier",
        "reference_role",
        "label_eligible",
        "nearest_world_anchor",
        "nearest_world_anchor_model",
        "nearest_world_anchor_distance_km",
        "source_file",
        "geometry",
    ]
    gpd.GeoDataFrame(references[point_columns], geometry="geometry", crs=references.crs).to_file(
        points_path, layer="verified_references", driver="GPKG"
    )
    gpd.GeoDataFrame(relations, geometry="geometry", crs=relations.crs).to_file(
        relations_path, layer="target_relations", driver="GPKG"
    )
    group_counts = _group_counts(references)
    relation_counts = _relation_counts(relations)
    summary = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "source": str(source_path),
        "evidence_rule": (
            "MRDS DEV_STAT in {Producer, Past Producer}; target commodity code present"
        ),
        "source_statuses": {
            "verified": sorted(VERIFIED_DEV_STATS),
            "excluded": ["Prospect", "Occurrence", "Unknown", "Plant"],
        },
        "target_commodity_records": int(len(all_target_records)),
        "verified_reference_records": int(len(references)),
        "unique_reference_sites": int(references["SITE_NAME"].nunique()),
        "target_relations": int(len(relations)),
        "groups": group_counts,
        "targets": relation_counts,
        "training_guardrail": {
            "label_eligible": False,
            "written_to_training_labels": False,
            "reason": "external reference deposits are evidence, not in-belt training positives",
        },
        "missing_scientific_fields": [
            "country",
            "host_rock",
            "deposit_type",
            "formation",
            "age",
            "alteration",
            "ore_minerals",
            "reserves_resources",
            "grade",
            "tonnage",
        ],
        "outputs": {
            "points": str(points_path),
            "relations": str(relations_path),
            "report": str(report_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report_path.write_text(
        _render_report(
            source_path,
            len(all_target_records),
            references,
            relations,
            targets,
            group_counts,
            relation_counts,
        ),
        encoding="utf-8",
    )
    return {
        "points": points_path,
        "relations": relations_path,
        "summary": summary_path,
        "report": report_path,
        "target_commodity_records": int(len(all_target_records)),
        "verified_reference_records": int(len(references)),
        "target_relations": int(len(relations)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build verified MRDS reference evidence")
    parser.add_argument("--source", default=DEFAULT_MRDS)
    parser.add_argument("--out-dir")
    args = parser.parse_args()
    for name, value in build_reference_inventory(args.source, args.out_dir).items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
