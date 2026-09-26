"""Tests for the verified reference-deposit evidence layer."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest

from src.labels.reference_study import (
    VERIFIED_DEV_STATS,
    build_reference_inventory,
    target_relations,
    verified_mrds_references,
)


def _write_mrds(path: Path) -> None:
    rows = [
        {
            "DEP_ID": "1",
            "SITE_NAME": "Producer Cu",
            "DEV_STAT": "Producer",
            "URL": "https://example.test/1",
            "CODE_LIST": "CU ZN",
            "geometry": gpd.points_from_xy([1.0], [1.0])[0],
        },
        {
            "DEP_ID": "2",
            "SITE_NAME": "Past Sn",
            "DEV_STAT": "Past Producer",
            "URL": "https://example.test/2",
            "CODE_LIST": "SN W",
            "geometry": gpd.points_from_xy([2.0], [2.0])[0],
        },
        {
            "DEP_ID": "3",
            "SITE_NAME": "Prospect Al",
            "DEV_STAT": "Prospect",
            "URL": "https://example.test/3",
            "CODE_LIST": "AL",
            "geometry": gpd.points_from_xy([3.0], [3.0])[0],
        },
        {
            "DEP_ID": "4",
            "SITE_NAME": "Plant",
            "DEV_STAT": "Plant",
            "URL": "https://example.test/4",
            "CODE_LIST": "FE",
            "geometry": gpd.points_from_xy([4.0], [4.0])[0],
        },
    ]
    gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326").to_file(
        path, driver="GPKG"
    )


def test_verified_references_include_all_target_commodity_producers(tmp_path):
    source = tmp_path / "mrds.gpkg"
    _write_mrds(source)

    result = verified_mrds_references(source)

    assert set(result["DEP_ID"]) == {"1", "2"}
    assert set(result["DEV_STAT"]) == VERIFIED_DEV_STATS
    assert set(result["groups"]) == {"copper_zinc", "tin_tungsten_tantalum"}
    assert not result["label_eligible"].any()


def test_target_relations_are_same_group_and_geographically_scientific(tmp_path):
    source = tmp_path / "mrds.gpkg"
    _write_mrds(source)
    references = verified_mrds_references(source)
    target = {
        "id": "cu_window",
        "group": "copper_zinc",
        "country": "test",
        "bbox_wgs84": {"min_lon": 0, "max_lon": 2, "min_lat": 0, "max_lat": 2},
        "confidence": "high",
        "deposit_model": "sediment-hosted Cu",
        "analogue": "type basin",
        "rationale": "Basin-margin redox traps",
        "signatures": {
            "spectral": ["iron oxide ratio"],
            "terrain": ["fault corridor"],
            "geophysics": ["gravity"],
        },
    }

    relations = target_relations(references, [target])

    assert len(relations) == 1
    row = relations.iloc[0]
    assert row["DEP_ID"] == "1"
    assert row["target_deposit_model"] == "sediment-hosted Cu"
    assert row["target_spectral_signatures"] == "iron oxide ratio"
    assert row["target_relation"] == "inside_permissive_window"
    assert row["distance_to_target_km"] == 0
    assert not row["label_eligible"]


def test_bbox_distance_is_zero_inside_and_great_circle_outside(tmp_path):
    source = tmp_path / "mrds.gpkg"
    _write_mrds(source)
    references = verified_mrds_references(source)
    target = {
        "id": "w_window",
        "group": "tin_tungsten_tantalum",
        "country": "test",
        "bbox_wgs84": {"min_lon": 0, "max_lon": 1, "min_lat": 0, "max_lat": 1},
        "confidence": "low",
        "deposit_model": "LCT pegmatite",
        "analogue": "type field",
        "rationale": "Granite contact",
        "signatures": {"spectral": [], "terrain": [], "geophysics": []},
    }

    relations = target_relations(references, [target])

    assert relations.iloc[0]["distance_to_target_km"] == pytest.approx(156.876, abs=0.01)
    assert relations.iloc[0]["target_relation"] == "global_reference_distance_only"


def test_inventory_exports_evidence_without_training_labels(tmp_path):
    source = tmp_path / "mrds.gpkg"
    out_dir = tmp_path / "evidence"
    _write_mrds(source)

    result = build_reference_inventory(source, out_dir)
    points = gpd.read_file(result["points"], layer="verified_references")
    relations = gpd.read_file(result["relations"], layer="target_relations")

    assert result["target_commodity_records"] == 3
    assert result["verified_reference_records"] == 2
    assert len(points) == 2
    assert not points["label_eligible"].any()
    assert not relations["label_eligible"].any()
    assert "not written to training labels" in result["report"].read_text(
        encoding="utf-8"
    ).lower()
    assert not list(out_dir.glob("labels_*.gpkg"))
