"""Tests for the analogue-scene registry bridge (configs/analogue_targets.yaml)."""
from __future__ import annotations

import rasterio
import yaml

from src.ingest.analogue_scene import (
    _dry_season_range,
    load_targets,
    scene_config,
    write_scene_config,
)


def test_load_targets_filters():
    all_t = load_targets()
    assert len(all_t) == 17
    hi = load_targets(confidence="high")
    assert len(hi) == 7
    assert all(t["confidence"] == "high" for t in hi)
    grp = load_targets(group="bauxite")
    assert len(grp) == 6
    one = load_targets(target_id="kab_burundi_north")
    assert len(one) == 1 and one[0]["crs"] == "EPSG:32736"


def test_dry_season_range_parsing():
    assert _dry_season_range("2023-06-01 to 2023-09-30") == (
        "2023-06-01",
        "2023-09-30",
    )
    # suffix annotations are ignored (primary window wins)
    assert _dry_season_range(
        "2023-06-01 to 2023-09-30 (supplement Jan-Feb for equatorial cloud)"
    ) == ("2023-06-01", "2023-09-30")


def test_scene_config_schema_matches_registry():
    for t in load_targets(confidence="high"):
        cfg = scene_config(t)
        assert cfg["grid"]["crs"] == t["crs"]
        assert cfg["grid"]["resolution_m"] == t["resolution_m"]
        assert cfg["grid"]["aoi_wgs84"] == dict(t["bbox_wgs84"])
        dates = cfg["sources"]["sentinel2"]["date_range"]
        assert dates["start"] <= dates["end"]
        assert cfg["_meta"]["target_id"] == t["id"]
        # guardrail survives into the generated config
        assert "NOT a label source" in cfg["_meta"]["guardrail"]


def test_write_scene_config_roundtrip(tmp_path):
    t = load_targets(target_id="usambara_east_extension")[0]
    path = write_scene_config(t, out_dir=tmp_path)
    assert path.name == "usambara_east_extension.yml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # registry remains the source of truth
    assert cfg["grid"]["aoi_wgs84"] == dict(t["bbox_wgs84"])


def test_export_aoi_geojson(tmp_path):
    import json

    from src.ingest.analogue_scene import export_aoi

    t = load_targets(target_id="kab_burundi_north")[0]
    path = export_aoi(t, out_dir=tmp_path)
    fc = json.loads(path.read_text(encoding="utf-8"))
    assert fc["type"] == "FeatureCollection"
    feat = fc["features"][0]
    ring = feat["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]  # closed ring
    lons = [c[0] for c in ring]
    lats = [c[1] for c in ring]
    b = t["bbox_wgs84"]
    assert (min(lons), min(lats), max(lons), max(lats)) == (
        b["min_lon"], b["min_lat"], b["max_lon"], b["max_lat"],
    )
    assert "NOT a label" in feat["properties"]["purpose"]


def test_scene_config_grid_builds():
    from src.preprocess.grid import GridSpec

    for t in load_targets(confidence="high"):
        grid = GridSpec.from_config(scene_config(t))
        assert grid.width > 0 and grid.height > 0
        assert grid.crs == t["crs"]


def test_dry_run_ingest_outputs_align(tmp_path_path=None):
    """The three ingested top scenes must have pixel-aligned DEM + S2."""
    from src.utils import project_path

    for tid, crs in [
        ("kab_burundi_north", "EPSG:32736"),
        ("lufilian_kolwezi_kambove", "EPSG:32735"),
        ("usambara_east_extension", "EPSG:32737"),
    ]:
        dem = project_path("data", "interim", "dem",
                           f"srtm_{tid}_aligned.tif")
        s2 = project_path("data", "interim", "sentinel2",
                          f"sentinel2_{tid}_aligned.tif")
        if not (dem.exists() and s2.exists()):
            continue  # ingest not run in this checkout — skip, not fail
        with rasterio.open(dem) as d, rasterio.open(s2) as s:
            assert d.crs == s.crs == crs
            assert (d.width, d.height) == (s.width, s.height)
            assert d.transform == s.transform
            assert s.count == 6 and d.count == 1