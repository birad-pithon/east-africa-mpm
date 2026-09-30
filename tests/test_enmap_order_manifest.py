"""Tests for the EnMAP order manifest (scripts/gen_enmap_order_manifest.py).

The manifest exists because one buffered polygon per ranked grid cell is not
one order: adjacent cells inside a single anomaly collapse to a scene. These
tests pin that collapse, the bbox padding, and the AOI file shape.
"""
from __future__ import annotations

import json

import pytest

from scripts.gen_enmap_order_manifest import (
    TIERS,
    bbox_of,
    build_scenes,
    cluster_indices,
    haversine_km,
    load_tier_targets,
    prune_stale_aois,
    render_manifest,
    season_context,
    write_aoi,
)


def _target(lon, lat, rank=1, prob=0.99, strategy="top_prob"):
    return {"lon": lon, "lat": lat, "rank": rank, "prob": prob,
            "strategy": strategy, "score": prob, "group": "demo"}


class TestHaversine:
    def test_zero_distance(self):
        assert haversine_km(30.0, -2.0, 30.0, -2.0) == pytest.approx(0.0)

    def test_one_degree_of_latitude(self):
        # 1 deg of latitude is ~111 km everywhere
        assert haversine_km(30.0, -2.0, 30.0, -1.0) == pytest.approx(111.2, abs=1.0)

    def test_symmetric(self):
        assert haversine_km(29.0, -1.0, 31.0, -3.0) == pytest.approx(
            haversine_km(31.0, -3.0, 29.0, -1.0))


class TestClusterIndices:
    def test_adjacent_points_merge(self):
        pts = [(30.0, -2.0), (30.02, -2.0), (30.04, -2.0)]
        assert len(cluster_indices(pts, 30.0)) == 1

    def test_distant_points_stay_apart(self):
        pts = [(30.0, -2.0), (35.0, -2.0)]
        assert len(cluster_indices(pts, 30.0)) == 2

    def test_single_linkage_chains(self):
        # 0.25 deg lon at -2 deg lat is ~28 km per hop, but the extremes are
        # ~56 km apart: only chaining links all three into one cluster.
        pts = [(30.0, -2.0), (30.25, -2.0), (30.5, -2.0)]
        assert len(cluster_indices(pts, 30.0)) == 1
        assert len(cluster_indices(pts, 60.0)) == 1
        # below the hop distance nothing links at all
        assert len(cluster_indices(pts, 10.0)) == 3

    def test_largest_cluster_first(self):
        pts = [(30.0, -2.0), (30.4, -2.0), (40.0, -2.0)]
        sizes = [len(c) for c in cluster_indices(pts, 30.0)]
        assert sizes == sorted(sizes, reverse=True)

    def test_every_point_lands_in_exactly_one_cluster(self):
        pts = [(30.0, -2.0), (30.02, -2.0), (35.0, -1.0)]
        flat = sorted(i for c in cluster_indices(pts, 30.0) for i in c)
        assert flat == [0, 1, 2]

    def test_empty_input(self):
        assert cluster_indices([], 30.0) == []


class TestBbox:
    def test_padding_expands_all_sides(self):
        b = bbox_of([(30.0, -2.0)], 5.0)
        assert b["min_lon"] < 30.0 < b["max_lon"]
        assert b["min_lat"] < -2.0 < b["max_lat"]

    def test_zero_padding_is_the_point(self):
        b = bbox_of([(30.0, -2.0)], 0.0)
        assert b == {"min_lon": 30.0, "max_lon": 30.0,
                     "min_lat": -2.0, "max_lat": -2.0}

    def test_covers_every_input_point(self):
        pts = [(29.5, -3.1), (30.2, -1.8)]
        b = bbox_of(pts, 2.0)
        for lon, lat in pts:
            assert b["min_lon"] < lon < b["max_lon"]
            assert b["min_lat"] < lat < b["max_lat"]


class TestLoadTierTargets:
    def _write(self, tmp_path, strategies):
        feats = [
            {"type": "Feature",
             "properties": {"lon": 30.0 + 0.1 * i, "lat": -2.0, "rank": rank,
                            "prob": prob, "strategy": strategy},
             "geometry": None}
            for i, (strategy, rank, prob) in enumerate(strategies)
        ]
        path = tmp_path / "tasking_demo.geojson"
        path.write_text(json.dumps({"type": "FeatureCollection",
                                    "features": feats}), encoding="utf-8")
        return path

    def test_splits_by_strategy(self, tmp_path):
        path = self._write(tmp_path, [("top_prob", 1, 0.9),
                                      ("margin", 2, 0.3),
                                      ("top_prob", 3, 0.8)])
        tiers = load_tier_targets(path)
        assert set(tiers) == set(TIERS)
        assert len(tiers["top_prob"]) == 2
        assert len(tiers["margin"]) == 1

    def test_sorts_each_tier_by_rank(self, tmp_path):
        path = self._write(tmp_path, [("top_prob", 5, 0.5),
                                      ("top_prob", 1, 0.9)])
        assert [t["rank"] for t in load_tier_targets(path)["top_prob"]] == [1, 5]

    def test_unknown_strategy_is_dropped(self, tmp_path):
        path = self._write(tmp_path, [("top_prob", 1, 0.9),
                                      ("something_else", 2, 0.3)])
        tiers = load_tier_targets(path)
        assert sum(len(v) for v in tiers.values()) == 1

    def test_no_features_gives_empty_tiers(self, tmp_path):
        path = self._write(tmp_path, [])
        assert all(v == [] for v in load_tier_targets(path).values())


class TestBuildScenes:
    def test_adjacent_cells_collapse_to_one_scene(self):
        """The whole point: bauxite's 10 exploit cells span ~6 x 8 km."""
        targets = [_target(38.25 + 0.005 * i, -4.80 + 0.004 * (i % 3),
                           rank=i + 1)
                   for i in range(10)]
        scenes = build_scenes("bauxite", "top_prob", targets, 30.0, 2.0)
        assert len(scenes) == 1
        assert scenes[0]["n_cells"] == 10

    def test_distant_cells_split(self):
        targets = [_target(29.6, -2.0, rank=1), _target(30.2, -1.2, rank=2)]
        scenes = build_scenes("tin_tungsten_tantalum", "top_prob", targets,
                              30.0, 2.0)
        assert len(scenes) == 2

    def test_scene_ids_are_sequential_and_unique(self):
        targets = [_target(29.6, -2.0, rank=1), _target(30.2, -1.2, rank=2)]
        scenes = build_scenes("tin_tungsten_tantalum", "top_prob", targets,
                              30.0, 2.0)
        ids = [s["scene_id"] for s in scenes]
        assert ids == ["tin_tungsten_tantalum_exploit_01",
                       "tin_tungsten_tantalum_exploit_02"]
        assert len(set(ids)) == len(ids)

    def test_scene_id_encodes_the_explore_tier(self):
        scenes = build_scenes("bauxite", "margin", [_target(38.27, -4.79)],
                              30.0, 2.0)
        assert scenes[0]["scene_id"] == "bauxite_explore_01"

    def test_aggregates_max_prob_and_best_rank(self):
        targets = [_target(38.25, -4.80, rank=7, prob=0.40),
                   _target(38.26, -4.80, rank=3, prob=0.95)]
        scene = build_scenes("bauxite", "top_prob", targets, 30.0, 2.0)[0]
        assert scene["max_prob"] == pytest.approx(0.95)
        assert scene["best_rank"] == 3

    def test_scenes_are_ordered_by_best_rank(self):
        targets = [_target(29.6, -2.0, rank=9), _target(30.2, -1.2, rank=2)]
        scenes = build_scenes("tin_tungsten_tantalum", "top_prob", targets,
                              30.0, 2.0)
        assert [s["best_rank"] for s in scenes] == sorted(
            s["best_rank"] for s in scenes)

    def test_tile_count_is_at_least_one(self):
        scene = build_scenes("bauxite", "top_prob", [_target(38.27, -4.79)],
                             30.0, 2.0)[0]
        assert scene["tiles"] >= 1

    def test_empty_targets_give_no_scenes(self):
        assert build_scenes("bauxite", "top_prob", [], 30.0, 2.0) == []


class TestSeasonContext:
    def test_inside_a_registered_window_names_it(self):
        # usambara_east_extension spans 38.3, -5.2 to 39.1, -4.4
        text = season_context("bauxite", 38.6, -4.8)
        assert "usambara_east_extension" in text
        assert "2023-06-01" in text

    def test_outside_every_window_is_flagged_for_verification(self):
        text = season_context("bauxite", 20.0, 10.0)
        assert "verify season" in text

    def test_never_silently_invents_a_season(self):
        """Outside the registry the answer must be doubt, not a date."""
        text = season_context("bauxite", 20.0, 10.0)
        assert "2023-06-01" not in text


class TestWriteAoi:
    def _scene(self):
        return build_scenes("bauxite", "top_prob", [_target(38.27, -4.79)],
                            30.0, 2.0)[0]

    def test_filename_matches_scene_id(self, tmp_path):
        scene = self._scene()
        assert write_aoi(scene, tmp_path).name == f"{scene['scene_id']}.geojson"

    def test_ring_is_closed_and_rectangular(self, tmp_path):
        ring = json.loads(
            write_aoi(self._scene(), tmp_path).read_text("utf-8")
        )["features"][0]["geometry"]["coordinates"][0]
        assert len(ring) == 5
        assert ring[0] == ring[-1]

    def test_properties_carry_tier_and_guardrail(self, tmp_path):
        props = json.loads(
            write_aoi(self._scene(), tmp_path).read_text("utf-8")
        )["features"][0]["properties"]
        assert props["tier"] == "exploit"
        assert props["n_cells"] == 1
        assert "NOT a label source" in props["purpose"]

    def test_bbox_matches_the_scene(self, tmp_path):
        scene = self._scene()
        ring = json.loads(
            write_aoi(scene, tmp_path).read_text("utf-8")
        )["features"][0]["geometry"]["coordinates"][0]
        lons = [p[0] for p in ring]
        lats = [p[1] for p in ring]
        assert min(lons) == scene["bbox"]["min_lon"]
        assert max(lats) == scene["bbox"]["max_lat"]


class TestPruneStaleAois:
    """A leftover AOI would order ground we no longer want, so drop it."""

    def _group(self, n=1, group="bauxite", tier="top_prob"):
        targets = [_target(38.27 + i * 0.6, -4.79) for i in range(n)]
        return {group: build_scenes(group, tier, targets, 30.0, 2.0)}

    def test_scene_that_no_longer_exists_is_deleted(self, tmp_path):
        scenes = self._group(3)
        for scene in scenes["bauxite"]:
            write_aoi(scene, tmp_path)
        assert len(list(tmp_path.glob("*.geojson"))) == 3

        removed = prune_stale_aois({"bauxite": scenes["bauxite"][:2]},
                                   tmp_path)
        assert [p.name for p in removed] == ["bauxite_exploit_03.geojson"]
        assert not (tmp_path / "bauxite_exploit_03.geojson").exists()
        assert len(list(tmp_path.glob("*.geojson"))) == 2

    def test_current_scenes_survive(self, tmp_path):
        scenes = self._group(3)
        for scene in scenes["bauxite"]:
            write_aoi(scene, tmp_path)
        assert prune_stale_aois(scenes, tmp_path) == []
        assert len(list(tmp_path.glob("*.geojson"))) == 3

    def test_groups_skipped_this_run_are_left_alone(self, tmp_path):
        tin = self._group(1, group="tin_tungsten_tantalum")
        for scene in tin["tin_tungsten_tantalum"]:
            write_aoi(scene, tmp_path)
        bauxite = self._group(1)
        for scene in bauxite["bauxite"]:
            write_aoi(scene, tmp_path)

        assert prune_stale_aois(bauxite, tmp_path) == []
        assert (tmp_path / "tin_tungsten_tantalum_exploit_01.geojson"
                ).exists()

    def test_files_we_did_not_write_are_never_touched(self, tmp_path):
        (tmp_path / "analogue_scene.geojson").write_text("{}", encoding="utf-8")
        (tmp_path / "bauxite_manual_01.geojson").write_text(
            "{}", encoding="utf-8")

        assert prune_stale_aois(self._group(1), tmp_path) == []
        assert (tmp_path / "analogue_scene.geojson").exists()
        assert (tmp_path / "bauxite_manual_01.geojson").exists()

    def test_missing_directory_is_not_an_error(self, tmp_path):
        assert prune_stale_aois(self._group(1), tmp_path / "absent") == []


class TestRenderManifest:
    def _manifest(self):
        scenes = build_scenes("bauxite", "top_prob", [_target(38.27, -4.79)],
                              30.0, 2.0)
        return render_manifest({"bauxite": scenes}, 2.0)

    def test_lists_the_scene_row(self):
        assert "`bauxite_exploit_01`" in self._manifest()

    def test_has_a_tracker_and_totals_section(self):
        text = self._manifest()
        assert "## Order tracker" in text
        assert "## Totals" in text
        assert "1 exploit scene(s) + 0 explore scene(s)" in text

    def test_says_ordering_is_not_scriptable(self):
        """The manifest must not imply a bot could submit these."""
        text = self._manifest()
        assert "no public API" in text
        assert "cannot be cancelled through EOWEB" in text

    def test_tells_the_reader_not_to_hand_over_a_password(self):
        assert "Never commit a portal password" in self._manifest()

    def test_prefers_archive_over_new_acquisition(self):
        assert "archive-first" in self._manifest()

    def test_reports_probe_wavelengths(self):
        # bauxite probes are 2.20 / 0.90 um
        assert "2.20" in self._manifest()


