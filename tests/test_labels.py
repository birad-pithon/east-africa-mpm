"""Tests for positive-label and background generation (src/labels).

Run with:  pytest tests/test_labels.py -v

Uses the real USGS MRDS Africa subset already downloaded into
``data/raw/usgs_africa_minerals.gpkg`` plus the curated seed table.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from src.labels import (
    SEED_DEPOSITS,
    belt_polygon,
    build_labels,
    mrds_in_belts,
    sample_background,
    seed_points,
)
from src.utils import load_config, project_path

BELTS = load_config(project_path("configs", "belts.yaml"))


# ─── Seed table ─────────────────────────────────────────────────────────

class TestSeeds:
    def test_all_brief_deposits_present(self):
        names = {d["name"] for d in SEED_DEPOSITS}
        assert {
            "Nyakabingo", "Musha", "Ntunga", "Gatumba",
            "Kipushi", "Kamoa-Kakula", "Lushoto", "Magamba",
        } <= names

    def test_seed_counts_per_group(self):
        assert len(seed_points("tin_tungsten_tantalum")) == 4   # Rwanda
        assert len(seed_points("copper_zinc")) == 2             # DRC
        assert len(seed_points("bauxite")) == 2                 # Tanzania

    def test_every_seed_inside_its_belt(self):
        """Each curated seed must fall within its configured belt polygon."""
        for d in SEED_DEPOSITS:
            poly = belt_polygon(d["belt"])
            pt = __import__("shapely").geometry.Point(d["lon"], d["lat"])
            assert poly.contains(pt), f"{d['name']} outside {d['belt']}"


# ─── MRDS filtering ─────────────────────────────────────────────────────

class TestMrdsInBelts:
    def test_copperbelt_returns_records(self):
        gdf, belts = mrds_in_belts("copper_zinc")
        assert len(gdf) > 0
        assert "central_african_copperbelt" in belts

    def test_kab_returns_records(self):
        gdf, _ = mrds_in_belts("tin_tungsten_tantalum")
        assert len(gdf) > 0

    def test_commodity_filter_respected(self):
        """Every returned record carries at least one group commodity code."""
        codes = {c.upper() for c in BELTS_CFG()["groups"]["copper_zinc"]["mrds_codes"]}
        gdf, _ = mrds_in_belts("copper_zinc")
        tokens = gdf["CODE_LIST"].fillna("").astype(str).str.upper().str.split()
        assert all(set(t) & codes for t in tokens)


def BELTS_CFG():
    return BELTS


# ─── Label building ─────────────────────────────────────────────────────

class TestBuildLabels:
    def test_copper_zinc_contains_seeds_and_mrds(self, tmp_path):
        pos, path = build_labels("copper_zinc",
                                 out_path=tmp_path / "labels.gpkg")
        assert path.exists()
        names = set(pos["name"].dropna())
        assert {"Kipushi", "Kamoa-Kakula"} <= names
        sources = set(pos["source"])
        assert {"seed", "mrds"} <= sources
        assert (pos["label"] == 1).all()

    def test_no_mrds_within_seed_merge_distance(self, tmp_path):
        """Dedup guarantee: no MRDS point sits on top of a curated seed."""
        from scipy.spatial import cKDTree

        pos, _ = build_labels("copper_zinc", seed_merge_m=2_000,
                              out_path=tmp_path / "labels.gpkg")
        epsg = "EPSG:327" + ("35" if pos.geometry.x.mean() < 12 else "36")
        # Copperbelt spans ~21-31E -> zones 34-36; use generic zone from mean
        from src.labels.build_labels import utm_epsg
        epsg = utm_epsg(float(pos.geometry.x.mean()))
        utm = pos.to_crs(epsg)
        seeds = utm[utm["source"] == "seed"]
        mrds = utm[utm["source"] == "mrds"]
        if len(seeds) and len(mrds):
            tree = cKDTree(np.column_stack(
                [seeds.geometry.x, seeds.geometry.y]))
            d, _ = tree.query(np.column_stack(
                [mrds.geometry.x, mrds.geometry.y]), k=1)
            assert (d > 2_000).all()

    def test_kab_group_has_rwanda_seeds(self, tmp_path):
        pos, _ = build_labels("tin_tungsten_tantalum",
                              out_path=tmp_path / "labels.gpkg")
        names = set(pos["name"].dropna())
        assert {"Nyakabingo", "Musha", "Ntunga", "Gatumba"} <= names


# ─── Background sampling ────────────────────────────────────────────────

class TestBackgroundSampling:
    def test_min_distance_from_positives_enforced(self, tmp_path):
        pos, _ = build_labels("bauxite", out_path=tmp_path / "labels.gpkg")
        bg = sample_background(pos, ["usambara_bauxite"], n=150,
                               min_dist_m=5_000, seed=1)
        from src.labels.build_labels import utm_epsg
        epsg = utm_epsg(float(pos.geometry.x.mean()))
        p_utm = pos.to_crs(epsg)
        b_utm = bg.to_crs(epsg)
        tree = cKDTree(np.column_stack([p_utm.geometry.x, p_utm.geometry.y]))
        d, _ = tree.query(np.column_stack([b_utm.geometry.x, b_utm.geometry.y]))
        assert (d > 5_000).all()

    def test_background_inside_belt_and_labelled(self, tmp_path):
        pos, _ = build_labels("bauxite", out_path=tmp_path / "labels.gpkg")
        bg = sample_background(pos, ["usambara_bauxite"], n=100, seed=2)
        poly = belt_polygon("usambara_bauxite")
        assert (bg["label"] == 0).all()
        assert (bg["source"] == "background").all()
        assert all(poly.contains(g) for g in bg.geometry)

    def test_requested_n_returned_when_feasible(self, tmp_path):
        pos, _ = build_labels("copper_zinc",
                              out_path=tmp_path / "labels.gpkg")
        bg = sample_background(pos, ["central_african_copperbelt"],
                               n=500, min_dist_m=5_000, seed=3)
        assert len(bg) == 500
