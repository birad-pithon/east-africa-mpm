"""Tests for the acquisition-request CLI (scripts/gen_tasking_requests.py).

These polygons feed ``scripts/gen_enmap_order_manifest.py``, whose AOIs are
submitted by hand to EOWEB - where an order cannot be changed and cannot be
cancelled through the portal at all. Which probability raster the ranking runs
on is therefore an ordering-safety property, not a convenience: the CLI must
resolve the raster that the *currently shipped* model produced.
"""
from __future__ import annotations

import json
import sys

import pytest

import scripts.gen_tasking_requests as gtr
from src.models.catalog import GROUP_CONFIGS, known_groups
from src.utils import project_path


class TestCatalog:
    def test_configs_come_from_the_trained_catalogue(self):
        assert gtr.CATALOG is GROUP_CONFIGS

    def test_every_known_group_is_requestable(self):
        for group in known_groups():
            assert group in gtr.CATALOG, f"{group} cannot be tasked"


class TestShippedProba:
    def test_resolves_the_winning_algorithm_from_metrics(self, tmp_path,
                                                          monkeypatch):
        models = tmp_path / "outputs" / "models"
        models.mkdir(parents=True)
        (models / "metrics_bauxite.json").write_text(
            json.dumps({"best_algo": "xgb"}), encoding="utf-8")
        monkeypatch.setattr(gtr, "project_path",
                            lambda *p: tmp_path.joinpath(*p))

        assert gtr.shipped_proba("bauxite") == \
            "outputs/models/proba_bauxite_xgb.tif"

    def test_follows_a_retrain_to_a_new_algorithm(self, tmp_path, monkeypatch):
        """The point of the resolver: metrics changes, raster path changes."""
        models = tmp_path / "outputs" / "models"
        models.mkdir(parents=True)
        monkeypatch.setattr(gtr, "project_path",
                            lambda *p: tmp_path.joinpath(*p))
        for algo in ("rf", "xgb", "lgbm"):
            (models / "metrics_copper_zinc.json").write_text(
                json.dumps({"best_algo": algo}), encoding="utf-8")
            assert gtr.shipped_proba("copper_zinc") == \
                f"outputs/models/proba_copper_zinc_{algo}.tif"

    def test_missing_metrics_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gtr, "project_path",
                            lambda *p: tmp_path.joinpath(*p))
        assert gtr.shipped_proba("bauxite") is None

    def test_metrics_without_best_algo_returns_none(self, tmp_path,
                                                    monkeypatch):
        models = tmp_path / "outputs" / "models"
        models.mkdir(parents=True)
        (models / "metrics_bauxite.json").write_text(json.dumps({}),
                                                     encoding="utf-8")
        monkeypatch.setattr(gtr, "project_path",
                            lambda *p: tmp_path.joinpath(*p))
        assert gtr.shipped_proba("bauxite") is None

    def test_shipped_raster_matches_metrics_and_exists(self):
        """Against real artifacts: the resolved path is the shipped model's."""
        for group in known_groups():
            metrics = project_path("outputs", "models",
                                   f"metrics_{group}.json")
            if not metrics.exists():
                pytest.skip(f"{group} not trained in this checkout")
            algo = json.loads(metrics.read_text(encoding="utf-8"))["best_algo"]
            rel = gtr.shipped_proba(group)
            assert rel == f"outputs/models/proba_{group}_{algo}.tif"
            if not project_path(rel).exists():
                pytest.skip(f"{rel} not generated in this checkout")
            assert project_path(rel).exists()


class TestMainGroupSelection:
    def test_neither_group_nor_all_is_a_usage_error(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["gen_tasking_requests.py"])
        with pytest.raises(SystemExit):
            gtr.main()

    def test_unknown_group_writes_nothing(self, tmp_path, monkeypatch,
                                          caplog):
        monkeypatch.setattr(sys, "argv",
                            ["gen_tasking_requests.py", "--group", "nope"])
        monkeypatch.setattr(gtr, "project_path",
                            lambda *p: tmp_path.joinpath(*p))
        gtr.main()                               # must not raise
        assert not (tmp_path / "outputs" / "maps" /
                    "tasking_nope.geojson").exists()
        assert "known groups" in caplog.text

    def test_explicit_proba_bypasses_the_resolver(self, tmp_path, monkeypatch):
        def boom(group):
            raise AssertionError("resolver must not run when --proba is given")

        monkeypatch.setattr(sys, "argv",
                            ["gen_tasking_requests.py", "--group", "bauxite",
                             "--proba", "outputs/models/custom.tif"])
        monkeypatch.setattr(gtr, "project_path",
                            lambda *p: tmp_path.joinpath(*p))
        monkeypatch.setattr(gtr, "shipped_proba", boom)

        gtr.main()          # raster is absent on purpose: nothing is written

        assert not (tmp_path / "outputs" / "maps" /
                    "tasking_bauxite.geojson").exists()
