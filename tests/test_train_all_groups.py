"""``--group all`` must cover the whole catalogue, on each group's own belt.

The training CLI used to default every group to ``configs/karagwe.yml``, so a
re-training sweep could silently put the Copperbelt labels on the Karagwe
feature grid. These tests pin the catalogue mapping, the per-group config
resolution and the fault isolation of the all-groups sweep.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import src.models.main as train_cli
from src.models.catalog import (
    ALL,
    GROUP_CONFIGS,
    config_for_group,
    known_groups,
    resolve_groups,
)
from src.utils import project_path


def _ok_result(group: str) -> dict:
    return {
        "metrics": {"best_algo": "rf",
                    "results": {"rf": {"mean_average_precision": 0.5}}},
        "metrics_path": Path(f"metrics_{group}.json"),
        "bundle_path": Path(f"model_{group}_rf.joblib"),
    }


def _exists(rel_path: str) -> bool:
    return project_path(*rel_path.split("/")).exists()


class TestCatalog:
    def test_every_declared_group_maps_to_an_existing_config(self):
        for group in known_groups():
            assert group in GROUP_CONFIGS, f"{group} missing from catalogue"
            assert _exists(GROUP_CONFIGS[group]), GROUP_CONFIGS[group]

    def test_all_expands_to_every_group_in_belts_yaml(self):
        assert [g for g, _ in resolve_groups(ALL)] == known_groups()

    def test_group_defaults_to_its_own_belt_config(self):
        # footgun guard: copper_zinc must never inherit the KAB config
        assert resolve_groups("copper_zinc") == [
            ("copper_zinc", GROUP_CONFIGS["copper_zinc"])
        ]

    def test_explicit_config_override_wins(self):
        assert config_for_group("bauxite", "configs/other.yml") == (
            "configs/other.yml")

    def test_single_group_never_expands(self):
        assert resolve_groups("bauxite", "configs/custom.yml") == [
            ("bauxite", "configs/custom.yml")]

    def test_unmapped_group_falls_back_with_a_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            got = config_for_group("mystery_metal")
        assert got == "configs/karagwe.yml"
        assert "mystery_metal" in caplog.text


@pytest.fixture
def spy(monkeypatch):
    """Stub out training, prediction and the prior; record every call."""
    calls: dict[str, list] = {"train": [], "predict": [], "rasters": [],
                             "prior": [], "tune": []}

    def fake_train_group(group, config_path=None, **kwargs):
        calls["train"].append((group, str(config_path), kwargs))
        return _ok_result(group)

    def fake_default_feature_rasters(group, config_path=None):
        calls["rasters"].append((group, str(config_path)))
        return [Path("features.tif")]

    def fake_predict_raster(bundle, rasters, **kwargs):
        calls["predict"].append(str(bundle))
        return Path("proba.tif")

    def fake_apply_prior(proba, group):
        calls["prior"].append(group)
        return Path(f"prior_{group}.tif"), {"sources": ["spectral"]}

    def fake_tune_group(group, **kwargs):
        calls["tune"].append((group, kwargs.get("config_path")))
        return {"best_algo": "xgb", "best_mean_ap": 0.6,
                "best_params": {}, "n_trials": 3}

    monkeypatch.setattr(train_cli, "train_group", fake_train_group)
    monkeypatch.setattr(train_cli, "default_feature_rasters",
                        fake_default_feature_rasters)
    monkeypatch.setattr(train_cli, "predict_raster", fake_predict_raster)

    import src.features.priors as priors
    import src.models.tuning as tuning

    monkeypatch.setattr(priors, "apply_prior_to_raster", fake_apply_prior)
    monkeypatch.setattr(
        priors, "lift_report",
        lambda base, prior, group: {"ap_base": 0.1, "ap_prior": 0.2,
                                    "lift": 0.1, "note": "in-sample"})
    monkeypatch.setattr(tuning, "tune_group", fake_tune_group)
    return calls


class TestSweepCLI:
    def test_all_trains_every_group_on_its_own_belt(self, spy):
        assert train_cli.main(["--group", ALL]) == 0
        assert [g for g, _, _ in spy["train"]] == known_groups()
        assert [c for _, c, _ in spy["train"]] == [GROUP_CONFIGS[g]
                                                   for g in known_groups()]

    def test_single_group_keeps_its_mapped_config(self, spy):
        assert train_cli.main(["--group", "copper_zinc"]) == 0
        assert spy["train"][0][:2] == (
            "copper_zinc", GROUP_CONFIGS["copper_zinc"])

    def test_predict_uses_the_resolved_config_for_features(self, spy):
        assert train_cli.main(["--group", "bauxite", "--predict"]) == 0
        assert spy["rasters"] == [("bauxite", GROUP_CONFIGS["bauxite"])]
        assert len(spy["predict"]) == 1

    def test_prior_only_runs_with_predict(self, spy):
        assert train_cli.main(["--group", "bauxite", "--prior"]) == 0
        assert spy["prior"] == []

    def test_prior_conditions_the_predicted_raster(self, spy):
        assert train_cli.main(["--group", ALL, "--predict",
                              "--prior"]) == 0
        assert spy["prior"] == known_groups()

    def test_one_failing_group_does_not_lose_the_others(self, spy, monkeypatch,
                                                       capsys):
        def flaky(group, config_path=None, **kwargs):
            spy["train"].append((group, str(config_path), kwargs))
            if group == known_groups()[-1]:
                raise FileNotFoundError(f"labels_{group}.gpkg missing")
            return _ok_result(group)

        monkeypatch.setattr(train_cli, "train_group", flaky)
        rc = train_cli.main(["--group", ALL])
        out = capsys.readouterr().out

        assert rc == 1
        # every group was attempted, and the healthy ones still reported
        assert [g for g, _, _ in spy["train"]] == known_groups()
        assert f"{len(known_groups()) - 1}/{len(known_groups())} trained" in out
        assert "FAILED: FileNotFoundError" in out

    def test_config_override_is_rejected_for_all(self, spy):
        assert train_cli.main(["--group", ALL,
                               "--config", "configs/karagwe.yml"]) == 2
        assert spy["train"] == []

    def test_unknown_group_is_rejected(self, spy):
        assert train_cli.main(["--group", "gold_placer"]) == 2
        assert spy["train"] == []

    def test_tune_mode_also_expands_all(self, spy):
        assert train_cli.main(["--group", ALL, "--tune"]) == 0
        assert [g for g, _ in spy["tune"]] == known_groups()
        assert spy["train"] == []

    def test_training_flags_are_forwarded(self, spy):
        train_cli.main(["--group", "bauxite", "--algos", "rf", "lgbm",
                        "--block-m", "8000", "--buffer-km", "3",
                        "--pu", "--ensemble", "--oversample"])
        kwargs = spy["train"][0][2]
        assert kwargs["algos"] == ["rf", "lgbm"]
        assert kwargs["block_m"] == 8000
        assert kwargs["buffer_km"] == 3
        assert kwargs["pu"] and kwargs["ensemble"] and kwargs["oversample"]
