"""Offline tests for src.models.loo: the full LOO rediscovery driver.

No network and no retraining: synthetic label frames, synthetic harness
result payloads, and ``tmp_path`` only — production artifacts under
``data/processed/`` and ``outputs/models/`` are never touched.
"""

from __future__ import annotations

import json
import sys

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from src.models.loo import (
    compact_record,
    fold_slug,
    folds_from_labels,
    loo_markdown,
    loo_report,
    main,
    run_loo,
    summarize,
)


def _labels() -> gpd.GeoDataFrame:
    """Five deposit points (EPSG:4326) with a name column."""
    return gpd.GeoDataFrame(
        {
            "name": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
            "geometry": [
                Point(30.0, -1.0),
                Point(30.5, -1.5),
                Point(31.0, -2.0),
                Point(31.5, -2.5),
                Point(32.0, -3.0),
            ],
        },
        crs="EPSG:4326",
    )


def _result(rank, *, best_algo="rf", ap=0.5, n_remaining=8, in_extent=True):
    """Minimal harness result payload shaped like run_holdout_test's."""
    return {
        "group": "demo",
        "best_algo": best_algo,
        "n_labels_removed": 1,
        "n_labels_remaining": n_remaining,
        "metrics": {"results": {best_algo: {"mean_average_precision": ap}}},
        "metrics_path": "metrics.json",
        "proba_raster": "proba.tif",
        "sites": [
            {
                "name": "Alpha",
                "lon": 30.0,
                "lat": -1.0,
                "in_extent": in_extent,
                "rank": rank,
                "percentile": (100.0 if rank is None else 100.0 * (1.0 - (rank - 1) / 1_000_000.0)),
                "rediscovered": bool(rank is not None and rank <= 100),
                "prob_at_holdout": 0.9,
                "n_valid_cells": 1_000_000,
                "distance_to_top1_km": 2.5,
                "n_top50_within_tolerance": 3,
            }
        ],
        # model the harness: only populated when rank <= 50
        "rediscovery": {
            "rank_in_top50": rank if (rank is not None and rank <= 50) else None,
            "distance_to_top1_km": 2.5,
            "num_candidates_within_5km": 3,
            "percentile": (100.0 if rank is None else 100.0 * (1.0 - (rank - 1) / 1_000_000.0)),
            "rediscovered": bool(rank is not None and rank <= 100),
        },
    }


def _fold(idx, name="Alpha"):
    """Fold descriptor as produced by folds_from_labels."""
    return {"fold_idx": idx, "label_row": idx, "name": name, "lon": 30.0, "lat": -1.0}


class TestFoldSlug:
    def test_spaces_become_underscores(self):
        assert fold_slug(3, "Tenke Fungurume") == "fold_03_Tenke_Fungurume"

    def test_width_zero_pads(self):
        assert fold_slug(7, "A", width=3).startswith("fold_007_")

    def test_unsafe_characters_are_replaced(self):
        slug = fold_slug(0, "A/B:C*D?E")
        payload = slug.split("fold_00_")[1]
        assert not set(payload) & set('/:*?\\<>|"')
        assert payload.replace("_", "") == "ABCDE"

    def test_collapses_repeated_separators(self):
        assert "__" not in fold_slug(0, "A  /  B").split("fold_00_")[1]

    def test_empty_name_falls_back(self):
        assert fold_slug(1, "   ") == "fold_01_unnamed"
        assert fold_slug(1, "///") == "fold_01_unnamed"

    def test_long_name_is_truncated(self):
        slug = fold_slug(2, "X" * 200)
        assert len(slug.split("fold_02_")[1]) == 48


class TestFoldsFromLabels:
    def test_indexes_in_extent_subset_only(self):
        labels = _labels()
        mask = np.array([True, False, True, False, True])
        folds = folds_from_labels(labels, mask)
        assert [f["fold_idx"] for f in folds] == [0, 1, 2]
        assert [f["name"] for f in folds] == ["Alpha", "Gamma", "Epsilon"]

    def test_label_row_preserves_original_position(self):
        labels = _labels()
        mask = np.array([False, False, True, True, False])
        folds = folds_from_labels(labels, mask)
        assert [f["label_row"] for f in folds] == [2, 3]

    def test_coordinates_are_floats(self):
        folds = folds_from_labels(_labels(), np.ones(5, dtype=bool))
        assert all(isinstance(f["lon"], float) for f in folds)
        assert folds[0]["lon"] == pytest.approx(30.0)

    def test_mask_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="mask length"):
            folds_from_labels(_labels(), np.array([True, False]))

    def test_empty_mask_yields_no_folds(self):
        assert folds_from_labels(_labels(), np.zeros(5, dtype=bool)) == []


class TestCompactRecord:
    def test_reads_full_rank_beyond_top50(self):
        """Regression: the rediscovery block hides ranks > 50."""
        rec = compact_record(_result(8_824), _fold(0))
        assert rec["rank"] == pytest.approx(8_824)
        assert rec["rank_in_top50"] is None
        assert rec["rediscovered"] is False

    def test_marks_rediscovered_small_rank(self):
        rec = compact_record(_result(1), _fold(0))
        assert rec["rank"] == pytest.approx(1)
        assert rec["rank_in_top50"] == 1
        assert rec["rediscovered"] is True

    def test_extracts_mean_ap_for_best_algo(self):
        rec = compact_record(_result(5, best_algo="lgbm", ap=0.431), _fold(0))
        assert rec["best_algo"] == "lgbm"
        assert rec["mean_ap"] == pytest.approx(0.431)

    def test_positives_trained_on_counts_sampled_rows(self):
        """Out-of-extent labels never reach the model, so metrics wins."""
        result = _result(5, n_remaining=17)  # 17 labels survive removal ...
        result["metrics"]["n_positives"] = 8  # ... but only 8 are in extent
        assert compact_record(result, _fold(0))["n_positives_trained_on"] == 8

    def test_positives_trained_on_falls_back_to_labels_remaining(self):
        result = _result(5, n_remaining=4)
        assert compact_record(result, _fold(0))["n_positives_trained_on"] == 4

    def test_missing_metrics_yields_nan(self):
        result = _result(5)
        result["metrics"] = {}
        assert np.isnan(compact_record(result, _fold(0))["mean_ap"])

    def test_absent_sites_falls_back_to_rediscovery(self):
        result = _result(3)
        result["sites"] = []
        rec = compact_record(result, _fold(0))
        assert rec["rank"] == pytest.approx(3)
        assert rec["in_extent"] is True

    def test_out_of_extent_site_is_nan_and_flagged(self):
        result = _result(None, in_extent=False)
        result["rediscovery"] = {}
        rec = compact_record(result, _fold(0))
        assert rec["in_extent"] is False
        assert np.isnan(rec["rank"])
        assert rec["rediscovered"] is False

    def test_non_finite_site_rank_falls_back_to_summary(self):
        """An inf in the site block uses the summary rank, not a NaN."""
        result = _result(5)
        result["sites"][0]["rank"] = float("inf")
        assert compact_record(result, _fold(0))["rank"] == pytest.approx(5)

    def test_non_finite_rank_everywhere_becomes_nan(self):
        result = _result(5)
        result["sites"][0]["rank"] = float("inf")
        result["rediscovery"]["rank_in_top50"] = None
        rec = compact_record(result, _fold(0))
        assert np.isnan(rec["rank"])
        assert rec["rediscovered"] is False
        # the site block's explicit flag still wins over the rank heuristic
        assert rec["in_extent"] is True

    def test_in_extent_heuristic_when_site_block_is_absent(self):
        result = _result(5)
        result["sites"] = []
        assert compact_record(result, _fold(0))["in_extent"] is True
        result["sites"] = []
        result["rediscovery"]["rank_in_top50"] = None
        assert compact_record(result, _fold(0))["in_extent"] is False

    def test_fold_identity_is_preserved(self):
        rec = compact_record(_result(2), _fold(4, "Gamma"))
        assert (rec["fold_idx"], rec["name"], rec["label_row"]) == (4, "Gamma", 4)


class TestSummarize:
    def test_rate_uses_scored_folds_only(self):
        records = [
            {
                "fold_idx": 0,
                "rank": 1,
                "rediscovered": True,
                "in_extent": True,
                "percentile": 100.0,
                "distance_to_top1_km": 1.0,
                "mean_ap": 0.5,
                "best_algo": "rf",
                "n_positives_trained_on": 8,
            },
            {
                "fold_idx": 1,
                "rank": 5_000,
                "rediscovered": False,
                "in_extent": True,
                "percentile": 99.99,
                "distance_to_top1_km": 50.0,
                "mean_ap": 0.4,
                "best_algo": "rf",
                "n_positives_trained_on": 8,
            },
            {
                "fold_idx": 2,
                "rank": float("nan"),
                "rediscovered": False,
                "in_extent": False,
                "percentile": float("nan"),
                "distance_to_top1_km": float("nan"),
                "mean_ap": float("nan"),
                "best_algo": "rf",
                "n_positives_trained_on": 8,
                "name": "Outside",
            },
        ]
        summary = summarize("demo", records)
        assert summary["n_folds"] == 3
        assert summary["n_folds_scored"] == 2
        assert summary["n_folds_unscored"] == 1
        assert summary["n_rediscovered"] == 1
        assert summary["rediscovery_rate"] == pytest.approx(0.5)
        assert summary["unscored_folds"] == ["Outside"]

    def test_rank_statistics(self):
        records = [
            {
                "fold_idx": i,
                "rank": r,
                "in_extent": True,
                "rediscovered": False,
                "percentile": 90.0,
                "distance_to_top1_km": 10.0,
                "mean_ap": 0.2,
                "best_algo": "xgb",
                "n_positives_trained_on": 5,
            }
            for i, r in enumerate([10, 20, 30, 1_000])
        ]
        summary = summarize("demo", records)
        assert summary["rank_min"] == pytest.approx(10)
        assert summary["rank_max"] == pytest.approx(1_000)
        assert summary["rank_median"] == pytest.approx(25.0)

    def test_top_1pct_rate_uses_percentile(self):
        records = [
            {
                "fold_idx": 0,
                "rank": 900,
                "in_extent": True,
                "rediscovered": False,
                "percentile": 99.5,
                "distance_to_top1_km": 5.0,
                "mean_ap": 0.3,
                "best_algo": "rf",
                "n_positives_trained_on": 8,
            },
            {
                "fold_idx": 1,
                "rank": 900_000,
                "in_extent": True,
                "rediscovered": False,
                "percentile": 98.0,
                "distance_to_top1_km": 90.0,
                "mean_ap": 0.3,
                "best_algo": "rf",
                "n_positives_trained_on": 8,
            },
        ]
        summary = summarize("demo", records)
        assert summary["n_top_1pct"] == 1
        assert summary["rate_top_1pct"] == pytest.approx(0.5)

    def test_counts_best_algorithms(self):
        records = [
            {
                "fold_idx": 0,
                "rank": 1,
                "in_extent": True,
                "rediscovered": True,
                "percentile": 100.0,
                "distance_to_top1_km": 1.0,
                "mean_ap": 0.5,
                "best_algo": "lgbm",
                "n_positives_trained_on": 2,
            },
            {
                "fold_idx": 1,
                "rank": 2,
                "in_extent": True,
                "rediscovered": True,
                "percentile": 100.0,
                "distance_to_top1_km": 1.0,
                "mean_ap": 0.5,
                "best_algo": "rf",
                "n_positives_trained_on": 3,
            },
        ]
        summary = summarize("demo", records)
        assert summary["best_algo_counts"] == {"lgbm": 1, "rf": 1}
        assert summary["n_positives_trained_on_min"] == 2

    def test_empty_records_gives_nan_rate(self):
        summary = summarize("demo", [])
        assert summary["n_folds"] == 0
        assert np.isnan(summary["rediscovery_rate"])
        assert np.isnan(summary["rank_median"])
        assert summary["n_rediscovered"] == 0


def _payload(group="demo", ranks=(1, 250, 5_000)):
    """Group payload as run_loo returns it, built from synthetic results."""
    records = [compact_record(_result(r), _fold(i, f"D{i}")) for i, r in enumerate(ranks)]
    return {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "group": group,
        "n_folds_total": len(records),
        "n_folds_done": len(records),
        "errors": {},
        "summary": summarize(group, records),
        "records": records,
    }


class TestReports:
    def test_json_report_holds_every_group(self, tmp_path):
        path = loo_report([_payload("a"), _payload("b")], tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["n_groups"] == 2
        assert [g["group"] for g in payload["groups"]] == ["a", "b"]

    def test_json_report_survives_nan_values(self, tmp_path):
        path = loo_report([_payload("empty", ranks=())], tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "NaN" in text  # emitted by the numpy-aware encoder, not a crash

    def test_markdown_contains_rate_and_fold_rows(self, tmp_path):
        path = loo_markdown([_payload("demo")], tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "Leave-one-deposit-out" in text
        assert "33.3 %" in text  # 1 of 3 folds rediscovered
        for name in ("D0", "D1", "D2"):
            assert name in text
        assert "| yes |" in text and "| no |" in text

    def test_markdown_reports_nan_as_not_available(self, tmp_path):
        text = loo_markdown([_payload("empty", ranks=())], tmp_path).read_text(encoding="utf-8")
        assert "n/a" in text

    def test_markdown_handles_no_groups(self, tmp_path):
        path = loo_markdown([], tmp_path)
        assert path.exists()


class TestRunLooResume:
    def test_no_folds_writes_summary_without_training(self, tmp_path):
        payload = run_loo("bauxite", folds=[], out_root=tmp_path)
        assert payload["records"] == []
        assert payload["summary"]["n_folds"] == 0
        saved = json.loads((tmp_path / "bauxite" / "loo_summary.json").read_text("utf-8"))
        assert saved["n_folds_done"] == 0

    def test_partial_state_skips_completed_fold(self, tmp_path):
        """A recorded fold must be reused, never retrained."""
        group_dir = tmp_path / "bauxite"
        group_dir.mkdir(parents=True)
        existing = compact_record(_result(42), _fold(0, "Lushoto"))
        (group_dir / "loo_partial.json").write_text(
            json.dumps(
                {"group": "bauxite", "n_folds_total": 3, "records": [existing], "errors": {}}
            ),
            encoding="utf-8",
        )
        payload = run_loo("bauxite", folds=[_fold(0, "Lushoto")], out_root=tmp_path)
        assert payload["n_folds_done"] == 1
        assert payload["records"][0]["rank"] == pytest.approx(42)

    def test_no_resume_ignores_partial_state(self, tmp_path, monkeypatch):
        """--no-resume forces a retrain path (stubbed) even with state."""
        group_dir = tmp_path / "bauxite"
        group_dir.mkdir(parents=True)
        (group_dir / "loo_partial.json").write_text(
            json.dumps({"records": [compact_record(_result(42), _fold(0))]}), encoding="utf-8"
        )

        import src.models.loo as loo_mod

        calls: list[int] = []

        def _fake(group, **kwargs):
            calls.append(kwargs.get("holdout_idx"))
            return _result(7)

        monkeypatch.setattr(loo_mod, "run_holdout_test", _fake)
        payload = run_loo("bauxite", folds=[_fold(0, "Lushoto")], out_root=tmp_path, resume=False)
        assert calls == [0]
        assert payload["records"][0]["rank"] == pytest.approx(7)

    def test_fold_failure_is_recorded_not_fatal(self, tmp_path, monkeypatch):
        import src.models.loo as loo_mod

        def _boom(group, **kwargs):
            raise RuntimeError("simulated training failure")

        monkeypatch.setattr(loo_mod, "run_holdout_test", _boom)
        payload = run_loo("bauxite", folds=[_fold(0, "Lushoto")], out_root=tmp_path)
        assert payload["records"] == []
        assert "0" in payload["errors"]
        assert "simulated" in payload["errors"]["0"]


class TestCli:
    def test_prints_help_without_arguments(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["loo_validation.py"])
        main()
        assert "leave-one-deposit-out" in capsys.readouterr().out

    def test_report_only_aggregates_saved_partial(self, tmp_path, monkeypatch):
        group_dir = tmp_path / "bauxite"
        group_dir.mkdir(parents=True)
        record = compact_record(_result(3), _fold(0, "Lushoto"))
        (group_dir / "loo_partial.json").write_text(
            json.dumps({"group": "bauxite", "n_folds_total": 1, "records": [record], "errors": {}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "loo_validation.py",
                "--group",
                "bauxite",
                "--report-only",
                "--out-root",
                str(tmp_path),
            ],
        )
        main()  # must not retrain, must not sys.exit
        assert (tmp_path / "loo_report.json").exists()
        assert (tmp_path / "LOO_VALIDATION.md").exists()
