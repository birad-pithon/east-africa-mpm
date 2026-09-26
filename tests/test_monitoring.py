"""Tests for the monitoring layer: calibration reports, reliability
diagrams, and model-card generation."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pytest

from src.models.calibration import calibration_report, plot_reliability
from src.models.model_card import build_model_card

# ── calibration report ────────────────────────────────────────────────

class TestCalibrationReport:
    def test_perfect_predictions_give_zero_error(self):
        # probabilities matching the true conditional rate in each group:
        # p=0.3 group has 30% positives, p=0.7 group 70% positives
        y = np.concatenate([np.zeros(70, dtype=int), np.ones(30, dtype=int),
                            np.zeros(30, dtype=int), np.ones(70, dtype=int)])
        p = np.concatenate([np.full(100, 0.3), np.full(100, 0.7)])
        rep = calibration_report(y, p, n_bins=10, min_bin_n=5)
        assert rep["brier"] == pytest.approx(0.21)
        assert rep["max_abs_gap"] == pytest.approx(0.0, abs=1e-9)
        assert rep["ece"] == pytest.approx(0.0, abs=1e-9)

    def test_constant_probability_has_bounded_gap(self):
        y = np.array([0] * 80 + [1] * 20)
        rep = calibration_report(y, np.full(100, 0.5), n_bins=5)
        assert rep["n_bins_usable"] == 1
        assert rep["max_abs_gap"] == pytest.approx(0.3)   # |0.5 - 0.2|
        assert 0.0 <= rep["ece"] <= 1.0

    def test_brier_known_value(self):
        y = np.array([1, 0])
        p = np.array([0.8, 0.4])
        rep = calibration_report(y, p, n_bins=2, min_bin_n=1)
        assert rep["brier"] == pytest.approx(0.5 * (0.04 + 0.16))

    def test_bin_table_counts_sum_to_n(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 500)
        p = rng.uniform(0, 1, 500)
        rep = calibration_report(y, p, n_bins=8, min_bin_n=1)
        assert sum(b["n"] for b in rep["bins"]) == 500
        assert len(rep["bins"]) >= 2

    def test_validation(self):
        with pytest.raises(ValueError, match="same non-empty"):
            calibration_report(np.array([]), np.array([]))
        with pytest.raises(ValueError, match="NaN"):
            calibration_report(np.array([1, 0]), np.array([0.5, np.nan]))
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            calibration_report(np.array([1, 0]), np.array([0.2, 1.4]))

    def test_reliability_plot_written(self, tmp_path):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, 300)
        p = rng.uniform(0, 1, 300)
        out = plot_reliability(y, p, tmp_path / "rel.png")
        assert out.exists() and out.stat().st_size > 0


# ── model card ────────────────────────────────────────────────────────

@pytest.fixture
def trained_artifacts(tmp_path):
    """Minimal bundle + metrics mimicking train_group output."""
    from sklearn.dummy import DummyClassifier

    bundle_path = tmp_path / "model_testgroup_rf.joblib"
    joblib.dump({
        "group": "testgroup", "algo": "rf",
        "model": DummyClassifier().fit([[0], [1]], [0, 1]),
        "feature_names": ["a:b", "c:d"],
        "provenance": {
            "timestamp_utc": "2026-08-28T00:00:00+00:00",
            "git": {"commit": "abc1234", "branch": "main", "dirty": False},
            "packages": {"python": "3.13.3", "scikit-learn": "1.7.0"},
            "input_sha256": {"labels_testgroup.gpkg": "deadbeef" * 4},
        },
    }, bundle_path)

    metrics_path = tmp_path / "metrics_testgroup.json"
    metrics_path.write_text(json.dumps({
        "group": "testgroup", "n_rows": 300, "n_positives": 12,
        "features": ["a:b", "c:d"], "buffer_km": 5, "block_m": 10000,
        "oversample": True, "anomaly_baseline": True, "best_algo": "rf",
        "results": {"rf": {
            "n_folds": 4, "mean_average_precision": 0.21,
            "mean_roc_auc": 0.61, "anomaly_baseline_mean_ap": 0.04,
        }},
        "provenance": {
            "git": {"commit": "abc1234", "branch": "main", "dirty": False},
            "packages": {"python": "3.13.3"},
            "input_sha256": {"labels_testgroup.gpkg": "d" * 64},
        },
        "calibration": {"brier": 0.09, "ece": 0.12, "max_abs_gap": 0.3,
                        "n_bins_usable": 4},
    }, indent=2), encoding="utf-8")
    return {"metrics": metrics_path, "bundle": bundle_path,
            "dir": tmp_path}


class TestModelCard:
    def test_card_written_from_artifacts(self, trained_artifacts):
        e = trained_artifacts
        card = build_model_card("testgroup",
                                metrics_path=e["metrics"],
                                bundle_path=e["bundle"])
        text = card.read_text(encoding="utf-8")
        assert card.name == "model_card_testgroup.md"
        for section in ("Intended use", "Training data",
                        "Evaluation protocol", "Results",
                        "Probability calibration", "Limitations",
                        "Provenance"):
            assert section in text
        assert "abc1234" in text                       # git commit
        assert "0.210" in text                         # rf AP
        assert "0.040" in text                         # anomaly baseline
        assert "deadbeef" not in text                  # metrics prov wins
        assert "dddddddddddddddd" in text              # 16-char hash prefix
        assert "Brier" in text and "0.090" in text

    def test_card_without_calibration(self, trained_artifacts):
        e = trained_artifacts
        metrics = json.loads(e["metrics"].read_text(encoding="utf-8"))
        metrics.pop("calibration")
        e["metrics"].write_text(json.dumps(metrics), encoding="utf-8")
        card = build_model_card("testgroup",
                                metrics_path=e["metrics"],
                                bundle_path=e["bundle"])
        assert "not yet computed" in card.read_text(encoding="utf-8")

    def test_missing_metrics_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_model_card("nope", metrics_path=tmp_path / "none.json")
