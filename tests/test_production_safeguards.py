"""Tests for production-readiness safeguards: run provenance,
feature-raster allowlist, and background extent clipping."""
from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box

from src.labels.build_labels import aoi_extent, sample_background, seed_points
from src.models.dataset import default_feature_rasters
from src.utils.provenance import collect_provenance, sha256_file

# provenance: deterministic hashing + config snapshot + package versions

class TestProvenance:
    def test_hashes_and_missing(self, tmp_path):
        f = tmp_path / "in.tif"
        f.write_bytes(b"raster-bytes")
        prov = collect_provenance(input_paths=[f, tmp_path / "nope.tif"])
        assert prov["input_sha256"]["in.tif"] == sha256_file(f)
        assert prov["input_sha256"]["nope.tif"] == "missing"

    def test_sha256_stable_and_correct(self, tmp_path):
        f = tmp_path / "x.bin"
        f.write_bytes(b"abc")
        assert sha256_file(f) == (
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )

    def test_git_and_packages_recorded(self):
        prov = collect_provenance()
        assert set(prov["git"]) == {"commit", "branch", "dirty"}
        assert prov["git"]["commit"]
        assert "numpy" in prov["packages"]
        assert prov["packages"]["python"]

    def test_config_snapshot(self):
        prov = collect_provenance(config_path="configs/karagwe.yml")
        assert "grid" in prov["config_snapshot"]


# feature allowlist: only configured rasters fed to the model (no stray TIFFs)

class TestFeatureAllowlist:
    def test_allowlisted_group_resolves_config_files(self):
        paths = default_feature_rasters("tin_tungsten_tantalum")
        names = {p.name for p in paths}
        assert names == {
            "srtm_kabar_aligned.tif",
            "sentinel2_kabar_aligned.tif",
            "terrain_features_kabar.tif",
            "geology_features_kabar.tif",
            "alteration_features_kabar.tif",
            "deposit_distance_tin_tungsten_tantalum.tif",
        }

    def test_stray_tif_excluded_from_allowlist(self, tmp_path, monkeypatch):
        """A file in data/interim NOT on the allowlist must be ignored."""
        from src.utils import project_path

        root = project_path("data", "interim")
        stray = root / "_stray_probe.tif"
        stray.write_bytes(b"x")
        try:
            paths = default_feature_rasters("tin_tungsten_tantalum")
            assert stray not in paths
        finally:
            stray.unlink()

    def test_unknown_group_falls_back_to_glob(self):
        paths = default_feature_rasters("group_without_allowlist")
        assert paths                       # glob fallback still returns files


# provenance inputs: the manifest must list the rasters that actually
# trained the model (per-belt allowlist), never every *.tif in data/interim

BELT_CONFIGS = [
    ("tin_tungsten_tantalum", "configs/karagwe.yml", "srtm_kabar_aligned.tif"),
    ("copper_zinc", "configs/copperbelt.yml", "srtm_copperbelt_aligned.tif"),
    ("bauxite", "configs/usambara.yml", "srtm_usambara_aligned.tif"),
]


class _StopAfterProvenance(Exception):
    """Sentinel used to abort ``train_group`` right after provenance."""


class TestProvenanceInputs:
    @pytest.mark.parametrize("group,config,expected", BELT_CONFIGS)
    def test_only_allowlisted_rasters_recorded(self, group, config, expected):
        """The per-belt config selects exactly the 6 belt rasters."""
        from src.models.train import provenance_input_paths

        paths = provenance_input_paths(group, config_path=config)
        gens = [p for p in paths if str(p).endswith(".gpkg")]
        rasters = {p.name for p in paths if str(p).endswith(".tif")}
        # label + background GPKG first, then the allowlisted rasters
        assert len(gens) == 2
        assert expected in rasters
        assert len(rasters) == 6
        # stray / other-belt / un-namespaced layers must NOT appear
        assert "terrain_features.tif" not in rasters
        assert "geology_features.tif" not in rasters
        for other in ("kab", "copperbelt", "usambara"):
            if other not in expected:
                assert not any(other in name for name in rasters)

    def test_no_allowlist_fallback_warning(self, caplog):
        """Regression: resolving provenance without ``config_path`` logged a
        bogus 'no explicit feature allowlist' warning and returned every
        TIFF under data/interim (mixed UTM grids included)."""
        from src.models.train import provenance_input_paths

        with caplog.at_level("WARNING"):
            provenance_input_paths("copper_zinc",
                                   config_path="configs/copperbelt.yml")
        assert "no explicit feature allowlist" not in caplog.text

    def test_explicit_raster_paths_take_precedence(self):
        from src.models.train import provenance_input_paths

        explicit = ["data/interim/features/terrain_features_copperbelt.tif"]
        paths = provenance_input_paths("copper_zinc",
                                       config_path="configs/copperbelt.yml",
                                       raster_paths=explicit)
        assert [str(p) for p in paths] == [
            str(paths[0]), str(paths[1]), *explicit,
        ]

    def test_train_group_resolves_provenance_with_passed_config(
            self, tmp_path, monkeypatch):
        """Wiring: ``train_group`` calls the helper. Stop it immediately
        after provenance so no model is fitted or artifact written."""
        import pandas as pd

        from src.models import train as tr

        monkeypatch.setattr(
            tr, "prepare_training_table",
            lambda *a, **k: (
                {"grid": {"crs": "EPSG:32736"}}, {}, ["rf"],
                pd.DataFrame({"a": [1.0, 2.0]}), np.array([1, 0]),
                None, np.zeros((2, 2)), 10_000.0, 5.0,
            ),
        )
        seen: dict = {}

        def spy(*args, **kwargs):
            seen["args"] = args
            seen.update(kwargs)
            raise _StopAfterProvenance

        monkeypatch.setattr(tr, "provenance_input_paths", spy)

        with pytest.raises(_StopAfterProvenance):
            tr.train_group("synth_prov",
                           config_path="configs/copperbelt.yml",
                           algos=["rf"], output_dir=tmp_path)
        assert seen["config_path"] == "configs/copperbelt.yml"
        assert seen["args"][0] == "synth_prov"


# background extent clip: strata sampled inside the raster AOI only

class TestBackgroundExtentClip:
    def test_extent_restricts_sampling(self):
        pos = seed_points("tin_tungsten_tantalum")
        belts = ["karagwe_ankole"]
        extent = box(29.4, -2.4, 30.6, -1.2)      # raster AOI
        bg = sample_background(pos, belts, n=60, seed=1, extent=extent)
        assert len(bg) > 0
        minx, miny, maxx, maxy = bg.total_bounds
        assert minx >= 29.4 and maxx <= 30.6
        assert miny >= -2.4 and maxy <= -1.2

    def test_extent_respects_strata_and_exclusion(self):
        pos = seed_points("tin_tungsten_tantalum")
        extent = box(29.4, -2.4, 30.6, -1.2)
        bg = sample_background(pos, ["karagwe_ankole"], n=80, seed=2, extent=extent,
                               min_dist_m=5_000)
        assert set(bg["stratum"]).issubset({"barren_halo", "greenfield"})
        # min distance from positives still enforced
        from scipy.spatial import cKDTree
        pos_utm = pos.to_crs("EPSG:32736")
        bg_utm = bg.to_crs("EPSG:32736")
        tree = cKDTree(np.column_stack(
            [pos_utm.geometry.x, pos_utm.geometry.y]))
        d, _ = tree.query(np.column_stack(
            [bg_utm.geometry.x, bg_utm.geometry.y]))
        assert (d > 5_000).all()

    def test_aoi_extent_from_config(self):
        ext = aoi_extent("configs/karagwe.yml")
        assert ext.bounds == (29.4, -2.4, 30.6, -1.2)

    def test_empty_intersection_raises(self):
        pos = seed_points("tin_tungsten_tantalum")
        far = box(0.0, 0.0, 1.0, 1.0)             # nowhere near the belt
        with pytest.raises(ValueError, match="empty"):
            sample_background(pos, ["karagwe_ankole"], n=10, seed=0, extent=far)
