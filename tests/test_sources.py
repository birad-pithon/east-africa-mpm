"""Tests for the data-source registry, BRGM geology, DEM and national-surveys modules.

Run with:  pytest tests/test_sources.py -v

These tests are network-free (no live downloads) and cover:
  * the master data-source registry (configs/data_sources.yaml)
  * the DEM module config + synthetic fallback on the common grid
  * the national-surveys manifest register/get round-trip
  * the BRGM module import + clip helper behaviour
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
import rasterio

# ─── 1. Master data-source registry ────────────────────────────────────

EXPECTED_LAYERS = {
    "Regional geological maps",
    "Known deposits / occurrences",
    "Aeromagnetic & gravity surveys",
    "Multispectral imagery",
    "Digital elevation model",
    "Stream sediment / soil geochemistry",
    "Cadastral / licence boundaries",
}


class TestDataSourceRegistry:
    def test_registry_loads(self):
        """data_sources.yaml parses and exposes a sources dict."""
        from src.utils import load_config

        cfg = load_config("configs/data_sources.yaml")
        assert "sources" in cfg
        assert len(cfg["sources"]) >= 15

    def test_all_data_layers_covered(self):
        """Every layer from the project brief is represented."""
        from src.utils import load_config

        cfg = load_config("configs/data_sources.yaml")
        layers = {s["layer"] for s in cfg["sources"].values()}
        assert EXPECTED_LAYERS <= layers

    def test_usgs_mrds_is_configured(self):
        """USGS MRDS is a configured (verified) source."""
        from src.utils import load_config

        cfg = load_config("configs/data_sources.yaml")
        mrds = cfg["sources"]["usgs_mrds"]
        assert mrds["status"] == "configured"
        assert mrds["url"].endswith("mrds-trim.zip")

    def test_national_surveys_present(self):
        """RMB, GST, CAMI and DGSM geology entries exist."""
        from src.utils import load_config

        cfg = load_config("configs/data_sources.yaml")
        for key in ("rmb_geology", "gst_geology", "cami_geology", "dgsm_geology"):
            assert key in cfg["sources"], f"missing {key}"
            assert cfg["sources"][key]["access"] == "direct-request"

    def test_dem_sources_present(self):
        from src.utils import load_config

        cfg = load_config("configs/data_sources.yaml")
        assert cfg["sources"]["srtm_dem"]["ee_collection"] == "USGS/SRTMGL1_003"
        assert cfg["sources"]["glo30_dem"]["ee_collection"] == "COPERNICUS/DEM/GLO30"


# ─── 2. DEM module ─────────────────────────────────────────────────────


class TestDemModule:
    def test_collections_mapping(self):
        from src.ingest.srtm_dem import EE_COLLECTIONS

        assert EE_COLLECTIONS["srtm"] == "USGS/SRTMGL1_003"
        assert EE_COLLECTIONS["glo30"] == "COPERNICUS/DEM/GLO30"

    def test_synthetic_dem_on_common_grid(self, common_grid, tmp_path):
        """Synthetic DEM fallback writes a 1-band raster on the common grid."""
        from src.ingest.srtm_dem import _create_synthetic_dem

        out = tmp_path / "syn_dem.tif"
        _create_synthetic_dem(
            out,
            common_grid["bounds_utm"],
            common_grid["transform"],
            common_grid["width"],
            common_grid["height"],
        )
        with rasterio.open(out) as src:
            assert src.crs.to_string() == common_grid["crs"]
            assert src.width == common_grid["width"]
            assert src.height == common_grid["height"]
            assert src.count == 1
            assert src.dtypes[0] == "float32"

    def test_pull_dem_creates_aligned_raster(self, tmp_path):
        """pull_dem_tile produces a grid-aligned file even without EE."""
        from src.ingest.srtm_dem import pull_dem_tile

        out = tmp_path / "dem_aligned.tif"
        path = pull_dem_tile("configs/karagwe.yml", source="srtm",
                             out_path=out)
        assert Path(path) == out
        assert Path(path).exists()
        with rasterio.open(path) as src:
            assert src.crs.to_string() == "EPSG:32736"
            assert src.count == 1

    def test_unknown_source_raises(self):
        from src.ingest.srtm_dem import pull_dem_tile

        with pytest.raises(ValueError):
            pull_dem_tile("configs/karagwe.yml", source="nope")


# ─── 3. National surveys / direct-request framework ────────────────────


class TestNationalSurveys:
    def test_request_sources_listed(self):
        from src.ingest.national_surveys import list_request_sources

        reg = list_request_sources()
        assert "rmb_geology" in reg
        assert "cami_cadastre" in reg

    def test_register_get_roundtrip(self, tmp_path):
        """register_local_data + get_survey_data round-trip with provenance."""
        from src.ingest import national_surveys as ns

        gdf = gpd.GeoDataFrame(
            {"name": ["test"]},
            geometry=gpd.points_from_xy([30.0], [-1.5]),
            crs="EPSG:4326",
        )
        sample = tmp_path / "rmb_sample.gpkg"
        gdf.to_file(sample, driver="GPKG")

        orig_base, orig_manifest = ns.NATIONAL_BASE, ns.MANIFEST_PATH
        ns.NATIONAL_BASE = tmp_path / "national"
        ns.MANIFEST_PATH = ns.NATIONAL_BASE / "MANIFEST.json"
        try:
            dest = ns.register_local_data(
                "rmb_geology", sample, crs="EPSG:4326", notes="test delivery"
            )
            assert dest.exists()
            got = ns.get_survey_data("rmb_geology")
            assert got is not None and got == dest
            assert "rmb_geology" in ns._load_manifest()["sources"]
        finally:
            ns.NATIONAL_BASE, ns.MANIFEST_PATH = orig_base, orig_manifest

    def test_print_requests_renders(self):
        from src.ingest.national_surveys import print_requests

        text = print_requests()
        assert "National survey" in text
        assert "rmb_geology" in text

    def test_get_unregistered_returns_none(self):
        from src.ingest.national_surveys import get_survey_data

        assert get_survey_data("definitely_not_registered") is None


# ─── 4. BRGM geology module ────────────────────────────────────────────


class TestBrgmModule:
    def test_import_and_bbox(self):
        from src.ingest.brgm_geology import EAST_AFRICA_BBOX, ingest_brgm_geology

        assert EAST_AFRICA_BBOX["min_lon"] < EAST_AFRICA_BBOX["max_lon"]
        assert callable(ingest_brgm_geology)

    def test_missing_local_archive_raises(self):
        """Without a local archive or a configured URL the module fails cleanly."""
        from src.ingest.brgm_geology import ingest_brgm_geology

        with pytest.raises((FileNotFoundError, KeyError, OSError)):
            ingest_brgm_geology(local_archive="definitely/missing/brgm.zip")


# ─── 5. DLR EOC OGC web map services ───────────────────────────────────


class TestDlrEocWmsRegistry:
    """The two DLR EOC WMS endpoints are registered with verified facts."""

    def _sources(self):
        from src.utils import load_config

        return load_config("configs/data_sources.yaml")["sources"]

    def test_both_services_registered(self):
        sources = self._sources()
        assert sources["dlr_eoc_imagery_wms"]["access"] == "wms"
        assert sources["dlr_eoc_land_wms"]["access"] == "wms"
        assert sources["dlr_eoc_imagery_wms"]["status"] == "pending"
        assert sources["dlr_eoc_land_wms"]["status"] == "configured"

    def test_urls_point_at_the_dlr_geoservice(self):
        sources = self._sources()
        for key in ("dlr_eoc_imagery_wms", "dlr_eoc_land_wms"):
            entry = sources[key]
            assert entry["url"].startswith("https://geoservice.dlr.de/eoc/")
            assert entry["url"].endswith("/wms")
            assert "GetCapabilities" in entry["capabilities_url"]

    def test_verified_layer_lists_are_populated(self):
        sources = self._sources()
        land = sources["dlr_eoc_land_wms"]["verified_layers"]
        names = {item["name"] for item in land}
        # The layers that actually rendered over the belts.
        assert {"SOILSUITE_SRC_AFR_P4Y", "TDM_FNF50", "WSF_2019",
                "TS_LANDSAT_2015"} <= names
        for item in land:
            assert set(item) == {"name", "style", "verdict"}

    def test_imagery_service_records_the_empty_enmap_layers(self):
        sources = self._sources()
        imagery = sources["dlr_eoc_imagery_wms"]
        verdicts = {item["name"]: item["verdict"]
                    for item in imagery["verified_layers"]}
        assert verdicts["ENMAP_HSI_L0_QL_SWIR"].startswith("empty")
        assert "evaluation_note" in imagery


class TestDlrEocWmsModule:
    """Module helpers, the verified table and blank-tile detection."""

    def test_verified_table_has_unique_wms_layers(self):
        from src.ingest.dlr_eoc_wms import SERVICE_URLS, VERIFIED_LAYERS

        wms_names = [spec["layer"] for spec in VERIFIED_LAYERS.values()]
        assert len(wms_names) == len(set(wms_names))
        for spec in VERIFIED_LAYERS.values():
            assert spec["service"] in SERVICE_URLS
            assert set(spec) >= {"service", "layer", "style", "title",
                                 "coverage", "use", "opaque", "status"}

    def test_registry_and_module_layers_agree(self):
        from src.ingest.dlr_eoc_wms import VERIFIED_LAYERS
        from src.utils import load_config

        sources = load_config("configs/data_sources.yaml")["sources"]
        by_wms_name = {spec["layer"]: spec for spec in VERIFIED_LAYERS.values()}
        for item in sources["dlr_eoc_land_wms"]["verified_layers"]:
            assert item["name"] in by_wms_name, f"{item['name']} not in module table"
            assert by_wms_name[item["name"]]["style"] == item["style"]
            assert by_wms_name[item["name"]]["status"] == "usable"

    def test_enmap_layers_flagged_empty(self):
        from src.ingest.dlr_eoc_wms import VERIFIED_LAYERS

        for key in ("enmap_l0_quicklook_footprints", "enmap_l0_quicklook_swir"):
            spec = VERIFIED_LAYERS[key]
            assert spec["status"] == "empty"
            assert set(spec["opaque"].values()) == {None}

    def test_dashboard_layers_are_usable_and_exclude_enmap(self):
        from src.ingest.dlr_eoc_wms import DASHBOARD_LAYERS, VERIFIED_LAYERS

        assert DASHBOARD_LAYERS, "dashboard needs at least one verified layer"
        for key in DASHBOARD_LAYERS:
            assert VERIFIED_LAYERS[key]["status"] == "usable"
            assert "enmap" not in key

    def test_mercator_conversion_known_values(self):
        from src.ingest.dlr_eoc_wms import bbox_to_mercator, wgs84_to_mercator

        assert wgs84_to_mercator(0.0, 0.0) == pytest.approx((0.0, 0.0), abs=1e-6)
        assert wgs84_to_mercator(180.0, 0.0)[0] == pytest.approx(
            20037508.342789244, rel=1e-9)
        x0, y0, x1, y1 = bbox_to_mercator((37.9, -5.6, 39.1, -4.1))
        assert x0 < x1 and y0 < y1
        # Literal expected bounds (EPSG:3857 metres) for the Usambara window.
        assert x0 == pytest.approx(4219008.701065068, rel=1e-9)
        assert x1 == pytest.approx(4352592.0900169965, rel=1e-9)
        assert y0 == pytest.approx(-624384.0442539449, rel=1e-9)
        assert y1 == pytest.approx(-456799.92850204714, rel=1e-9)

    def test_getmap_url_contains_wms_parameters(self):
        from src.ingest.dlr_eoc_wms import build_getmap_url

        url = build_getmap_url("soilsuite_src_africa", (37.9, -5.6, 39.1, -4.1),
                               width=256, height=256)
        for fragment in ("service=WMS", "version=1.3.0", "request=GetMap",
                         "SOILSUITE_SRC_AFR_P4Y",
                         "soilsuite-src-afr-p4y-fin", "EPSG%3A3857",
                         "width=256", "format=image%2Fpng"):
            assert fragment in url, fragment
        assert url.startswith("https://geoservice.dlr.de/eoc/land/wms?")

    def test_getmap_url_accepts_wms_layer_name(self):
        from src.ingest.dlr_eoc_wms import build_getmap_url

        by_key = build_getmap_url("tandemx_forest_nonforest", (0, 0, 1, 1))
        by_name = build_getmap_url("TDM_FNF50", (0, 0, 1, 1))
        assert by_key == by_name

    def test_unknown_layer_raises(self):
        from src.ingest.dlr_eoc_wms import build_getmap_url

        with pytest.raises(KeyError):
            build_getmap_url("not_a_layer", (0, 0, 1, 1))

    def test_unknown_belt_raises_before_any_request(self):
        from src.ingest.dlr_eoc_wms import fetch_belt_context

        with pytest.raises(KeyError):
            fetch_belt_context("atlantis")

    def test_blank_tile_detection(self):
        import io

        from PIL import Image

        from src.ingest.dlr_eoc_wms import is_blank_tile, opaque_fraction

        empty = io.BytesIO()
        Image.new("RGBA", (16, 16), (0, 0, 0, 0)).save(empty, format="PNG")
        filled = io.BytesIO()
        Image.new("RGBA", (16, 16), (10, 200, 90, 255)).save(filled, format="PNG")

        assert opaque_fraction(empty.getvalue()) == 0.0
        assert is_blank_tile(empty.getvalue())
        assert opaque_fraction(filled.getvalue()) == 1.0
        assert not is_blank_tile(filled.getvalue())

    def test_layer_summary_documents_both_services(self):
        from src.ingest.dlr_eoc_wms import layer_summary

        text = layer_summary()
        assert "geoservice.dlr.de/eoc/imagery/wms" in text
        assert "geoservice.dlr.de/eoc/land/wms" in text
        assert "enmap_l0_quicklook_swir" in text

