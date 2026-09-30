"""Ingestion modules for the East Africa MPM pipeline."""

from src.ingest.brgm_geology import ingest_brgm_geology
from src.ingest.dlr_eoc_wms import (
    DASHBOARD_LAYERS,
    VERIFIED_LAYERS,
    fetch_belt_context,
    fetch_wms_map,
    layer_summary,
    save_wms_map,
)
from src.ingest.national_surveys import (
    get_survey_data,
    list_request_sources,
    print_requests,
    register_local_data,
)
from src.ingest.sentinel2_ee import (
    aoi_from_config,
    bounds_from_config,
    init_ee,
    pull_sentinel2_tile,
)
from src.ingest.srtm_dem import EE_COLLECTIONS, pull_dem_tile
from src.ingest.usgs_africa import download_zip, ingest_usgs_africa
from src.ingest.worldgeol import BELT_BBOX, ROCK_TYPE_LABELS, ingest_world_geology

__all__ = [
    "ingest_usgs_africa",
    "download_zip",
    "pull_sentinel2_tile",
    "init_ee",
    "aoi_from_config",
    "bounds_from_config",
    "pull_dem_tile",
    "EE_COLLECTIONS",
    "ingest_brgm_geology",
    "ingest_world_geology",
    "BELT_BBOX",
    "ROCK_TYPE_LABELS",
    "list_request_sources",
    "register_local_data",
    "get_survey_data",
    "print_requests",
    "VERIFIED_LAYERS",
    "DASHBOARD_LAYERS",
    "layer_summary",
    "fetch_wms_map",
    "save_wms_map",
    "fetch_belt_context",
]
