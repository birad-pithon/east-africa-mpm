# East Africa MPM — Mineral Prospectivity Mapping

A modular Python project for mineral **prospectivity mapping** across the
East African region, using a machine-learning pipeline that takes raw
geospatial data through ingestion, preprocessing, feature engineering,
label generation, model training, spatial validation, and map export.

**Target commodities:** tin, tungsten, tantalum (Karagwe-Ankole Belt);
copper, zinc (Central African Copperbelt); bauxite (lateritic plateaus, Tanzania).

## Project Structure

```
east-africa-mpm/
|-- data/
|   |-- raw/              # Untouched source downloads (MRDS geodatabase, S2 tiles)
|   |-- interim/          # Reprojected / aligned layers (common 30 m grid)
|   `-- processed/        # Feature-ready, model-ready datasets
|-- notebooks/            # Exploratory analysis, QA visualisation
|-- src/
|   |-- ingest/           # Per-source download & access scripts
|   |   ├── usgs_africa.py     # USGS MRDS global database → Africa subset
|   |   ├── sentinel2_ee.py    # Sentinel-2 composite via Google Earth Engine
|   |   ├── srtm_dem.py        # SRTM 30 m / Copernicus GLO-30 DEM (via EE)
|   |   ├── brgm_geology.py    # BRGM 1:10M Africa geology (hosts unreachable; kept for swap-in)
|   |   ├── worldgeol.py       # USGS/GSC world geology WFS — production lithology layer
|   |   └── national_surveys.py# RMB/GST/CAMI/DGSM request + provenance manifest
|   |-- preprocess/       # Reprojection, resampling, grid alignment (+ grid_check)
|   |-- features/         # Commodity-specific feature engineering (+ geology_features)
|   |-- labels/           # Known-deposit labels + field-verified label merging
|   |-- models/           # Training, calibration reports, model cards
|   |-- validate/         # Spatial cross-validation utilities + leakage certification
|   |-- predict/          # Inference + prospectivity map export
|   `-- utils/            # Shared grid / reprojection / provenance helpers
|-- configs/
|   |-- karagwe.yml       # KAB (Sn/W/Ta) — CRS, resolution, sources, model
|-- outputs/
|   |-- models/           # Trained model artefacts
|   `-- maps/             # Exported GeoTIFF / interactive HTML maps
|-- tests/                # Unit & integration tests (see below)
|-- conftest.py           # Shared pytest fixtures (config, grid, data)
|-- pyproject.toml        # Project metadata + dependencies
|-- README.md
`-- .gitignore
```

## Data Sources

Every data source consumed by the pipeline is registered centrally in
**`configs/data_sources.yaml`** — the single source of truth for purpose,
provider, access method, format, target commodities and integration status.
Seven data layers are covered:

| # | Data layer | Purpose | Sources |
|---|---|---|---|
| 1 | **Regional geological maps** | Lithology & structure base | **USGS/GSC world geology WFS** (`configured`); BRGM 1:10M Africa (hosts unreachable); RMB, GST, CAMU/CAMI, DGSM (direct request) |
| 2 | **Known deposits / occurrences** | Positive training labels | USGS MRDS (download, *verified*); USGS Africa compilation; deposit-genesis papers |
| 3 | **Aeromagnetic & gravity** | Structural / intrusion detection | NCEI EMAG2v3, BGI WGM2012; airborne data from NI 43-101 / JORC filings |
| 4 | **Multispectral imagery** | Laterite & alteration mapping | Sentinel-2 (EE, *verified*); Landsat 8/9; ASTER (EarthExplorer) |
| 5 | **Digital elevation model** | Plateau geomorphology, lineaments | SRTM 30 m / Copernicus GLO-30 (EE) |
| 6 | **Stream sediment / soil geochemistry** | Cu-Zn-Sn anomaly surfaces | National geochemical databases (direct request); historical reports |
| 7 | **Cadastral / licence boundaries** | Exclude licensed ground | DRC CAMI cadastre; Tanzania Mining Commission; RMB registry |

**Analogue-informed target windows:** `configs/analogue_targets.yaml`
registers 17 permissive scene-selection windows across the 7 countries,
derived by transferring deposit-model criteria from ~15 world-class
analogue districts (see `ANALOGUE_DEPOSITS_EVALUATION.md` and section 20
of `DATA_SOURCES_SUMMARY.md`). Windows are permissive extents only —
never labels. The `src/ingest/analogue_scene.py` module turns a window
into an ingest-ready scene config (`configs/scenes/<id>.yml`) and runs
the SRTM + Sentinel-2 dry-run; EnMAP order parameters for the 7
high-confidence targets live in `ENMAP_ORDERING_CHECKLIST.md`.

**Ingestion modules** (`src/ingest/`):

- `usgs_africa.py` — USGS MRDS (`mrds-trim.zip`, *verified*) → Africa subset
- `sentinel2_ee.py` — Sentinel-2 dry-season composite via Earth Engine
- `srtm_dem.py` — SRTM 30 m / Copernicus GLO-30 DEM (EE), grid-aligned
- `analogue_scene.py` — registry loader: window → scene config / dry-run ingest
- `worldgeol.py` — **USGS/GSC General geologic map of the world** via WFS
  1.1.0 (`geology` polygons + `contacts` lines), clipped and cached to
  `data/raw/world_geology/` — the *production* lithology layer, since the
  BRGM/SIGAfrique hosts are unreachable; swap back by re-pointing
  `configs/data_sources.yaml` when they return
- `brgm_geology.py` — BRGM 1:10M Africa geology (higher resolution than
  worldgeol when obtainable; currently offline)
- `national_surveys.py` — direct-request queue + provenance `MANIFEST.json`

**Direct-request tracking.** Sources without a public bulk URL (national
geological surveys, geochemistry, licence cadastres) are tracked through a
provenance manifest under `data/raw/national/`. When an authority delivers
a file, register it so the rest of the pipeline can find it:

```bash
python -m src.ingest.national_surveys --print-requests   # see pending queue
python -m src.ingest.national_surveys \
    --register rmb_geology "data/raw/national/rmb/rmb_geology.gpkg" --crs EPSG:32736
python -m src.ingest.national_surveys --get rmb_geology # -> stored path
```

> **USGS "Compilation of Geospatial Data for Mineral Industries … of
> Africa"** — registry entry `usgs_africa_compilation` (attribute `pending-url`).
> It is a ScienceBase data release (ESRI File Geodatabase); drop the sciencebase
> download URL into `configs/data_sources.yaml` to activate a dedicated pull.

> **OneGeology** aggregator is being wound down in 2026 — treat as *archival
> only*. Prefer the national surveys directly (Rwandan RWB; Tanzania GTS;
> DRC CAMU; Uganda DGSM).

## Common Analysis Grid

All layers are harmonised to a single grid before modelling:

- **CRS:** `EPSG:32736` (UTM Zone 36S) — config-driven, suitable for Rwanda/KAB.
- **Resolution:** 30 m (matches SRTM / Landsat Sentinel-2 composite).
- **AOI:** Karagwe-Ankole Belt box in WGS-84 (config): `(29.4, -2.4) → (30.6, -1.2)`.
- **Origin snapping:** grid origin is snapped to exact multiples of the
  resolution, so raster tiles from any source align pixel-to-pixel.

Grid math lives in `src/utils/__init__.py` (`make_grid_transform`,
`wgs84_to_utm`, `grid_dimensions`, `reproject_raster_to_grid`).

## First-Session Deliverables — Status

1. **Repository scaffold** — complete (structure above).
2. **Environment** — `pyproject.toml` declares the full geospatial + ML
   stack (numpy, pandas, geopandas, rasterio, xarray, rioxarray, pyproj,
   shapely, rasterstats, earthengine-api, scikit-learn, xgboost, lightgbm,
   matplotlib, leafmap, folium, pyyaml).
3. **USGS Africa mineral ingestion** — `src/ingest/usgs_africa.py`
   downloads `mrds-trim.zip`, extracts, filters to Africa (bbox) and to
   target commodities (Sn/W/Ta/Cu/Zn/Al via `CODE_LIST`), and writes
   `data/raw/usgs_africa_minerals.gpkg` (680 records covering Africa).
4. **Sentinel-2 pull via Earth Engine** — `src/ingest/sentinel2_ee.py`
   builds a cloud-masked, dry-season median composite over the KAB AOI and
   exports/reprojects it to the common grid.
5. **DEM ingestion** — `src/ingest/srtm_dem.py` pulls SRTM 30 m or
   Copernicus GLO-30 and aligns it to the same grid.
6. **Geology base layer** — `src/ingest/worldgeol.py` ingests the
   USGS/GSC 1:35M world geology via WFS (production lithology/contacts
   layer; `src/ingest/brgm_geology.py` remains for the higher-res BRGM
   1:10M map whose hosts are currently unreachable).
7. **National survey framework** — `src/ingest/national_surveys.py` tracks
   the RMB / GST / CAMI / DGSM direct-request queue and a provenance
   manifest for locally-delivered data.
8. **Data-source registry** — `configs/data_sources.yaml` encodes all seven
   data layers and 21 sources from the project brief.
9. **Grid-alignment confirmation** — tests in `tests/test_ingest_align.py`
   and `tests/test_sources.py` verify USGS points reproject into `EPSG:32736`
   and rasters land exactly on the common 30 m grid
   (`pytest` → **140 passed** at last full run, incl. the five
   pipeline-safeguard suites).

## Quick Start

```bash
# 1. Install the environment
pip install -e .                    # from project root

# 2. Ingest USGS MRDS mineral data for Africa
python -m src.ingest.usgs_africa --config configs/karagwe.yml

# 3. Authenticate Earth Engine once, then pull imagery + DEM
earthengine authenticate
python -m src.ingest.sentinel2_ee --config configs/karagwe.yml
python -m src.ingest.srtm_dem   --config configs/karagwe.yml --source srtm

# 4. Pull the production geology layer (USGS/GSC world geology via WFS)
python -m src.ingest.worldgeol

# 5. Build labels + AOI-clipped background for a commodity group
python -m src.labels.build_labels --group tin_tungsten_tantalum \
    --background 320 --clip-aoi configs/karagwe.yml

# 6. Train + export probability map, calibration report, and model card
python -m src.models.main --group tin_tungsten_tantalum \
    --oversample --anomaly-baseline --predict

# 7. Review which national surveys need a direct request, and register
#    any data you receive from them
python -m src.ingest.national_surveys --print-requests

# 8. Run the test suite (140 tests incl. pipeline-safeguard suites)
pytest -v
```

## Next Steps

Where the pipeline goes from here, in priority order:

1. **More in-extent positives** (highest impact on AP). Tin has 18
   positives but only 9 fall inside the raster footprint. Per-group AOI
   configs (copper_zinc at lon 24–28 and bauxite at lon 38.3 currently
   have *zero* raster coverage) + the Safeguard-5 field loop to grow
   verified labels.
2. **Higher-resolution national geology** — request the RMB / GST /
   DGSM / CAMI 1:50k–1:100k maps via `configs/data_sources.yaml`
   (`access: direct-request`); they drop straight into
   `src.features.geology_features` with no code changes and should
   cross the current ~1:35M GSC lithology resolution threshold.
3. **Groundproofing** — rank unexplored cells (`src.predict.main`),
   route top candidates through `field_update` when results come back.
4. The five pipeline safeguards, provenance, calibration, and model-card
   layers are complete and tested (140 passing).

## Pipeline Safeguards

Five failure modes identified in the project brief are actively defended
in code — each is enforced, logged, and unit-tested:

1. **Spatial leakage** — buffered block CV only (`src.validate`); the
   fold plan is *certified* (`certify_folds`) for train/test separation
   before any model is fit. A violation aborts the run.
2. **Class imbalance** — deterministic SMOTE on training folds only
   (`--oversample`); test folds are never touched.
3. **Sampling bias** — background drawn in two strata: a `barren_halo`
   ring (5–25 km around known deposits, `halo_fraction`) plus
   `greenfield` far-field, correcting the "model learns
   distance-to-camp" bias. Background is clipped to the raster AOI
   (`--clip-aoi`) by construction.
4. **Resolution/grid mismatch** — `src.preprocess.grid_check`
   (`assert_common_grid`) compares CRS/affine/shape against the
   reference grid and raises `GridMismatchError` before misaligned
   layers are stacked. Feature rasters additionally resolve through an
   explicit per-group allowlist (`features.rasters_by_group` in
   `configs/karagwe.yml`), so stray TIFFs cannot enter training.
5. **Feedback loop** — field-verified labels are merged into the label
   sets via `src.labels.field_update` (dedup + stale-label override +
   versioned `label_manifest.json`), so retraining automatically picks
   up ground truth.

Reproducibility: every training run embeds full **provenance** in
`metrics_<group>.json` and the model bundle — git commit/branch/dirty
flag, SHA-256 of every input file, package versions, and a config
snapshot (`src/utils/provenance.py`).

## Labels (`src/labels/`)

Positive labels merge two sources per commodity group:

1. **Curated seed deposits** — Nyakabingo, Musha, Ntunga, Gatumba (Rwanda
   Sn-W-Ta); Kipushi, Kamoa-Kakula (DRC Cu-Zn); Lushoto, Magamba
   (Tanzania bauxite) — literature-geocoded in `src/labels/known_deposits.py`.
2. **USGS MRDS occurrences** filtered by *belt polygon ∩ commodity code*
   (belt boundaries in `configs/belts.yaml`, hand-digitised & flagged for
   refinement against national survey maps).

MRDS points within 2 km of a seed are treated as duplicates (seed wins).
Background negatives are rejection-sampled inside the belt union with a
5 km minimum distance from every positive — guarding against the
sampling-bias failure mode.

```bash
python -m src.labels.build_labels --group copper_zinc --background 5000
# -> data/processed/labels_copper_zinc.gpkg + background_copper_zinc.gpkg
```

Current label counts: Sn/W/Ta **18** positives; Cu/Zn **78**; bauxite **3**
(bauxite is seed-driven until national geochemical/laterite mapping lands).

## Model Training (`src/models/`)

Per-commodity baselines with **buffered blocked spatial CV only**
(`src.validate`) — random k-fold is never used. Primary metric is
average precision (PR-AUC), which stays honest at small positive-label
counts; the fold plan is certified for spatial separation before
fitting, and an IsolationForest-on-barren **anomaly baseline** is
reported as the null reference every model must beat.

```bash
# train + compare RF / XGBoost / LightGBM for one group
python -m src.models.main --group tin_tungsten_tantalum \
    --oversample --anomaly-baseline

# custom CV geometry, subset of algorithms, + full-grid probability map
python -m src.models.main --group tin_tungsten_tantalum \
    --algos rf xgb --block-m 8000 --buffer-km 3 --oversample \
    --anomaly-baseline --predict
```

Outputs (`outputs/models/`) per run:

- `metrics_<group>.json` — per-fold + summary metrics, calibration
  summary, and full run **provenance** (git, input SHA-256, packages,
  config snapshot)
- `model_<group>_<algo>.joblib` — best-by-AP bundle (estimator +
  feature names + CV config + provenance)
- `proba_<group>_<algo>.tif` — full-grid P(deposit) raster, grid-aligned
- `calibration_<group>.json` + `reliability_<group>.png` — Brier / ECE /
  reliability diagram computed from **out-of-fold** predictions
- `model_card_<group>.md` — generated summary for reviewers (intended
  use, data, protocol, results, calibration, limitations, provenance)

**Current real-data result (Sn-W-Ta, 18 positives / 320 AOI-clipped
backgrounds, 25 features incl. geology):** RF best at **mean AP ≈ 0.079**
over 5 certified buffered-block folds, ROC-AUC ≈ 0.531 — vs the ~0.027
class base rate and ~0.047 anomaly baseline, i.e. real but modest signal.
An earlier reported AP of 0.43 was inflated: it ranked "far from any
known camp" over geologically comparable ground precisely the sampling
bias Safeguard 3 and the AOI-clipped background are built to expose.

## Candidate Ranking & Maps (`src/predict/`)

Ranks the top-N **unexplored** cells from any probability raster:
cells inside licence polygons are excluded, a greedy minimum-spacing
keeps one target per anomaly instead of N clustered pixels, and NaN /
low-probability cells are dropped.

```bash
python -m src.predict.main \
    --proba outputs/models/proba_tin_tungsten_tantalum_rf.tif \
    --group tin_tungsten_tantalum --n 25 --map
# -> outputs/maps/top25_tin_tungsten_tantalum_candidates.geojson
# -> outputs/maps/top25_tin_tungsten_tantalum_map.html  (leafmap/folium)
```

Licence layers (CAMI / Tanzanian Mining Commission / RMB cadastres) are
supplied as GeoJSON/GPKG via `--licences`; without one, ranking runs on
the unmasked field.

## Web Dashboard

A Streamlit-based web dashboard is available to explore trained model metrics,
probability rasters, reliability diagrams, model cards, and ranked exploration
candidates interactively.

**Install extras** (once):

```bash
pip install -e ".[dashboard]"
```

**Run the dashboard**:

```bash
streamlit run app.py
```

The dashboard opens in your browser and contains six tabs:

| Tab | Content |
|---|---|
| **Overview** | Summary metrics cards, configuration, feature list, algorithm comparison bar chart |
| **Cross-Validation** | Per-fold AP / ROC-AUC bar charts, fold detail table, spatial-CV certification note |
| **Probability Map** | Interactive folium map with the probability raster overlay and candidate markers |
| **Candidates** | Top-N ranked target table, scatter map, CSV/GeoJSON download buttons |
| **Calibration** | Brier score, ECE, reliability bin table, reliability diagram chart + PNG |
| **Model Card** | Full markdown model card for the selected commodity group |

Use the sidebar to switch between commodity groups (Sn-W-Ta, Cu-Zn, Bauxite).

| Layer | Tools | Status |
|---|---|---|
| Language / environment | Python 3.11+, `venv` / conda | ✅ Python 3.13 + `pip install -e .` |
| Geospatial core | rasterio, geopandas, GDAL/OGR, shapely, pyproj | ✅ installed & used throughout |
| Earth observation access | `earthengine-api` (GEE), Copernicus Data Space API (`sentinelhub`), `landsatxplore` (USGS archives) | ✅ installed; GEE + CDSE wired in `src/ingest/` |
| Machine learning | scikit-learn, xgboost, lightgbm | ✅ installed (PyTorch deferred — see extras) |
| Spatial validation | `verde`, or custom spatial k-fold / buffered blocks (**not** sklearn's random CV) | ✅ `src/validate/spatial_cv.py`: `SpatialBlockCV` (grid blocks) + `BufferedSpatialCV` (KD-tree exclusion buffer), both sklearn-`cv=` compatible |
| Visualisation / QA | matplotlib, leafmap / folium; QGIS for manual review | ✅ installed (QGIS is external/manual) |
| Data management | DVC (optional) for versioning large rasters | ⏳ optional extra: `pip install -e ".[dvc]"` |

Optional extras (declared in `pyproject.toml`, not installed by default):

```bash
pip install -e ".[cnn]"   # PyTorch + torchvision — only for image-patch CNNs later
pip install -e ".[dvc]"   # DVC dataset versioning
pip install -e ".[dashboard]"  # Streamlit web dashboard
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for coding standards, testing
conventions, and how to add new data sources or commodities.

## License

Specify the intended license here.
