# EnMAP ordering checklist — analogue target scenes

**Date:** 2026-09-23
**Scope:** the 7 high-confidence windows in `configs/analogue_targets.yaml`
**Companion:** `ANALOGUE_DEPOSITS_EVALUATION.md`, `DATA_SOURCES_SUMMARY.md` §18/§20

Facts below verified against https://www.enmap.org/data_access,
https://www.enmap.org/data_tools/foreground_mission and
https://www.enmap.org/data_tools/planning on 2026-09-23.

## 0. Credential boundary — what works WITHOUT a login (verified 2026-09-23)

| Step | Anonymous? | Notes |
|---|---|---|
| Per-target AOI GeoJSON: `python -m src.ingest.analogue_scene --aoi --confidence high` → `data/raw/enmap/aois/*.geojson` | **yes** | offline; drag into the EOWEB map or use §4 coordinates |
| Parameter sheets (bbox, dry season, probes) — §4 | **yes** | offline, from `configs/analogue_targets.yaml` |
| Browse the EOC Geoservice catalogue (`geoservice.dlr.de`) — EnMAP HSI L1B / L1C global datasets are listed with OGC services "Online" | **yes** | catalogue/quicklook discovery only; no scripted scene-footprint search found anonymously (OGC endpoint URLs are only published inside the JS map UI) |
| Search the EnMAP archive + download scenes (EOWEB GeoPortal) | **no** | free registration required (§1) |
| On-demand L1B/L1C/L2A processing order | **no** | EOWEB login |
| New/targeted acquisition proposal (`planning.enmap.org`) | **no** | account + proposal form (§3) |
| Campaign questionnaire (`survey.hifis.dkfz.de/371151`) | web form | confirmation is by e-mail with ENMAP_APPLICATION_SP@dlr.de |

**Bottom line:** everything short of placing the order can be prepared
without credentials (AOIs, parameter sheets, ingest configs, the SRTM +
S2 dry-runs). The archive search in §2 and the order/proposal buttons in
§3 require the free accounts from §1 — registration takes minutes and
no data behind it is paid or restricted beyond licence/AUP acceptance.

## 1. Accounts (one-time)

1. Register at the **EnMAP Data Access Portal** — screencast "How to
   register and assign to user roles" on the Data & Access page;
   accept the Licence / AUP (links on the same page).
2. Register at the **EOWEB GeoPortal** (https://eoweb.dlr.de/) with the
   same institutional e-mail — this is the archive search/download
   entry point.
3. Register at **https://planning.enmap.org** (Instrument Planning
   Portal) — required only for *new/targeted* acquisition proposals
   ("How to submit a data proposal" screencast).

## 2. Archive-first search (EOWEB GeoPortal) — do this before ordering

For each target below, in this order:

1. Open EOWEB GeoPortal → search EnMAP archive → set the **AOI to the
   target bbox** (table §4; paste min/max lon/lat directly).
2. Filter: product level **L2A** (surface reflectance is what
   `src.features.hyperspectral` consumes), date range = the target's
   **dry-season window**, ascending cloud cover.
3. No L2A but L1B/L1C present? Re-run with on-demand processing:
   EOWEB regenerates **L1B/L1C/L2A with the most up-to-date processor**
   (processing + delivery within a maximum of 6 days after downlink).
4. Also check the free **EOC Geoservice ARD** bulk route (L2A ARD,
   GeoTIFF, UTM zone of scene centre) and **EO-Lab** L2A explorer if
   EOWEB search is slow.
5. Download the **GeoTIFF** variant (the ingest expects band
   descriptions with wavelengths; GeoTIFF + XML metadata).

## 3. New / targeted acquisitions (when the archive has no scene)

The Foreground Mission (March 2024 onward) schedules 1000 km flightlines
**primarily over Germany, March–October** — our East Africa windows do
not conflict with FG stripes, so standalone requests are the right
route:

1. Submit an **EnMAP proposal + EO request at
   https://planning.enmap.org** (this is the regular acquisition
   planning route stated by the EnMAP team).
2. For campaigns needing close-to-concordant acquisitions, additionally
   complete the campaign questionnaire
   (https://survey.hifis.dkfz.de/371151?lang=en) **~6 weeks before**
   the planned date and confirm with
   `ENMAP_APPLICATION_SP@dlr.de` (~2 weeks before). Short-term requests
   are possible depending on scheduling.
3. Per request, attach: target bbox (§4), preferred dry-season dates
   (§4), on-ground purpose statement (alteration-mineralogy validation
   of a permissive MPM window — cite the analogue deposit model),
   maximum acceptable cloud cover (≤10 % recommended for SWIR work).
4. Acquired scenes become visible to registered users **~6 days after
   acquisition** in the EOWEB archive — re-run §2 search to pick them up.

## 4. Per-target order parameters (7 high-confidence windows)

| # | target id | group | bbox (min_lon, min_lat, max_lon, max_lat) | CRS | dry season (S2 composite) | EnMAP probes (µm) | order priority |
|---|---|---|---|---|---|---|---|
| 1 | kab_burundi_north | Sn-W-Ta | 29.3, -3.5, 30.3, -2.4 | EPSG:32736 | 2023-06-01 → 2023-09-30 | 2.20, 2.35 | 1 (ingest dry-run done) |
| 2 | lufilian_kolwezi_kambove | Cu-Zn | 25.2, -11.2, 27.0, -10.0 | EPSG:32735 | 2023-05-01 → 2023-09-30 | 2.20, 2.33, 0.86 | 1 (ingest dry-run done) |
| 3 | usambara_east_extension | bauxite | 38.3, -5.2, 39.1, -4.4 | EPSG:32737 | 2023-06-01 → 2023-09-30 | 2.20, 0.90 | 1 (ingest dry-run done) |
| 4 | kab_tanzania_karagwe | Sn-W-Ta | 30.5, -2.8, 31.8, -1.4 | EPSG:32736 | 2023-06-01 → 2023-09-30 | 2.20, 2.35 | 2 |
| 5 | kab_uganda_southwest | Sn-W-Ta | 29.7, -1.4, 30.6, -0.5 | EPSG:32736 | 2023-06-01 → 2023-09-30 (+ Jan–Feb) | 2.20, 2.35 | 2 |
| 6 | lufilian_likasi_tenke | Cu-Zn | 26.2, -11.2, 27.4, -10.2 | EPSG:32735 | 2023-05-01 → 2023-09-30 | 2.20, 2.33, 0.86 | 2 |
| 7 | zambia_copperbelt_north | Cu-Zn | 27.6, -13.2, 28.8, -12.0 | EPSG:32735 | 2023-05-01 → 2023-09-30 | 2.20, 2.33, 0.86 | 2 |

Notes:

- A window wider than one EnMAP scene (scene swath ~30 km) needs
  **multiple tiles**: order per tile and list them in the tracking
  table (§5); `lufilian_kolwezi_kambove` (~1.8° lon) will need ~4 tiles.
- The bbox is the *permissive* extent; order scenes covering the
  structural/lithology corridor inside it first (see each target's
  `rationale` and `signatures` in the registry).

## 5. Ordering tracker (fill in as orders are placed)

| target id | scene/product ID | product level | ordered on | delivered | cloud % | ingest path |
|---|---|---|---|---|---|---|
| kab_burundi_north | _TBD_ | L2A | | | | `data/raw/enmap/` |
| lufilian_kolwezi_kambove | _TBD_ | L2A | | | | `data/raw/enmap/` |
| usambara_east_extension | _TBD_ | L2A | | | | `data/raw/enmap/` |
| kab_tanzania_karagwe | _TBD_ | L2A | | | | `data/raw/enmap/` |
| kab_uganda_southwest | _TBD_ | L2A | | | | `data/raw/enmap/` |
| lufilian_likasi_tenke | _TBD_ | L2A | | | | `data/raw/enmap/` |
| zambia_copperbelt_north | _TBD_ | L2A | | | | `data/raw/enmap/` |

## 6. Ingest after download

Scene configs for all 7 targets are generated from the registry:

```bash
python -m src.ingest.analogue_scene --list
python -m src.ingest.analogue_scene --write --confidence high   # configs/scenes/<id>.yml
```

Align a downloaded cube onto the target's common grid (nearest-neighbour,
wavelength band descriptions preserved, clouds → NaN):

```bash
python -m src.ingest.enmap \
  --input data/raw/enmap/enmap_scene.tif \
  --config configs/scenes/kab_burundi_north.yml \
  --out data/interim/hyperspectral/enmap_kab_burundi_north_aligned.tif
```

Then run the band-depth / SAM probes at the wavelengths in §4 with
`src.features.hyperspectral` (T7).

**Guardrail:** ordered scenes feed *feature extraction only*. EnMAP
coverage of a window never creates a training label — labels remain
MRDS / national-survey / field-validated via `src.labels.build_labels`,
otherwise spatial CV would train on the same domain shift the registry
exists to avoid.

