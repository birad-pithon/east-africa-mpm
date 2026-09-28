# Model card â€” tin_tungsten_tantalum

*Generated from `metrics_tin_tungsten_tantalum.json`; algorithm `rf` (best of rf, xgb, lgbm).*

## Intended use

Gridded mineral prospectivity screening: rank 30 m cells of the study AOI by P(deposit) to prioritise field verification for the `tin_tungsten_tantalum` commodity group. Output is a *screening prior*, not a resource estimate; every high-scoring cell requires ground truth before any decision.

## Geological context

Karagwe-Ankole Belt (Mesoproterozoic Kibaran orogen): Sn-W-Ta mineralisation hosted in pegmatites, quartz veins and greisenised granites of the KIB province. Target styles are cassiterite-bearing granite-related pegmatites and wolframite-quartz vein systems.

Feature groups and the geological evidence they encode: lithology + one-hot classes = host-rock type; distance-to-* = proximity to contact zones / fertile units; dist_to_contact_m = structural proximity; terrain (slope/aspect/TPI/relief/plateau) = geomorphic position; Sentinel-2 bands = alteration/laterite surface signatures.


## Training data

* labelled rows: **327** (8 positives, base rate 0.024)
* features: 36 â€” alteration_features_kabar, deposit_distance_tin_tungsten_tantalum, geology_features_kabar, sentinel2_kabar_aligned, srtm_kabar_aligned, terrain_features_kabar
* class balancing: class weights

## Evaluation protocol

* spatial block CV, 5 folds, block 10000 m, train/test buffer 5 km â€” fold plan certified (`src.validate.certify_folds`) before fitting
* primary metric: average precision (PR-AUC); ROC-AUC secondary

## Results

| algorithm | mean AP | mean ROC-AUC |
|---|---|---|
| rf | 0.029 | 0.295 |
| xgb | 0.029 | 0.294 |
| lgbm | 0.026 | 0.161 |

Anomaly-baseline (IsolationForest-on-barren) reference AP: **n/a** â€” supervised models must beat this and the class base rate to demonstrate signal.

## Probability calibration

* Brier score: **0.045**; ECE (10 usable bins): **0.117**; max |pred-obs| gap: 0.296
* reliability diagram: `reliability_tin_tungsten_tantalum.png`
* read P(deposit) as a *ranking prior*; absolute values are not calibrated probabilities unless ECE is small

## Limitations & known caveats

* this is a coarse regional-screening prior, not a drill target list - each high-scoring cell must be validated against local geological maps (1:50k national surveys) and field observation before permitting or drilling
* positive labels are USGS MRDS occurrences + curated seeds â€” exploration-biased; barren-but-unexplored ground may hide deposits (background strata mitigate, not eliminate)
* lithology source is the ~1:35M GSC world map â€” coarse; replace with national 1:50k layers when available
* only 8 positives â€” fold metrics have high variance; treat single-fold numbers with caution

## Provenance

* git: commit `ea322f247acd54c610ca2beb48baa95e3a3a8140` (branch `main`, dirty=True)
* trained: 2026-09-28T07:35:34+00:00
* key versions: python 3.13.3, scikit-learn 1.9.0, pandas 3.0.5, geopandas 1.1.4
* input SHA-256:
  * `labels_tin_tungsten_tantalum.gpkg`: `6dc610123d8ecb4e...`
  * `background_tin_tungsten_tantalum.gpkg`: `ef4de3d79ed22ad2...`
  * `srtm_kabar_aligned.tif`: `413779db5124ba2c...`
  * `sentinel2_kabar_aligned.tif`: `cbfcfa82852d792f...`
  * `terrain_features_kabar.tif`: `8d6d2fa74a73608a...`
  * `geology_features_kabar.tif`: `a311452c7e9105e3...`
  * `alteration_features_kabar.tif`: `1eec9699fdde46d8...`
  * `deposit_distance_tin_tungsten_tantalum.tif`: `e7b85cce9bf6c918...`
