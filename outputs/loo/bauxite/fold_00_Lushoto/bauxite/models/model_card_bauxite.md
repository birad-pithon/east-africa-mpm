# Model card â€” bauxite

*Generated from `metrics_bauxite.json`; algorithm `lgbm` (best of rf, xgb, lgbm).*

## Intended use

Gridded mineral prospectivity screening: rank 30 m cells of the study AOI by P(deposit) to prioritise field verification for the `bauxite` commodity group. Output is a *screening prior*, not a resource estimate; every high-scoring cell requires ground truth before any decision.

## Geological context

Usambara lateritic bauxite province (Tanga, Tanzania): supergene Al(OH)3 enrichment on plateau-capping weathering profiles over basement gneisses.

Feature groups and the geological evidence they encode: lithology + one-hot classes = host-rock type; distance-to-* = proximity to contact zones / fertile units; dist_to_contact_m = structural proximity; terrain (slope/aspect/TPI/relief/plateau) = geomorphic position; Sentinel-2 bands = alteration/laterite surface signatures.


## Training data

* labelled rows: **302** (2 positives, base rate 0.007)
* features: 36 â€” alteration_features_usambara, deposit_distance_bauxite, geology_features_usambara, sentinel2_usambara_aligned, srtm_usambara_aligned, terrain_features_usambara
* class balancing: class weights

## Evaluation protocol

* spatial block CV, 5 folds, block 10000 m, train/test buffer 5 km â€” fold plan certified (`src.validate.certify_folds`) before fitting
* primary metric: average precision (PR-AUC); ROC-AUC secondary

## Results

| algorithm | mean AP | mean ROC-AUC |
|---|---|---|
| rf | 0.600 | 0.967 |
| xgb | 0.267 | 0.972 |
| lgbm | 0.667 | 0.984 |

Anomaly-baseline (IsolationForest-on-barren) reference AP: **n/a** â€” supervised models must beat this and the class base rate to demonstrate signal.

## Probability calibration

* Brier score: **0.007**; ECE (10 usable bins): **0.005**; max |pred-obs| gap: 0.033
* reliability diagram: `reliability_bauxite.png`
* read P(deposit) as a *ranking prior*; absolute values are not calibrated probabilities unless ECE is small

## Limitations & known caveats

* this is a coarse regional-screening prior, not a drill target list - each high-scoring cell must be validated against local geological maps (1:50k national surveys) and field observation before permitting or drilling
* positive labels are USGS MRDS occurrences + curated seeds â€” exploration-biased; barren-but-unexplored ground may hide deposits (background strata mitigate, not eliminate)
* lithology source is the ~1:35M GSC world map â€” coarse; replace with national 1:50k layers when available
* only 2 positives â€” fold metrics have high variance; treat single-fold numbers with caution

## Provenance

* git: commit `a9b8e06cecf9cbf10eba66b59b2f5495186c85b3` (branch `main`, dirty=True)
* trained: 2026-09-26T08:39:44+00:00
* key versions: python 3.13.3, scikit-learn 1.9.0, pandas 3.0.5, geopandas 1.1.4
* input SHA-256:
  * `labels_bauxite.gpkg`: `ebaeed76126855c7...`
  * `background_bauxite.gpkg`: `bf0796ba37e2764e...`
  * `srtm_usambara_aligned.tif`: `9962263d3917fdca...`
  * `sentinel2_usambara_aligned.tif`: `c7ef1af4c2fc491c...`
  * `terrain_features_usambara.tif`: `a81a41206f831373...`
  * `geology_features_usambara.tif`: `6e19308b422a09b5...`
  * `alteration_features_usambara.tif`: `8969b0e5e6803a9b...`
  * `deposit_distance_bauxite.tif`: `8ff2036d46214131...`
