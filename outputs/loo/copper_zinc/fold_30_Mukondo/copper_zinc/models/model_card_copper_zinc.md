# Model card â€” copper_zinc

*Generated from `metrics_copper_zinc.json`; algorithm `rf` (best of rf, xgb, lgbm).*

## Intended use

Gridded mineral prospectivity screening: rank 30 m cells of the study AOI by P(deposit) to prioritise field verification for the `copper_zinc` commodity group. Output is a *screening prior*, not a resource estimate; every high-scoring cell requires ground truth before any decision.

## Geological context

Central African Copperbelt (Neoproterozoic Lufilian arc): sediment-hosted stratiform Cu-Co and Cu-Zn-(Pb) mineralisation within Roan Group metasediments.

Feature groups and the geological evidence they encode: lithology + one-hot classes = host-rock type; distance-to-* = proximity to contact zones / fertile units; dist_to_contact_m = structural proximity; terrain (slope/aspect/TPI/relief/plateau) = geomorphic position; Sentinel-2 bands = alteration/laterite surface signatures.


## Training data

* labelled rows: **95** (37 positives, base rate 0.389)
* features: 36 â€” alteration_features_copperbelt, deposit_distance_copper_zinc, geology_features_copperbelt, sentinel2_copperbelt_aligned, srtm_copperbelt_aligned, terrain_features_copperbelt
* class balancing: class weights

## Evaluation protocol

* spatial block CV, 5 folds, block 10000 m, train/test buffer 5 km â€” fold plan certified (`src.validate.certify_folds`) before fitting
* primary metric: average precision (PR-AUC); ROC-AUC secondary

## Results

| algorithm | mean AP | mean ROC-AUC |
|---|---|---|
| rf | 0.924 | 0.928 |
| xgb | 0.922 | 0.910 |
| lgbm | 0.921 | 0.900 |

Anomaly-baseline (IsolationForest-on-barren) reference AP: **n/a** â€” supervised models must beat this and the class base rate to demonstrate signal.

## Probability calibration

* Brier score: **0.087**; ECE (5 usable bins): **0.049**; max |pred-obs| gap: 0.162
* reliability diagram: `reliability_copper_zinc.png`
* read P(deposit) as a *ranking prior*; absolute values are not calibrated probabilities unless ECE is small

## Limitations & known caveats

* this is a coarse regional-screening prior, not a drill target list - each high-scoring cell must be validated against local geological maps (1:50k national surveys) and field observation before permitting or drilling
* positive labels are USGS MRDS occurrences + curated seeds â€” exploration-biased; barren-but-unexplored ground may hide deposits (background strata mitigate, not eliminate)
* lithology source is the ~1:35M GSC world map â€” coarse; replace with national 1:50k layers when available
* only 37 positives â€” fold metrics have high variance; treat single-fold numbers with caution

## Provenance

* git: commit `ea322f247acd54c610ca2beb48baa95e3a3a8140` (branch `main`, dirty=True)
* trained: 2026-09-28T05:49:02+00:00
* key versions: python 3.13.3, scikit-learn 1.9.0, pandas 3.0.5, geopandas 1.1.4
* input SHA-256:
  * `labels_copper_zinc.gpkg`: `20a14b0ee6eb65bd...`
  * `background_copper_zinc.gpkg`: `cd8f5bcbc3f9cdeb...`
  * `srtm_copperbelt_aligned.tif`: `e1b2a48305ed1823...`
  * `sentinel2_copperbelt_aligned.tif`: `4c6fb15d7a40557d...`
  * `terrain_features_copperbelt.tif`: `dd310c04019e7c19...`
  * `geology_features_copperbelt.tif`: `ac24b869552b2351...`
  * `alteration_features_copperbelt.tif`: `040e6c0166bb23bc...`
  * `deposit_distance_copper_zinc.tif`: `ccfd73b69f999351...`
