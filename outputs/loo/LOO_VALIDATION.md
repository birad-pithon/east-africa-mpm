# Leave-one-deposit-out (LOO) rediscovery validation

Generated: 2026-09-28 09:36 UTC

## Method

Every in-extent verified deposit is withheld once, in turn. For each fold the harness:

1. writes reduced labels with the withheld deposit removed by name **and** within 2 km (so alias rows cannot leak);
2. rebuilds the deposit-distance prior from those reduced labels, so the withheld cell no longer reads distance 0;
3. retrains and predicts into an isolated per-fold directory;
4. ranks the withheld deposit against every valid cell.

**Rediscovery** = the best cell within 5 km of the withheld deposit ranks in the global top 100. The rank is the raw count of valid cells with a strictly higher probability (deterministic, no declustering).

## Rediscovery rate by group

| group | folds | scored | rediscovered | rate | median rank | median pct | in top 1% | median top-1 dist (km) | median AP | min. positives trained on |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bauxite | 3 | 3 | 3 | 100.0 % | 26 | 100.00 % | 100.0 % | 7.4 | 0.625 | 2 |
| copper_zinc | 39 | 39 | 23 | 59.0 % | 1 | 100.00 % | 92.3 % | 46.9 | 0.935 | 36 |
| tin_tungsten_tantalum | 9 | 9 | 2 | 22.2 % | 9818 | 99.95 % | 100.0 % | 48.1 | 0.059 | 8 |

## bauxite — per-fold detail

| fold | deposit | best | rank | percentile | rediscovered | top-1 dist (km) | AP | trained on |
|---:|---|---|---:|---:|:--:|---:|---:|---:|
| 0 | Lushoto | lgbm | 2 | 100.00 | yes | 7.1 | 0.667 | 2 |
| 1 | Magamba | xgb | 26 | 100.00 | yes | 9.3 | 0.583 | 2 |
| 2 | Mombo | rf | 53 | 100.00 | yes | 7.4 | n/a | 2 |

## copper_zinc — per-fold detail

| fold | deposit | best | rank | percentile | rediscovered | top-1 dist (km) | AP | trained on |
|---:|---|---|---:|---:|:--:|---:|---:|---:|
| 0 | Kambove West | rf | 1 | 100.00 | yes | 52.2 | 0.955 | 38 |
| 1 | Kazibizi | xgb | 4 | 100.00 | yes | 23.8 | 0.928 | 37 |
| 2 | M'Sesa | xgb | 4 | 100.00 | yes | 22.3 | 0.928 | 37 |
| 3 | (Facility) Shituru Smelter | rf | 1 | 100.00 | yes | 80.9 | 0.907 | 36 |
| 4 | Kakanda | rf | 1 | 100.00 | yes | 16.4 | 0.942 | 38 |
| 5 | Shituru | xgb | 1025 | 100.00 | no | 64.5 | 0.927 | 38 |
| 6 | (Facility) Shituru Copper Refinery | rf | 1 | 100.00 | yes | 80.9 | 0.907 | 36 |
| 7 | Tenke Fungurume | xgb | 8824 | 99.99 | no | 163.3 | 0.949 | 37 |
| 8 | Disele | rf | 1 | 100.00 | yes | 15.1 | 0.924 | 37 |
| 9 | Bangwe | rf | 1 | 100.00 | yes | 4.0 | 0.922 | 37 |
| 10 | Fwalu | xgb | 8824 | 99.99 | no | 163.1 | 0.949 | 37 |
| 11 | Judeira | rf | 1 | 100.00 | yes | 97.6 | 0.948 | 38 |
| 12 | Kababankola | rf | 1 | 100.00 | yes | 12.0 | 0.924 | 36 |
| 13 | Kabolela | rf | 1 | 100.00 | yes | 26.7 | 0.945 | 38 |
| 14 | Kakanda East | rf | 1 | 100.00 | yes | 10.6 | 0.924 | 36 |
| 15 | Kamatanda | rf | 1 | 100.00 | yes | 79.1 | 0.907 | 36 |
| 16 | Kaminafitwe | rf | 1 | 100.00 | yes | 99.7 | 0.957 | 38 |
| 17 | Kasala | rf | 240731 | 99.64 | no | 117.0 | 0.942 | 38 |
| 18 | Kikuswe | rf | 138145 | 99.80 | no | 120.3 | 0.941 | 38 |
| 19 | Kileba | xgb | 298 | 100.00 | no | 17.4 | 0.930 | 38 |
| 20 | Kimbwe | rf | 277055 | 99.59 | no | 126.7 | 0.946 | 38 |
| 21 | Kinsevere | rf | 3188432 | 95.28 | no | 153.1 | 0.974 | 38 |
| 22 | Kipoi Central | rf | 1 | 100.00 | yes | 126.0 | 0.931 | 37 |
| 23 | Kipoi North | rf | 1 | 100.00 | yes | 125.1 | 0.931 | 37 |
| 24 | Kiwana | rf | 1 | 100.00 | yes | 8.5 | 0.934 | 38 |
| 25 | Kwatebala | xgb | 2104 | 100.00 | no | 46.9 | 0.939 | 38 |
| 26 | Lufomboshi | rf | 1 | 100.00 | yes | 16.5 | 0.924 | 37 |
| 27 | Luishia | rf | 272680 | 99.60 | no | 82.0 | 0.963 | 38 |
| 28 | Luita | xgb | 6 | 100.00 | yes | 121.7 | 0.932 | 38 |
| 29 | Mambilima | rf | 103124 | 99.85 | no | 23.6 | 0.947 | 38 |
| 30 | Mukondo | rf | 1 | 100.00 | yes | 7.9 | 0.924 | 37 |
| 31 | Shangolowe | rf | 1 | 100.00 | yes | 20.7 | 0.937 | 38 |
| 32 | Saafi | rf | 1 | 100.00 | yes | 12.2 | 0.924 | 36 |
| 33 | Taratara | rf | 1 | 100.00 | yes | 7.4 | 0.935 | 36 |
| 34 | Mwandinkomba | rf | 91543 | 99.86 | no | 27.5 | 0.950 | 38 |
| 35 | Sefu | rf | 109430 | 99.84 | no | 31.8 | 0.924 | 38 |
| 36 | Munaka | rf | 207703 | 99.69 | no | 114.0 | 0.945 | 38 |
| 37 | Musonge | rf | 4326159 | 93.59 | no | 179.2 | 0.946 | 38 |
| 38 | Shinkolobwe Signal | rf | 5983733 | 91.14 | no | 42.7 | 0.974 | 38 |

## tin_tungsten_tantalum — per-fold detail

| fold | deposit | best | rank | percentile | rediscovered | top-1 dist (km) | AP | trained on |
|---:|---|---|---:|---:|:--:|---:|---:|---:|
| 0 | Nyakabingo | rf | 1 | 100.00 | yes | 1.1 | 0.053 | 8 |
| 1 | Musha | rf | 4530 | 99.98 | no | 65.7 | 0.029 | 8 |
| 2 | Ntunga | xgb | 9818 | 99.95 | no | 37.5 | 0.036 | 8 |
| 3 | Gatumba | xgb | 94244 | 99.52 | no | 48.1 | 0.077 | 8 |
| 4 | Nyamalilo | xgb | 13966 | 99.93 | no | 87.4 | 0.061 | 8 |
| 5 | Somirwa Mine | lgbm | 179835 | 99.09 | no | 16.9 | 0.103 | 8 |
| 6 | Somirwa | xgb | 7 | 100.00 | yes | 10.7 | 0.043 | 8 |
| 7 | Ruhuma | lgbm | 24407 | 99.88 | no | 81.3 | 0.059 | 8 |
| 8 | Somirwa (Societe Miniere De Rwanda) | xgb | 2522 | 99.99 | no | 59.0 | 0.065 | 8 |

## Caveats

* **Rates are not comparable across groups.** `bauxite` has only 3 in-extent seeds, so each fold trains on 2 positives — its rate is near-meaningless statistically. `copper_zinc` trains on ~38 positives over a 67.5 M-cell AOI. Read the `trained on` column.
* **A miss is not proof of failure.** A withheld deposit in the 99.9th percentile that still lands below rank 100 shows the surface concentrating where it should; the percentile column shows that.
* **Background points are reused unchanged.** Background sampling excluded a buffer around the original labels, so a small hole in background density survives near each withheld deposit. It encodes no holdout information into any feature.
* **`partial` means the group's run is unfinished.** The rate beside it covers only the folds done so far and is not a final rediscovery rate — re-run to completion before quoting it.
* **A `n/a` AP means the fold could not be scored**, not a bad score. With only 2 positives in the group, a blocked fold's training batch can hold a single class (and other folds hold no positive test label), so cross-validated average precision is undefined for that fold. Rediscovery rank is unaffected: it comes from the full-group retrain, not the CV fold.
* **Folds are independent single-deposit removals**, not a nested resampling of model selection: the algorithm choice is re-made inside each fold from the same 3-algorithm comparison.
