# Leave-one-deposit-out (LOO) rediscovery validation

Generated: 2026-09-26 10:13 UTC

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

## bauxite — per-fold detail

| fold | deposit | best | rank | percentile | rediscovered | top-1 dist (km) | AP | trained on |
|---:|---|---|---:|---:|:--:|---:|---:|---:|
| 0 | Lushoto | lgbm | 2 | 100.00 | yes | 7.1 | 0.667 | 2 |
| 1 | Magamba | xgb | 26 | 100.00 | yes | 9.3 | 0.583 | 2 |
| 2 | Mombo | rf | 53 | 100.00 | yes | 7.4 | n/a | 2 |

## Caveats

* **Rates are not comparable across groups.** `bauxite` has only 3 in-extent seeds, so each fold trains on 2 positives — its rate is near-meaningless statistically. `copper_zinc` trains on ~38 positives over a 67.5 M-cell AOI. Read the `trained on` column.
* **A miss is not proof of failure.** A withheld deposit in the 99.9th percentile that still lands below rank 100 shows the surface concentrating where it should; the percentile column shows that.
* **Background points are reused unchanged.** Background sampling excluded a buffer around the original labels, so a small hole in background density survives near each withheld deposit. It encodes no holdout information into any feature.
* **Folds are independent single-deposit removals**, not a nested resampling of model selection: the algorithm choice is re-made inside each fold from the same 3-algorithm comparison.
