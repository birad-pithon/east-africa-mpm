#!/usr/bin/env python3
"""Legacy CLI shim for holdout validation.

Historically this file "held out" a deposit by mutating ``SEED_DEPOSITS``
in memory — but ``train_group`` re-reads the label GeoPackage and the
deposit-distance prior from disk, so the held-out deposit leaked into
both and the test only ever proved the model could rediscover a deposit
it was trained on. The leakage-free implementation now lives in
:mod:`src.models.holdout` (isolated artifacts under ``outputs/holdout/``);
this script forwards every legacy flag to its ``main()``.

Usage (unchanged):
    python holdout_validation.py --group tin_tungsten_tantalum
    python holdout_validation.py --group copper_zinc
    python holdout_validation.py --group bauxite
    python holdout_validation.py --all  # run all three groups
"""

from src.models.holdout import main

if __name__ == "__main__":
    main()
