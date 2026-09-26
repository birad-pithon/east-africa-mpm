#!/usr/bin/env python3
"""CLI shim for the full leave-one-deposit-out (LOO) study.

``holdout_validation.py`` certifies one withheld deposit per group; a
rediscovery *rate* needs every in-extent seed withheld in turn. The
driver lives in :mod:`src.models.loo`; this script forwards to its
``main()``.

Usage:
    python loo_validation.py --all
    python loo_validation.py --group copper_zinc --max-folds 5
    python loo_validation.py --all --report-only
"""

from src.models.loo import main

if __name__ == "__main__":
    main()
