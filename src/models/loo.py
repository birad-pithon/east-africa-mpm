"""Full leave-one-deposit-out (LOO) rediscovery-rate driver.

:mod:`src.models.holdout` validates a *single* held-out deposit per
invocation and keeps one report per group — enough to answer "can the
model rediscover this deposit?", but not "how often does it rediscover
a deposit?".

This module drives that harness once per fold so **every in-extent seed
is withheld in turn**:

* each fold's artifacts live in an isolated
  ``outputs/loo/<group>/fold_<NN>_<slug>/`` tree (reduced labels,
  rebuilt deposit-distance prior, model bundle, probability raster,
  per-fold report) — nothing is overwritten between folds;
* a compact per-fold record (rank / percentile / rediscovery / AP) is
  appended to ``loo_partial.json`` **after every fold**, so a crashed or
  killed run resumes instead of restarting (``--no-resume`` disables);
* the aggregate write-up is ``loo_report.json`` + ``LOO_VALIDATION.md``.

Rediscovery metric (unchanged from the harness, so results stay
comparable): a fold is *rediscovered* iff the best cell within
``TOLERANCE_KM`` of the withheld deposit ranks within the global
``TOP_N`` of the probability surface.  Rank is the raw count of valid
cells with a strictly higher probability, i.e. a deterministic global
rank over every cell.

Interpretation caveats (real, not leaks):

* **bauxite is degenerate.** Only 3 in-extent seeds exist, so a fold
  trains on 2 positives.  Its rate is reported but carries almost no
  statistical weight — read ``n_positives_trained_on`` per fold.
* **copper_zinc loses 39 of 78 labels** to "outside the raster extent",
  so its folds train on ~38 positives over a 67.5 M-cell AOI.
* a withheld deposit is removed by name **and** within ``COLOCATION_KM``
  (alias rows); its own cell in the prior no longer reads distance 0.

CLI::

    python -m src.models.loo --all
    python -m src.models.loo --group copper_zinc --max-folds 5
    python -m src.models.loo --all --algos rf            # 1 algo = ~3x faster
    python -m src.models.loo --all --report-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.models.dataset import default_feature_rasters
from src.models.holdout import (
    GROUP_CONFIGS,
    TOLERANCE_KM,
    TOP_N,
    _extent_mask,
    _json_default,
    _load_labels,
    run_holdout_test,
)
from src.utils import project_path

logger = logging.getLogger(__name__)

__all__ = [
    "LOO_ROOT",
    "folds_from_labels",
    "enumerate_folds",
    "fold_slug",
    "compact_record",
    "summarize",
    "loo_report",
    "loo_markdown",
    "run_loo",
    "main",
]

#: root for every fold's artifacts (never ``outputs/holdout/``).
LOO_ROOT: Path = project_path("outputs", "loo")

#: secondary context thresholds reported alongside the primary rate.
PCT_TOP1 = 99.0  # percentile >= 99  -> inside the best 1 % of cells
PCT_TOP01 = 99.9  # percentile >= 99.9 -> inside the best 0.1 %


def fold_slug(idx: int, name: str, width: int = 2) -> str:
    """Filesystem-safe ``fold_<NN>_<slug>`` directory name for one fold."""
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in str(name).strip()]
    slug = "".join(keep).strip("_") or "unnamed"
    while "__" in slug:
        slug = slug.replace("__", "_")
    return f"fold_{int(idx):0{width}d}_{slug[:48]}"


def folds_from_labels(labels, in_extent) -> list[dict]:
    """Build the fold list from labels + a boolean in-extent mask.

    Pure function (no I/O) so it is unit-testable.  ``fold_idx`` indexes
    the *in-extent* subset — exactly what ``run_holdout_test``'s
    ``holdout_idx`` expects — while ``label_row`` records the original
    GeoPackage row for traceability.
    """
    mask = np.asarray(in_extent, dtype=bool)
    if len(mask) != len(labels):
        raise ValueError(f"in_extent mask length {len(mask)} != labels length {len(labels)}")
    names = (
        labels["name"].astype(str)
        if "name" in labels.columns
        else labels.index.to_series().astype(str)
    )
    folds: list[dict] = []
    for fold_idx, (row_pos, row) in enumerate(labels[mask].iterrows()):
        geom = row["geometry"]
        if geom is None or geom.is_empty:
            logger.warning("skipping empty label geometry at row %s", row_pos)
            continue
        folds.append(
            {
                "fold_idx": fold_idx,
                "label_row": int(row_pos),
                "name": str(names.loc[row_pos]),
                "lon": float(geom.x),
                "lat": float(geom.y),
            }
        )
    return folds


def enumerate_folds(group: str, config_path: str | Path | None = None) -> list[dict]:
    """Every in-extent seed for *group*, in label-GeoPackage order.

    Uses the same footprint reference (first allowlisted raster) and the
    same allowlist resolution as the harness, so the fold count matches
    ``n_seeds_in_extent`` in a holdout report.
    """
    config = str(config_path or GROUP_CONFIGS[group])
    labels = _load_labels(group)
    rasters = default_feature_rasters(group, config_path=config)
    if not rasters:
        raise ValueError(f"no allowlisted feature rasters for {group!r}")
    extent = _extent_mask(labels, rasters[0])
    return folds_from_labels(labels, extent)


def _mean_ap(metrics: dict, algo: str | None) -> float:
    """Mean average precision of *algo* from a ``metrics`` payload."""
    if not isinstance(metrics, dict) or algo is None:
        return float("nan")
    entry = (metrics.get("results") or {}).get(algo)
    if not isinstance(entry, dict):
        return float("nan")
    value = entry.get("mean_average_precision")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def compact_record(result: dict, fold: dict) -> dict:
    """Reduce a harness result to the fields the LOO study aggregates.

    Drops the multi-MB ``metrics``/``feature_rasters`` payloads while
    keeping every rediscovery number plus the best-algo CV score.

    The full global ``rank`` is read from ``sites[0]`` — the summary
    ``rediscovery`` block only carries it when the rank is <= 50, so
    relying on that alone would turn every mid-table fold into a NaN.
    """
    sites = result.get("sites") or []
    site = sites[0] if sites else {}
    rediscovery = result.get("rediscovery") or {}
    metrics = result.get("metrics") or {}
    best_algo = result.get("best_algo")

    def _num(value):
        """Coerce to a finite float, else NaN (never inf/-inf)."""
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return out if np.isfinite(out) else float("nan")

    def _pick(primary, fallback):
        """First finite value of the two, else NaN."""
        first = _num(primary)
        return first if np.isfinite(first) else _num(fallback)

    rank = _pick(site.get("rank"), rediscovery.get("rank_in_top50"))
    if "in_extent" in site:
        in_extent = bool(site["in_extent"])
    else:
        # no site block (older report): a finite rank proves it was scored
        in_extent = bool(np.isfinite(rank))
    # a fold we could not rank can never count as a rediscovery
    rediscovered = (
        bool(site.get("rediscovered", rediscovery.get("rediscovered")))
        if np.isfinite(rank)
        else False
    )
    return {
        "fold_idx": fold["fold_idx"],
        "label_row": fold.get("label_row"),
        "name": fold["name"],
        "lon": fold["lon"],
        "lat": fold["lat"],
        "best_algo": best_algo,
        "in_extent": in_extent,
        "rank": rank,
        "rank_in_top50": (int(rank) if np.isfinite(rank) and rank <= 50 else None),
        "rediscovered": rediscovered,
        "percentile": _pick(site.get("percentile"), rediscovery.get("percentile")),
        "prob_at_holdout": _num(site.get("prob_at_holdout")),
        "n_valid_cells": int(site.get("n_valid_cells") or 0),
        "distance_to_top1_km": _pick(
            site.get("distance_to_top1_km"), rediscovery.get("distance_to_top1_km")
        ),
        "n_candidates_within_5km": int(
            site.get("n_top50_within_tolerance")
            or rediscovery.get("num_candidates_within_5km")
            or 0
        ),
        "n_labels_removed": int(result.get("n_labels_removed") or 0),
        "n_labels_remaining": int(result.get("n_labels_remaining") or 0),
        # metrics.n_positives = rows that actually reached the model, which is
        # lower than n_labels_remaining whenever labels fall outside the raster
        "n_positives_trained_on": int(
            metrics.get("n_positives") or result.get("n_labels_remaining") or 0
        ),
        "mean_ap": _mean_ap(metrics, best_algo),
        "metrics_path": result.get("metrics_path"),
        "proba_raster": result.get("proba_raster"),
    }


def _quantile(values: list[float], q: float) -> float:
    """``numpy`` quantile that tolerates an all-NaN input."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def summarize(
    group: str,
    records: list[dict],
    top_n: int = TOP_N,
    tolerance_km: float = TOLERANCE_KM,
) -> dict:
    """Rediscovery statistics for one group's folds.

    Unscored folds (held-out site outside the probability footprint) are
    excluded from the rate denominator and counted separately — they must
    never be silently folded into a "miss".
    """
    scored = [r for r in records if r.get("in_extent")]
    unscored = [r for r in records if not r.get("in_extent")]
    ranks = [r.get("rank", float("nan")) for r in scored]
    finite_ranks = [float(r) for r in ranks if np.isfinite(r)]
    pcts = [r.get("percentile", float("nan")) for r in scored]
    dists = [r.get("distance_to_top1_km", float("nan")) for r in scored]
    aps = [r.get("mean_ap", float("nan")) for r in scored]
    n_hits = sum(1 for r in scored if r.get("rediscovered"))
    n_top1pct = sum(1 for p in pcts if np.isfinite(p) and p >= PCT_TOP1)
    n_top01pct = sum(1 for p in pcts if np.isfinite(p) and p >= PCT_TOP01)

    algos: dict[str, int] = {}
    for r in scored:
        key = str(r.get("best_algo"))
        algos[key] = algos.get(key, 0) + 1

    denom = len(scored)
    return {
        "group": group,
        "n_folds": len(records),
        "n_folds_scored": denom,
        "n_folds_unscored": len(unscored),
        "n_rediscovered": n_hits,
        "rediscovery_rate": (n_hits / denom) if denom else float("nan"),
        "n_top_1pct": n_top1pct,
        "rate_top_1pct": (n_top1pct / denom) if denom else float("nan"),
        "n_top_0p1pct": n_top01pct,
        "rate_top_0p1pct": (n_top01pct / denom) if denom else float("nan"),
        "rank_median": _quantile(finite_ranks, 0.5),
        "rank_mean": (float(np.mean(finite_ranks)) if finite_ranks else float("nan")),
        "rank_min": min(finite_ranks) if finite_ranks else float("nan"),
        "rank_max": max(finite_ranks) if finite_ranks else float("nan"),
        "percentile_median": _quantile(pcts, 0.5),
        "percentile_mean": (
            float(np.nanmean(np.asarray(pcts, dtype=float))) if denom else float("nan")
        ),
        "distance_to_top1_km_median": _quantile(dists, 0.5),
        "distance_to_top1_km_mean": (
            float(np.nanmean(np.asarray(dists, dtype=float))) if denom else float("nan")
        ),
        "mean_ap_median": _quantile(aps, 0.5),
        "mean_ap_min": (float(np.nanmin(np.asarray(aps, dtype=float))) if denom else float("nan")),
        "mean_ap_max": (float(np.nanmax(np.asarray(aps, dtype=float))) if denom else float("nan")),
        "best_algo_counts": algos,
        "n_positives_trained_on_min": (
            min((r.get("n_positives_trained_on", 0) for r in scored), default=0)
        ),
        "top_n": int(top_n),
        "tolerance_km": float(tolerance_km),
        "unscored_folds": [r.get("name") for r in unscored],
    }


def _write_partial(
    path: Path,
    group: str,
    records: list[dict],
    n_folds_total: int,
    errors: dict[str, str],
) -> Path:
    """Rewrite the crash-safe running aggregate after every fold."""
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "group": group,
        "n_folds_total": int(n_folds_total),
        "n_folds_done": len(records),
        "errors": errors,
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def run_loo(
    group: str,
    algos: list[str] | None = None,
    resume: bool = True,
    out_root: Path | str | None = None,
    max_folds: int | None = None,
    tolerance_km: float = TOLERANCE_KM,
    top_n: int = TOP_N,
    config_path: str | Path | None = None,
    folds: list[dict] | None = None,
) -> dict:
    """Withhold every in-extent seed of *group* once, then aggregate.

    Artifacts per fold live in ``<out_root>/<group>/<fold_slug>/<group>/``
    (the harness appends the trailing group level).  ``loo_partial.json``
    in the group directory is rewritten after each fold, so an
    interrupted run resumes instead of restarting.

    Returns the group payload (``summary`` + ``records`` + ``errors``).
    """
    root = Path(out_root or LOO_ROOT)
    group_dir = root / group
    group_dir.mkdir(parents=True, exist_ok=True)
    partial = group_dir / "loo_partial.json"

    if folds is None:
        folds = enumerate_folds(group, config_path)
    if max_folds is not None:
        folds = folds[: int(max_folds)]
    logger.info("LOO %s: %d fold(s) to run", group, len(folds))

    records: list[dict] = []
    errors: dict[str, str] = {}
    if resume and partial.exists():
        try:
            payload = json.loads(partial.read_text(encoding="utf-8"))
            records = list(payload.get("records") or [])
            errors = dict(payload.get("errors") or {})
        except (OSError, ValueError) as exc:
            logger.warning("ignoring unreadable %s: %s", partial, exc)
    done = {int(r["fold_idx"]) for r in records if "fold_idx" in r}
    # a recorded fold is a successful one: drop failure notes it may have
    # left behind on an earlier attempt, so retries do not report stale errors
    errors = {str(k): v for k, v in errors.items() if not (str(k).isdigit() and int(k) in done)}

    for n, fold in enumerate(folds, start=1):
        idx = fold["fold_idx"]
        harness_dir = group_dir / fold_slug(idx, fold["name"]) / group
        report = harness_dir / "holdout_report.json"
        label = f"[{n}/{len(folds)}] {group} fold {idx} ({fold['name']})"

        if idx in done:
            logger.info("%s - already recorded, skipping", label)
            continue

        cached = None
        if resume and report.exists():
            try:
                payload = json.loads(report.read_text(encoding="utf-8"))
                results = payload.get("results") or []
                cached = results[0] if results else None
            except (OSError, ValueError) as exc:
                logger.warning("%s - unreadable cached report: %s", label, exc)
        if cached is not None:
            records.append(compact_record(cached, fold))
            logger.info("%s - reused cached report", label)
        else:
            logger.info("%s - retraining (all other seeds kept)", label)
            try:
                result = run_holdout_test(
                    group,
                    holdout_idx=idx,
                    algos=algos,
                    predict=True,
                    config_path=config_path,
                    tolerance_km=tolerance_km,
                    top_n=top_n,
                    out_root=harness_dir.parent,
                )
                records.append(compact_record(result, fold))
            except Exception as exc:  # noqa: BLE001 - per-fold isolation
                logger.error("%s - FAILED: %s", label, exc, exc_info=True)
                errors[str(idx)] = f"{fold['name']}: {exc}"
                _write_partial(partial, group, records, len(folds), errors)
                continue

        # a fold that failed earlier but now succeeds (or reuses a good
        # cached report) must drop its stale failure from the resume state
        errors.pop(str(idx), None)
        rec = records[-1]
        logger.info(
            "%s - best=%s rank=%s pct=%s rediscovered=%s",
            label,
            rec.get("best_algo"),
            rec.get("rank"),
            rec.get("percentile"),
            rec.get("rediscovered"),
        )
        _write_partial(partial, group, records, len(folds), errors)

    summary = summarize(group, records, top_n=top_n, tolerance_km=tolerance_km)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "group": group,
        "n_folds_total": len(folds),
        "n_folds_done": len(records),
        "errors": errors,
        "summary": summary,
        "records": records,
    }
    (group_dir / "loo_summary.json").write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )
    rate = summary["rediscovery_rate"]
    logger.info(
        "LOO %s: rediscovered %d/%d %s, median rank %s",
        group,
        summary["n_rediscovered"],
        summary["n_folds_scored"],
        f"({100.0 * rate:.1f} %)" if np.isfinite(rate) else "(n/a)",
        summary["rank_median"],
    )
    return payload


def loo_report(groups: list[dict], out_dir: Path | str) -> Path:
    """Write the aggregate ``loo_report.json`` for every group payload."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "loo_report.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_groups": len(groups),
        "groups": groups,
    }
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    logger.info("LOO aggregate report -> %s", path)
    return path


def _fmt(value, spec: str = ".4g", suffix: str = "") -> str:
    """Format a possibly-NaN number for the Markdown report."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(num):
        return "n/a"
    return f"{num:{spec}}{suffix}"


def _pct(value) -> str:
    """Format a 0-1 rate as a percentage, or ``n/a``."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{100.0 * num:.1f} %" if np.isfinite(num) else "n/a"


def loo_markdown(groups: list[dict], out_dir: Path | str) -> Path:
    """Write ``LOO_VALIDATION.md``: method, rate table, per-fold detail."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "LOO_VALIDATION.md"
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = [
        "# Leave-one-deposit-out (LOO) rediscovery validation",
        "",
        f"Generated: {generated}",
        "",
        "## Method",
        "",
        "Every in-extent verified deposit is withheld once, in turn. For each fold the harness:",
        "",
        "1. writes reduced labels with the withheld deposit removed by name "
        "**and** within 2 km (so alias rows cannot leak);",
        "2. rebuilds the deposit-distance prior from those reduced labels, so "
        "the withheld cell no longer reads distance 0;",
        "3. retrains and predicts into an isolated per-fold directory;",
        "4. ranks the withheld deposit against every valid cell.",
        "",
        "**Rediscovery** = the best cell within 5 km of the withheld deposit "
        "ranks in the global top 100. The rank is the raw count of valid "
        "cells with a strictly higher probability (deterministic, no "
        "declustering).",
        "",
        "## Rediscovery rate by group",
        "",
        "| group | folds | scored | rediscovered | rate | median rank | "
        "median pct | in top 1% | median top-1 dist (km) | median AP | "
        "min. positives trained on |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for payload in groups:
        summary = payload.get("summary") or {}
        lines.append(
            f"| {summary.get('group')} "
            f"| {summary.get('n_folds', 0)} "
            f"| {summary.get('n_folds_scored', 0)} "
            f"| {summary.get('n_rediscovered', 0)} "
            f"| {_pct(summary.get('rediscovery_rate'))} "
            f"| {_fmt(summary.get('rank_median'), '.0f')} "
            f"| {_fmt(summary.get('percentile_median'), '.2f', ' %')} "
            f"| {_pct(summary.get('rate_top_1pct'))} "
            f"| {_fmt(summary.get('distance_to_top1_km_median'), '.1f')} "
            f"| {_fmt(summary.get('mean_ap_median'), '.3f')} "
            f"| {summary.get('n_positives_trained_on_min', 0)} |"
        )
    lines.append("")

    for payload in groups:
        records = payload.get("records") or []
        lines += [
            f"## {payload.get('group')} — per-fold detail",
            "",
            "| fold | deposit | best | rank | percentile | rediscovered | "
            "top-1 dist (km) | AP | trained on |",
            "|---:|---|---|---:|---:|:--:|---:|---:|---:|",
        ]
        for rec in sorted(records, key=lambda r: r.get("fold_idx", 0)):
            lines.append(
                f"| {rec.get('fold_idx')} "
                f"| {rec.get('name')} "
                f"| {rec.get('best_algo')} "
                f"| {_fmt(rec.get('rank'), '.0f')} "
                f"| {_fmt(rec.get('percentile'), '.2f')} "
                f"| {'yes' if rec.get('rediscovered') else 'no'} "
                f"| {_fmt(rec.get('distance_to_top1_km'), '.1f')} "
                f"| {_fmt(rec.get('mean_ap'), '.3f')} "
                f"| {rec.get('n_positives_trained_on', 0)} |"
            )
        lines.append("")

    lines += [
        "## Caveats",
        "",
        "* **Rates are not comparable across groups.** `bauxite` has only 3 "
        "in-extent seeds, so each fold trains on 2 positives — its rate is "
        "near-meaningless statistically. `copper_zinc` trains on ~38 "
        "positives over a 67.5 M-cell AOI. Read the `trained on` column.",
        "* **A miss is not proof of failure.** A withheld deposit in the "
        "99.9th percentile that still lands below rank 100 shows the surface "
        "concentrating where it should; the percentile column shows that.",
        "* **Background points are reused unchanged.** Background sampling "
        "excluded a buffer around the original labels, so a small hole in "
        "background density survives near each withheld deposit. It encodes "
        "no holdout information into any feature.",
        "* **Folds are independent single-deposit removals**, not a nested "
        "resampling of model selection: the algorithm choice is re-made "
        "inside each fold from the same 3-algorithm comparison.",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("LOO markdown -> %s", path)
    return path


def main() -> None:
    """CLI entry point for the full leave-one-deposit-out study."""
    parser = argparse.ArgumentParser(
        description="Full leave-one-deposit-out rediscovery rate: withholds "
        "every in-extent seed once per group and aggregates ranks"
    )
    parser.add_argument("--group", choices=sorted(GROUP_CONFIGS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--algos",
        nargs="+",
        default=None,
        choices=["rf", "xgb", "lgbm"],
        help="algorithms to train per fold (default: the config comparison list)",
    )
    parser.add_argument(
        "--max-folds", type=int, default=None, help="cap folds per group (smoke testing)"
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="ignore cached per-fold reports / partial state"
    )
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--tolerance-km", type=float, default=TOLERANCE_KM)
    parser.add_argument("--config", default=None, help="config override (single-group runs only)")
    parser.add_argument(
        "--report-only", action="store_true", help="re-aggregate saved partials without retraining"
    )
    parser.add_argument("--out-root", default=None, help="artifact root (default: outputs/loo)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not args.group and not args.all:
        parser.print_help()
        return
    if args.all and args.config:
        logger.warning("--config ignored with --all (per-group belt configs)")
        args.config = None
    names = sorted(GROUP_CONFIGS) if args.all else [args.group]
    root = Path(args.out_root) if args.out_root else LOO_ROOT

    groups: list[dict] = []
    for group in names:
        logger.info("=" * 70)
        if args.report_only:
            partial = root / group / "loo_partial.json"
            if not partial.exists():
                logger.error("no saved partial for %s: %s", group, partial)
                continue
            payload = json.loads(partial.read_text(encoding="utf-8"))
            records = payload.get("records") or []
            groups.append(
                {
                    "generated_at": payload.get("generated_at"),
                    "group": group,
                    "n_folds_total": payload.get("n_folds_total", len(records)),
                    "n_folds_done": len(records),
                    "errors": payload.get("errors") or {},
                    "summary": summarize(
                        group, records, top_n=args.top_n, tolerance_km=args.tolerance_km
                    ),
                    "records": records,
                }
            )
            continue
        groups.append(
            run_loo(
                group,
                algos=args.algos,
                resume=not args.no_resume,
                out_root=root,
                max_folds=args.max_folds,
                tolerance_km=args.tolerance_km,
                top_n=args.top_n,
                config_path=args.config,
            )
        )

    if not groups:
        logger.error("no groups processed")
        sys.exit(1)

    loo_report(groups, root)
    loo_markdown(groups, root)
    logger.info("=" * 70)
    logger.info(
        "LOO REDISCOVERY SUMMARY (tolerance %.1f km, top-%d)", args.tolerance_km, args.top_n
    )
    for payload in groups:
        summary = payload["summary"]
        logger.info(
            "  %-24s %d/%d rediscovered %s",
            summary["group"],
            summary["n_rediscovered"],
            summary["n_folds_scored"],
            _pct(summary["rediscovery_rate"]),
        )
    failed = {p["group"]: p["errors"] for p in groups if p.get("errors")}
    if failed:
        logger.error("folds failed: %s", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
