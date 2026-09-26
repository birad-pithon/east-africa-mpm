"""Field-feedback loop: ingest verified ground truth into the labels.

Implements SOW safeguard #5 (No feedback loop / static training set).

Prospectivity models decay: deposits get discovered, barren prospects
get confirmed barren, and a model trained on yesterday's MRDS snapshot
never learns any of it. This module closes the loop:

1. A geologist exports field-verification results as a point layer
   (GeoPackage / GeoJSON / Shapefile / CSV) carrying at least a
   ``status`` column (``positive`` / ``barren`` -- case-insensitive)
   and optionally ``notes``, ``date``, ``commodities``.
2. :func:`merge_field_labels` validates the input, de-duplicates points
   within ``dedup_m`` of an already-labelled location (re-labelling
   opposite-class hits instead of adding near-duplicates), and appends
   the remainder to ``labels_<group>.gpkg`` / ``background_<group>.gpkg``.
3. A versioned manifest (``label_manifest.json``) records every merge.
   Retraining via ``train_group`` picks up the enriched labels
   automatically because it reads the same GeoPackages.

CLI::

    python -m src.labels.field_update --group copper_zinc \\
        --new field_results_2026Q3.gpkg [--dedup-m 2000]
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.utils import project_path

logger = logging.getLogger(__name__)

__all__ = ["merge_field_labels", "read_manifest", "LabelManifest"]

_POSITIVE_TOKENS = {"positive", "deposit", "occurrence",
                    "1", "true", "yes"}
_BARREN_TOKENS = {"barren", "negative", "no_mineralisation",
                  "0", "false", "no"}
STATUS_CANDIDATES = ("status", "result", "verified", "class",
                     "field_status", "label")


class LabelManifest:
    """Thin reader/writer for the label version manifest."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists() else {"groups": {}}
        )

    def record(self, group: str, entry: dict) -> None:
        g = self.data["groups"].setdefault(
            group, {"version": 0, "history": []})
        g["version"] += 1
        entry["version"] = g["version"]
        g["history"].append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2),
                             encoding="utf-8")

    def version(self, group: str) -> int:
        return int(self.data.get("groups", {}).get(
            group, {}).get("version", 0))


def read_manifest() -> dict:
    """Return the full label manifest (empty if none yet)."""
    p = project_path("data", "processed", "label_manifest.json")
    return (json.loads(p.read_text(encoding="utf-8"))
            if p.exists() else {"groups": {}})


def _normalise_status(frame: gpd.GeoDataFrame) -> np.ndarray:
    """Map heterogeneous status strings to {1, 0}; raise if unusable."""
    col = next((c for c in frame.columns
                if c.lower() in STATUS_CANDIDATES), None)
    if col is None:
        raise ValueError(
            f"no status column found (looked for {STATUS_CANDIDATES}); "
            "field results must record verified positive/barren per point"
        )
    out = np.full(len(frame), -1, dtype=int)
    vals = frame[col].astype(str).str.strip().str.lower()
    out[vals.isin(_POSITIVE_TOKENS)] = 1
    out[vals.isin(_BARREN_TOKENS)] = 0
    unknown = int((out == -1).sum())
    if unknown:
        raise ValueError(
            f"{unknown} points have unrecognised status values "
            f"{sorted(set(vals[out == -1]))[:5]}...; expected e.g. "
            "'positive' / 'barren'"
        )
    return out

def merge_field_labels(
    group: str,
    new_points_path: Path | str,
    dedup_m: float = 2_000.0,
    labels_gpkg: Path | str | None = None,
    background_gpkg: Path | str | None = None,
    manifest_path: Path | str | None = None,
) -> dict:
    """Merge verified field points into a group's label GeoPackages.

    Points within ``dedup_m`` of an existing same-class label are
    dropped as duplicates; within ``dedup_m`` of an *opposite-class*
    label they override the stale row (field truth wins).

    Returns a summary dict and bumps the manifest version.
    """
    processed = project_path("data", "processed")
    pos_path = Path(labels_gpkg or processed / f"labels_{group}.gpkg")
    bg_path = Path(background_gpkg or processed / f"background_{group}.gpkg")
    new_path = Path(new_points_path)
    if not pos_path.exists():
        raise FileNotFoundError(
            f"{pos_path} missing - run src.labels.build_labels first")
    if not new_path.exists():
        raise FileNotFoundError(f"{new_path} missing")

    new_pts = gpd.read_file(new_path)
    if new_pts.crs is None:
        raise ValueError(f"{new_path.name} has no CRS; cannot place points")
    new_pts = new_pts.to_crs("EPSG:4326")
    if len(new_pts) == 0:
        raise ValueError(f"{new_path.name} contains no points")
    y_new = _normalise_status(new_pts)

    existing = {
        1: gpd.read_file(pos_path),
        0: (gpd.read_file(bg_path) if bg_path.exists()
            else gpd.GeoDataFrame(
                columns=["label", "geometry"], geometry="geometry",
                crs="EPSG:4326")),
    }
    for cls, frame in existing.items():
        frame["y"] = cls

    # KD-tree per existing class, in a metric CRS for metre-true dedup
    utm = new_pts.estimate_utm_crs() or "EPSG:32736"
    new_utm = new_pts.to_crs(utm)
    trees: dict[int, cKDTree | None] = {}
    for cls, frame in existing.items():
        f = frame.to_crs(utm)
        trees[cls] = (
            cKDTree(np.column_stack([f.geometry.x, f.geometry.y]))
            if len(f) else None
        )

    keep_rows: list[int] = []
    drops: dict[int, list[int]] = {1: [], 0: []}
    n_dup = 0
    for i, (pt, cls) in enumerate(zip(new_utm.geometry, y_new, strict=False)):
        dup = False
        for other_cls in (1, 0):
            tree = trees[other_cls]
            if tree is None:
                continue
            d, j = tree.query([pt.x, pt.y], k=1)
            if d <= dedup_m:
                if other_cls == cls:
                    dup = True              # same truth already recorded
                    n_dup += 1
                else:
                    drops[other_cls].append(int(j))   # stale label
                    trees[other_cls] = None           # drop once
                break
        if not dup:
            keep_rows.append(i)

    added = new_pts.iloc[keep_rows].copy()
    added["y"] = y_new[keep_rows]

    def _rebuild(frame: gpd.GeoDataFrame, cls: int,
                 new_rows: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        stale = drops.get(cls, [])
        if stale:
            frame = frame.drop(index=frame.index[stale], errors="ignore")
        if len(new_rows):
            extra = {c: new_rows[c] for c in
                     ("commodities", "notes") if c in new_rows.columns}
            # NOTE: pass geometry as an array - a Series would be
            # index-aligned against the dict's fresh RangeIndex and
            # silently produce None geometries when new_rows retains
            # its original (filtered) index.
            rows = gpd.GeoDataFrame(
                {**extra,
                 "name": [f"field_{group}_{i}"
                          for i in range(len(new_rows))],
                 "source": "field_verified",
                 "reference": str(new_path.name),
                 "y": cls},
                geometry=new_rows.geometry.values, crs="EPSG:4326")
            frame = pd.concat([frame, rows], ignore_index=True)
        frame = gpd.GeoDataFrame(frame, geometry="geometry",
                                 crs="EPSG:4326")
        frame["label"] = cls
        return frame.reset_index(drop=True)

    pos_out = _rebuild(existing[1].drop(columns=["y"], errors="ignore"),
                       1, added[added["y"] == 1])
    bg_out = _rebuild(existing[0].drop(columns=["y"], errors="ignore"),
                      0, added[added["y"] == 0])

    pos_out.to_file(pos_path, driver="GPKG")
    bg_out.to_file(bg_path, driver="GPKG")

    summary = {
        "group": group,
        "source": str(new_path),
        "merged_at": datetime.now(timezone.utc).isoformat(),
        "received": int(len(new_pts)),
        "added_positives": int((added["y"] == 1).sum()),
        "added_background": int((added["y"] == 0).sum()),
        "dropped_duplicates": n_dup,
        "relabelled_stale_rows": {str(k): len(v)
                                  for k, v in drops.items() if v},
        "labels_total": int(len(pos_out)),
        "background_total": int(len(bg_out)),
    }

    manifest = LabelManifest(Path(manifest_path)
                             if manifest_path
                             else processed / "label_manifest.json")
    manifest.record(group, summary)
    logger.info(
        "[%s] field merge: +%d pos +%d bg, %d dup dropped, stale "
        "relabelled %s -> labels=%d bg=%d (manifest v%d)",
        group, summary["added_positives"], summary["added_background"],
        n_dup, summary["relabelled_stale_rows"],
        summary["labels_total"], summary["background_total"],
        manifest.version(group))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge field-verified ground truth into labels")
    parser.add_argument("--group", required=True)
    parser.add_argument("--new", required=True,
                        help="field results (GPKG/GeoJSON/SHP/CSV)")
    parser.add_argument("--dedup-m", type=float, default=2_000.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    summary = merge_field_labels(args.group, args.new,
                                 dedup_m=args.dedup_m)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
