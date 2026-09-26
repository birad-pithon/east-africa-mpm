"""Model-card generation for prospectivity model bundles.

Produces a human-readable ``model_card_<group>.md`` next to the trained
bundle, covering the sections reviewers actually need: intended use,
training data, evaluation protocol + metrics (with the anomaly
baseline reference), probability calibration, and full run
provenance. Generated from the artifacts that ``train_group`` already
writes â€” no manual duplication.

CLI::

    python -m src.models.model_card --group tin_tungsten_tantalum
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib

from src.utils import project_path

logger = logging.getLogger(__name__)

__all__ = ["build_model_card"]


def _fmt(value, nd: int = 3, dash: str = "n/a") -> str:
    if value is None:
        return dash
    try:
        return f"{float(value):.{nd}f}"
    except (TypeError, ValueError):
        return str(value)


def build_model_card(
    group: str,
    metrics_path: str | Path | None = None,
    bundle_path: str | Path | None = None,
    out_path: str | Path | None = None,
) -> Path:
    """Render ``model_card_<group>.md`` from metrics + bundle artifacts.

    Parameters
    ----------
    group : commodity-group key.
    metrics_path : defaults to ``outputs/models/metrics_<group>.json``.
    bundle_path : defaults to the bundle referenced by the metrics.
    out_path : defaults to ``outputs/models/model_card_<group>.md``.

    Returns
    -------
    Path of the written card.
    """
    out_dir = Path(metrics_path).parent if metrics_path \
        else project_path("outputs", "models")
    metrics_path = Path(metrics_path) if metrics_path \
        else out_dir / f"metrics_{group}.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics not found: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    if bundle_path is None:
        bundle_path = out_dir / (
            f"model_{group}_{metrics['best_algo']}.joblib")
    bundle = joblib.load(bundle_path)
    prov = metrics.get("provenance") or bundle.get("provenance") or {}
    git = prov.get("git", {})
    algo = metrics["best_algo"]
    results = metrics["results"]

    best = results[algo]
    baseline_ap = best.get("anomaly_baseline_mean_ap")
    calib = metrics.get("calibration")

    lines: list[str] = []
    add = lines.append
    add(f"# Model card â€” {group}")
    add("")
    add(f"*Generated from `{metrics_path.name}`; algorithm `{algo}` "
        f"(best of {', '.join(results)}).*")
    add("")
    add("## Intended use")
    add("")
    add("Gridded mineral prospectivity screening: rank 30 m cells of the "
        "study AOI by P(deposit) to prioritise field verification for "
        f"the `{group}` commodity group. Output is a *screening prior*, "
        "not a resource estimate; every high-scoring cell requires "
        "ground truth before any decision.")
    add("")
    add("## Geological context")
    add("")
    _ctx = {
        "tin_tungsten_tantalum": (
            "Karagwe-Ankole Belt (Mesoproterozoic Kibaran orogen): "
            "Sn-W-Ta mineralisation hosted in pegmatites, quartz veins "
            "and greisenised granites of the KIB province. Target styles "
            "are cassiterite-bearing granite-related pegmatites and "
            "wolframite-quartz vein systems."
        ),
        "copper_zinc": (
            "Central African Copperbelt (Neoproterozoic Lufilian arc): "
            "sediment-hosted stratiform Cu-Co and Cu-Zn-(Pb) mineralisation "
            "within Roan Group metasediments."
        ),
        "bauxite": (
            "Usambara lateritic bauxite province (Tanga, Tanzania): "
            "supergene Al(OH)3 enrichment on plateau-capping weathering "
            "profiles over basement gneisses."
        ),
    }
    add(_ctx.get(group,
                 "Group-specific geological context not yet documented."))
    add("")
    add("Feature groups and the geological evidence they encode: "
        "lithology + one-hot classes = host-rock type; "
        "distance-to-* = proximity to contact zones / fertile units; "
        "dist_to_contact_m = structural proximity; "
        "terrain (slope/aspect/TPI/relief/plateau) = geomorphic position; "
        "Sentinel-2 bands = alteration/laterite surface signatures.")
    add("")
    add("")
    add("## Training data")
    add("")
    add(f"* labelled rows: **{metrics['n_rows']}** "
        f"({metrics['n_positives']} positives, base rate "
        f"{_fmt(metrics['n_positives'] / max(1, metrics['n_rows']))})")
    add(f"* features: {len(metrics['features'])} â€” "
        f"{', '.join(sorted({f.split(':')[0] for f in metrics['features']}))}")
    add(f"* class balancing: "
        f"{'SMOTE per training fold' if metrics.get('oversample') else 'class weights'}")
    add("")
    add("## Evaluation protocol")
    add("")
    add(f"* spatial block CV, {best['n_folds']} folds, block "
        f"{_fmt(metrics['block_m'], 0)} m, train/test buffer "
        f"{_fmt(metrics['buffer_km'], 0)} km â€” fold plan certified "
        "(`src.validate.certify_folds`) before fitting")
    add("* primary metric: average precision (PR-AUC); ROC-AUC secondary")
    add("")
    add("## Results")
    add("")
    add("| algorithm | mean AP | mean ROC-AUC |")
    add("|---|---|---|")
    for name, res in results.items():
        add(f"| {name} | {_fmt(res.get('mean_average_precision'))} | "
            f"{_fmt(res.get('mean_roc_auc'))} |")
    add("")
    add(f"Anomaly-baseline (IsolationForest-on-barren) reference AP: "
        f"**{_fmt(baseline_ap)}** â€” supervised models must beat this and "
        "the class base rate to demonstrate signal.")
    add("")
    add("## Probability calibration")
    add("")
    if calib:
        add(f"* Brier score: **{_fmt(calib['brier'])}**; "
            f"ECE ({calib['n_bins_usable']} usable bins): "
            f"**{_fmt(calib['ece'])}**; max |pred-obs| gap: "
            f"{_fmt(calib['max_abs_gap'])}")
        add(f"* reliability diagram: `reliability_{group}.png`")
        add("* read P(deposit) as a *ranking prior*; absolute values are "
            "not calibrated probabilities unless ECE is small")
    else:
        add("* not yet computed â€” run training with calibration enabled")
    add("")
    add("## Limitations & known caveats")
    add("")
    add("* this is a coarse regional-screening prior, not a drill "
        "target list - each high-scoring cell must be validated against "
        "local geological maps (1:50k national surveys) and field "
        "observation before permitting or drilling")
    add("* positive labels are USGS MRDS occurrences + curated seeds â€” "
        "exploration-biased; barren-but-unexplored ground may hide "
        "deposits (background strata mitigate, not eliminate)")
    add("* lithology source is the ~1:35M GSC world map â€” coarse; "
        "replace with national 1:50k layers when available")
    add(f"* only {metrics['n_positives']} positives â€” fold metrics have "
        "high variance; treat single-fold numbers with caution")
    add("")
    add("## Provenance")
    add("")
    add(f"* git: commit `{git.get('commit', 'unknown')}` "
        f"(branch `{git.get('branch', '?')}`, "
        f"dirty={git.get('dirty', '?')})")
    add(f"* trained: {prov.get('timestamp_utc', 'unknown')}")
    pkgs = prov.get("packages", {})
    add(f"* key versions: python {pkgs.get('python', '?')}, "
        f"scikit-learn {pkgs.get('scikit-learn', '?')}, "
        f"pandas {pkgs.get('pandas', '?')}, "
        f"geopandas {pkgs.get('geopandas', '?')}")
    hashes = prov.get("input_sha256", {})
    if hashes:
        add("* input SHA-256:")
        for name, digest in hashes.items():
            add(f"  * `{name}`: `{digest[:16]}...`")
    add("")

    out_path = Path(out_path) if out_path \
        else out_dir / f"model_card_{group}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Model card -> %s", out_path)
    return out_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate a model card from training artifacts")
    parser.add_argument("--group", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    out_dir = Path(args.out_dir) if args.out_dir else None
    card = build_model_card(args.group, metrics_path=(
        out_dir / f"metrics_{args.group}.json") if out_dir else None)
    print(f"OK model card -> {card}")


if __name__ == "__main__":
    main()
