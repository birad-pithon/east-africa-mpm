"""CLI: rank top-N unexplored candidate cells outside licence areas.

    python -m src.predict.main \
        --proba outputs/models/proba_tin_tungsten_tantalum_xgb.tif \
        --group tin_tungsten_tantalum --n 50 \
        [--licences data/processed/licences_kab.geojson] [--map]
"""
from __future__ import annotations

import argparse
import logging

from src.predict.rank import rank_candidates
from src.predict.viz import candidates_to_geojson, render_interactive_map
from src.utils import project_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank top-N candidate cells from a probability raster"
    )
    parser.add_argument("--proba", required=True,
                        help="P(deposit) GeoTIFF from src.models.predict")
    parser.add_argument("--group", default="",
                        help="commodity-group tag written into outputs")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--licences", default=None,
                        help="licence polygon layer (GeoJSON/GPKG); cells "
                             "inside polygons are excluded")
    parser.add_argument("--min-prob", type=float, default=0.0)
    parser.add_argument("--spacing-cells", type=int, default=2,
                        help="greedy min separation between picks")
    parser.add_argument("--out-dir", default=None,
                        help="default: outputs/maps")
    parser.add_argument("--map", action="store_true",
                        help="also render interactive leafmap HTML")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")

    out_dir = (args.out_dir and project_path(args.out_dir)) or \
        project_path("outputs", "maps")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = rank_candidates(
        args.proba, n=args.n, licence_path=args.licences,
        min_prob=args.min_prob, spacing_cells=args.spacing_cells,
    )

    stem = f"top{args.n}_{args.group}" if args.group else f"top{args.n}"
    geojson = candidates_to_geojson(df, out_dir / f"{stem}_candidates.geojson",
                                    group=args.group)
    print(f"\nOK {len(df)} candidates -> {geojson}")
    if len(df):
        print(df.head(10).to_string(index=False))

    if args.map:
        html = render_interactive_map(
            args.proba, geojson, out_dir / f"{stem}_map.html",
            group=args.group or "prospectivity",
        )
        print(f"OK interactive map -> {html}")


if __name__ == "__main__":
    main()
