"""Positive-label and background-sample generation for the MPM pipeline."""

from src.labels.analogs import (
    GROUP_CODES,
    WORLD_ANALOGUES,
    african_out_of_belt_analogues,
    world_class_analogues,
    write_analogues,
)
from src.labels.build_labels import (
    belt_polygon,
    build_labels,
    mrds_in_belts,
    sample_background,
    seed_points,
    utm_epsg,
)
from src.labels.field_update import LabelManifest, merge_field_labels, read_manifest
from src.labels.known_deposits import SEED_DEPOSITS

_REFERENCE_EXPORTS = {
    "VERIFIED_DEV_STATS",
    "verified_mrds_references",
    "target_relations",
    "build_reference_inventory",
}


def __getattr__(name: str):
    if name in _REFERENCE_EXPORTS:
        from src.labels import reference_study
        return getattr(reference_study, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "build_labels",
    "mrds_in_belts",
    "sample_background",
    "seed_points",
    "belt_polygon",
    "utm_epsg",
    "SEED_DEPOSITS",
    "merge_field_labels",
    "read_manifest",
    "LabelManifest",
    "GROUP_CODES",
    "WORLD_ANALOGUES",
    "african_out_of_belt_analogues",
    "world_class_analogues",
    "write_analogues",
    "VERIFIED_DEV_STATS",
    "verified_mrds_references",
    "target_relations",
    "build_reference_inventory",
]
