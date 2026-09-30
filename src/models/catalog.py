"""Canonical commodity-group catalogue: group key -> belt pipeline config.

``configs/belts.yaml`` is the source of truth for *which* commodity groups
exist (belts + MRDS codes). This module pairs every group with the pipeline
config that owns its AOI grid, feature-raster allowlist and model params, and
expands the ``--group all`` shorthand the CLIs use to cover the entire
catalogue in one command.

Without this mapping the training CLI defaulted every group to
``configs/karagwe.yml``, so ``--group copper_zinc`` silently trained on the
wrong AOI / allowlist unless ``--config`` was remembered.
"""
from __future__ import annotations

import logging

from src.utils import load_config, project_path

logger = logging.getLogger(__name__)

__all__ = [
    "ALL",
    "DEFAULT_CONFIG",
    "GROUP_CONFIGS",
    "config_for_group",
    "known_groups",
    "resolve_groups",
]

#: value accepted by ``--group`` to mean "every group in the catalogue"
ALL = "all"

#: last resort when a group has no explicit mapping (KAB is the original AOI)
DEFAULT_CONFIG = "configs/karagwe.yml"

#: group key (configs/belts.yaml `groups:`) -> pipeline config
GROUP_CONFIGS: dict[str, str] = {
    "tin_tungsten_tantalum": "configs/karagwe.yml",
    "copper_zinc": "configs/copperbelt.yml",
    "bauxite": "configs/usambara.yml",
}


def known_groups() -> list[str]:
    """Every commodity group declared in ``configs/belts.yaml``."""
    cfg = load_config(project_path("configs", "belts.yaml"))
    return list(cfg.get("groups", {}))


def config_for_group(group: str, config_path: str | None = None) -> str:
    """Pipeline config to use for *group*.

    An explicit *config_path* always wins (per-belt re-runs, tests, alternate
    AOIs); otherwise the catalogue mapping is used.
    """
    if config_path:
        return str(config_path)
    if group in GROUP_CONFIGS:
        return GROUP_CONFIGS[group]
    logger.warning(
        "no belt config mapped to group '%s' in src.models.catalog "
        "- falling back to %s (wrong AOI/allowlist is possible)",
        group, DEFAULT_CONFIG)
    return DEFAULT_CONFIG


def resolve_groups(group: str,
                   config_path: str | None = None) -> list[tuple[str, str]]:
    """``[(group, config_path), ...]`` to run for a ``--group`` value.

    ``"all"`` expands to every group declared in ``configs/belts.yaml``, in
    catalogue order; anything else stays a single-group request.
    """
    if group != ALL:
        return [(group, config_for_group(group, config_path))]

    groups = known_groups()
    if not groups:
        raise RuntimeError(
            "configs/belts.yaml declares no 'groups:' entries - nothing to "
            "train")
    unmapped = [g for g in groups if g not in GROUP_CONFIGS]
    if unmapped:
        logger.warning(
            "group(s) %s have no config mapping in src.models.catalog - "
            "using %s for them", ", ".join(unmapped), DEFAULT_CONFIG)
    return [(g, config_for_group(g, config_path)) for g in groups]
