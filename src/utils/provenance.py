"""Run provenance capture for reproducible training.

``collect_provenance`` assembles everything needed to reconstruct a
training run:

* VCS state — commit SHA, branch, dirty flag (best effort; absent when
  git is unavailable);
* package versions of the geospatial / ML stack;
* SHA-256 digests of every input file consumed by the run
  (label GeoPackages, background, feature rasters);
* a snapshot of the pipeline config.

The dict is embedded in ``outputs/models/metrics_<group>.json`` and the
model bundle so any published metric can be traced back to exact inputs.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["collect_provenance", "sha256_file"]

# Packages worth pinning in the provenance record (missing ones skipped).
_TRACKED_PACKAGES = [
    "numpy", "pandas", "scipy", "geopandas", "rasterio", "shapely",
    "pyproj", "PyYAML", "requests", "scikit-learn", "xgboost", "lightgbm",
    "imbalanced-learn", "ruff", "pytest",
]


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 hex digest of *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _run_git(args: list[str]) -> str | None:
    """Run one git command; return stripped stdout or None on failure."""
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:                                   # noqa: BLE001
        return None


def git_state() -> dict:
    """Commit / branch / dirty flag; all-unknown when git is unavailable."""
    commit = _run_git(["rev-parse", "HEAD"]) or "unknown"
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"
    dirty_out = _run_git(["status", "--porcelain"])
    dirty = True if dirty_out is None else bool(dirty_out.strip())
    return {"commit": commit, "branch": branch, "dirty": dirty}


def package_versions() -> dict[str, str]:
    """Installed versions of the tracked stack (missing packages skipped)."""
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
    }
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def collect_provenance(
    config_path: str | Path | None = None,
    input_paths: list[str | Path] | None = None,
) -> dict:
    """Build the provenance record for a training / inference run.

    Parameters
    ----------
    config_path : pipeline YAML used for the run (snapshotted verbatim).
    input_paths : files whose SHA-256 digests should be recorded
        (labels, background, feature rasters). Missing files are noted
        but do not fail the run.

    Returns
    -------
    dict suitable for embedding in ``metrics_<group>.json``.
    """
    inputs: dict[str, str] = {}
    for p in input_paths or []:
        p = Path(p)
        if p.exists():
            try:
                inputs[p.name] = sha256_file(p)
            except OSError as exc:
                inputs[p.name] = f"hash-error: {exc}"
        else:
            inputs[p.name] = "missing"

    config_snapshot = None
    if config_path is not None and Path(config_path).exists():
        try:
            config_snapshot = Path(config_path).read_text(encoding="utf-8")
        except OSError:
            config_snapshot = None

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "git": git_state(),
        "packages": package_versions(),
        "input_sha256": inputs,
        **({"config_snapshot": config_snapshot}
           if config_snapshot is not None else {}),
    }
