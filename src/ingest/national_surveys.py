"""National survey / direct-request data framework (RMB, GST, CAMI, DGSM).

Several key layers (high-resolution national geology, aeromagnetics,
geochemistry, licence boundaries) have **no public bulk download**; they
must be requested from national authorities. This module provides:

* a registry of every such source (from ``configs/data_sources.yaml``)
  with its purpose and contact,
* ``print_requests()`` to generate a request tracking table, and
* a provenance ``MANIFEST.json`` under ``data/raw/national/`` so locally
  delivered files are tracked (source, received date, CRS, notes) and can
  be recalled by the rest of the pipeline via ``get_survey_data()``.

Usage::

    from src.ingest.national_surveys import (
        list_request_sources, register_local_data, get_survey_data,
    )
    register_local_data("rmb_geology", "data/raw/national/rmb/rmb_geology.gpkg",
                        crs="EPSG:32736", notes="Received 2026-08-26 via RWB")
    path = get_survey_data("rmb_geology")

CLI::

    python -m src.ingest.national_surveys --print-requests
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from src.utils import load_config, project_path

logger = logging.getLogger(__name__)

NATIONAL_BASE = project_path("data", "raw", "national")
MANIFEST_PATH = NATIONAL_BASE / "MANIFEST.json"

# Only these access types are surfaced as "request-or-manual" sources
ACCESS_TYPES = ("direct-request", "portal", "manual")


def _load_registry() -> dict[str, dict[str, Any]]:
    """Load the sources registry and return the national/request sources."""
    cfg = load_config(project_path("configs", "data_sources.yaml"))
    sources = cfg["sources"]
    return {
        key: src for key, src in sources.items() if src.get("access") in ACCESS_TYPES
    }


def _load_manifest() -> dict[str, Any]:
    """Return the manifest dict (empty if none)."""
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"sources": {}}


def _save_manifest(manifest: dict[str, Any]) -> None:
    NATIONAL_BASE.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _today() -> str:
    from datetime import date

    return date.today().isoformat()


def list_request_sources() -> dict[str, dict[str, Any]]:
    """Return all sources that require a direct request or manual ingest."""
    return _load_registry()


def print_requests() -> str:
    """Build a markdown summary of pending request sources + received data.

    Returns
    -------
    str
        Human-readable markdown summary.

    Notes
    -----
    The table lists every source requiring a direct request, portal export,
    or manual ingest; the "Already received" section lists everything that
    has been registered locally via ``register_local_data``.
    """
    reg = list_request_sources()
    lines = [
        "# National survey / direct-request data",
        "",
        "| source id | layer | provider | access | status |",
        "|---|---|---|---|---|",
    ]
    for key, src in sorted(reg.items()):
        lines.append(
            f"| `{key}` | {src.get('layer', '')} | "
            f"{src.get('provider', '')} | {src.get('access', '')} | "
            f"{src.get('status', '')} |"
        )

    contacts = sorted(
        {s.get("contact_url", "") for s in reg.values() if s.get("contact_url")}
    )
    lines += ["", "## Contact URLs", ""]
    lines += [f"- {c}" for c in contacts]

    manifest = _load_manifest()
    received = manifest.get("sources", {})
    lines += ["", "## Already received / registered", ""]
    if received:
        for s_id, meta in received.items():
            lines.append(
                f"- `{s_id}` -> {meta.get('path', '')} "
                f"(received {meta.get('received', '')})"
            )
    else:
        lines.append("- (none yet)")
    return "\n".join(lines) + "\n"


def register_local_data(
    source_id: str,
    path: str | Path,
    crs: str | None = None,
    notes: str | None = None,
) -> Path:
    """Register a locally-provided dataset under ``data/raw/national/``.

    The file is copied into a per-survey directory and the manifest is
    updated with provenance metadata.

    Parameters
    ----------
    source_id : str
        Key in ``configs/data_sources.yaml`` (e.g. ``"rmb_geology"``).
    path : str | Path
        Path to the delivered file (shapefile, gpkg, geojson, tif, ...).
    crs : str | None
        Optional CRS string for provenance.
    notes : str | None
        Optional free-text provenance notes.

    Returns
    -------
    Path
        The canonical stored path.
    """
    src_path = Path(path)
    if not src_path.exists():
        raise FileNotFoundError(f"File not found: {src_path}")

    dest_dir = NATIONAL_BASE / source_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src_path.name
    shutil.copy2(src_path, dest)

    manifest = _load_manifest()
    manifest.setdefault("sources", {})[source_id] = {
        "path": str(dest),
        "received": _today(),
        "crs": crs,
        "notes": notes,
    }
    _save_manifest(manifest)
    logger.info("Registered %s -> %s", source_id, dest)
    return dest


def get_survey_data(source_id: str) -> Path | None:
    """Return the canonical path for a registered national dataset.

    Returns
    -------
    Path | None
        The stored path if registered, else ``None``.
    """
    manifest = _load_manifest()
    meta = manifest["sources"].get(source_id)
    if not meta:
        return None
    p = Path(meta["path"])
    return p if p.exists() else None


def main():
    """CLI entry point for the national surveys utility."""
    parser = argparse.ArgumentParser(
        description="National survey / direct-request data utility"
    )
    parser.add_argument(
        "--print-requests",
        action="store_true",
        help="Print the request/manifest summary",
    )
    parser.add_argument(
        "--register",
        nargs=2,
        metavar=("SOURCE_ID", "PATH"),
        help="Register a locally delivered dataset",
    )
    parser.add_argument(
        "--get", metavar="SOURCE_ID", help="Print the stored path for a source"
    )
    parser.add_argument("--crs", default=None, help="CRS for --register")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.print_requests:
        print(print_requests())
    elif args.register:
        dest = register_local_data(args.register[0], args.register[1], crs=args.crs)
        print(f"Registered: {dest}")
    elif args.get:
        pth = get_survey_data(args.get)
        print(pth if pth else f"No data registered for {args.get}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
