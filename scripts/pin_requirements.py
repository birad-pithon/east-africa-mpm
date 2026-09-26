"""Regenerate ``requirements-lock.txt`` with pinned installed versions.

Usage::

    python scripts/pin_requirements.py
"""
from __future__ import annotations

import re
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    req_text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    names = [
        m.group(1)
        for line in req_text.splitlines()
        if (m := re.match(r"([A-Za-z0-9_.-]+)>=", line))
    ]
    lines = [
        "# Auto-generated lock file - pinned versions of requirements.txt",
        "# Regenerate with: python scripts/pin_requirements.py",
        "",
    ]
    missing = []
    for name in names:
        try:
            lines.append(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            missing.append(name)
    (ROOT / "requirements-lock.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"locked {len(names) - len(missing)}/{len(names)} packages")
    if missing:
        print("not installed (left unpinned):", missing)


if __name__ == "__main__":
    sys.exit(main())