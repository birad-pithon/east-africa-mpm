"""Regenerate ``requirements-lock.txt`` with pinned installed versions.

Names come from three places, so that a pin can never silently disappear from
the lock file (a dropped pin means CI installs an unpinned, drifting version):

1. ``requirements.txt`` - the core install set,
2. the extras CI installs (``dev``, ``bayes``) - read from ``pyproject.toml``,
3. whatever the previous lock file already pinned.

A name that is not installed locally keeps its previous pin when it has one and
is reported; only genuinely new, unpinned names are skipped - and those are
written into the lock file as a comment, so ``tests/test_dependency_consistency.py``
can tell "unpinned on purpose" from "a pin was dropped by accident".

Usage::

    python scripts/pin_requirements.py
"""
from __future__ import annotations

import os
import re
import sys
from importlib import metadata
from pathlib import Path

try:                                   # tomllib is 3.11+; project floor is 3.10
    import tomllib
except ModuleNotFoundError:            # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None                 # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
CI_EXTRAS = ("dev", "bayes")           # what .github/workflows/ci.yml installs


def _requirement_names(text: str) -> list[str]:
    """Distribution names required by a requirements-style blob."""
    names = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        m = re.match(r"([A-Za-z0-9_.-]+)\s*[=<>!~]", line)
        if m:
            names.append(m.group(1))
    return names


def _extra_names() -> list[str]:
    if tomllib is None:
        print("warning: no TOML parser available; extras names not scanned")
        return []
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"].get("optional-dependencies", {})
    return _requirement_names("\n".join(
        n for extra in CI_EXTRAS for n in extras.get(extra, [])
    ))


def main() -> None:
    names = _requirement_names((ROOT / "requirements.txt").read_text(encoding="utf-8"))
    names += _extra_names()

    lock_path = ROOT / "requirements-lock.txt"
    previous: dict[str, str] = {}
    if lock_path.exists():
        for line in lock_path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"([A-Za-z0-9_.-]+)==(\S+)", line.strip())
            if m:
                previous[m.group(1)] = m.group(2)
    names += list(previous)            # never drop an existing pin

    lines = [
        "# Auto-generated lock file - pinned versions of requirements.txt",
        "# Regenerate with: python scripts/pin_requirements.py",
        "#",
        "# landsatxplore is deliberately absent: it requires shapely<2, which",
        "# cannot coexist with geopandas 1.x (shapely>=2.0) and made the whole",
        "# install unsatisfiable. It lives in the [landsat] extra instead.",
        "",
    ]
    kept, kept_previous, missing = 0, [], []
    seen: set[str] = set()
    for name in names:
        key = name.lower().replace("_", "-")
        if key in seen:
            continue
        seen.add(key)
        try:
            lines.append(f"{name}=={metadata.version(name)}")
            kept += 1
        except metadata.PackageNotFoundError:
            if name in previous:
                lines.append(f"{name}=={previous[name]}")
                kept_previous.append(name)
            else:
                missing.append(name)

    if missing:
        # Record them in the file itself: test_dependency_consistency reads this
        # block to tell "unpinned on purpose" from "pin dropped by accident".
        lines += [
            "",
            "# Not pinned: absent from the environment that regenerates this file,",
            "# so CI installs them (and their transitive deps) unpinned from PyPI:",
        ]
        lines += [f"#   {name}" for name in missing]

    tmp = lock_path.with_name("requirements-lock.txt.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, lock_path)         # never leave a half-written lock behind

    print(f"locked {kept} packages from this environment")
    if kept_previous:
        print("kept previous pin (not installed here):", ", ".join(kept_previous))
    if missing:
        print("NOT PINNED - install them or pin by hand:", ", ".join(missing))


if __name__ == "__main__":
    sys.exit(main())
