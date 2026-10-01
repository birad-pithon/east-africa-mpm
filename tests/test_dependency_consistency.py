"""The pinned dependency set must be internally satisfiable.

CI stopped at ``pip install -r requirements-lock.txt -r requirements.txt`` on
*every* run since the first commit: ``landsatxplore`` publishes
``shapely>=1.7,<2.0`` while ``geopandas`` publishes ``shapely>=2.0.0``, so the
set had no solution and the resolver aborted before the linter or the tests ever
ran. Nothing in the repository imported landsatxplore.

These checks reproduce that failure offline, in seconds, from the requirement
metadata already installed - no network, no resolver run.
"""
from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path

import tomllib
from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def _pins(text: str) -> dict[str, str]:
    """``{name: version}`` from a requirements-style file."""
    pins = {}
    for line in text.splitlines():
        if (m := re.match(r"([A-Za-z0-9_.-]+)==(\S+)", line.strip())):
            pins[m.group(1)] = m.group(2)
    return pins


def _declared_requires(name: str) -> list[Requirement] | None:
    """Requirements an *installed* distribution declares, markers applied."""
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None
    reqs = []
    for raw in dist.requires or []:
        req = Requirement(raw)
        if req.marker and not req.marker.evaluate():
            continue
        reqs.append(req)
    return reqs


def _violations(pins: dict[str, str],
                requires_of: dict[str, list[Requirement] | None]) -> list[str]:
    """Pins that contradict a requirement declared by another pin.

    Only packages that are themselves part of the pinned set are asked what
    they need - an unpinned package (e.g. landsatxplore now that it lives in an
    extra) cannot veto anything.
    """
    bad = []
    for name in pins:
        reqs = requires_of.get(name)
        if reqs is None:
            continue
        for req in reqs:
            dep = req.name
            if dep not in pins or not req.specifier:
                continue                      # not pinned here / no constraint
            if not req.specifier.contains(Version(pins[dep])):
                bad.append(
                    f"{name} requires {req}, but the lock pins "
                    f"{dep}=={pins[dep]}"
                )
    return bad


LOCK = _pins((ROOT / "requirements-lock.txt").read_text(encoding="utf-8"))
REQUIREMENTS = (ROOT / "requirements.txt").read_text(encoding="utf-8")
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
CORE_DEPS = [Requirement(r).name for r in PYPROJECT["project"]["dependencies"]]
EXTRA_DEPS = {
    extra: [Requirement(r).name for r in reqs]
    for extra, reqs in PYPROJECT["project"].get("optional-dependencies", {}).items()
}


def test_lock_file_pins_the_whole_core_set():
    assert len(LOCK) > 20, "requirements-lock.txt looks truncated"
    missing = [d for d in CORE_DEPS if d not in LOCK]
    assert not missing, f"core dependencies missing from the lock: {missing}"


def test_pinned_versions_are_mutually_satisfiable():
    """Every pinned package must accept the version pinned for its own deps."""
    requires_of = {name: _declared_requires(name) for name in LOCK}
    uninstalled = [n for n, r in requires_of.items() if r is None]
    assert len(uninstalled) < len(LOCK) / 2, (
        f"too many pinned packages are not installed to judge ({uninstalled})"
    )
    assert not _violations(LOCK, requires_of), "\n".join(
        _violations(LOCK, requires_of)
    )


def test_detector_catches_the_landsatxplore_conflict():
    """Regression guard for the bug that broke every CI run."""
    # Published metadata of the two packages, hard-coded so this test does not
    # depend on what happens to be installed.
    requires_of = {
        "landsatxplore": [Requirement("shapely (>=1.7,<2.0)")],
        "geopandas": [Requirement("shapely>=2.0.0")],
        "shapely": [],
    }
    broken = {"shapely": "2.1.2", "geopandas": "1.1.4", "landsatxplore": "0.15.0"}
    found = _violations(broken, requires_of)
    assert len(found) == 1, found
    assert "landsatxplore" in found[0] and "shapely" in found[0]

    fixed = {"shapely": "2.1.2", "geopandas": "1.1.4"}
    assert not _violations(fixed, requires_of)


def test_landsatxplore_stays_out_of_the_installable_set():
    """It can only ever be a separate-venv extra (it needs shapely<2)."""
    for name in ("requirements.txt", "requirements-lock.txt"):
        text = (ROOT / name).read_text(encoding="utf-8")
        live = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
        assert not any("landsatxplore" in ln for ln in live), f"{name} pins it"
    core_blob = "\n".join(PYPROJECT["project"]["dependencies"])
    assert "landsatxplore" not in core_blob
    assert EXTRA_DEPS.get("landsat") == ["landsatxplore"]


def test_requirements_files_and_pyproject_do_not_drift():
    """No duplicate bounds, and neither side lists a package the other lacks."""
    req_names = {
        Requirement(re.sub(r"^\s*", "", ln).split("#")[0]).name
        for ln in REQUIREMENTS.splitlines()
        if re.match(r"\s*[A-Za-z0-9_.-]+\s*[=<>!~]", ln)
    }
    for dep in CORE_DEPS:
        assert dep in req_names, f"{dep} is a pyproject core dep but not in requirements.txt"
    known = set(req_names) | set(CORE_DEPS) | {
        n for names in EXTRA_DEPS.values() for n in names
    }
    orphan = req_names - known
    assert not orphan, f"requirements.txt lists packages pyproject never mentions: {orphan}"



# --- CI workflow vs pyproject.toml vs the lock file ----------------------
# CI installs `pip install -e ".[dev,bayes]"`. That step, the extras declared in
# pyproject.toml and the pins in requirements-lock.txt have to agree: an extra
# that CI installs but the lock never pinned leaves CI on whatever PyPI serves
# that day, which is how the Jupyter stack drifted while CI still looked green.
CI_YAML = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
LOCK_TEXT = (ROOT / "requirements-lock.txt").read_text(encoding="utf-8")


def _norm(name: str) -> str:
    """Same name normalisation scripts/pin_requirements.py applies."""
    return name.lower().replace("_", "-")


def _ci_extras() -> list[str]:
    match = re.search(r'-e\s+"\.\[([\w,\s-]+)\]"', CI_YAML)
    assert match, 'ci.yml no longer has a `pip install -e ".[extras]"` step'
    return [n.strip() for n in match.group(1).split(",") if n.strip()]


# Names the generator could not pin, read straight out of the comment block it
# writes for them, so this guardrail cannot drift away from that script.
UNPINNED_BY_GENERATOR = {_norm(n) for n in
                         re.findall(r"^#\s{2,}([A-Za-z][\w.-]*)$", LOCK_TEXT,
                                    re.MULTILINE)}
LOCK_NORMALISED = {_norm(name) for name in LOCK}


def test_ci_installs_only_declared_extras():
    """No typo'd or deleted extra may slip into the CI install line."""
    installed = _ci_extras()
    assert "dev" in installed, "ci.yml must install the dev extra: pytest lives there"
    undeclared = set(installed) - set(EXTRA_DEPS)
    assert not undeclared, f"ci.yml installs extras pyproject never declares: {sorted(undeclared)}"
    # landsatxplore requires shapely<2 and would make CI unsatisfiable again.
    assert "landsat" not in installed, "the landsat extra must stay out of CI"


def test_ci_extra_requirements_are_pinned_or_documented():
    """Everything CI installs through an extra is pinned, or says why not."""
    drifting = [
        f"{extra}: {dep}"
        for extra in _ci_extras()
        for dep in EXTRA_DEPS.get(extra, [])
        if _norm(dep) not in LOCK_NORMALISED
        and _norm(dep) not in UNPINNED_BY_GENERATOR
    ]
    assert not drifting, (
        "installed by CI but neither pinned nor documented as unpinned in "
        f"requirements-lock.txt: {drifting}"
    )

    dupes: dict[str, list[str]] = {}
    for raw in PYPROJECT["project"]["dependencies"]:
        dupes.setdefault(Requirement(raw).name, []).append(raw)
    clash = {k: v for k, v in dupes.items() if len(v) > 1}
    assert not clash, f"pyproject lists a dependency more than once: {clash}"
