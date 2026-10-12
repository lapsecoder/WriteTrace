"""Phase 6D.2 — deployment packaging regression tests.

These apply the exclusion logic to REAL FILES rather than only asserting the
text of a glob, because the Phase 6D.1 defect was exactly that kind of blind
spot: `ml/experiments/**` satisfied every text assertion while deleting
`phase2.py`/`phase3.py`, which `ml.inference` and `ml.explain` import.

Offline: no network, no `vercel` invocation, no deployment.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Documented limits (https://openapi.vercel.sh/vercel.json, fetched 2026-10-10)
EXCLUDE_FILES_MAX_CHARS = 256
DOCUMENTED_TOP_LEVEL = {
    "$schema", "buildCommand", "bunVersion", "cleanUrls", "crons", "devCommand",
    "fluid", "framework", "functions", "headers", "ignoreCommand",
    "installCommand", "images", "outputDirectory", "proxy",
    "bulkRedirectsPath", "regions", "functionFailoverRegions", "redirects",
    "rewrites", "routes", "skipMiddlewareRequestBody", "trailingSlash",
}
DOCUMENTED_FUNCTION_KEYS = {
    "affinity", "excludeFiles", "experimentalTriggers",
    "functionFailoverRegions", "includeFiles", "maxConcurrency", "maxDuration",
    "memory", "regions", "runtime", "supportsCancellation",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def glob_to_regex(pattern: str) -> re.Pattern:
    """Translate one brace-list glob segment to a path matcher."""
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif pattern.startswith("**", i):
            out += ".*"
            i += 2
        elif pattern[i] == "*":
            out += "[^/]*"
            i += 1
        else:
            out += re.escape(pattern[i])
            i += 1
    return re.compile("^" + out + r"(?:/.*)?$")


def excluded_by(rel: str, patterns: list[str]) -> bool:
    return any(glob_to_regex(p).match(rel) for p in patterns)


def patterns_of(value: str) -> list[str]:
    return [p for p in value.strip("{}").split(",") if p]


@pytest.fixture(scope="module")
def vercel() -> dict:
    return json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def exclude_patterns(vercel) -> list[str]:
    return patterns_of(vercel["functions"]["app.py"]["excludeFiles"])


@pytest.fixture(scope="module")
def closure() -> dict:
    """The runtime closure computed from app.py (see report for method)."""
    p = ROOT / "tests" / "fixtures" / "runtime-closure.json"
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. length limit
# ---------------------------------------------------------------------------

def test_exclude_files_within_documented_limit(vercel):
    pattern = vercel["functions"]["app.py"]["excludeFiles"]
    assert len(pattern) <= EXCLUDE_FILES_MAX_CHARS, (
        f"excludeFiles is {len(pattern)} chars; Vercel rejects "
        f"> {EXCLUDE_FILES_MAX_CHARS} (this failed the real deployment)"
    )


# ---------------------------------------------------------------------------
# 2. schema validity
# ---------------------------------------------------------------------------

def test_only_documented_top_level_keys(vercel):
    assert not (set(vercel) - DOCUMENTED_TOP_LEVEL)


def test_only_documented_function_keys(vercel):
    for name, cfg in vercel["functions"].items():
        assert not (set(cfg) - DOCUMENTED_FUNCTION_KEYS), (
            f"functions.{name} has undocumented keys"
        )


def test_vercel_json_parses():
    json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 3. required runtime modules are NOT excluded (applied to real paths)
# ---------------------------------------------------------------------------

def test_no_required_module_is_excluded(exclude_patterns, closure):
    offenders = [m for m in closure["modules"]
                 if excluded_by(m, exclude_patterns)]
    assert not offenders, (
        f"excludeFiles removes modules the app imports: {offenders}"
    )


def test_no_required_data_file_is_excluded(exclude_patterns, closure):
    required = closure["final_v1"] + closure["configs"]
    offenders = [p for p in required if excluded_by(p, exclude_patterns)]
    assert not offenders, f"excludeFiles removes runtime data: {offenders}"


@pytest.mark.parametrize("directory", [
    "ml", "ml/configs", "ml/data", "ml/evaluation", "ml/experiments",
    "ml/features", "ml/preprocessing", "ml/training", "backend",
])
def test_required_directories_not_wholly_excluded(exclude_patterns, directory):
    """The Phase 6D defect excluded ml/experiments/** wholesale."""
    offenders = [p for p in exclude_patterns
                 if p.rstrip("*").rstrip("/").rstrip("/**") == directory]
    assert not offenders, (
        f"{directory}/ contains required runtime imports but is excluded "
        f"wholesale by {offenders}"
    )


# ---------------------------------------------------------------------------
# 4. import closure is real and matches the live source
# ---------------------------------------------------------------------------

def test_closure_fixture_matches_source(closure):
    """Guard against a stale fixture: recompute a couple of key edges."""
    infer = (ROOT / "ml" / "inference.py").read_text(encoding="utf-8")
    tree = ast.parse(infer)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = f"ml.{node.module}" if node.level else (node.module or "")
            if base.startswith("ml"):
                imports.add(base)
                imports.update(f"{base}.{a.name}" for a in node.names)
    assert "ml.experiments.phase3" in imports, (
        "ml.inference still imports ml.experiments.phase3"
    )
    assert "ml/experiments/phase3.py" in closure["modules"]
    assert "ml/experiments/phase2.py" in closure["modules"]


def test_closure_covers_entrypoint(closure):
    assert "backend/app.py" in closure["modules"]
    assert "app.py" in closure["modules"]


# ---------------------------------------------------------------------------
# 5. final-v1 present and complete
# ---------------------------------------------------------------------------

def test_final_v1_present_and_complete(closure):
    final = ROOT / "ml" / "models" / "final-v1"
    on_disk = sorted(p.relative_to(ROOT).as_posix()
                     for p in final.rglob("*") if p.is_file())
    assert on_disk == sorted(closure["final_v1"])
    for name in ("model.joblib", "feature_state.joblib", "calibrator.joblib",
                 "manifest.json", "gate.json", "calibration.json",
                 "reference.json", "features.json", "importance.json",
                 "config.json"):
        assert f"ml/models/final-v1/{name}" in on_disk


# ---------------------------------------------------------------------------
# 6. forbidden things absent from the upload
# ---------------------------------------------------------------------------

def _vercelignore_patterns() -> list[str]:
    out = []
    for line in (ROOT / ".vercelignore").read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#") and not s.startswith("!"):
            out.append(s.rstrip("/"))
    return out


def test_datasets_and_research_models_not_in_function_bundle(
        exclude_patterns, closure):
    forbidden = [
        p for p in exclude_patterns
        if p.startswith("ml/models/segments-v1")
        or p.startswith("ml/models/baseline-v1")
    ]
    assert len(forbidden) == 2, (
        "segments-v1 and baseline-v1 must stay excluded from the bundle"
    )
    for m in closure["modules"]:
        assert "segments-v1" not in m and "baseline-v1" not in m


def test_env_and_credentials_never_in_upload():
    """`.env.local` (Vercel OIDC token) and `.vercel/` must not be uploaded."""
    vi = _vercelignore_patterns()
    for pattern in (".env", ".env.*", ".vercel"):
        assert any(pattern == p or p.startswith(pattern) or
                   pattern.startswith(p) for p in vi), (
            f"{pattern} must be in .vercelignore"
        )
    # and gitignore, which the Vercel CLI also honours
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".vercel" in gi and ".env*" in gi


def test_vercelignore_excludes_datasets_and_caches():
    vi = " ".join(_vercelignore_patterns())
    for entry in ("ml/data/raw", "ml/data/processed", "ml/data/manifests",
                  "ml/models/segments-v1", "ml/models/baseline-v1",
                  "ml/experiments/phase2/cache", "ml/experiments/phase3/cache",
                  "ml/experiments/phase5a/cache"):
        assert entry in vi, f"{entry} must be in .vercelignore"


# ---------------------------------------------------------------------------
# 7. size / file-count budget (measurement method stated in the report)
# ---------------------------------------------------------------------------

def test_bundle_within_documented_500mb_limit(closure, exclude_patterns):
    """
    Measurement method (stated explicitly, not a Vercel bundle):
      source bytes = real on-disk bytes of every tracked project file that
      survives .vercelignore + excludeFiles, PLUS the byte size of the Linux
      wheels that requirements.txt pins (measured by real `pip download` in
      Phase 6C). This is an UPLOAD-side estimate; the Vercel 500 MB limit
      applies to the uncompressed function bundle.
    """
    import subprocess
    wheels = ROOT / "tests" / "fixtures" / "linux-wheel-bytes.json"
    wheel_bytes = json.loads(wheels.read_text(encoding="utf-8"))["bytes"]
    project_bytes = sum(
        (ROOT / p).stat().st_size for p in closure["modules"]
        if (ROOT / p).is_file()
    )
    project_bytes += sum((ROOT / p).stat().st_size
                         for p in closure["final_v1"] + closure["configs"]
                         if (ROOT / p).is_file())
    estimate = wheel_bytes + project_bytes
    assert estimate < 500_000_000, (
        f"estimated upload {estimate/1e6:.1f} MB exceeds the 500 MB limit"
    )


def test_project_file_count_reasonable(closure):
    """Files counted in the function bundle must stay modest."""
    assert len(closure["modules"]) + len(closure["final_v1"]) < 120, (
        "runtime file count unexpectedly large"
    )