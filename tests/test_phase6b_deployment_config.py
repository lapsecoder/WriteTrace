"""Phase 6B — Vercel deployment configuration tests.

These run entirely offline. They do NOT deploy, do NOT contact Vercel, and do
NOT require an account or payment method. They assert the configuration is
internally consistent and matches the validated Phase 6A measurements:

  * vercel.json is valid JSON with the expected single-origin shape
  * the FastAPI entrypoint re-exports backend.app:app (one authoritative app)
  * requirements.txt excludes torch/transformers (bundle-limit critical)
  * requirements.txt pins the spaCy model from its release URL (not PyPI)
  * segments-v1 / baseline-v1 / datasets are excluded from the deployment
  * no wildcard CORS is configured anywhere
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# vercel.json
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vercel() -> dict:
    return json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))


def test_vercel_json_is_valid_json(vercel):
    assert isinstance(vercel, dict)


def test_single_origin_fastapi_preset(vercel):
    """Static site + Python function on ONE project/origin."""
    assert vercel["framework"] == "fastapi"
    # The browser calls the same-origin /api path, so no CORS is needed.
    assert "rewrites" not in vercel, (
        "No rewrite/catch-all should be needed: the FastAPI preset serves the "
        "app directly and the frontend calls the same-origin /api path."
    )
    assert "routes" not in vercel


def test_output_directory_is_absent(vercel):
    """Phase 6D.1: `outputDirectory` was removed.

    It previously had to be "frontend/dist". Combining `outputDirectory` with
    the FastAPI preset is undocumented (Phase 6D.0), and Vercel's FastAPI guide
    states you should not set an output directory for FastAPI. The built site is
    now served by FastAPI via `app.frontend()` in the root `app.py`.
    See tests/test_phase6d1_frontend_routing.py.
    """
    assert "outputDirectory" not in vercel


def test_build_command_builds_the_frontend(vercel):
    cmd = vercel["buildCommand"]
    assert "frontend" in cmd and "build" in cmd


def test_function_duration_and_excludes(vercel):
    fn = vercel["functions"]["app.py"]
    assert 1 <= fn["maxDuration"] <= 300, "Hobby max duration is 300s"
    excluded = fn["excludeFiles"]

    # Research-only models are trimmed from the function bundle...
    for must_exclude in ("ml/models/segments-v1/**", "ml/models/baseline-v1/**"):
        assert must_exclude in excluded, f"{must_exclude} must not ship"

    # ...and datasets never reach the deployment at all (Phase 6D.2 moved these
    # from excludeFiles to .vercelignore: repeating them pushed the pattern past
    # Vercel's documented 256-character limit).
    vercelignore = (ROOT / ".vercelignore").read_text(encoding="utf-8")
    for dataset in ("ml/data/raw/", "ml/data/processed/",
                    "ml/data/manifests/"):
        assert dataset in vercelignore, f"{dataset} must not ship"

    # Nothing in the runtime import closure may be excluded.
    assert "ml/experiments/**" not in excluded, (
        "ml/experiments contains modules the app imports"
    )


def test_security_headers_present(vercel):
    sources = {h["source"]: h for h in vercel["headers"]}
    assert "/(.*)" in sources
    keys = {x["key"] for x in sources["/(.*)"]["headers"]}
    assert {"X-Content-Type-Options", "X-Frame-Options"} <= keys
    assert "/_astro/(.*)" in sources, "hashed Astro assets should be immutable"


# ---------------------------------------------------------------------------
# app.py entrypoint
# ---------------------------------------------------------------------------

def test_app_py_reexports_the_authoritative_app():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = False
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "backend.app":
            imported = any(a.name == "app" for a in node.names)
    assert imported, "app.py must re-export backend.app:app"
    # No duplicate FastAPI() construction: one authoritative application only.
    assert "FastAPI(" not in src, (
        "app.py must not build a second FastAPI instance; backend/app.py is "
        "the single authoritative application."
    )


def test_app_py_does_not_register_api_routes():
    """Routing must stay in backend/app.py; the entrypoint only re-exports."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert not re.search(r"@(app|router)\.(get|post|put|delete)", src)


# ---------------------------------------------------------------------------
# requirements.txt
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def requirements_text() -> str:
    return (ROOT / "requirements.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def requirement_names(requirements_text) -> set[str]:
    names = set()
    for line in requirements_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(re.split(r"[=<>\[\s;]", line, maxsplit=1)[0].lower())
    return names


def test_requirements_excludes_heavy_optional_deps(requirement_names):
    """torch/transformers would alone breach the 500 MB Python bundle limit."""
    for heavy in ("torch", "transformers", "pyarrow"):
        assert heavy not in requirement_names, (
            f"{heavy} must not be a deployment dependency (bundle size)"
        )


def test_requirements_excludes_dev_only_deps(requirement_names):
    for dev in ("pytest", "httpx"):
        assert dev not in requirement_names, f"{dev} is dev-only"


def test_requirements_has_core_runtime(requirement_names):
    for needed in ("fastapi", "pydantic", "numpy", "pandas",
                   "scikit-learn", "scipy", "joblib", "spacy"):
        assert needed in requirement_names, f"{needed} is required at runtime"


def test_spacy_model_installed_from_release_url(requirements_text):
    """en_core_web_sm is not on PyPI; it must come from its GitHub release."""
    assert "en_core_web_sm-3.8.0-py3-none-any.whl" in requirements_text
    assert "github.com/explosion/spacy-models" in requirements_text


def test_requirements_are_version_pinned(requirements_text):
    """Unpinned deps make a non-reproducible deployment."""
    for line in requirements_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("http"):
            continue
        assert "==" in line, f"requirement must be pinned exactly: {line}"


# ---------------------------------------------------------------------------
# packaging invariants
# ---------------------------------------------------------------------------

def test_vercelignore_blocks_datasets_and_research_models():
    text = (ROOT / ".vercelignore").read_text(encoding="utf-8")
    for entry in ("ml/data/raw/", "ml/data/processed/",
                  "ml/models/segments-v1/", "ml/models/baseline-v1/"):
        assert entry in text, f"{entry} must be ignored"


def test_final_model_is_not_ignored():
    """The deployed model MUST be uploaded: final-v1 ships, the others do not."""
    text = (ROOT / ".vercelignore").read_text(encoding="utf-8")
    assert "ml/models/final-v1/" not in text
    assert not (ROOT / "ml" / "models" / "final-v1").exists() is False


def test_no_wildcard_cors_emitted():
    """Production is same-origin; a wildcard would be a security regression.

    "*" is allowed to EXIST in backend/app.py, but only inside the allowlist
    parser, where it triggers the fail-closed refusal. It must never reach an
    Access-Control-Allow-Origin header value.
    """
    path = ROOT / "backend" / "app.py"
    src = path.read_text(encoding="utf-8")

    # 1. the wildcard refusal is present
    assert '"*" in origins' in src and "return []" in src, (
        "backend/app.py must refuse a wildcard and fail closed"
    )

    # 2. the ACAO header is only ever set from the incoming request origin,
    #    never from a literal or from the configured list.
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any("access-control-allow-origin" in t.lower() for t in targets):
                assert "origin" in ast.unparse(node.value), (
                    "Access-Control-Allow-Origin must echo the request origin"
                )

    # 3. credentials are never enabled
    assert "allow_credentials=True" not in src


def test_deployment_files_exist():
    for rel in ("vercel.json", "requirements.txt", ".vercelignore", "app.py"):
        assert (ROOT / rel).is_file(), f"missing deployment file: {rel}"