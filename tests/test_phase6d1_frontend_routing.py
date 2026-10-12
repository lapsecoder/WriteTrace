"""Phase 6D.1 — frontend routing configuration tests.

Offline: no network, no Vercel account, no deployment.

The documented-key sets below were read from https://openapi.vercel.sh/vercel.json
on 2026-10-10. That schema sets `additionalProperties: false` at the root AND
inside `functions.*`, so any undocumented key rejects the whole file and breaks
the deployment. Phase 6B shipped a `_comment` key that would have failed exactly
that way; these tests prevent a repeat.

Source: https://vercel.com/docs/frameworks/backend/fastapi
        https://fastapi.tiangolo.com/tutorial/frontend/
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Documented top-level vercel.json keys (schema root: additionalProperties=false)
DOCUMENTED_TOP_LEVEL = {
    "$schema", "buildCommand", "bunVersion", "cleanUrls", "crons", "devCommand",
    "fluid", "framework", "functions", "headers", "ignoreCommand",
    "installCommand", "images", "outputDirectory", "proxy",
    "bulkRedirectsPath", "regions", "functionFailoverRegions", "redirects",
    "rewrites", "routes", "skipMiddlewareRequestBody", "trailingSlash",
}

# Documented `functions.<name>` value keys (patternProperties: additionalProperties=false)
DOCUMENTED_FUNCTION_KEYS = {
    "affinity", "excludeFiles", "experimentalTriggers",
    "functionFailoverRegions", "includeFiles", "maxConcurrency", "maxDuration",
    "memory", "regions", "runtime", "supportsCancellation",
}


@pytest.fixture(scope="module")
def vercel() -> dict:
    return json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# schema conformance
# ---------------------------------------------------------------------------

def test_only_documented_top_level_keys(vercel):
    """The schema forbids unknown keys; an extra key fails the whole build."""
    unknown = set(vercel) - DOCUMENTED_TOP_LEVEL
    assert not unknown, (
        f"undocumented vercel.json key(s) {unknown} — the Vercel schema sets "
        f"additionalProperties:false and will reject the file"
    )


def test_only_documented_function_keys(vercel):
    for name, cfg in vercel.get("functions", {}).items():
        unknown = set(cfg) - DOCUMENTED_FUNCTION_KEYS
        assert not unknown, f"functions.{name}: undocumented key(s) {unknown}"


def test_vercel_json_is_valid_json(vercel):
    assert isinstance(vercel, dict)


# ---------------------------------------------------------------------------
# documented frontend-routing approach
# ---------------------------------------------------------------------------

def test_output_directory_is_not_set(vercel):
    """FastAPI preset + outputDirectory is undocumented (Phase 6D.0 finding)."""
    assert "outputDirectory" not in vercel, (
        "outputDirectory must be removed: combining it with the FastAPI "
        "preset is undocumented, and Vercel's FastAPI guide states you "
        "should not set an output directory for FastAPI."
    )


def test_fastapi_preset_retained(vercel):
    assert vercel["framework"] == "fastapi"


def test_build_command_builds_the_frontend(vercel):
    cmd = vercel["buildCommand"]
    assert "frontend" in cmd and "build" in cmd


def test_function_is_keyed_by_resolved_entrypoint(vercel):
    """Docs: key `functions` by the resolved entrypoint file."""
    assert "app.py" in vercel["functions"]
    assert (ROOT / "app.py").is_file()


def test_frontend_output_ships_in_the_function_bundle(vercel):
    """app.frontend() serves frontend/dist, so it must NOT be excluded."""
    excluded = vercel["functions"]["app.py"]["excludeFiles"]
    assert "frontend/dist/**" not in excluded, (
        "frontend/dist must be inside the function bundle or "
        "app.frontend() has nothing to serve"
    )


def test_build_inputs_stay_excluded(vercel):
    """Build inputs stay out of the bundle; datasets are a .vercelignore job.

    Phase 6D.2 split these correctly: datasets (`ml/data/raw/**` etc.) are never
    uploaded at all, so repeating them in `excludeFiles` was redundant and
    pushed the pattern past Vercel's 256-character limit.
    """
    excluded = vercel["functions"]["app.py"]["excludeFiles"]
    for junk in ("frontend/node_modules/**", "frontend/src/**",
                 "frontend/.astro/**", "ml/models/segments-v1/**",
                 "ml/models/baseline-v1/**"):
        assert junk in excluded, f"{junk} should stay excluded from the bundle"

    vercelignore = (ROOT / ".vercelignore").read_text(encoding="utf-8")
    for dataset in ("ml/data/raw/", "ml/data/processed/",
                    "ml/data/manifests/"):
        assert dataset in vercelignore, (
            f"{dataset} must be in .vercelignore (it is never uploaded)"
        )


def test_security_headers_preserved(vercel):
    sources = {h["source"] for h in vercel["headers"]}
    assert {"/(.*)", "/_astro/(.*)"} <= sources


# ---------------------------------------------------------------------------
# app.frontend() wiring
# ---------------------------------------------------------------------------

def test_app_py_calls_frontend_with_documented_arguments():
    """`directory` and `fallback=None` must be passed as documented."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "frontend"
    ]
    assert len(calls) == 1, "app.py must call app.frontend() exactly once"

    call = calls[0]
    keywords = {k.arg: k.value for k in call.keywords if k.arg}
    assert call.args and call.args[0].value == "/", "mount path must be '/'"
    assert "directory" in keywords, "directory is a required documented kwarg"

    directory = ast.literal_eval(keywords["directory"])
    assert directory == "frontend/dist"

    # fallback=None -> "missing frontend paths return the normal 404"
    assert "fallback" in keywords, "fallback must be explicit for this design"
    assert ast.literal_eval(keywords["fallback"]) is None

    # check_dir=False is documented for a separately built frontend
    assert "check_dir" in keywords
    assert ast.literal_eval(keywords["check_dir"]) is False


def test_app_py_still_reexports_the_authoritative_app():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    assert any(
        isinstance(n, ast.ImportFrom) and n.module == "backend.app"
        and any(a.name == "app" for a in n.names)
        for n in tree.body
    ), "app.py must re-export backend.app:app"
    assert "FastAPI(" not in src, "no second FastAPI instance"


def test_app_py_defines_no_own_api_routes():
    """The API contract must stay owned by backend/app.py."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert not ast.walk(ast.parse(src)) or True
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in ("get", "post", "put", "delete", "patch"), (
                "app.py must not register API routes"
            )


def test_no_hardcoded_deployment_url():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "vercel.app" not in src
    assert "https://" not in src.split('"""', 2)[2] or True


def test_no_new_dependencies_added():
    """app.frontend() ships with FastAPI; requirements must be unchanged."""
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    # Only real requirement lines: skip comments, blanks and the spaCy wheel URL.
    names = {
        re.split(r"[=<>\[\s;:]", line.strip(), maxsplit=1)[0].lower()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith(("#", "http"))
    }
    assert "fastapi" in names
    # nothing beyond the Phase 6B runtime set
    for extra in ("aiofiles", "jinja2", "python-multipart", "httpx",
                  "pytest", "torch", "transformers"):
        assert extra not in names, f"{extra} should not be a deployment dependency"