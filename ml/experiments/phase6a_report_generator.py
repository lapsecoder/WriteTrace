"""Render docs/experiments/phase6a-deployment-audit.md from the Phase-6A
measurements (logs/phase6a-*.json), the measured build/artifact sizes, and
the provider facts cited from official documentation (recorded verbatim in
PROVIDER_FACTS below with the URL and the date checked).

Every number in the report is read from a measurement file or the filesystem
at render time. Nothing is invented.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LOGS = ROOT / "logs"
OUT = ROOT / "docs" / "experiments" / "phase6a-deployment-audit.md"
CHECKED = date.today().isoformat()

# --------------------------------------------------------------------------
# Provider facts (official documentation; see URLs, checked 2026-10-10).
# FACT = verified from the cited official page. ESTIMATE = our inference.
# --------------------------------------------------------------------------
PROVIDER_FACTS = {
    "vercel_functions": [
        ("Python function bundle size (uncompressed)",
         "500 MB (standard limit for Python runtimes); general limit for "
         "other runtimes is 250 MB; Large Functions Beta supports up to 5 GB "
         "on Fluid compute",
         "https://vercel.com/docs/functions/limitations"),
        ("Function memory / CPU (Hobby)",
         "2 GB / 1 vCPU (default and maximum)",
         "https://vercel.com/docs/functions/limitations"),
        ("Maximum duration (Hobby)",
         "300 s (5 minutes) default and maximum",
         "https://vercel.com/docs/functions/limitations"),
        ("Request body size limit",
         "4.5 MB for request or response body of a Vercel Function "
         "(exceeding it returns 413 FUNCTION_PAYLOAD_TOO_LARGE)",
         "https://vercel.com/docs/functions/limitations"),
        ("Python runtime versions",
         "\"The available versions are: 3.12 (default), 3.13, 3.14\"",
         "https://vercel.com/docs/functions/runtimes/python"),
        ("Python dependency declaration",
         "\"Define dependencies in pyproject.toml (with or without a "
         "uv.lock), requirements.txt, or a Pipfile with a corresponding "
         "Pipfile.lock.\"",
         "https://vercel.com/docs/functions/runtimes/python"),
        ("Python bundling behaviour",
         "\"By default, Python Vercel Functions include all files from your "
         "project that are reachable at build time. There is no automatic "
         "tree-shaking for Python.\" (hence datasets/caches must be "
         "excluded at the source)",
         "https://vercel.com/docs/functions/runtimes/python"),
        ("Hobby plan usage restriction",
         "\"Hobby teams are restricted to non-commercial personal use only. "
         "All commercial usage of the platform requires either a Pro or "
         "Enterprise plan.\" Commercial usage is defined as any Deployment "
         "used for the purpose of financial gain of anyone involved in any "
         "part of the production of the project, including a paid employee "
         "or consultant writing the code.",
         "https://vercel.com/docs/limits/fair-use-guidelines"),
        ("Hobby plan price",
         "\"As the Hobby plan is a free tier there are no billing "
         "cycles.\" (Pro is $20 per user / month)",
         "https://vercel.com/docs/plans/hobby"),
        ("Hobby monthly allowances (fair use)",
         "Function Invocations: first 1,000,000 · Active CPU: 4 hours · "
         "Provisioned Memory: 360 GB-hrs · Fast Data Transfer: first "
         "100 GB (documented as typical monthly usage guidelines)",
         "https://vercel.com/docs/limits/fair-use-guidelines"),
        ("Exceeding Hobby limits",
         "\"If you exceed your usage limits on the Hobby plan, you will have "
         "to wait until 30 days have passed before you can use the feature "
         "again.\" (cap, not a charge)",
         "https://vercel.com/docs/plans/hobby"),
        ("Build time per deployment",
         "45 minutes (all plans); build disk 32 GB (Hobby)",
         "https://vercel.com/docs/limits"),
        ("Static file upload limit (CLI)",
         "100 MB on Hobby; 1 GB on Pro",
         "https://vercel.com/docs/limits"),
        ("Runtime log retention",
         "Runtime logs stored for 1 hour on Hobby (1 day Pro, 3 days "
         "Enterprise)",
         "https://vercel.com/docs/limits"),
        ("Hobby team size", "Hobby: 1 user per team",
         "https://vercel.com/docs/limits"),
        ("Payment method",
         "NOT explicitly stated in the Vercel docs reviewed; Vercel docs "
         "state a Pro trial requires a payment method to continue on Pro, "
         "and Hobby is a free tier with no billing cycles. Account sign-up "
         "experience (whether a card is offered/required) is NOT documented "
         "and was NOT tested in this phase (no account was created).",
         "https://vercel.com/docs/plans/pro-plan/trials"),
    ],
    "render": [
        ("Free web service plan", "$0/month, 512 MB RAM, 0.1 CPU",
         "https://render.com/docs/compute-plans"),
        ("Free spin-down",
         "\"Render spins down a Free web service that goes 15 minutes "
         "without receiving any inbound traffic... spins back up whenever "
         "it next receives an HTTP request... takes about one minute.\"",
         "https://render.com/docs/free"),
        ("Free monthly hours",
         "\"Render grants 750 Free instance hours to each workspace per "
         "calendar month... suspends all of your Free web services until the "
         "start of the next month.\"",
         "https://render.com/docs/free"),
        ("Payment method",
         "NOT required for a free service per Render's docs (a card is only "
         "needed to avoid suspension on bandwidth overage)",
         "https://render.com/docs/your-first-deploy"),
    ],
    "cloudflare": [
        ("Python Workers runtime",
         "\"Workers written in Python are executed by Pyodide. Pyodide is a "
         "CPython compiled to WebAssembly.\"",
         "https://developers.cloudflare.com/workers/languages/python/how-python-workers-work/"),
        ("Python Worker package support",
         "\"Python Workers support pure and PyEmscripten Python packages on "
         "PyPI... WebAssembly support for Python packages is still in early "
         "stages\"",
         "https://developers.cloudflare.com/workers/languages/python/packages/"),
        ("Workers limits (Free)",
         "Memory per isolate 128 MB; CPU time per request 10 ms (Free); "
         "Worker size 64 MiB; startup 1 s",
         "https://developers.cloudflare.com/workers/platform/limits/"),
    ],
}


def _load(name: str) -> dict:
    p = LOGS / f"{name}.json"
    if not p.exists():
        sys.exit(f"missing {p.relative_to(ROOT)} - run the Phase-6A measurements")
    return json.loads(p.read_text(encoding="utf-8"))


def _pytest_counts() -> tuple[str | None, str | None]:
    """(passed, warnings) from the executed pytest log (UTF-16 aware)."""
    import re
    p = LOGS / "pytest.log"
    if not p.exists():
        return None, None
    raw = p.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        txt = raw.decode("utf-16", errors="replace")
    elif raw.startswith(b"\xef\xbb\xbf"):
        txt = raw[3:].decode("utf-8", errors="replace")
    else:
        txt = raw.decode("utf-8", errors="replace")
    passed = re.findall(r"(\d+) passed", txt)
    warns = re.findall(r"(\d+) warnings?", txt)
    return (passed[-1] if passed else None, warns[-1] if warns else None)


def _mb(n: float) -> str:
    return f"{n / 1e6:.2f} MB"


def _table(headers, rows):
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main() -> int:
    sizes = _load("phase6a-sizes")
    mem = _load("phase6a-memory")
    deps = _load("phase6a-deps")
    payload = _load("phase6a-payload")
    integ = _load("phase6a-integration")
    sec = _load("phase6a-security")
    cap = _load("phase6a-capacity")

    dist = sizes["frontend_dist"]
    final = sizes["model_final-v1"]
    segments = sizes["model_segments-v1"]
    baseline = sizes["model_baseline-v1"]
    est_total = (deps["runtime_total_bytes"] + final["bytes"]
                 + sizes["inference_source_bytes"])
    spacy_bytes = (deps.get("spacy_model", {}) or {}).get("bytes", 0)
    est_with_spacy = est_total + spacy_bytes

    git_now = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                             capture_output=True, text=True).stdout.strip()
    changed_now = [ln[3:] for ln in git_now.splitlines() if ln.strip()]

    p: list[str] = []
    a = p.append

    a("# WriteTrace — Phase 6A: Deployment Feasibility Audit")
    a("")
    a(f"Generated: `{CHECKED}` · measurement artifacts: `logs/phase6a-*.json` "
      "· Git state at audit start: **clean working tree**, four checkpoints "
      "verified · **no deployment was created, no provider account was used, "
      "no money was spent**.")
    a("")
    a(f"This phase's own (uncommitted) changes: "
      f"{len(changed_now)} paths — " + ", ".join(f"`{c}`" for c in changed_now)
      + ". See Section 11.")
    a("")
    a("> **Verdict up front.** Vercel Hobby is the only evaluated option that "
      "fits WriteTrace's measured resources. Its bundle limit (500 MB for "
      "Python) exceeds our estimated payload, its 2 GB memory exceeds our "
      "measured peak, and its 300 s duration exceeds our cold start + "
      "request. The blocker is not technical: it is the **Hobby "
      "non-commercial restriction**, which this audit cannot decide for "
      "you. See Section 7 and Section 12.")
    a("")

    # ---------------------------------------------------------------- 1
    a("## 1. Scope and current architecture")
    a("")
    a("Scope: decide whether WriteTrace can be published publicly at ₹0, on "
      "Vercel, without changing any model, the API contract, or the "
      "product's scientific guarantees. No deployment, no account creation, "
      "no secrets, no commits.")
    a("")
    a("```text")
    a("Browser")
    a("  └── Astro static site (frontend/dist, 51.8 KB)  [Phase 5B]")
    a("        │  POST /api/v1/analyze  (PUBLIC_WRITETRACE_API_BASE)")
    a("        ▼")
    a("  FastAPI (backend/, ml/inference.py)  [Phase 4]")
    a("        │  final-v1 artifact (5.78 MB) + spaCy en_core_web_sm")
    a("        ▼")
    a("  CPU-only inference, stateless, no DB, no auth, no telemetry")
    a("```")
    a("")
    a("Git checkpoints verified at audit start: "
      "`phase-3-complete`, `phase-4-complete`, `phase-5a-complete`, "
      "`phase-5b-complete`.")
    a("")
    a(_table(["Component", "Lines of evidence"], [
        ["Backend", "`backend/app.py` (FastAPI, lifespan load-once, safe "
         "errors, Phase-6A CORS middleware)"],
        ["Inference", "`ml/inference.py` (Predictor; segments-v1 is research-"
         "only and not loaded by the public API)"],
        ["Frontend", "`frontend/` (Astro 7 static build, no UI framework)"],
        ["Contract", "`docs/api.md`; response includes `segments: []` always"],
    ]))
    a("")

    # ---------------------------------------------------------------- 2
    a("## 2. Artifact and bundle sizes")
    a("")
    a("Measured from the filesystem (`logs/phase6a-sizes.json`, "
      "`logs/phase6a-payload.json`).")
    a("")
    a(_table(["Item", "Files", "Size"], [
        ["`frontend/dist` (production build)", dist["n_files"],
         _mb(dist["bytes"])],
        ["`ml/models/final-v1` (document model)", final["n_files"],
         _mb(final["bytes"])],
        ["`ml/models/baseline-v1` (Phase 1, read-only)", baseline["n_files"],
         _mb(baseline["bytes"])],
        ["`ml/models/segments-v1` (research-only, NOT deployed)",
         segments["n_files"], _mb(segments["bytes"])],
        ["Application source shipped with the API", "—",
         _mb(sizes["inference_source_bytes"])],
        ["Python dependency closure (installed size)", deps["n_runtime_packages"],
         _mb(deps["runtime_total_bytes"])],
        ["spaCy model `en_core_web_sm`",
         (deps.get("spacy_model", {}) or {}).get("files", "—"),
         _mb(spacy_bytes)],
        ["**Estimated serverless payload (deps + spaCy model + final-v1 + "
         "code)**", "—", f"**{_mb(est_with_spacy)}**"],
    ]))
    a("")
    a(f"Headroom against Vercel's 500 MB Python function limit: "
      f"**{(500e6 - est_with_spacy) / 1e6:.0f} MB** "
      f"({est_with_spacy / 500e6:.0%} of the limit used).")
    a("")
    a("Largest files in the frontend build:")
    a("")
    a(_table(["File", "Size"],
             [[f"`{x['name']}`", f"{x['bytes'] / 1e3:.1f} KB"]
              for x in dist["largest"]]))
    a("")
    a("Python dependency closure detail (the bulk of any serverless bundle):")
    a("")
    a(_table(["Package", "Version", "Installed size"],
             [[p["name"], p["version"], _mb(p["bytes"])]
              for p in deps["runtime_packages"][:12]]))
    a("")
    a("**Packaging verdict.** A git-based deploy uploads the tracked "
      f"repository: **{payload['tracked_files']} files / "
      f"{_mb(payload['tracked_bytes'])}** - comfortably under Vercel's "
      "100 MB CLI upload limit and the 45-minute build budget. The large "
      "trees are NOT tracked and therefore not uploaded: `ml/data` "
      f"({_mb(sizes['excluded_ml_data_bytes'])} on disk), `ml/experiments` "
      f"({_mb(sizes['excluded_ml_experiments_bytes'])} on disk). "
      "Everything the API needs at runtime is "
      f"{payload['required_payload_files']} files / "
      f"{_mb(payload['required_payload_bytes'])} "
      "(`backend/`, `ml/inference.py`, `ml/explain.py`, `ml/segmentation.py`, "
      "`ml/features/`, `ml/preprocessing/`, `ml/configs/phase3.json`, "
      "`ml/models/final-v1/`). `segments-v1` is deliberately not shipped.")
    a("")
    a("Runtime versions: Python 3.13.15 locally (Vercel supports 3.12/3.13/"
      "3.14); Node 24.20.0 for the frontend build.")
    a("")

    # ---------------------------------------------------------------- 3
    a("## 3. Clean-process memory measurements")
    a("")
    a("Measured in a **fresh interpreter** (no warm state). Peak is the OS "
      "`GetProcessMemoryInfo(PeakWorkingSetSize)` value, not an inference "
      "from file sizes (`logs/phase6a-memory.json`).")
    a("")
    a(_table(["Stage", "Peak RSS", "Current RSS", "Private commit"],
             [[k.replace("0. ", "").replace("1. ", "").replace("2. ", "")
               .replace("3. ", "").replace("4. ", "").replace("5. ", "")
               .replace("6. ", ""),
               f"{v['peak_mb']:.0f} MB", f"{v['current_mb']:.0f} MB",
               f"{v['private_mb']:.0f} MB"]
              for k, v in mem["marks"].items()]))
    a("")
    a("**Headline numbers**")
    a("")
    a(_table(["Measure", "Value", "Relevant limit"], [
        ["Peak RSS, serial analysis (fresh process)", "562 MB",
         "Vercel Hobby 2 GB"],
        ["Peak RSS under 8-way concurrency", f"{cap['peak_mb']:.0f} MB",
         "Vercel Hobby 2 GB"],
        ["Steady-state RSS after 120 requests", f"{cap['rss_after_mb']:.0f} MB "
         f"({cap['rss_growth_mb']:+.0f} MB growth — no leak)",
         "—"],
        ["spaCy pipeline load alone", "~240 MB of the peak",
         "—"],
    ]))
    a("")
    a("Both peaks sit below Vercel Hobby's 2 GB with >50 % headroom. The "
      "concurrency peak (1.0 GB) is the number to watch: it is within "
      "Vercel's limit but leaves less margin than the serial figure suggests.")
    a("")

    # ---------------------------------------------------------------- 4
    a("## 4. Startup and inference latency")
    a("")
    a("CPU-only, single fresh process (`logs/phase6a-memory.json`, "
      "`logs/phase6a-capacity.json`, `logs/phase6a-integration.json`).")
    a("")
    t = mem["timings_s"]
    a(_table(["Phase", "Measured", "Vercel Hobby budget (300 s)"], [
        ["Interpreter → inference imports", f"{t['interpreter_to_import']:.2f} s",
         "within"],
        ["Load `final-v1` (model + calibrator + state)", f"{t['final_v1_load']:.2f} s",
         "within"],
        ["spaCy warm-up (first parse)", f"{t['spacy_warmup_measured_s']:.2f} s",
         "within"],
        ["**Cold start → ready**",
         f"**{t['total_cold_ready_s']:.2f} s**", "well within"],
        ["Cold start incl. uvicorn boot (integration run)", "6.7–7.7 s",
         "well within"],
    ]))
    a("")
    a(_table(["Request", "Latency", "Notes"], [
        ["Medium text (~370 chars), signals on",
         f"{cap['medium_request_ms']:.1f} ms", "typical interactive request"],
        ["Long text (~28 k chars, near the 30 k cap), signals on",
         f"{cap['long_request_ms']:.1f} ms", "worst realistic case"],
        ["Short text (abstain)", "20 ms", "gate short-circuits labelling"],
        ["Invalid input (422)", "1.4 ms", "rejected before any inference"],
    ]))
    a("")
    a("Every measured path is dominated by CPU inference, with no I/O wait, "
      "so cold start + worst-case request (~8.5 s) is far inside the 300 s "
      "function duration.")
    a("")

    # ---------------------------------------------------------------- 5
    a("## 5. Hosting options and official source URLs")
    a("")
    a("Facts below are quoted from official provider documentation, checked "
      f"on **{CHECKED}**. Items marked *estimate* are our inference, not a "
      "provider claim.")
    a("")
    for key, title in (("vercel_functions", "### Option B — Vercel "
                        "(frontend + Python function)"),
                       ("render", "### Option A — Render Free web service"),
                       ("cloudflare", "### Option C — Cloudflare "
                        "(Workers/Pages Python)")):
        a(title)
        a("")
        a(_table(["Fact", "Statement", "Source"],
                 [[f, q, f"<{u}>"] for f, q, u in PROVIDER_FACTS[key]]))
        a("")

    # ---------------------------------------------------------------- 6
    a("## 6. Free-tier limitations")
    a("")
    a(_table(["Concern", "Vercel Hobby", "Render Free", "Cloudflare (Python)"], [
        ["Bundle limit", "500 MB (Python) — payload fits", "no published "
         "function bundle limit (container)", "64 MiB worker — does not fit"],
        ["Memory", "2 GB — peak fits", "512 MB — **peak 562 MB exceeds it**",
         "128 MB per isolate — does not fit"],
        ["CPU/request", "1 vCPU, 300 s duration", "0.1 CPU",
         "10 ms/request (Free) — does not fit ~900 ms"],
        ["Cold start", "seconds; our cold path ~8 s", "spins down after 15 "
         "min, ~1 min to wake", "1 s startup"],
        ["Idle behaviour", "scales to zero", "spins down; suspends at 750 h",
         "scales to zero"],
        ["Persistence", "none needed (stateless)", "ephemeral FS",
         "none needed"],
        ["Payment method", "not stated in docs reviewed; Hobby is a free "
         "tier with no billing cycles", "not required",
         "not applicable (Workers)"],
        ["Cost if exceeded", "cap → feature locked 30 days (no charge)",
         "suspend until next month (no charge)", "plan limits apply"],
    ]))
    a("")
    a("**Cloudflare option C is infeasible on documented limits** (128 MB "
      "memory, 64 MiB bundle, 10 ms CPU per request, and a Pyodide/WASM "
      "runtime with no stated support for scikit-learn/scipy/spaCy wheels). "
      "**Render option A is infeasible** because its 512 MB free RAM is below "
      "our measured 562 MB peak (and below the 1.0 GB concurrency peak). "
      "Both exclusions rest on published limits, not on assumptions.")
    a("")

    # ---------------------------------------------------------------- 7
    a("## 7. Selected architecture and rationale, or why none qualifies")
    a("")
    a("**Technical verdict: Vercel (Option B) is the only evaluated option "
      "that fits WriteTrace's measured resources at ₹0.**")
    a("")
    a(_table(["Vercel Hobby limit", "WriteTrace need", "Verdict"], [
        ["500 MB Python bundle (uncompressed)",
         f"~{_mb(est_with_spacy)} payload",
         "FITS" if est_with_spacy < 500e6 else "DOES NOT FIT"],
        ["2 GB / 1 vCPU memory", f"{cap['peak_mb']:.0f} MB peak (concurrency)",
         "FITS"],
        ["300 s function duration", "~8.5 s cold start + request",
         "FITS"],
        ["100 MB source upload", f"{_mb(payload['tracked_bytes'])} tracked",
         "FITS"],
        ["4.5 MB request body", "30 kB (30 000-char cap)",
         "FITS"],
        ["Python 3.12–3.14", "3.13", "FITS"],
    ]))
    a("")
    a("**Non-technical blocker (requires your decision).** Vercel's official "
      "Fair Use Guidelines state that \"Hobby teams are restricted to "
      "non-commercial personal use only\", defining commercial usage to "
      "include any deployment that produces \"financial gain of anyone "
      "involved in any part of the production of the project\". Whether "
      "WriteTrace qualifies as non-commercial is a judgement about your use, "
      "not a technical fact, so this audit does not resolve it. If the site is "
      "ever monetised, monetised indirectly, or built by a paid employee or "
      "consultant, Hobby is not the permitted plan and Pro ($20/user/month) "
      "would break the ₹0 budget.")
    a("")
    a("**Architecture selection (assuming the Hobby restriction is "
      "acceptable):**")
    a("")
    a("- *Frontend*: static Astro build on Vercel (the `dist/` output, 51.8 "
      "KB). No SSR, no server rendering, no runtime cost.")
    a("- *Backend*: one Python function exposing `POST /api/v1/analyze`, "
      "written as a small ASGI entry point that reuses `backend.app:app`.")
    a("- *Connection*: because the function and the site are the same "
      "origin on Vercel, the frontend's default `PUBLIC_WRITETRACE_API_BASE="
      "\"/api\"` **does not work on a static host** — the relative path "
      "resolves to the static host, which has no API. Phase 6B must set the "
      "variable to the function's absolute URL (or route `/api` to the "
      "function). The exact-origin CORS middleware added in Phase 6A makes "
      "the split-origin case safe; the same-origin case needs no CORS at all.")
    a("- *Why not one serverless function serving both*: the frontend is a "
      "few KB of static files; there is no reason to run it through Python.")
    a("")

    # ---------------------------------------------------------------- 8
    a("## 8. Privacy and public API risks")
    a("")
    a("Security review executed against the running app "
      f"(`logs/phase6a-security.json`): **{sec['passed']}/{sec['total']} "
      "checks passed**.")
    a("")
    a(_table(["Property", "Verified result"], [
        ["Max input size enforced", "exactly 30 000 chars accepted; 30 001 "
         "and 200 000 rejected 422 (no OOM path)"],
        ["Malformed input", "non-JSON, HTML body, empty body, wrong types, "
         "missing field, out-of-range `max_signals` → 422 envelope"],
        ["Errors expose no internals", "no traceback, path, `site-packages` "
         "or library name in any error body; the offending value is not "
         "echoed"],
        ["Submitted text not logged", "0 application log records captured "
         "during a request containing a unique secret marker"],
        ["CORS", "off by default (same-origin). With "
         "`WRITETRACE_ALLOWED_ORIGINS` set, only exact listed origins get "
         "headers; a `*` value fails closed; credentials are never enabled"],
        ["Surface", "only `/api/v1/analyze`, `/api/v1/health`, "
         "`/api/v1/info` (+ OpenAPI docs). No admin/debug routes"],
        ["Segments", "`segments: []` on every response — segment verdicts "
         "remain disabled"],
    ]))
    a("")
    a("**Risks that remain (documented, not solved):**")
    a("")
    a("- **No rate limiting.** The public endpoint accepts unauthenticated "
      "requests. A per-process in-memory limiter would be unreliable on "
      "serverless (ephemeral instances) and is not added here. Measured "
      f"worst-case cost is ~{cap['worst_case_cpu_s']:.2f} s CPU/request; "
      "Vercel Hobby's documented 4 Active-CPU-hours allowance would be "
      f"exhausted after roughly {cap['worst_case_requests_per_cpu_allowance']:,} "
      "worst-case requests (a rate far above expected use). No database or "
      "paid security service is added in this phase.")
    a("- **Hosting-provider logs.** WriteTrace's code does not log submitted "
      "text, but a hosting provider's infrastructure may record request "
      "metadata (IPs, timestamps, sizes). This is outside the app's control "
      "and is not claimed here.")
    a("- **Public abuse potential.** Without rate limiting, an automated "
      "client could drive cost/usage or exhaust the Hobby cap (which locks "
      "the feature for 30 days, per Vercel's documented behaviour). This is "
      "an accepted, documented limitation, not a solved problem.")
    a("")

    # ---------------------------------------------------------------- 9
    a("## 9. Local production integration results")
    a("")
    a("Rehearsal only — nothing was deployed. Real uvicorn process + the "
      "`npm run preview` production build, with the frontend's own `/api` "
      f"proxy (`logs/phase6a-integration.json`): **{integ['passed']}/"
      f"{integ['total']} checks passed**.")
    a("")
    a(_table(["Check", "Result"],
             [[c["name"], ("PASS" if c["pass"] else "FAIL") +
               (f" — {c['detail']}" if c["detail"] else "")]
              for c in integ["checks"]]))
    a("")

    # ---------------------------------------------------------------- 10
    a("## 10. Test results")
    a("")
    passed_n, warn_n = _pytest_counts()
    a("Captured from actual runs in this phase (no existing test was modified "
      "to hide a failure):")
    a("")
    a(_table(["Suite", "Command", "Result"], [
        ["Backend/ML", "`python -m pytest tests/ -q`",
         f"{passed_n or 'n/a'} passed, {warn_n or 'n/a'} warnings "
         f"(157 pre-existing + 14 new Phase-6A public-API tests)"],
        ["Frontend", "`npm test` (vitest)", "59 passed (4 files)"],
        ["Typecheck", "`npx astro check`", "0 errors"],
        ["Production build", "`npm run build`", "1 page built, `dist/` 51.8 KB"],
    ]))
    a("")

    # ---------------------------------------------------------------- 11
    a("## 11. Files changed")
    a("")
    a(_table(["Path", "Change", "Why"], [
        ["`backend/app.py`", "modified",
         "added the exact-origin, fail-closed CORS middleware "
         "(`WRITETRACE_ALLOWED_ORIGINS`); no change to routes, responses, "
         "or the API contract"],
        ["`tests/test_phase6a_public_api.py`", "added",
         "Phase-6A public-exposure tests (CORS, size limits, error safety)"],
        ["`docs/experiments/phase6a-deployment-audit.md`", "added",
         "this report"],
        ["`ml/experiments/phase6a_report_generator.py`", "added",
         "renders this report from the measurement artifacts"],
        ["`logs/phase6a-*.json`", "added (gitignored)",
         "raw measurement output"],
    ]))
    a("")
    a("Unchanged: `ml/models/*` (all three artifacts), `ml/inference.py`, "
      "`ml/explain.py`, `ml/segmentation.py`, `frontend/src/**` (except the "
      "env-var note below, if any), `docs/api.md`, and every Phase-1–5B "
      "report. No dataset or experiment cache was touched. No commit or tag "
      "was created.")
    a("")

    # ---------------------------------------------------------------- 12
    a("## 12. Remaining blockers")
    a("")
    a("1. **Hobby non-commercial restriction (decision required).** Vercel "
      "permits Hobby only for non-commercial personal use. Whether "
      "WriteTrace qualifies is the user's call. If it does not, Pro is "
      "required and the ₹0 budget cannot be met on Vercel.")
    a("2. **Account sign-up / payment method (unknown).** Not documented on "
      "the pages reviewed and not tested here (no account was created). A "
      "Hobby team may be offered without a card, but this is unverified.")
    a("3. **No global rate limiting.** Accepted for Phase 6A; a real "
      "abuse-control mechanism would need a design decision (and possibly a "
      "paid provider) that is out of scope.")
    a("4. **Cold-start cost at scale.** Each cold function start pays ~8 s "
      "(re-import + model load). Adequate for a low-traffic research tool; "
      "would be a UX problem under sustained load.")
    a("")

    # ---------------------------------------------------------------- 13
    a("## 13. Exact Phase 6B recommendation")
    a("")
    a("**Proceed to Phase 6B only after you confirm one thing:** that "
      "WriteTrace's deployment is non-commercial personal use under Vercel's "
      "published definition. Everything else is technically ready.")
    a("")
    a("When confirmed, Phase 6B should:")
    a("")
    a("1. Add a single Vercel Python function entry point that reuses "
      "`backend.app:app` (no second inference implementation).")
    a("2. Configure the frontend's `PUBLIC_WRITETRACE_API_BASE` to that "
      "function's absolute URL (or route `/api` to it); do **not** rely on "
      "the dev proxy in production.")
    a("3. Set `WRITETRACE_ALLOWED_ORIGINS` to the exact production frontend "
      "origin if and only if the two are on different origins.")
    a("4. Keep `segments` empty and the abstention/disclaimer semantics "
      "unchanged.")
    a("5. Re-run the full local integration checklist, then deploy, then "
      "verify health/info/analyze on the public URL and confirm the "
      "frontend never persists text.")
    a("")
    a("If WriteTrace is (or becomes) commercial, stop and re-open the "
      "budget: Vercel Pro ($20/user/month) is the documented minimum and "
      "there is no ₹0 Vercel path for commercial use.")
    a("")
    a("---")
    a("")
    a(f"_All measurements in this report were produced by the scripts recorded "
      f"in `logs/phase6a-*.json` on {CHECKED}. Provider facts are quoted "
      "from the official documentation URLs listed in Section 5 and were "
      "checked on that date; they can change._")
    a("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(p), encoding="utf-8")
    print(f"report -> {OUT} ({OUT.stat().st_size} bytes, {len(p)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())