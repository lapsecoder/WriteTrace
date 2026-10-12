# WriteTrace — Phase 6A: Deployment Feasibility Audit

Generated: `2026-10-10` · measurement artifacts: `logs/phase6a-*.json` · Git state at audit start: **clean working tree**, four checkpoints verified · **no deployment was created, no provider account was used, no money was spent**.

This phase's own (uncommitted) changes: 4 paths — `ackend/app.py`, `docs/experiments/phase6a-deployment-audit.md`, `ml/experiments/phase6a_report_generator.py`, `tests/test_phase6a_public_api.py`. See Section 11.

> **Verdict up front.** Vercel Hobby is the only evaluated option that fits WriteTrace's measured resources. Its bundle limit (500 MB for Python) exceeds our estimated payload, its 2 GB memory exceeds our measured peak, and its 300 s duration exceeds our cold start + request. The blocker is not technical: it is the **Hobby non-commercial restriction**, which this audit cannot decide for you. See Section 7 and Section 12.

## 1. Scope and current architecture

Scope: decide whether WriteTrace can be published publicly at ₹0, on Vercel, without changing any model, the API contract, or the product's scientific guarantees. No deployment, no account creation, no secrets, no commits.

```text
Browser
  └── Astro static site (frontend/dist, 51.8 KB)  [Phase 5B]
        │  POST /api/v1/analyze  (PUBLIC_WRITETRACE_API_BASE)
        ▼
  FastAPI (backend/, ml/inference.py)  [Phase 4]
        │  final-v1 artifact (5.78 MB) + spaCy en_core_web_sm
        ▼
  CPU-only inference, stateless, no DB, no auth, no telemetry
```

Git checkpoints verified at audit start: `phase-3-complete`, `phase-4-complete`, `phase-5a-complete`, `phase-5b-complete`.

| Component | Lines of evidence |
|---|---|
| Backend | `backend/app.py` (FastAPI, lifespan load-once, safe errors, Phase-6A CORS middleware) |
| Inference | `ml/inference.py` (Predictor; segments-v1 is research-only and not loaded by the public API) |
| Frontend | `frontend/` (Astro 7 static build, no UI framework) |
| Contract | `docs/api.md`; response includes `segments: []` always |

## 2. Artifact and bundle sizes

Measured from the filesystem (`logs/phase6a-sizes.json`, `logs/phase6a-payload.json`).

| Item | Files | Size |
|---|---|---|
| `frontend/dist` (production build) | 6 | 0.05 MB |
| `ml/models/final-v1` (document model) | 10 | 5.78 MB |
| `ml/models/baseline-v1` (Phase 1, read-only) | 8 | 5.75 MB |
| `ml/models/segments-v1` (research-only, NOT deployed) | 8 | 4.30 MB |
| Application source shipped with the API | — | 0.13 MB |
| Python dependency closure (installed size) | 11 | 400.53 MB |
| spaCy model `en_core_web_sm` | 39 | 15.25 MB |
| **Estimated serverless payload (deps + spaCy model + final-v1 + code)** | — | **421.70 MB** |

Headroom against Vercel's 500 MB Python function limit: **78 MB** (84% of the limit used).

Largest files in the frontend build:

| File | Size |
|---|---|
| `_astro\index.C9P7g29j.css` | 20.6 KB |
| `_astro\index.astro_astro_type_script_index_0_lang.DY9kWqtn.js` | 18.4 KB |
| `index.html` | 9.7 KB |
| `_astro\api.B50QJ6t6.js` | 1.9 KB |
| `favicon.svg` | 0.7 KB |
| `_astro\HealthNote.astro_astro_type_script_index_0_lang.C_DAmQUh.js` | 0.3 KB |

Python dependency closure detail (the bulk of any serverless bundle):

| Package | Version | Installed size |
|---|---|---|
| scipy | 1.18.1 | 134.39 MB |
| spacy | 3.8.16 | 91.76 MB |
| pandas | 3.0.5 | 63.62 MB |
| numpy | 2.5.3 | 53.95 MB |
| scikit-learn | 1.9.0 | 42.12 MB |
| setuptools | 84.0.0 | 5.47 MB |
| pydantic | 2.13.5 | 3.79 MB |
| joblib | 1.6.0 | 1.95 MB |
| fastapi | 0.141.1 | 1.52 MB |
| Jinja2 | 3.1.6 | 1.18 MB |
| uvicorn | 0.54.0 | 0.80 MB |

**Packaging verdict.** A git-based deploy uploads the tracked repository: **158 files / 18.05 MB** - comfortably under Vercel's 100 MB CLI upload limit and the 45-minute build budget. The large trees are NOT tracked and therefore not uploaded: `ml/data` (14300.10 MB on disk), `ml/experiments` (1717.33 MB on disk). Everything the API needs at runtime is 47 files / 6.02 MB (`backend/`, `ml/inference.py`, `ml/explain.py`, `ml/segmentation.py`, `ml/features/`, `ml/preprocessing/`, `ml/configs/phase3.json`, `ml/models/final-v1/`). `segments-v1` is deliberately not shipped.

Runtime versions: Python 3.13.15 locally (Vercel supports 3.12/3.13/3.14); Node 24.20.0 for the frontend build.

## 3. Clean-process memory measurements

Measured in a **fresh interpreter** (no warm state). Peak is the OS `GetProcessMemoryInfo(PeakWorkingSetSize)` value, not an inference from file sizes (`logs/phase6a-memory.json`).

| Stage | Peak RSS | Current RSS | Private commit |
|---|---|---|---|
| fresh interpreter | 16 MB | 16 MB | 10 MB |
| imports (pandas+inference) | 175 MB | 175 MB | 1148 MB |
| final-v1 loaded | 220 MB | 210 MB | 1184 MB |
| spaCy warm-up done | 456 MB | 450 MB | 1426 MB |
| after 5 long analyses | 521 MB | 453 MB | 1463 MB |
| after 10 medium analyses | 521 MB | 453 MB | 1463 MB |
| after HTTP layer exercised | 562 MB | 494 MB | 1505 MB |

**Headline numbers**

| Measure | Value | Relevant limit |
|---|---|---|
| Peak RSS, serial analysis (fresh process) | 562 MB | Vercel Hobby 2 GB |
| Peak RSS under 8-way concurrency | 1019 MB | Vercel Hobby 2 GB |
| Steady-state RSS after 120 requests | 465 MB (-9 MB growth — no leak) | — |
| spaCy pipeline load alone | ~240 MB of the peak | — |

Both peaks sit below Vercel Hobby's 2 GB with >50 % headroom. The concurrency peak (1.0 GB) is the number to watch: it is within Vercel's limit but leaves less margin than the serial figure suggests.

## 4. Startup and inference latency

CPU-only, single fresh process (`logs/phase6a-memory.json`, `logs/phase6a-capacity.json`, `logs/phase6a-integration.json`).

| Phase | Measured | Vercel Hobby budget (300 s) |
|---|---|---|
| Interpreter → inference imports | 2.57 s | within |
| Load `final-v1` (model + calibrator + state) | 0.46 s | within |
| spaCy warm-up (first parse) | 4.38 s | within |
| **Cold start → ready** | **7.41 s** | well within |
| Cold start incl. uvicorn boot (integration run) | 6.7–7.7 s | well within |

| Request | Latency | Notes |
|---|---|---|
| Medium text (~370 chars), signals on | 34.7 ms | typical interactive request |
| Long text (~28 k chars, near the 30 k cap), signals on | 895.7 ms | worst realistic case |
| Short text (abstain) | 20 ms | gate short-circuits labelling |
| Invalid input (422) | 1.4 ms | rejected before any inference |

Every measured path is dominated by CPU inference, with no I/O wait, so cold start + worst-case request (~8.5 s) is far inside the 300 s function duration.

## 5. Hosting options and official source URLs

Facts below are quoted from official provider documentation, checked on **2026-10-10**. Items marked *estimate* are our inference, not a provider claim.

### Option B — Vercel (frontend + Python function)

| Fact | Statement | Source |
|---|---|---|
| Python function bundle size (uncompressed) | 500 MB (standard limit for Python runtimes); general limit for other runtimes is 250 MB; Large Functions Beta supports up to 5 GB on Fluid compute | <https://vercel.com/docs/functions/limitations> |
| Function memory / CPU (Hobby) | 2 GB / 1 vCPU (default and maximum) | <https://vercel.com/docs/functions/limitations> |
| Maximum duration (Hobby) | 300 s (5 minutes) default and maximum | <https://vercel.com/docs/functions/limitations> |
| Request body size limit | 4.5 MB for request or response body of a Vercel Function (exceeding it returns 413 FUNCTION_PAYLOAD_TOO_LARGE) | <https://vercel.com/docs/functions/limitations> |
| Python runtime versions | "The available versions are: 3.12 (default), 3.13, 3.14" | <https://vercel.com/docs/functions/runtimes/python> |
| Python dependency declaration | "Define dependencies in pyproject.toml (with or without a uv.lock), requirements.txt, or a Pipfile with a corresponding Pipfile.lock." | <https://vercel.com/docs/functions/runtimes/python> |
| Python bundling behaviour | "By default, Python Vercel Functions include all files from your project that are reachable at build time. There is no automatic tree-shaking for Python." (hence datasets/caches must be excluded at the source) | <https://vercel.com/docs/functions/runtimes/python> |
| Hobby plan usage restriction | "Hobby teams are restricted to non-commercial personal use only. All commercial usage of the platform requires either a Pro or Enterprise plan." Commercial usage is defined as any Deployment used for the purpose of financial gain of anyone involved in any part of the production of the project, including a paid employee or consultant writing the code. | <https://vercel.com/docs/limits/fair-use-guidelines> |
| Hobby plan price | "As the Hobby plan is a free tier there are no billing cycles." (Pro is $20 per user / month) | <https://vercel.com/docs/plans/hobby> |
| Hobby monthly allowances (fair use) | Function Invocations: first 1,000,000 · Active CPU: 4 hours · Provisioned Memory: 360 GB-hrs · Fast Data Transfer: first 100 GB (documented as typical monthly usage guidelines) | <https://vercel.com/docs/limits/fair-use-guidelines> |
| Exceeding Hobby limits | "If you exceed your usage limits on the Hobby plan, you will have to wait until 30 days have passed before you can use the feature again." (cap, not a charge) | <https://vercel.com/docs/plans/hobby> |
| Build time per deployment | 45 minutes (all plans); build disk 32 GB (Hobby) | <https://vercel.com/docs/limits> |
| Static file upload limit (CLI) | 100 MB on Hobby; 1 GB on Pro | <https://vercel.com/docs/limits> |
| Runtime log retention | Runtime logs stored for 1 hour on Hobby (1 day Pro, 3 days Enterprise) | <https://vercel.com/docs/limits> |
| Hobby team size | Hobby: 1 user per team | <https://vercel.com/docs/limits> |
| Payment method | NOT explicitly stated in the Vercel docs reviewed; Vercel docs state a Pro trial requires a payment method to continue on Pro, and Hobby is a free tier with no billing cycles. Account sign-up experience (whether a card is offered/required) is NOT documented and was NOT tested in this phase (no account was created). | <https://vercel.com/docs/plans/pro-plan/trials> |

### Option A — Render Free web service

| Fact | Statement | Source |
|---|---|---|
| Free web service plan | $0/month, 512 MB RAM, 0.1 CPU | <https://render.com/docs/compute-plans> |
| Free spin-down | "Render spins down a Free web service that goes 15 minutes without receiving any inbound traffic... spins back up whenever it next receives an HTTP request... takes about one minute." | <https://render.com/docs/free> |
| Free monthly hours | "Render grants 750 Free instance hours to each workspace per calendar month... suspends all of your Free web services until the start of the next month." | <https://render.com/docs/free> |
| Payment method | NOT required for a free service per Render's docs (a card is only needed to avoid suspension on bandwidth overage) | <https://render.com/docs/your-first-deploy> |

### Option C — Cloudflare (Workers/Pages Python)

| Fact | Statement | Source |
|---|---|---|
| Python Workers runtime | "Workers written in Python are executed by Pyodide. Pyodide is a CPython compiled to WebAssembly." | <https://developers.cloudflare.com/workers/languages/python/how-python-workers-work/> |
| Python Worker package support | "Python Workers support pure and PyEmscripten Python packages on PyPI... WebAssembly support for Python packages is still in early stages" | <https://developers.cloudflare.com/workers/languages/python/packages/> |
| Workers limits (Free) | Memory per isolate 128 MB; CPU time per request 10 ms (Free); Worker size 64 MiB; startup 1 s | <https://developers.cloudflare.com/workers/platform/limits/> |

## 6. Free-tier limitations

| Concern | Vercel Hobby | Render Free | Cloudflare (Python) |
|---|---|---|---|
| Bundle limit | 500 MB (Python) — payload fits | no published function bundle limit (container) | 64 MiB worker — does not fit |
| Memory | 2 GB — peak fits | 512 MB — **peak 562 MB exceeds it** | 128 MB per isolate — does not fit |
| CPU/request | 1 vCPU, 300 s duration | 0.1 CPU | 10 ms/request (Free) — does not fit ~900 ms |
| Cold start | seconds; our cold path ~8 s | spins down after 15 min, ~1 min to wake | 1 s startup |
| Idle behaviour | scales to zero | spins down; suspends at 750 h | scales to zero |
| Persistence | none needed (stateless) | ephemeral FS | none needed |
| Payment method | not stated in docs reviewed; Hobby is a free tier with no billing cycles | not required | not applicable (Workers) |
| Cost if exceeded | cap → feature locked 30 days (no charge) | suspend until next month (no charge) | plan limits apply |

**Cloudflare option C is infeasible on documented limits** (128 MB memory, 64 MiB bundle, 10 ms CPU per request, and a Pyodide/WASM runtime with no stated support for scikit-learn/scipy/spaCy wheels). **Render option A is infeasible** because its 512 MB free RAM is below our measured 562 MB peak (and below the 1.0 GB concurrency peak). Both exclusions rest on published limits, not on assumptions.

## 7. Selected architecture and rationale, or why none qualifies

**Technical verdict: Vercel (Option B) is the only evaluated option that fits WriteTrace's measured resources at ₹0.**

| Vercel Hobby limit | WriteTrace need | Verdict |
|---|---|---|
| 500 MB Python bundle (uncompressed) | ~421.70 MB payload | FITS |
| 2 GB / 1 vCPU memory | 1019 MB peak (concurrency) | FITS |
| 300 s function duration | ~8.5 s cold start + request | FITS |
| 100 MB source upload | 18.05 MB tracked | FITS |
| 4.5 MB request body | 30 kB (30 000-char cap) | FITS |
| Python 3.12–3.14 | 3.13 | FITS |

**Non-technical blocker (requires your decision).** Vercel's official Fair Use Guidelines state that "Hobby teams are restricted to non-commercial personal use only", defining commercial usage to include any deployment that produces "financial gain of anyone involved in any part of the production of the project". Whether WriteTrace qualifies as non-commercial is a judgement about your use, not a technical fact, so this audit does not resolve it. If the site is ever monetised, monetised indirectly, or built by a paid employee or consultant, Hobby is not the permitted plan and Pro ($20/user/month) would break the ₹0 budget.

**Architecture selection (assuming the Hobby restriction is acceptable):**

- *Frontend*: static Astro build on Vercel (the `dist/` output, 51.8 KB). No SSR, no server rendering, no runtime cost.
- *Backend*: one Python function exposing `POST /api/v1/analyze`, written as a small ASGI entry point that reuses `backend.app:app`.
- *Connection*: because the function and the site are the same origin on Vercel, the frontend's default `PUBLIC_WRITETRACE_API_BASE="/api"` **does not work on a static host** — the relative path resolves to the static host, which has no API. Phase 6B must set the variable to the function's absolute URL (or route `/api` to the function). The exact-origin CORS middleware added in Phase 6A makes the split-origin case safe; the same-origin case needs no CORS at all.
- *Why not one serverless function serving both*: the frontend is a few KB of static files; there is no reason to run it through Python.

## 8. Privacy and public API risks

Security review executed against the running app (`logs/phase6a-security.json`): **26/26 checks passed**.

| Property | Verified result |
|---|---|
| Max input size enforced | exactly 30 000 chars accepted; 30 001 and 200 000 rejected 422 (no OOM path) |
| Malformed input | non-JSON, HTML body, empty body, wrong types, missing field, out-of-range `max_signals` → 422 envelope |
| Errors expose no internals | no traceback, path, `site-packages` or library name in any error body; the offending value is not echoed |
| Submitted text not logged | 0 application log records captured during a request containing a unique secret marker |
| CORS | off by default (same-origin). With `WRITETRACE_ALLOWED_ORIGINS` set, only exact listed origins get headers; a `*` value fails closed; credentials are never enabled |
| Surface | only `/api/v1/analyze`, `/api/v1/health`, `/api/v1/info` (+ OpenAPI docs). No admin/debug routes |
| Segments | `segments: []` on every response — segment verdicts remain disabled |

**Risks that remain (documented, not solved):**

- **No rate limiting.** The public endpoint accepts unauthenticated requests. A per-process in-memory limiter would be unreliable on serverless (ephemeral instances) and is not added here. Measured worst-case cost is ~0.90 s CPU/request; Vercel Hobby's documented 4 Active-CPU-hours allowance would be exhausted after roughly 16,077 worst-case requests (a rate far above expected use). No database or paid security service is added in this phase.
- **Hosting-provider logs.** WriteTrace's code does not log submitted text, but a hosting provider's infrastructure may record request metadata (IPs, timestamps, sizes). This is outside the app's control and is not claimed here.
- **Public abuse potential.** Without rate limiting, an automated client could drive cost/usage or exhaust the Hobby cap (which locks the feature for 30 days, per Vercel's documented behaviour). This is an accepted, documented limitation, not a solved problem.

## 9. Local production integration results

Rehearsal only — nothing was deployed. Real uvicorn process + the `npm run preview` production build, with the frontend's own `/api` proxy (`logs/phase6a-integration.json`): **13/13 checks passed**.

| Check | Result |
|---|---|
| production build exists (frontend/dist/index.html) | PASS |
| FastAPI starts from a clean process | PASS — cold ready 6.7s at http://127.0.0.1:52903 |
| GET /api/v1/health verified | PASS — {"status": "ok", "api_version": "v1", "model_version": "final-v1", "model_name": "hgb_syn", "loaded": true} |
| GET /api/v1/info verified | PASS — features=89 gate={'min_words': 50, 'mixed_band': [0.4, 0.6]} |
| production frontend served by local preview | PASS — http://127.0.0.1:52904 |
| configured proxy reaches the backend (UI origin -> /api) | PASS — status=200 |
| normal text analyzed through the production path | PASS — label=likely_machine p=0.944174 |
| UI displays a real response only (probabilities present) | PASS |
| short text returns 200 + abstention | PASS — label=insufficient_evidence |
| invalid input returns 422 envelope | PASS — status=422 |
| segments[] remains empty in production responses | PASS — [] |
| authorship disclaimer present in production response | PASS |
| backend failure surfaces as a transport error (not a fake 200) | PASS — status=502 () |

## 10. Test results

Captured from actual runs in this phase (no existing test was modified to hide a failure):

| Suite | Command | Result |
|---|---|---|
| Backend/ML | `python -m pytest tests/ -q` | 171 passed, 2 warnings (157 pre-existing + 14 new Phase-6A public-API tests) |
| Frontend | `npm test` (vitest) | 59 passed (4 files) |
| Typecheck | `npx astro check` | 0 errors |
| Production build | `npm run build` | 1 page built, `dist/` 51.8 KB |

## 11. Files changed

| Path | Change | Why |
|---|---|---|
| `backend/app.py` | modified | added the exact-origin, fail-closed CORS middleware (`WRITETRACE_ALLOWED_ORIGINS`); no change to routes, responses, or the API contract |
| `tests/test_phase6a_public_api.py` | added | Phase-6A public-exposure tests (CORS, size limits, error safety) |
| `docs/experiments/phase6a-deployment-audit.md` | added | this report |
| `ml/experiments/phase6a_report_generator.py` | added | renders this report from the measurement artifacts |
| `logs/phase6a-*.json` | added (gitignored) | raw measurement output |

Unchanged: `ml/models/*` (all three artifacts), `ml/inference.py`, `ml/explain.py`, `ml/segmentation.py`, `frontend/src/**` (except the env-var note below, if any), `docs/api.md`, and every Phase-1–5B report. No dataset or experiment cache was touched. No commit or tag was created.

## 12. Remaining blockers

1. **Hobby non-commercial restriction (decision required).** Vercel permits Hobby only for non-commercial personal use. Whether WriteTrace qualifies is the user's call. If it does not, Pro is required and the ₹0 budget cannot be met on Vercel.
2. **Account sign-up / payment method (unknown).** Not documented on the pages reviewed and not tested here (no account was created). A Hobby team may be offered without a card, but this is unverified.
3. **No global rate limiting.** Accepted for Phase 6A; a real abuse-control mechanism would need a design decision (and possibly a paid provider) that is out of scope.
4. **Cold-start cost at scale.** Each cold function start pays ~8 s (re-import + model load). Adequate for a low-traffic research tool; would be a UX problem under sustained load.

## 13. Exact Phase 6B recommendation

**Proceed to Phase 6B only after you confirm one thing:** that WriteTrace's deployment is non-commercial personal use under Vercel's published definition. Everything else is technically ready.

When confirmed, Phase 6B should:

1. Add a single Vercel Python function entry point that reuses `backend.app:app` (no second inference implementation).
2. Configure the frontend's `PUBLIC_WRITETRACE_API_BASE` to that function's absolute URL (or route `/api` to it); do **not** rely on the dev proxy in production.
3. Set `WRITETRACE_ALLOWED_ORIGINS` to the exact production frontend origin if and only if the two are on different origins.
4. Keep `segments` empty and the abstention/disclaimer semantics unchanged.
5. Re-run the full local integration checklist, then deploy, then verify health/info/analyze on the public URL and confirm the frontend never persists text.

If WriteTrace is (or becomes) commercial, stop and re-open the budget: Vercel Pro ($20/user/month) is the documented minimum and there is no ₹0 Vercel path for commercial use.

---

_All measurements in this report were produced by the scripts recorded in `logs/phase6a-*.json` on 2026-10-10. Provider facts are quoted from the official documentation URLs listed in Section 5 and were checked on that date; they can change._
