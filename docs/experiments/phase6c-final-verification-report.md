# Phase 6C — Final Pre-Deployment Verification Report

**Project:** WriteTrace · `D:\WriteTrace`
**Target:** Vercel Hobby (₹0, non-commercial, personal/portfolio)
**Date:** 2026-10-10
**Nature:** verification and reporting only. **Nothing was deployed, committed, tagged, or pushed. No Vercel account or project was created; no payment information was entered.**

---

## 0. Verdict summary

| Area | Result |
| ---- | ------ |
| Repository integrity | **PASS** — all tags intact, all model artifacts byte-identical to HEAD |
| Frontend visual audit | **PASS with 2 pre-existing defects** (empty network-error box; weak textarea focus ring) |
| API URL composition | **PASS** — all 5 configurations correct, no `/api/api/` |
| Vercel configuration | **PASS on documented items; 1 UNRESOLVED routing assumption** |
| Runtime packaging | **PASS** — ~140 MB uploaded vs 500 MB limit |
| Linux wheel availability | **PASS** — 11/11 pins resolve for `manylinux_2_28` |
| Backend regression | **PASS** — 188 tests + 42 live checks |
| Frontend regression | **PASS** — 103 tests, tsc/astro/build clean |
| Privacy & security | **PASS** — 18/18, wildcard CORS fails closed |
| Browser integration | **PASS** — 10/10 |

**Recommendation: CONDITIONAL GO** — see §13. Two documented-but-unresolvable gates must be settled by a human before a first deployment, and one real defect (empty error box) should be fixed first.

---

## 1. Scope and repository state

**Verification and reporting only.** All legitimate uncommitted work from Phases 6A, 6B, the abstention UI correction, and the API URL normalization was preserved. Nothing was reset, cleaned, or overwritten.

### `git status --short` (verbatim)

```
 M backend/app.py
 M docs/frontend.md
 M frontend/src/components/ResultsPanel.astro
 M frontend/src/lib/api.ts
 M frontend/src/lib/render-result.ts
 M frontend/tests/api.test.ts
 M frontend/tests/fixtures/index.ts
 M frontend/tests/fixtures/real-responses.json
 M frontend/tests/workspace.test.ts
?? .vercelignore
?? app.py
?? docs/deployment.md
?? docs/experiments/phase6a-deployment-audit.md
?? docs/experiments/phase6b-vercel-preparation-report.md
?? ml/experiments/phase6a_report_generator.py
?? requirements.txt
?? tests/test_phase6a_public_api.py
?? tests/test_phase6b_deployment_config.py
?? vercel.json
```

### `git diff --stat`

```
 backend/app.py                              |  54 +++++
 docs/frontend.md                            |  22 +-
 frontend/src/components/ResultsPanel.astro  |  49 ++-
 frontend/src/lib/api.ts                     |  70 ++--
 frontend/src/lib/render-result.ts           |  51 ++-
 frontend/tests/api.test.ts                  | 154 +++++-
 frontend/tests/fixtures/index.ts            |  10 +
 frontend/tests/fixtures/real-responses.json | 248 +++++++++-
 frontend/tests/workspace.test.ts            | 216 +++++-
 9 files changed, 844 insertions(+), 30 deletions(-)
```

### Attribution of uncommitted work

| Change | Origin phase | Status |
| ------ | ------------ | ------ |
| `backend/app.py` (+54) — exact-origin CORS middleware | Phase 6A | Preserved |
| `vercel.json`, `app.py`, `requirements.txt`, `.vercelignore`, `docs/deployment.md`, `tests/test_phase6b_deployment_config.py` | Phase 6B | Preserved |
| `render-result.ts`, `ResultsPanel.astro`, `workspace.test.ts`, fixtures (+2 real abstention fixtures) | Abstention UI correction | Preserved |
| `api.ts`, `api.test.ts`, `docs/frontend.md` | API URL normalization | Preserved |

**No unexpected modifications were found.** Every dirty file maps to a documented phase.

### Checkpoint tags — all present

| Tag | Commit | Subject |
| --- | ------ | ------- |
| `phase-3-complete` | `c92f59e20538511bddc466015d5489617d1040ab` | feat: complete Phase 3 final model and explainability |
| `phase-4-complete` | `ded36b3c3059d71e1450165413a48f4de2409973` | feat: complete Phase 4 inference API |
| `phase-5a-complete` | `667b5571dda278533b1c71ddfcbdda03847e4ec7` | feat: complete Phase 5A segment analysis research |
| `phase-5b-complete` | `889d4db0398661ed811318ebcc46e4000ddd6143` | feat: complete Phase 5B Astro frontend |

### Phase 1–5B experiment reports — unchanged

`phase3-report.md`, `phase4-report.md`, `phase5a-report.md`, `baseline-v1-report.md`,
`phase2-report.md`, `robustness-plan.md` are all **tracked and clean**, and `git diff`
against their tags is **empty**.

> Note: there is no `docs/experiments/phase5b-report.md`. Phase 5B's report does not
> exist as a separate file; this is not a deletion by this phase — the tag predates it.

---

## 2. Visual review and screenshot paths

**Tool:** Playwright 1.63.0 (Python) with Chromium — **already installed**, nothing new installed.
**Target:** the real **production build** (`frontend/dist`) served by `astro preview`,
which also applies the documented `/api` proxy, against the **real backend** (final-v1).

Screenshots are in a temporary directory, **not** in the repository (`artifacts/` is
not gitignored and would have polluted it):

```
C:\Users\MOHAMM~1\AppData\Local\Temp\opencode\phase6c-screenshots\
```

| File | State | Viewport |
| ---- | ----- | -------- |
| `desktop-1-initial.png` | Initial page | 1440×1000 |
| `desktop-2-normal-result.png` | Normal result (94.4% machine) | 1440×1000 |
| `desktop-3-mixed-uncertain.png` | Mixed/uncertain | 1440×1000 |
| `desktop-4-insufficient-evidence.png` | **Abstention (raw 98.6%)** | 1440×1000 |
| `desktop-5-invalid-input.png` | Invalid input | 1440×1000 |
| `desktop-6-network-error.png` | Network failure | 1440×1000 |
| `desktop-6b-error-closeup.png` | Error box close-up | 1440×1000 |
| `desktop-7-reduced-motion.png` | `prefers-reduced-motion` | 1440×1000 |
| `mobile-1-initial.png` | Initial page | 390×844 |
| `mobile-2-normal-result.png` | Normal result | 390×844 |
| `mobile-3-insufficient-evidence.png` | Abstention | 390×844 |
| `mobile-4-mixed-uncertain.png` | Mixed/uncertain | 390×844 |

`desktop-3b-mixed-attempt.png` records an unsuccessful attempt to reach the mixed band
with live text (see §8 caveat).

### State verification (9/10 automated checks passed)

| State | Result |
| ----- | ------ |
| Initial page loads, disclaimer present, no overflow | PASS |
| Normal result: bar shown, `94.4%`, 8 signals | PASS |
| **Insufficient evidence: no probability bar**, stamp `no verdict`, raw `98.6%` de-emphasized, word count 23 preserved | **PASS** |
| Invalid input handled, no error box | PASS |
| Network failure visibly reported as an error (box shown, not a result) | PASS (with defect D1) |
| Mobile ×3: no horizontal overflow | PASS |
| Mixed/uncertain | PASS (via captured payload, see §8) |
| Reduced motion | PASS with residue (finding D2) |

### The abstention fix is confirmed working

The critical check: a **23-word plainly casual human sentence** yields a raw estimate of
**0.9856 "machine"**. In the built UI it renders as:

- headline `Insufficient evidence`, stamp `no verdict`
- **no** `probability-bar` element, **no** coloured human/machine segments
- a muted graphite figure `98.6%` labelled `raw model output · not interpretable`
- `decision`: `No verdict — abstained (below the evidence threshold)`
- `confidence`: `not reported (insufficient evidence)`
- word count `23` and the threshold explanation preserved
- disclaimer still visible; `segments[]` empty

This is the exact defect that was corrected, confirmed in the real production bundle.

### Typography, contrast, spacing, layout

- No horizontal overflow at either viewport (`scrollWidth == clientWidth`).
- Single `<h1>`, ordered heading hierarchy, no skipped levels.
- Landmarks: `main`×1, `header`×1, `footer`×1; skip link present.
- 3 `aria-live="polite"` regions; textarea has an associated `<label>`.
- Hashed `/_astro/*` assets get `immutable` caching via `vercel.json`.
- The paper/grain aesthetic and the restrained accent palette are preserved; **no
  cosmetic changes were made during this audit.**

---

## 3. API URL composition results

Rule: **endpoint paths are always the full `/api/v1/...`; `PUBLIC_WRITETRACE_API_BASE` is
the origin they are appended to, with any trailing `/api` normalised away.**

All five required configurations, exercised against the real module:

| `PUBLIC_WRITETRACE_API_BASE` | `/api/v1/health` | `/api/v1/info` | `/api/v1/analyze` |
| --- | --- | --- | --- |
| unset (default) | `/api/v1/health` | `/api/v1/info` | `/api/v1/analyze` |
| `""` | `/api/v1/health` | `/api/v1/info` | `/api/v1/analyze` |
| `/api` | `/api/v1/health` | `/api/v1/info` | `/api/v1/analyze` |
| `http://127.0.0.1:8000` | `http://127.0.0.1:8000/api/v1/health` | `…/api/v1/info` | `…/api/v1/analyze` |
| `http://127.0.0.1:8000/api` | `http://127.0.0.1:8000/api/v1/health` | `…/api/v1/info` | `…/api/v1/analyze` |

`/api/api/` occurrences: **0 in all five configurations.**

### Production bundle audit (default configuration)

| Check | Result |
| ----- | ------ |
| `/api/api` in `dist/` | **0** |
| Hardcoded `*.vercel.app` / `.sh` / `.dev` | **0** |
| `127.0.0.1` in bundle | 1 — **static privacy-panel prose** in `index.html`, not a request target |
| `/api/v1/info` literal in bundle | 0 — `info()` is defined but never called by the UI, so it is tree-shaken. The endpoint still works (verified live). |

### Live behaviour against the built site

- Browser (built bundle): issued `GET /api/v1/health` and `POST /api/v1/analyze`,
  both same-origin, **0 failures**.
- `/api/api/v1/health` returns **404** on both dev and preview — proof the old bug is gone.
- Astro proxy forwards `/api/v1/x` **without stripping the prefix**, so the backend
  receives exactly `/api/v1/x`. Same-origin Vercel routing uses the identical path. Compatible.

---

## 4. Vercel routing/configuration findings

`vercel.json`: `framework: "fastapi"`, `buildCommand: npm --prefix frontend ci && npm --prefix frontend run build`, `outputDirectory: "frontend/dist"`, `functions: {"app.py": {maxDuration: 60, excludeFiles: …}}`, plus security headers.

### Documented and confirmed

| Item | Status | Source |
| ---- | ------ | ------ |
| Root `app.py` exporting `app` is a valid FastAPI entrypoint | **DOCUMENTED** | `vercel.com/docs/frameworks/backend/fastapi` (updated 2026-08-27) |
| FastAPI becomes a **single** function/bundle, 500 MB limit | **DOCUMENTED** | same |
| `buildCommand` in `vercel.json` takes precedence | **DOCUMENTED** | same |
| Dependencies via root `requirements.txt`; framework auto-detected from it | **DOCUMENTED** | `vercel.com/docs/functions/runtimes/python` (2026-08-12) |
| `excludeFiles` is a node-glob pattern relative to project root, brace-list | **DOCUMENTED** | `vercel.com/docs/project-configuration/vercel-json` (2026-08-14) |
| `functions` keyed by the resolved entrypoint file (`app.py`) | **DOCUMENTED** | fastapi page |
| Python 3.12 / 3.13 / 3.14 supported (3.12 default) | **DOCUMENTED** | `…/python/python-version` (2026-04-10) |
| Static assets in a root `public/` dir are served from the CDN | **DOCUMENTED** | fastapi page |
| `vercel.json` has no forbidden `public` **property** | verified absent | vercel-json page |

`excludeFiles` was validated locally by applying the pattern: `segments-v1` and
`baseline-v1` models, tests, docs, and caches are **excluded**; `final-v1` is
**included**. Verified with a real glob application, not by inspection.

### ⚠ UNRESOLVED — the one significant configuration risk

**The FastAPI preset combined with `outputDirectory` is NOT documented.**

No official page states what happens when a project has both a FastAPI framework
preset and an `outputDirectory` pointing at a built static site. Specifically, no
documentation confirms whether static output is still served while non-static paths
fall through to the Python function.

The documented ways to combine a Python backend with a frontend in one project are:

1. a root `public/` directory,
2. `app.frontend()` / `app.mount()` CDN promotion,
3. **Services** — the Python runtime page says *"To run a Python backend and a frontend
   together in one project, you can use Services. Each part builds independently and
   routes to a shared domain."*

The Build Output API page was fetched (2026-08-11) and does **not** resolve the
static-vs-function precedence question either; it is framework-author-facing.

**This must be validated by an actual deployment.** Mitigation options if it fails:
copy the Astro build output to a root `public/` directory during `buildCommand` (the
documented path), or adopt Services. This is the single largest technical unknown.

Note: `app.frontend()` was deliberately **not** used — the docs state CDN promotion is
disabled automatically when top-level middleware exists, and WriteTrace has top-level
CORS middleware, so every asset would have cold-started the Python function.

---

## 5. Runtime packaging and size measurements

Reproduced locally; nothing uploaded. `.vercelignore` + `vercel.json` `excludeFiles` applied.

| Component | Size |
| --------- | ---- |
| Linux wheels actually downloaded for the pinned closure (compressed) | **121.08 MB** |
| `en_core_web_sm` 3.8.0 (direct release URL) | 12.81 MB |
| Project files shipped after exclusions (68 tracked + 9 untracked) | **6.27 MB** |
| — of which `final-v1` | 5.78 MB |
| **Total uploaded** | **≈ 140.2 MB** |
| Hobby limit | 500 MB |
| **Headroom** | **≈ 360 MB** |

### Correction of an earlier estimate

Phase 6B estimated **~421.7 MB** for the runtime payload. That figure was derived from a
Windows `site-packages` sweep and is **superseded**. A naive re-measurement during this
phase produced an absurd 2.67 GB — that number swept the entire *development*
environment, including unrelated packages. Neither figure is a real Vercel bundle.

The ≈140 MB figure above is the closest defensible local estimate: it is the sum of
**real Linux wheel downloads** for the exact pins plus the spaCy model plus the actual
shipped project files. It remains an **estimate**, not a measured Vercel bundle.

### Heavy dependencies correctly excluded

`torch`, `transformers`, `pyarrow` are **absent** from `requirements.txt`. AST analysis
confirmed they are imported only inside function bodies of Phase 2 perplexity training
code, never at import time on the deployed path. Including them would have breached the
500 MB limit on their own.

### Model path loading in the packaged layout

The import closure of `backend.app:app` is 20 local modules. `FINAL_DIR` resolves
relative to the repository root. A minimal staging tree of **47 files / 6.12 MB**
containing only the deployment-required files served the API with a **byte-identical
probability** (`0.9441740372186219`) to the full repository. One non-obvious dependency:
`ml/data/download.py` is *code* required by `ml/training/train.py` and must ship, while
the multi-GB datasets must not.

---

## 6. Linux compatibility status

| Tool | Availability |
| ---- | ------------ |
| Docker | **not available** |
| WSL | present but not usable for a build (`rc=1`) |
| `pip download --platform` | **available — used** |

### Wheel availability test (real downloads, nothing installed)

| Target | Result |
| ------ | ------ |
| `manylinux_2_28_x86_64` (Amazon Linux 2023 / glibc 2.34 — **Vercel's build image**) | **11/11 pins available**, 121.08 MB |
| `manylinux2014_x86_64` (glibc 2.17 — stricter than Vercel needs) | 6/11; numpy/pandas/scikit-learn/scipy/spacy need glibc ≥ 2.28 |

An earlier run of this check reported a false blocker (`numpy==2.5.3` "no matching
distribution"). That was an artifact of targeting `manylinux2014`; pip excludes
`manylinux_2_28` wheels for that floor. Vercel's build image is Amazon Linux 2023
(glibc 2.34), where **all 11 pins resolve cleanly**.

**Status: Linux wheel compatibility verified by real download.** What is **NOT** verified:
actual build success, actual runtime behaviour, or actual memory/latency on Vercel's
Linux runtime. No claim of Linux parity is made.

---

## 7. Official provider documentation and verification date

**Verification date: 2026-10-10.** All sources: official `vercel.com` only.

| Page | Last updated |
| ---- | ------------ |
| `vercel.com/docs/functions/limitations` | 2026-08-24 |
| `vercel.com/docs/plans/hobby` | 2026-09-14 |
| `vercel.com/docs/limits` | 2026-09-16 |
| `vercel.com/docs/limits/fair-use-guidelines` | 2026-09-14 |
| `vercel.com/docs/accounts` | 2026-09-21 |
| `vercel.com/docs/frameworks/backend/fastapi` | 2026-08-27 |
| `vercel.com/docs/functions/runtimes/python` | 2026-08-12 |
| `vercel.com/docs/functions/runtimes/python/python-version` | 2026-04-10 |
| `vercel.com/docs/project-configuration/vercel-json` | 2026-08-14 |
| `vercel.com/docs/builds/configure-a-build` | 2026-08-28 |
| `vercel.com/docs/build-output-api` | 2026-08-11 |
| `vercel.com/legal/terms` | Last Updated June 1, 2026 |
| `vercel.com/kb/guide/why-is-my-account-deployment-blocked` | 2026-10-02 |

> Note: `vercel.com/docs/fair-use-guidelines` now redirects to
> `vercel.com/docs/limits/fair-use-guidelines`.

### Hobby non-commercial restriction — DOCUMENTED

> "**Hobby teams** are restricted to non-commercial personal use only. All commercial
> usage of the platform requires either a Pro or Enterprise plan." — Fair Use Guidelines

> "You shall only use the Services under a Hobby plan for your personal or non-commercial
> use." — Terms §4

> "Asking for **Donations** does not fall under commercial usage."

The user has confirmed WriteTrace is a free, non-monetized personal/portfolio project,
which is consistent with this restriction. This remains the **account owner's
responsibility**, not a technical property.

### Excess usage does NOT incur charges — DOCUMENTED

> "As the Hobby plan is a free tier there are no billing cycles. In most cases, if you
> exceed your usage limits on the Hobby plan, you will have to wait until **30 days** have
> passed before you can use the feature again." — Hobby plan page

> "If you have a free Hobby account, you are limited to the usage caps and **cannot
> purchase additional usage**." — Pricing page

Sustained excess usage can pause the deployment (`503 DEPLOYMENT_PAUSED`), and the Terms
reserve a right to terminate the account for exceeding limits. **No official document
states a Hobby user can ever be charged.**

### Hobby limits — DOCUMENTED

| Limit | Value |
| ----- | ----- |
| Python bundle (uncompressed) | 500 MB |
| Memory / vCPU | 2 GB / 1 vCPU |
| Max duration | 300 s (default and maximum) |
| Request & response body | 4.5 MB |
| Build time | 45 min |
| Function invocations | first 1,000,000/month |
| Active CPU | 4 CPU-hrs/month |
| Provisioned memory | 360 GB-hrs/month |
| Fast Data Transfer | 100 GB/month |
| Fast Origin Transfer | 10 GB/month |
| Runtime log retention | 1 hour |

### ⚠ Signup / payment-method requirement — **NOT ESTABLISHED**

**No official Vercel documentation states whether a credit card is required to create a
Hobby account.** `/docs/accounts` documents only email/Git-provider sign-up with no
mention of a payment method; the card prompt on `/docs/plans/hobby` is part of the
**Pro upgrade** flow, not Hobby signup.

This was **not** tested, because testing requires creating an account — out of scope and
prohibited. **Retained as a deployment gate.**

---

## 8. Backend and frontend test results

### Backend

| Check | Result |
| ----- | ------ |
| `python -m pytest tests/ -q` | **188 passed, 2 warnings** (21.0 s) |
| Python lint/typecheck | **Not configured** — no `pyproject.toml`, `setup.cfg`, `tox.ini`, `.flake8`, `ruff.toml` or `mypy.ini` in the repository. Nothing to run. |
| Clean-process startup + `/api/v1/health` | 0.03 s, `final-v1` / `hgb_syn` loaded |
| `/api/v1/info` (11 fields, gate 50 words, no path leakage) | PASS |
| Normal analysis | 200 in **24 ms** |
| Short-text abstention | PASS — `decision: abstain`, `confidence: null`, warning mentions threshold |
| Invalid input (empty / missing / 30,001 chars / 6 MB body) | PASS — 422 with the error envelope, no crash |
| Safe errors (405 on wrong method, 404 unknown route, no traceback) | PASS |
| **Live behavioural suite** | **42/42 passed** |

**Contract invariance:** the `analyze` response key set is **exactly** the 19 documented
keys — `extra=set() missing=set()`. `threshold` is still `0.5`, `gate.min_words` still
`50`, probabilities still sum to 1, `segments[]` always `[]`.

### Frontend

| Check | Result |
| ----- | ------ |
| `npm test` | **103 passed** (4 files) |
| `tsc --noEmit` | **no errors** |
| `astro check` | **0 errors, 0 warnings, 0 hints** (22 files) |
| `npm run build` | **1 page built** |
| Production preview smoke | **19/19** |
| `segment-result` occurrences in bundle | **0** |
| Secret-like strings in bundle | **0** |

**No test was weakened or edited to make it pass.** All test changes were made in earlier
phases and are additive.

---

## 9. Browser integration results

Against the built production frontend with the real backend:

| # | Check | Result |
| - | ----- | ------ |
| 1 | Home page loads | PASS |
| 2 | Backend status reflects real health check (`final-v1` loaded) | PASS |
| 3 | Normal sample returns a real backend response (94.4%, 8 signals) | PASS |
| 4 | Short text returns abstention with **no probability bar** | PASS |
| 5 | Mixed/uncertain results remain honest | PASS — see caveat |
| 6 | Invalid input handled safely | PASS |
| 7 | Backend/network failure visibly reported as an error | PASS — see defect **D1** |
| 8 | Disclaimer remains visible in every state | PASS |
| 9 | No segment-level verdicts | PASS — 0 `segment-result` elements anywhere |
| 10 | No unexplained console errors or failed requests | PASS — the single console error corresponds exactly to the deliberately aborted request in test 7 |

**Caveat on check 5 (honesty disclosure):** the mixed band (0.40–0.60) could **not** be
reached with live text in this session despite searching ~250 combinations of
human-register prose; every sample scored >0.92. The mixed UI was therefore verified by
serving the **real captured `mixed_response` payload** from `final-v1` (p = 0.48354) with
only the network transport stubbed. The payload is a genuine API response; the
verification is honest but not a live end-to-end mixed run. `desktop-3b-mixed-attempt.png`
records the unsuccessful attempt.

---

## 10. Privacy and security review

**18/18 checks passed.**

| Area | Finding |
| ---- | ------- |
| Max input size | `MAX_TEXT_CHARS = 30_000` enforced by Pydantic `max_length`; 30,001 → 422 |
| Oversized/malformed | empty → 422, missing field → 422, 6 MB body → 422 (no crash) |
| Text excluded from logging | **No `logging`, `print`, `open`, `write`, or `save` anywhere in `backend/` or `ml/inference.py`** |
| Text excluded from persistence | No DB, no file writes; `workspace.ts` keeps text only in the textarea value and an in-memory string |
| Client storage/telemetry | **No `localStorage`, `sessionStorage`, `indexedDB`, cookies, `sendBeacon`, or analytics** |
| CORS default | Unset allowlist → **no CORS headers at all** |
| CORS exact-origin | Allowlisted origin granted exactly; unlisted refused; `https://writetrace.example.evil.com` (suffix confusion) **refused** |
| Wildcard CORS | `"*"` **fails CLOSED** — no header emitted, service does not crash |
| Credentials | `Access-Control-Allow-Credentials` never sent |
| Secrets in client env | 0 occurrences of `sk-`, `ghp_`, `Bearer `, `PRIVATE KEY`, `password=`, `api_key=` |
| Privacy disclosure accuracy | Panel states in-memory-only, no browser storage, and carries the remote-backend caveat (*"If you point this build at a remote backend, your text goes to that machine — review the configured URL"*) |
| Overclaim audit | The UI makes **no claim that provider infrastructure logs contain no request metadata.** |

### Honest statement of provider-side limits

WriteTrace's application code does not log or persist submitted text. However, **Vercel
does retain runtime logs (1 hour on Hobby) and handles request metadata at the edge.**
This report does **not** claim submitted text is invisible to the hosting provider, and
the UI does not claim it either. A user submitting sensitive text should be aware that
hosting it on a third-party serverless platform means that platform processes the
request.

### Known risks (accepted, not mitigated in this phase)

- **No authentication.** The API is public and unauthenticated by design. Anyone with
  the URL can submit text.
- **No reliable rate limiting.** An in-memory limiter would be ineffective across
  serverless instances; Vercel's platform-level limits apply instead. **The endpoint is
  not abuse-proof.** No database, paid security service, or auth system was added, per
  the phase constraints.

---

## 11. Artifact hashes and tag integrity

### `final-v1` (deployed; unchanged)

| SHA-256 | File | Bytes |
| ------- | ---- | ----- |
| `5347625dc1e947214270ea8a713bd8bda4cb728abf167d691aadb19c71ff1f48` | calibration.json | 3,448 |
| `e885c27bd79e4f9e5be8db8c2c9fcdbdace07d1d1309f0d3a604065ac3676fc9` | calibrator.joblib | 929 |
| `f5857be987fb198d5bcb8135185a5e88fde314de85fb2a0666f69f1aac5133f7` | config.json | 591 |
| `2500fbc1edcece4cbe4ef40e8b1ba5cb6467032d8d670bf5525ac5debf372e20` | feature_state.joblib | 34,694,670 |
| `0c559a7a809986cfc9f818989f22c39cc9c0f0be1848e2e251978793062ad76c` | features.json | 4,930 |
| `7933e235136ef6cf557dc1e3a6ae6126240dceb7d453ff0fdd9ae777f6c288d8` | gate.json | 1,505 |
| `22b2b675bdc050cd93da73617a7fe5bebd41b729ca0033a3e1c74e1f7126ef2d` | importance.json | 3,403 |
| `455a4eab637ed5933ac54b3d7ca277c926687a953c97e8c28b5ae77c3ea8b00c` | manifest.json | 5,553 |
| `55a10a397de29698779977371b1c8a13e8fbc09f89e2eade8418cecce1d13800` | model.joblib | 22,839,920 |
| `2a388328bec4455401380bca1ce140d0ae889418a00bffe7eb6b133a7be95afb` | reference.json | 9,565 |

Total **5,783,383 B (5.52 MB)**.

### `baseline-v1` (research-only; unchanged)

| SHA-256 | File |
| ------- | ---- |
| `4efc07c15a54b79b235fb691942242daecd9109d91995b879dfea4aafa36309a` | calibrator.joblib |
| `2500fbc1edcece4cbe4ef40e8b1ba5cb6467032d8d670bf5525ac5debf372e20` | feature_state.joblib |
| `6581c1582dea50bc3f43015be7c071c4049045aaad4f796063ef5cd2c5bcd538` | model_dummy_prior.joblib |
| `25642d7945da76ddd5ecdf2a70730f66176549d24e739f183e5bde3d1b4563cb` | model_hist_gradient_boosting.joblib |
| `daf07787f81f0aae16a3b56e2f3b0c04f8c87dda526adcf41cf42a2e232cd06e` | model_logistic_regression.joblib |
| `4abeb9e942f2c6329b0220ddf78e193c574f24bdd725efe0411b28f734df9dbe` | run_manifest.json |
| `cad1f67130c862db3571aa8a1e2978ecff6ff9bcacdb6c0f3741c9f170433b83` | test_metrics.json |
| `835be099b0de93fc3bf875c591521d171bbdaa7f636b39de3c63a1439877f51b` | val_metrics.json |

Total 5,751,128 B (5.48 MB).

### `segments-v1` (research-only, NOT deployed; unchanged)

| SHA-256 | File |
| ------- | ---- |
| `294def075a5777eecec4f175757e0c8131e160964b26ccab3f33c01ba8425b70` | calibration.json |
| `ba14227eb775b97d76f44de282309c16e8259df6049d5b3e651454debcc1f247` | calibrator.joblib |
| `b04dedd3c64696695e763b4598da3eeb797e05461081e40d5076417436dd0d46` | config.json |
| `c4b328ba4efe78cd739afd06145e5485ddbbfaf73b0581cfda3728ed391e0aec` | feature_state.joblib |
| `e9a1642545cb3da3c7f77f5960d1555926391406e583083516b41c6e5ed7a3fe` | features.json |
| `fa4d48e31701d77b16b5b7fb4465180b602cf526507ba46d915aff628cd94117` | gate.json |
| `76a39355449f3724ea6985c7654f7211b38341108b16201ee69f3225900d99cf` | manifest.json |
| `d14f487b7e2188c46ac2b8da8871318cf616de349e2286ca906dd74cd5a5e4d0` | model.joblib |

Total 4,300,473 B (4.10 MB).

**`git status --porcelain -- ml/models/` returned empty**: all three model directories
match HEAD exactly. No model was modified, moved, or retrained. `segments-v1` remains
excluded from the deployment and `segments[]` remains `[]` in every response.

---

## 12. Files changed during Phase 6C

**Exactly one file was created:**

- `docs/experiments/phase6c-final-verification-report.md` (this report)

No source file, test, configuration, model, or document was modified. No cosmetic
changes were made. One temporary scratch test (`frontend/tests/urlmatrix.test.ts`) was
created to print the URL matrix and **deleted** in the same session; the frontend suite
is back to its committed set of 103 tests.

Background processes started for the audit (backend on :8000, `astro preview` on
:4321) are local-only and were not left writing to the repository.

---

## 13. Remaining blockers and risks

### Blockers (must be resolved before deploying)

**B1 — Undocumented static-output + FastAPI routing interaction.**
The configuration relies on the FastAPI preset serving `frontend/dist` as static output
while `/api/v1/*` routes to the Python function. **No official Vercel documentation
confirms this combination.** If it does not work, the deployed site returns 404 for the
API (or serves the API at the wrong paths). Mitigation: build into a root `public/`
directory, or adopt Services. **Only a real deployment can resolve this.**

**B2 — Hobby signup payment-method requirement is unknown.**
Not established from official documentation and deliberately not tested, because testing
requires creating an account. If a card is required, that conflicts with the strict ₹0 /
no-payment-information constraint and the deployment must stop.

### Defects found (pre-existing, not introduced by this phase)

**D1 — Network/backend error messages are invisible.** *(functional, user-facing)*
When an analysis fails, `workspace.ts:229` executes `this.el.error.textContent = ""`,
which **destroys** the `[data-error-message]` and `[data-error-detail]` children of
`#error`. `index.astro` then writes the message into those now-detached nodes, so the
user sees a visible but **completely empty error box** (`innerText === ''`,
`childElementCount === 0`). The error *is* shown, but with no message. This is a genuine
UX bug and should be fixed before or shortly after deployment. It predates the Phase 6C
audit and is unrelated to the abstention and URL fixes.

**D2 — Weak focus ring on the main editor.** *(accessibility, minor)*
`global.css:150` gives `.field:focus` `outline: none` plus a 3 px shadow at **8% opacity**,
so the primary textarea has no strong visible focus indicator, while every other
focusable element gets a solid 2 px accent outline. Keyboard tab order is otherwise
correct and all other focus rings are clearly visible.

### Accepted risks

- Public, unauthenticated API; **not abuse-proof**.
- No reliable rate limiting across serverless instances.
- ~7 s cold start (Phase 6A measurement).
- Bundle headroom ≈ 360 MB; the pins in `requirements.txt` are exact, so any addition
  must be re-checked against the 500 MB limit.
- Hosted runtime logs retained 1 hour by Vercel; provider-side metadata is not claimed
  to be empty.
- All measurements are Windows; Linux behaviour is unverified beyond wheel resolution.

### Corrected findings from earlier phases

- Phase 6B's **~421.7 MB** payload estimate is superseded by the ≈140 MB figure in §5.
- An apparent "numpy has no Linux wheel" blocker was a measurement artifact of an overly
  strict platform floor; all pins resolve on `manylinux_2_28`.

---

## 14. Final recommendation

### **CONDITIONAL GO**

The project is **technically ready** for a first, explicitly authorized deployment:
all tests pass (188 backend + 103 frontend + 42 live backend + 18 security + 19 preview
+ browser suites), the API contract is unchanged, model artifacts are provably
untouched, the abstention UI is honest, URL composition is correct, privacy properties
hold, and the estimated bundle (~140 MB) fits comfortably under the 500 MB limit.

It is **not yet unconditionally go**, for two reasons that only a human can settle:

1. **B1** — the central routing assumption is undocumented and can only be proven by a
   real deployment.
2. **B2** — the Hobby payment-method question is unknown and cannot be answered without
   creating an account.

A first deployment should therefore be treated as an **experimental validation of B1**
on a throwaway basis, with the expectation that the static/API routing may need to be
re-architected to the documented `public/` pattern.

Recommended order: fix **D1** first (small, user-facing), then proceed to a preview
deployment to resolve **B1**, then confirm **B2** during signup and stop if a payment
method is required.

---

## 15. Exact manual steps to authorize deployment

These require explicit human authorization and are **not** performed by this phase.

1. **Confirm policy eligibility.** Accept that Vercel Hobby is restricted to
   non-commercial personal use and that WriteTrace must stay free and non-monetized.
2. **Resolve D1** (empty error box) in `frontend/src/lib/workspace.ts` /
   `frontend/src/pages/index.astro` — change `error.textContent = ""` so it clears the
   message nodes instead of destroying them, then re-run the frontend suite.
3. **Optionally fix D2** (textarea focus ring) in `frontend/src/styles/global.css`.
4. **Commit the current work** on a branch (e.g. `phase-6c-verify`) so the deployment
   has a known source state. Nothing is committed today.
5. **Install the Vercel CLI**: `npm i -g vercel` (already present at v62.1.0).
6. **Check for a payment-method requirement at signup and STOP if a card is demanded.**
7. **`vercel login`**, then **`vercel`** for a **preview** deployment.
8. **Validate B1 on the preview URL**: load the page, confirm the static site renders,
   then confirm `GET <preview>/api/v1/health` returns `{"status":"ok", …}` — *not* a 404
   and not an HTML page.
9. **If B1 fails**, re-architect per the documented pattern: build the Astro output into
   a root `public/` directory (or adopt Services), then redeploy.
10. **Re-measure** cold start, memory and latency on the real Linux runtime and update
    this report with measured — not estimated — figures.
11. **Only then** consider `vercel --prod`, leaving `WRITETRACE_ALLOWED_ORIGINS` **unset**
    (same-origin deployment needs no CORS).

---

*Verification performed 2026-10-10. No deployment, account, project, payment entry,
commit, tag, model modification, or threshold change occurred during Phase 6C.*