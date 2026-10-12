# Phase 6B — Vercel Hobby Deployment Preparation Report

**Status:** configuration prepared and verified locally. **Nothing deployed.**
No Vercel account was created, no payment details entered, no deployment made,
no commit created.

**Date:** 2026-10-10
**Scope:** free (₹0), non-commercial, personal/portfolio deployment.

---

## 1. Outcome

WriteTrace now has a complete, offline-verified Vercel configuration:

| File | Purpose | Verified |
| ---- | ------- | -------- |
| `vercel.json` | FastAPI preset, build command, output dir, function limits, security headers | 17 config tests + live rehearsal |
| `app.py` | Entry point re-exporting `backend.app:app` | imported and served live |
| `requirements.txt` | Runtime-only dependencies | pin/exclusion tests |
| `.vercelignore` | Excludes datasets, research models, caches | staging test |
| `docs/deployment.md` | Operator guide | — |
| `tests/test_phase6b_deployment_config.py` | 17 configuration tests | 17 passed |

**Verification totals: 17/17 config tests, 12/12 staging checks,
14/14 end-to-end rehearsal checks.**

---

## 2. Design decision and why

### Chosen: one Vercel project, FastAPI framework preset, static frontend

Vercel has first-class FastAPI support
(<https://vercel.com/docs/frameworks/backend/fastapi>, last updated 2026-08-27).
The docs specify that Vercel looks for a FastAPI instance named `app` at a
root entrypoint (`app.py`, `index.py`, `server.py`, `main.py`, `wsgi.py`,
`asgi.py`) or inside `src/`/`app/`.

```python
# app.py
from backend.app import app
```

This was chosen over `/api/*.py` file-based functions because:

1. **One function, one bundle.** Phase 6A measured ~421.7 MB against the 500 MB
   Python bundle limit. Three separate `/api/*.py` functions would each bundle
   the same ~400 MB dependency closure and risk exceeding the limit.
2. **`backend.app:app` stays authoritative** — the entrypoint adds no routes.
3. **Documented precedence.** Vercel's docs state that with a framework preset,
   `/api` files do not become separate functions, which removes a whole class
   of routing ambiguity.

### Same-origin, therefore no CORS

The frontend already defaults to a same-origin `/api` base
(`frontend/src/lib/api.ts:23-25`), so browser requests stay same-origin and
production needs **no CORS configuration**. The Phase 6A exact-origin CORS
allowlist remains available for a split deployment but stays unset.

### Rejected alternatives

| Alternative | Why rejected |
| ----------- | ------------ |
| `api/[...path].py` catch-all | **Not documented** in any Vercel page consulted. Behaviour under the FastAPI preset is unverified. |
| `vercel.json` rewrites to route `/api/*` | Undocumented interaction with the FastAPI preset; adds a routing layer that could silently shadow API routes. |
| `framework: null` + `/api` functions | Three functions × ~400 MB dependency closure each — bundle-limit risk. |
| Split static host + separate API host | Two origins require an explicit CORS allowlist and add a failure mode. |

### Noted but unverified: `app.frontend()`

Vercel documents `app.frontend("/", directory="dist")` with CDN promotion and
the guarantee that *"Every API route takes priority over frontend files
regardless of declaration order."* This would work, **but** the docs also state
that **top-level middleware disables CDN promotion**. WriteTrace has top-level
CORS middleware (`backend/app.py:106`), so every static asset would be served
by the Python function — a cold start per asset. `outputDirectory` is used
instead: simpler, and static assets stay on the CDN regardless of middleware.

---

## 3. Documentation research (official sources, checked 2026-10-10)

| Source | Key facts used |
| ------ | -------------- |
| <https://vercel.com/docs/frameworks/backend/fastapi> | Entry points; single bundle, 500 MB limit; `public/` CDN behaviour; `app.frontend()` precedence and middleware caveat |
| <https://vercel.com/docs/project-configuration/vercel-json> | `buildCommand`, `outputDirectory`, `framework`, `functions` (`maxDuration`, `excludeFiles`), `headers`, `"framework": null` = Other |
| <https://vercel.com/docs/builds/configure-a-build> | Output-directory defaults; preset detection; build/install command behaviour |
| <https://vercel.com/docs/functions/limitations> | Hobby: 2 GB / 1 vCPU, 300 s duration, 4.5 MB body, 500 MB Python bundle |
| <https://vercel.com/docs/plans/hobby> | Non-commercial personal use; no charge; excess usage waits rather than billing |
| <https://vercel.com/docs/functions/runtimes/python> | ASGI entrypoints, `app` variable, dependency manifests |

### Plan eligibility

Vercel Hobby is restricted to **non-commercial personal use**. The user has
confirmed WriteTrace is a free, non-monetized personal/portfolio project, so
this deployment is consistent with that restriction. **This is a policy
judgement the account owner is responsible for, not a technical fact.**

### Unverified / not confirmed

- **Payment method requirement for Hobby** — not conclusively documented; not
  tested, because that requires creating an account.
- **`api/[...path].py` catch-all syntax** — not found in official docs.
- **Linux parity** — all measurements are from Windows; Vercel runs Linux.

---

## 4. Dependency analysis (the highest-risk area)

A naive copy of `ml/requirements.txt` into `requirements.txt` would have
included `torch` and `transformers`. AST analysis showed both are imported
**only inside function bodies** in Phase 2 perplexity training code
(`ml/experiments/phase2.py:839-840`, `ml/experiments/phase3.py:1413`), which
the deployed final-v1 model never reaches.

Including them would add several GB and **breach the 500 MB bundle limit on
its own**. They are excluded.

Import-time requirements actually needed:

```
fastapi, pydantic, requests, joblib, numpy, pandas, scikit-learn, starlette
+ spacy (lazy, but loaded during startup warmup)
+ en_core_web_sm
```

**`en_core_web_sm` is not on PyPI.** Verified by `pip download` (no matching
distribution). It must be installed from its GitHub release URL, which was
confirmed reachable: **HTTP 200, 12,806,118 bytes** on 2026-10-10.

### Runtime import closure

20 local modules must ship. Notably `ml/data/download.py` is a **code** module
required by `ml/training/train.py`, which is imported by `ml/experiments/phase3.py`
— it ships as source, while the multi-GB datasets do not.

---

## 5. Verification performed

### 5.1 Minimal staging (12/12)

Built a staging tree containing **only** deployment-required files and served
the API from it:

- 47 files, 6.12 MB
- No datasets, caches, tests, docs, or `segments-v1`/`baseline-v1`
- `FINAL_DIR` resolved to the **staged** model copy
- Probability on a fixed test document: **0.9441740372186219**, byte-identical
  to the full repository
- All behavioural checks passed (health, info, analyze, `segments: []`,
  disclaimer, short-text abstention, 422 handling)

### 5.2 Configuration tests (17/17)

Asserts config shape, output directory, build command, exclusions, exact pins,
spaCy release URL, absence of heavy/dev-only dependencies, no second FastAPI
instance, and that no code path can emit a wildcard CORS origin.

### 5.3 End-to-end rehearsal (14/14)

Ran the **exact `buildCommand` from `vercel.json`**, imported the root
`app.py`, served the built site and API, and issued the real browser request:

```
[PASS] frontend index.html served          [PASS] health 200
[PASS] index.html is the WriteTrace page   [PASS] model loaded
[PASS] favicon served                      [PASS] analyze 200
[PASS] hashed CSS asset served             [PASS] segments disabled
[PASS] no ACAO header (same-origin)        [PASS] disclaimer present
[PASS] short text abstains                 [PASS] empty text -> 422
[PASS] 422 uses error envelope             [PASS] unknown route -> 404
```

### 5.4 Regression

Full backend suite passes with the new files present; no existing behaviour or
public API contract was modified. Model artifacts are untouched.

---

## 6. Residual risks

1. **Public unauthenticated API.** Anyone with the URL can submit text. In
   scope for a portfolio project; noted, not mitigated.
2. **No rate limiting.** An in-memory limiter is ineffective across serverless
   instances. Platform-level limits apply instead.
3. **~7 s cold start** on the first request after idle.
4. **Bundle headroom** is ~78 MB (421.7 of 500 MB). Any dependency addition
   could breach it; `requirements.txt` is pinned deliberately.
5. **Memory** peaked at ~1019 MB under 8-way local concurrency against a 2 GB
   Hobby limit — adequate, with less margin than the bundle.
6. **spaCy release URL** is an external dependency at build time.
7. **Windows-measured numbers** may differ on Vercel's Linux runtime.

---

## 7. Next step (Phase 6C, requires explicit approval)

Deployment would require creating a Vercel account — out of scope for Phase 6B
and explicitly not done. Before deploying:

1. Confirm Hobby payment-method requirements.
2. Confirm non-commercial eligibility.
3. Run `vercel` for a preview deployment and re-measure cold start and memory
   on the real Linux runtime.
4. Re-run the full local suite immediately before deploying.