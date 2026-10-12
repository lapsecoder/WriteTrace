# Phase 6D.0 — Vercel Routing Feasibility Decision

**Project:** WriteTrace · `D:\WriteTrace`
**Date:** 2026-10-10
**Scope:** research and decision record only. **No configuration was changed, no
account or project created, no payment details entered, nothing deployed,
committed or tagged.**

Resolves blocker **B1** raised in `phase6c-final-verification-report.md`.

---

## 1. Question

The current `vercel.json` combines a **FastAPI framework preset** with an
**`outputDirectory`** (`frontend/dist`) and expects the platform to serve the
Astro static site from the CDN while routing `/api/v1/*` to the Python
function. Phase 6B selected this combination; Phase 6C could not find
documentation confirming it works.

**This document determines whether that combination is documented, and what the
smallest documented alternative is.**

---

## 2. Current configuration under review

`vercel.json`:

```json
{
  "framework": "fastapi",
  "buildCommand": "npm --prefix frontend ci && npm --prefix frontend run build",
  "outputDirectory": "frontend/dist",
  "functions": { "app.py": { "maxDuration": 60, "excludeFiles": "{…}" } },
  "headers": [ … ]
}
```

Supporting files: root `app.py` (re-exports `backend.app:app`), root
`requirements.txt`, `.vercelignore`, `frontend/astro.config.mjs` (static
output, `/api` dev proxy).

---

## 3. Official sources and verification date

**Verified: 2026-10-10** (one agent reported 2026-10-11 for the same date in
UTC; all pages are `last_updated` dated below). Official `vercel.com` only.

| Source | `last_updated` |
| ------ | -------------- |
| <https://vercel.com/docs/functions/runtimes/python/api-directory> | 2026-08-12 |
| <https://vercel.com/docs/frameworks/backend/fastapi> | 2026-08-27 |
| <https://vercel.com/docs/functions/runtimes/python> | 2026-08-12 |
| <https://vercel.com/docs/builds/configure-a-build> | 2026-08-28 |
| <https://vercel.com/docs/project-configuration/vercel-json> | 2026-08-14 |
| <https://vercel.com/docs/services> | 2026-08-10 |
| <https://vercel.com/docs/services/routing> | 2026-06-30 |
| <https://vercel.com/docs/services/config-reference> | 2026-06-30 |
| <https://vercel.com/docs/build-output-api> | 2026-08-11 |
| <https://vercel.com/docs/build-output-api/services> | 2026-09-15 |
| <https://vercel.com/docs/build-output-api/configuration> | 2026-07-27 |
| <https://vercel.com/docs/build-output-api/primitives> | 2025-03-04 *(stale)* |
| <https://vercel.com/docs/frameworks/frontend/astro> | 2026-08-26 |
| <https://vercel.com/docs/frameworks> | 2026-08-11 |
| <https://vercel.com/docs/routing> | 2026-08-11 |
| <https://vercel.com/kb/guide/ship-a-fastapi-app-on-vercel> | 2026-08-24 |
| <https://vercel.com/changelog/fastapi-frontends-and-static-files-served-from-the-cdn> | 2026-09-10 |

---

## 4. Finding: the current configuration is **UNDOCUMENTED**, and one official statement suggests it may not work

### 4.1 `outputDirectory` is never documented with a backend framework preset

Text-searched for `outputDirectory` / `output directory` on the FastAPI page
(2026-08-27), the Python runtime page (2026-08-12), and the Astro page
(2026-08-26): **zero occurrences on all three**.

The FastAPI page documents exactly two static-serving mechanisms —
`public/` and `app.frontend()`/`app.mount()`. `outputDirectory` is not among
them.

Generic statements exist, but never in combination with a backend preset:

> "After building a project, most frameworks output the resulting build in a
> directory. Only the contents of this **Output Directory** will be served
> statically by Vercel. If Vercel detects a framework, the output directory will
> automatically be configured."
> — <https://vercel.com/docs/builds/configure-a-build#output-directory>

And an official counter-statement, from Vercel's own FastAPI guide:

> "Because Vercel ships zero-configuration detection for FastAPI, **you don't
> set a build command or output directory**. Vercel reads your project, finds the
> file that exports your FastAPI `app`, and applies the correct build settings."
> — <https://vercel.com/kb/guide/ship-a-fastapi-app-on-vercel>

### 4.2 The load-bearing risk: "the framework application handles all requests"

From the Python `/api` directory reference (2026-08-12), section *Framework
preset precedence* — **fetched and quoted verbatim during this phase**:

> "A Python framework preset takes precedence over file-based functions. When
> Vercel detects a framework preset, **the framework application handles all
> requests**, and files under `/api` don't become separate Vercel Functions."

The statement is scoped to `/api` files, not to static assets. But taken
literally it is the only official statement about what a framework preset does
with *all* requests — and it does not carve out static output.

**This materially escalates the Phase 6C risk assessment.** Phase 6C rated B1 as
"the API may not route". The correct worst case is worse: if the preset claims
all requests and ignores `outputDirectory`, then **`/` returns 404 and the entire
static site is unreachable**, not just the API.

### 4.3 Conclusion

The current configuration is **experimental — plausible, untested, and
contradicted in spirit by official guidance.** It must not be treated as
supported.

---

## 5. Documented architectures that can serve both halves on one origin

All four are documented. Only A and B are FastAPI-specific; C is the generic
multi-runtime pattern; D is what the Python docs explicitly point at.

### Option A — `app.frontend()` (recommended)

```
app.py:  app.frontend("/", directory="frontend/dist", fallback=None)
```

Documented behaviour (<https://vercel.com/docs/frameworks/backend/fastapi>):

> "`app.frontend()` registers a low-priority frontend build. **Every API route
> takes priority over frontend files regardless of declaration order.**"

> `fallback` = `None` → *"Unmatched paths reach the function"*

This is the only documented mechanism with an **explicit precedence guarantee
that API routes win**, which is precisely WriteTrace's requirement.

### Option B — root `public/` directory

> "Place files in a `public/` directory at your project root. Vercel serves them
> from the CDN at the matching root URL paths. For example, `public/logo.svg` is
> served at `/logo.svg`."
> — <https://vercel.com/docs/frameworks/backend/fastapi>

Requires the build command to copy `frontend/dist` into `public/`. Simple, but
**a full built site inside `public/` is not explicitly documented**, and no
SPA/`index.html` fallback rule is documented for `public/` (unlike
`app.frontend()`'s `fallback` table).

### Option C — Services (Beta)

> "Services let you deploy multiple backends and frontends within a single
> Vercel project. For example, a Next.js frontend and a FastAPI backend in the
> same repository deploy together with shared routing, environment variables,
> and a unique domain."
> — <https://vercel.com/docs/services>

Documented rewrite-based routing, `services` key with per-service `framework`
and `outputDirectory`, and `framework: "services"` at project level.
**"Services (Beta) are available on all plans"** — Hobby-eligible.

Caveats: Beta; top-level `buildCommand`/`outputDirectory`/`functions` are
**invalid** and must move into services; and *"Routing into a service is final…
Vercel does not fall back to your other top-level rewrites."*

### Option D — Astro preset + Python service (Services, frontend-side)

The Astro page documents SSR adapters, but also warns:

> "**Rewrites only work for static files with Astro.** … You should not use
> `vercel.json` to rewrite URL paths with astro projects; doing so produces
> inconsistent behavior, and is not officially supported."
> — <https://vercel.com/docs/frameworks/frontend/astro>

This makes a rewrite-driven Astro+Services combination explicitly discouraged.

---

## 6. Comparison

| | **A — `app.frontend()`** | **B — `public/`** | **C — Services** | **Current (`outputDirectory`)** |
|---|---|---|---|---|
| Documented for FastAPI + built frontend | **Yes** | **Yes** | Yes (generic) | **No** |
| Explicit API-beats-static precedence | **Yes ("every API route takes priority")** | Not stated | Yes (rewrite order) | Not stated |
| Config surface | 1 Python line + drop `outputDirectory` | 1 build-command change | Project preset + `services` + rewrites | current `vercel.json` |
| Needs a Beta feature | No | No | **Yes** | No |
| Static asset delivery | CDN (if `cdn`) or function | **CDN** | Per-service | Unknown |
| Handles unmatched paths | **Yes (`fallback=None`)** | Not documented | Yes, but **no fallthrough** | Unknown |
| Risk if wrong | Low | Medium | Medium-high | **High — could break the whole site** |

### Bundle, memory, routing and cost implications

- **Bundle:** unchanged across all options. A framework preset is a *single*
  function: *"The FastAPI application becomes a single bundle, which has a
  standard bundle size limit of 500MB"* (FastAPI page). Phase 6C estimated
  **≈140 MB** uploaded against that limit. Option A additionally keeps
  `frontend/dist` (≈52 KB) in the bundle unless `exclude = true` — negligible.
- **Memory:** unchanged (2 GB / 1 vCPU on Hobby). No option changes the
  measured ~562 MB serial / ~1019 MB concurrent peaks.
- **Routing:** the only dimension that differs. A and C are explicit; B is
  implied; the current configuration is undocumented.
- **Cost:** **no option changes cost.** All are Hobby-eligible
  (Services: *"available on all plans"*). Exceeding Hobby limits pauses
  features for up to 30 days rather than billing.

### Fit for WriteTrace specifically

The frontend is a **single-page Astro static build** — one `index.html`, no
client-side router, no SPA fallback requirement. Only `/`, `/favicon.svg` and
`/_astro/*` must be served. That is the simplest possible case, and it is
exactly the shape `app.frontend("/", directory=..., fallback=None)` describes.
Option A's `fallback=None` is also appropriate: unmatched paths should reach
FastAPI and return its structured 404.

---

## 7. Required configuration changes (recommended — **not applied**)

Phase 6D.0 is research-only. The following is the proposed Phase 6D.1 change set:

1. **`app.py`** — after importing `app`, add one line:
   ```python
   app.frontend("/", directory="frontend/dist", fallback=None, check_dir=False)
   ```
   `check_dir=False` because with `exclude = true` the directory is removed from
   the bundle and the import-time existence check would fail.
   *This is a deployment entrypoint change only. It does not touch
   `backend/app.py`, the inference pipeline, the API contract, or any UI.*
2. **`vercel.json`** — **remove** `"outputDirectory": "frontend/dist"`.
   Keep `framework`, `buildCommand`, `functions.app.py` (correct per docs:
   *"keyed by your resolved entrypoint file"*), and `headers`.
3. **`pyproject.toml`** (new, optional) — only if CDN promotion is wanted:
   ```toml
   [tool.vercel.fastapi.static]
   cdn = true
   ```
   Required because WriteTrace has **top-level middleware** (the Phase 6A CORS
   handler), and the docs state: *"Top-level middleware: All static mounts and
   frontends stay in the function."* Skipping this is safe — assets then load
   through the function, which is slower but correct.
4. **`.gitignore`** — no change needed (`frontend/dist` is already ignored).

Expected diff: ~4 lines. No dependency, model, backend or UI change.

---

## 8. Remaining unknowns

1. **The current configuration cannot be resolved without deploying.** No
   official document states its behaviour; only a real build settles it.
2. **No documented precedence rule exists for a static-file/function path
   collision.** The Build Output API pages define `static/` and `functions/` as
   siblings but state no tie-break. `primitives` and `features` are ~19 months
   stale (2025-03-04).
3. **`public/` containing a complete built site is not explicitly documented**,
   nor is SPA fallback behaviour for it. Relevant only to fallback Option B.
4. **`[tool.vercel.fastapi.static]` `cdn = true` bypasses middleware for CDN
   files.** Safe here (CORS is irrelevant for same-origin static assets) but it
   is a behavioural choice, not a no-op.
5. **Blocker B2 is unchanged:** whether Hobby signup requires a payment method
   remains **not established** from official documentation, and cannot be
   tested without creating an account.
6. **Linux runtime behaviour remains unverified** beyond wheel resolution.

---

## 9. Recommendation

### **REVISE LOCALLY, THEN DEPLOY — do not retain the current configuration.**

Do **not** deploy with `outputDirectory` + FastAPI preset as it stands. The
combination is undocumented, Vercel's own FastAPI guide says you should *not*
set an output directory for FastAPI, and the Python runtime reference says the
framework application "handles all requests" — meaning the plausible failure is
a completely unreachable site, not a partially working one.

**Recommended path:**

1. **Phase 6D.1** — apply the ~4-line change in §7 (Option A,
   `app.frontend()` + `fallback=None`, drop `outputDirectory`).
2. Re-run the local suite and the production-preview checks to confirm nothing
   regressed locally.
3. **Authorize a preview deployment** purely to verify Option A on the real
   platform — this is the only way to close B1.
4. If Option A fails, fall back to **Option B** (`public/`) before considering
   **Option C** (Services, Beta).
5. Confirm **B2** during signup and **stop** if a payment method is demanded.

This keeps the strict ₹0 budget and the existing inference pipeline entirely
intact — the change touches only the deployment entrypoint and one config key.

---

*Research only. No configuration was modified, no account or project created,
no payment details entered, nothing deployed, committed or tagged. No model,
backend inference behaviour, API contract or frontend UI was altered.*