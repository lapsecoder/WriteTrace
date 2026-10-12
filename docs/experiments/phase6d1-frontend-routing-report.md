# Phase 6D.1 — Documented Frontend Routing Report

**Project:** WriteTrace · `D:\WriteTrace`
**Date:** 2026-10-10
**Scope:** local verification only. **Nothing deployed, no Vercel project
created, no payment details entered, nothing committed, tagged or pushed, no
Git history rewritten.**

Implements the Phase 6D.0 recommendation: replace the undocumented FastAPI
preset + `outputDirectory` combination with the documented `app.frontend()`
approach.

---

## 1. Official sources

| Source | `last_updated` | Used for |
| ------ | -------------- | -------- |
| <https://fastapi.tiangolo.com/tutorial/frontend/> | live docs | `app.frontend()` signature, precedence, `fallback`, `check_dir` |
| <https://vercel.com/docs/frameworks/backend/fastapi> | 2026-08-27 | FastAPI static serving, CDN promotion, middleware caveat |
| <https://vercel.com/docs/functions/runtimes/python/api-directory> | 2026-08-12 | framework preset precedence |
| <https://vercel.com/docs/project-configuration/vercel-json> | 2026-08-14 | valid `vercel.json` keys |
| <https://openapi.vercel.sh/vercel.json> | live schema | authoritative allowed-key set |

### Verified API, not guessed

`app.frontend()` is a **FastAPI** API, not a Vercel one. Signature confirmed
against the installed FastAPI 0.141.1 and the official FastAPI tutorial:

```python
app.frontend(
    path: str,                                    # required
    *,
    directory: str | os.PathLike[str],            # required
    fallback: Literal["auto","index.html","404.html"] | None = "auto",
    check_dir: bool | Literal["auto"] = "auto",
) -> None
```

Decisive documented guarantees:

> "**FastAPI checks _path operations_ first. The frontend files are checked only
> if no normal route matched, so your API won't be affected.**"

> `fallback=None` → "Then missing frontend paths return the normal `404`."

> `check_dir=False` → "If your frontend files are created later, for example by a
> separate build step after the app object is created…"

> "This is useful with frontend tools that generate static HTML files for each
> page, like **Astro**."

---

## 2. Configuration changes

### 2.1 `app.py` — serve the frontend from FastAPI

```python
app.frontend("/", directory="frontend/dist", fallback=None, check_dir=False)
```

Argument justification:

| Argument | Value | Why |
| -------- | ----- | --- |
| `path` | `"/"` | Frontend at the site root; API routes keep precedence regardless. |
| `directory` | `"frontend/dist"` | `astro.config.mjs` sets no `outDir`, so Astro's default `dist/` applies. Relative to the project root, which is Vercel's CWD. |
| `fallback` | `None` | Single-page site, **no** client-side router, Astro emits **no `404.html`**. `auto` would serve `index.html` with 200 for unknown paths. |
| `check_dir` | `False` | Documented for a separately-built frontend — our `buildCommand` produces the directory. |

### 2.2 `vercel.json` — three changes

1. **Removed `"outputDirectory": "frontend/dist"`** — the undocumented part.
2. **Removed `frontend/dist/**` from `excludeFiles`** — it must now ship inside
   the function bundle or `app.frontend()` has nothing to serve.
3. **Removed two invalid keys** (see §4).

`framework`, `buildCommand`, `functions.app.py` (`maxDuration`, `excludeFiles`)
and the security `headers` are unchanged.

### 2.3 `frontend/dist/**` bundle consequence

`frontend/dist` (6 files, 54,217 B) is now part of the function bundle instead
of being uploaded as a separate static deployment. `frontend/src/**`,
`frontend/node_modules/**` and `frontend/.astro/**` remain excluded, as do the
datasets and research-only models.

**No new dependency.** `app.frontend()` ships with FastAPI; `requirements.txt`
is unchanged.

---

## 3. Why `cdn = true` was **not** enabled

Requirement 5 restricted this to cases where official docs establish the option
is *required* and *compatible*. Neither holds:

- **Not required.** The FastAPI page documents promotion as default behaviour,
  but also documents that when it does not happen the files simply stay in the
  function and are still served: *"Top-level middleware: All static mounts and
  frontends stay in the function."* Correctness does not depend on the CDN.
- **Not free.** *"CDN-served files bypass the function entirely, so middleware
  handlers and `Depends()` guards do not run for them."*

WriteTrace has top-level CORS middleware, so promotion auto-disables. The
function then serves ~54 KB of static files directly. That is slower and
correct. The setting is also available only via `[tool.vercel.fastapi.static]`
in `pyproject.toml`, which the repository does not have — adding it for an
optimisation that is not needed would be unnecessary configuration.

**Consequence to record:** on Hobby, every static asset request runs through
the Python function, including the ~7 s cold start after idle. This is a
performance characteristic, not a correctness problem, and can be revisited.

---

## 4. Incidental critical fix — invalid `vercel.json` keys

While validating against the live schema, a **latent deployment-blocking bug**
from Phase 6B was found.

`https://openapi.vercel.sh/vercel.json` sets `additionalProperties: false` at
the **root** and inside `functions.*`. Verified by fetching the schema:

```
root additionalProperties               : False
functions.<name> additionalProperties  : False
functions.<name> documented value keys : affinity, excludeFiles,
  experimentalTriggers, functionFailoverRegions, includeFiles,
  maxConcurrency, maxDuration, memory, regions, runtime,
  supportsCancellation
```

Phase 6B had added a top-level `"_comment"` key. That key is **not** a
documented property, so under `additionalProperties: false` the whole file
would be rejected — the deployment would have failed at configuration
validation, before any code ran.

I had also briefly added a `_excludeFilesNote` key inside `functions.app.py`
while editing; the same rule rejects it. **Both were removed.** The rationale
now lives in this report and `docs/deployment.md` instead.

Post-fix schema conformance: every top-level key and every `functions.app.py`
key is documented.

---

## 5. Tests

### 5.1 New — `tests/test_phase6d1_frontend_routing.py` (15 tests)

Offline, no network, no account. Covers:

- **Schema conformance** — only documented top-level keys; only documented
  `functions.*` keys (the regression test for §4).
- `outputDirectory` absent; `framework` retained; `functions` keyed by the
  resolved entrypoint `app.py`.
- `frontend/dist` **not** excluded; build inputs and research models still are.
- `app.py` calls `app.frontend()` exactly once, on `/`, with
  `directory="frontend/dist"`, `fallback=None`, `check_dir=False`.
- `app.py` still re-exports `backend.app:app`, builds no second `FastAPI`
  instance, registers no API routes, and hardcodes no deployment URL.
- `requirements.txt` gained no dependency.

### 5.2 Updated — `tests/test_phase6b_deployment_config.py`

`test_output_directory_is_astro_dist` asserted the **old, now-removed**
`outputDirectory`. It **correctly failed** after the change and was rewritten to
`test_output_directory_is_absent`, pointing at the Phase 6D.1 test file. The
test was not weakened to pass — it encoded a superseded design.

### 5.3 Results

| Suite | Result |
| ----- | ------ |
| `python -m pytest tests/ -q` | **203 passed**, 2 warnings |
| `npm test` | **133 passed** |
| `tsc --noEmit` | **no errors** |
| `astro check` | **0 errors, 0 warnings, 0 hints** (22 files) |
| `npm run build` | **1 page built** |
| Clean-process entrypoint checks | **20/20** |
| Browser (production preview) | **17/17** |

---

## 6. Clean-process verification (`uvicorn app:app`, the real entrypoint)

Started as a fresh process exactly as Vercel does — **20/20 passed**.

**API contract preserved:**

| Check | Result |
| ----- | ------ |
| `/api/v1/health` 200, `final-v1` loaded | PASS |
| `/api/v1/info` 200 | PASS |
| `/api/v1/analyze` 200 | PASS |
| Response schema **exactly** the 19 documented keys | PASS |
| `segments[]` empty | PASS |
| `threshold` still 0.5, `gate.min_words` still 50 | PASS |
| Short text abstains (`confidence: null`) | PASS |
| Empty text → 422 with the error envelope | PASS |

**Frontend + routing, same origin:**

| Check | Result |
| ----- | ------ |
| `GET /` serves the built site (200) | PASS |
| `/` is the WriteTrace page | PASS |
| `/favicon.svg` served | PASS |
| `/_astro/index.*.css` served | PASS |
| **API route beats frontend files** (`/api/v1/analyze` → JSON) | PASS |
| `/definitely-missing` → **404** (no `index.html` fallback) | PASS |
| `/api/v1/does-not-exist` → **404** | PASS |
| `/dashboard` → **404** | PASS |
| No wildcard CORS; no ACAO on API or static responses | PASS |

---

## 7. What is verified locally vs. what remains unverified

**Verified locally (real entrypoint, real HTTP, real model):** `app.frontend()`
serves the built site; API routes take precedence; `fallback=None` yields 404s
for unmatched paths; the API contract, thresholds, abstention semantics and
`segments[]` are unchanged; CORS remains fail-closed; `vercel.json` conforms to
the live schema.

**NOT verified — requires an actual Vercel deployment:**

1. **Whether Vercel promotes the frontend to the CDN.** Local runs exercise
   FastAPI only. Vercel's build-time promotion analysis was not reproduced.
2. **Build-time import ordering.** `check_dir=False` removes the failure mode
   regardless, but Vercel still imports `app.py` during the build; that was not
   exercised.
3. **Whether `frontend/dist` is present in the bundle at runtime.** The
   `excludeFiles` change is correct per the docs and verified by reading the
   config, but the bundle is only built on Vercel.
4. **Cold start, memory and latency on the Linux runtime.**
5. **Blocker B2** — Hobby signup payment-method requirement, still
   unestablished and untestable without creating an account.
6. **The previous configuration was never deployed**, so there is no
   comparative data; this change replaces an untested design rather than a
   proven-working one.

---

## 8. Integrity

- `ml/models/` **clean** — matches HEAD exactly.
  `final-v1/model.joblib` = `55a10a397de29698779977371b1c8a13e8fbc09f89e2eade8418cecce1d13800`,
  identical to the Phase 6C record.
- Checkpoint tags intact: `phase-3-complete`, `phase-4-complete`,
  `phase-5a-complete`, `phase-5b-complete`.
- HEAD unchanged at `889d4db`; no commit, tag, push or history rewrite.

---

## 9. Files changed

| File | Change |
| ---- | ------ |
| `app.py` | Added the `app.frontend(...)` call and updated the docstring |
| `vercel.json` | Removed `outputDirectory`; removed `frontend/dist/**` from `excludeFiles`; removed the invalid `_comment` / `_excludeFilesNote` keys |
| `docs/deployment.md` | Documented the new routing, the `cdn` decision, and removed stale `outputDirectory` references |
| `tests/test_phase6d1_frontend_routing.py` | **New** — 15 configuration tests |
| `tests/test_phase6b_deployment_config.py` | One test rewritten for the superseded design |

Untouched: models, `backend/`, `ml/`, API contract, thresholds, abstention
semantics, all frontend UI, `requirements.txt`, `.vercelignore`,
`frontend/astro.config.mjs`, `frontend/package.json`.

---

## 10. Remaining uncertainty and recommendation

### Recommendation: **PROCEED to an authorized preview deployment**

The documented architecture is now implemented, schema-validated against the
live Vercel schema, and verified locally end to end. Blocker **B1** is resolved
on the design side: the configuration now rests on explicit Vercel and FastAPI
documentation rather than an undocumented interaction.

This is **not** an unconditional go. Blocker **B1** can only be *closed* by a
real deployment, and **B2** (Hobby payment-method requirement) is still
unresolved.

Recommended sequence:

1. **Authorize a preview deployment** to close B1 — verify `/` serves the site
   and `/api/v1/health` returns JSON (not HTML, not 404).
2. **Confirm B2 during signup and stop** if a payment method is demanded.
3. Re-measure cold start and memory on the real Linux runtime and update the
   Phase 6C report with measured rather than estimated figures.
4. Consider enabling `[tool.vercel.fastapi.static] cdn = true` afterwards if
   static-asset latency through the function proves painful.

**Not recommended:** deploying to production before a preview validates the
routing, since that is the one behaviour no local test can prove.

---

*Local verification only. No configuration beyond the four files listed above
was altered. Nothing deployed, committed, tagged or pushed.*