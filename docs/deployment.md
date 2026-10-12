# WriteTrace — Deployment Guide

How to deploy WriteTrace to **Vercel Hobby** (free, ₹0), and what was verified
before deployment. Phase 6B prepared and verified this configuration locally;
**nothing has been deployed**.

---

## 1. Architecture: one project, one origin

WriteTrace is deployed as a **single Vercel project** serving both halves of
the app from one origin:

| Path             | Served by                                  |
| ---------------- | ------------------------------------------ |
| `/`              | Astro static build, served by FastAPI (`app.frontend()`) |
| `/_astro/*`      | Hashed assets, immutable cache              |
| `/api/v1/*`      | FastAPI function (root `app.py`)            |

Because both halves share an origin, the browser makes **same-origin**
requests and production needs **no CORS configuration at all**.

```
browser ──► https://<deployment>.vercel.app/
              ├── /                → CDN (static Astro output)
              └── /api/v1/analyze  → FastAPI function → final-v1 model
```

### Why not a split deployment?

A static host plus a separate API host would be two projects, two origins, and
would require an explicit CORS allowlist. Same-origin is simpler, removes an
entire class of browser/CORS failure, and keeps the privacy story intact
(no third-party origin sees the submitted text).

---

## 2. Files that make up the deployment

| File              | Purpose                                                         |
| ----------------- | --------------------------------------------------------------- |
| `vercel.json`     | Framework preset, build command, function limits, headers       |
| `app.py`          | Vercel entrypoint; re-exports `backend.app:app` **and** calls `app.frontend()` to serve the built site |
| `requirements.txt`| Runtime-only Python dependencies (bundle-size critical)          |
| `.vercelignore`   | Excludes datasets, research models, and caches                  |

`backend/app.py` remains the single source of truth for the API contract.

### How the frontend is served (Phase 6D.1)

The built Astro site is served by **FastAPI**, not by Vercel's `outputDirectory`:

```python
# app.py
app.frontend("/", directory="frontend/dist", fallback=None, check_dir=False)
```

- **Why not `outputDirectory`:** combining it with the FastAPI preset is
  **undocumented**. Vercel's FastAPI guide states you should not set an output
  directory for FastAPI, and the Python runtime reference says a framework
  preset means *"the framework application handles all requests"*. See
  `docs/experiments/phase6d0-vercel-routing-decision.md`.
- **API precedence is guaranteed by FastAPI:** *"FastAPI checks path operations
  first. The frontend files are checked only if no normal route matched, so your
  API won't be affected."*
- **`fallback=None`** — WriteTrace is a single-page site with no client-side
  router and generates no `404.html`, so unmatched paths return a normal 404
  instead of silently serving `index.html` with a 200.
- **`check_dir=False`** — the documented setting when the frontend is produced
  by a separate build step, which is exactly the case here.
- **`frontend/dist` must ship** in the function bundle, so it is deliberately
  **not** in `excludeFiles`.

### `cdn` option — deliberately not enabled

Vercel promotes frontend files to the CDN by default, but
*"top-level middleware: all static mounts and frontends stay in the function."*
WriteTrace has top-level CORS middleware, so promotion is disabled automatically
and the function serves the ~54 KB of static files itself.

`[tool.vercel.fastapi.static] cdn = true` would force CDN promotion, but it is
**not required for correctness** and would bypass middleware for those files.
It is therefore not used. If static-asset latency later matters, enabling it is
a one-line `pyproject.toml` change.

---

## 3. How the model is deployed

The API loads `ml/models/final-v1/` (5.78 MB). It ships in the deployment.

**Not shipped** (research-only, license/size reasons):

- `ml/models/segments-v1/` — segmentation research; the API returns `segments: []`
- `ml/models/baseline-v1/` — earlier winner, superseded by final-v1
- `ml/data/**` — datasets are reproducible from `scripts/` and must never
  reach a hosting provider
- test suites, docs, experiment caches

---

## 4. Runtime dependencies (why `requirements.txt` is not `ml/requirements.txt`)

`requirements.txt` at the root lists only what a **cold start must import**,
verified by AST analysis of the deployed module closure:

```
fastapi, pydantic, starlette, uvicorn,
numpy, pandas, scikit-learn, scipy, joblib, spacy
+ en_core_web_sm (from its GitHub release URL)
```

Deliberately excluded, with reasons:

| Excluded        | Why                                                            |
| --------------- | -------------------------------------------------------------- |
| `torch`         | Imported lazily inside Phase 2 perplexity training functions; never reached by final-v1. Multiple GB — would breach the 500 MB bundle limit by itself. |
| `transformers`  | Same as above.                                                  |
| `pyarrow`       | Lazily imported by the dataset builder only.                   |
| `pytest`        | Dev-only.                                                      |
| `httpx`         | Dev/test client only.                                          |

> `en_core_web_sm` is **not on PyPI**. It is installed from its GitHub release
> URL, pinned to 3.8.0. That URL was verified reachable (HTTP 200,
> 12,806,118 bytes) on 2026-10-10. If it ever disappears, the build fails at
> dependency installation — pin a new version rather than unpinning.

---

## 5. Measured limits (Phase 6A)

| Metric                       | Measured   | Vercel Hobby limit |
| ---------------------------- | ---------- | ------------------ |
| Python bundle (estimated)    | ~421.7 MB  | 500 MB             |
| Memory, serial peak          | ~562 MB    | 2048 MB            |
| Memory, 8-way concurrency    | ~1019 MB   | 2048 MB            |
| Longest request              | ~0.9 s     | 300 s max duration |
| Function memory growth       | none over 120 requests | — |

`maxDuration` is set to 60 s — far above the ~0.9 s worst case.

---

## 6. Deploying

```bash
npm i -g vercel
vercel            # preview deployment
vercel --prod     # production
```

### Required project settings

| Setting      | Value                  |
| ------------ | ---------------------- |
| Framework    | FastAPI                |
| Build command| `npm --prefix frontend ci && npm --prefix frontend run build` |
| Output dir   | *(unset — served by `app.frontend()`)* |

These are all pinned in `vercel.json`, so the dashboard can stay on defaults.

### Environment variables

| Variable | Required | Purpose |
| -------- | -------- | ------- |
| `WRITETRACE_ALLOWED_ORIGINS` | **No** | Leave unset. Same-origin deployment needs no CORS. Only set this if the API is ever split onto a different origin, with **exact** origins (a wildcard is refused). |
| `PUBLIC_WRITETRACE_API_BASE` | **No** | Unset defaults to same-origin, so the page requests `/api/v1/...` on its own host. This is correct for this deployment. |

### Local verification before deploying

```bash
python -m pytest tests/ -q                        # full suite
npm --prefix frontend test                        # frontend tests
python -m uvicorn app:app --port 8000             # root entrypoint
```

---

## 7. Verification performed in Phase 6B

All offline and local — **no deployment, no account, no payment details**:

1. **Import closure analysis** — AST walk proved the 20 local modules that must
   ship, and that `torch`/`transformers` are lazy (never imported at cold start).
2. **Minimal staging test** — a 47-file / 6.12 MB runtime tree, containing *only*
   what the deployment needs, served the API with **byte-identical probabilities**
   to the full repository.
3. **Configuration tests** — 17 tests assert the config shape, exclusions,
   pinned dependencies, and that no wildcard CORS can be emitted.
4. **End-to-end rehearsal** — ran the real `buildCommand`, imported the root
   entrypoint, served the built site and the API, and made the exact browser
   request: **14/14 checks passed**, including no ACAO header (same-origin).

---

## 8. Security posture

- **No authentication.** The API is public and unauthenticated by design
  (portfolio/free-tier scope). Anyone who finds the URL can submit text.
- **No rate limiting.** An in-memory limiter would not be reliable across
  serverless instances. Vercel's platform-level protections apply. Consider
  Vercel WAF or Cloudflare if abuse becomes a problem.
- **Static security headers** are set in `vercel.json`: `X-Content-Type-Options`,
  `Referrer-Policy`, `X-Frame-Options`.
- **Text privacy**: submitted text is never logged, stored, or forwarded
  anywhere. Serverless request logs do not include bodies.
- **`segments` stays empty** in every response.

## 9. Known limitations

- Payment-method requirements for Hobby are not fully documented by Vercel;
  this has **not** been tested because no account was created.
- Cold starts are ~7 s (model + spaCy load). Vercel keeps instances warm
  between requests, but the first request after idle is slow.
- Local measurements were taken on Windows; Vercel's Linux runtime may differ
  slightly in memory and latency.