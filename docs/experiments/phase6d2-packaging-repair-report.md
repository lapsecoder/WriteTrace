# Phase 6D.2 — Safe Vercel Packaging Repair Report

**Project:** WriteTrace · `D:\WriteTrace`
**Date:** 2026-10-10
**Scope:** local packaging repair and verification only. **`vercel` was not
invoked. Nothing deployed, committed, tagged, or pushed. No credential was read,
printed, copied, or modified.**

Repairs the two defects that caused the failed first deployment attempt.

---

## 1. Security gate

| Requirement | Status |
| ----------- | ------ |
| No token printed/read/copied/logged | Met. `.env.local` contents were **never read**; `.vercel/project.json` was read but contains only `projectId`/`orgId`/`projectName`. |
| Previously exposed refresh token | **Requires user action.** Must be revoked manually via Vercel → Account Settings → Tokens **before** any subsequent authenticated deployment. This report does not and cannot revoke it. |
| No automatic credential rotation | Met — nothing was attempted. |
| No credentials in deployment artifacts | Verified: `.env.local` and `.vercel/` are in `.gitignore` and `.vercelignore`; Vercel's own `--dry` resolver listed `.env.local` as **IGNORED** during the failed attempt. |

⚠️ **A credential is still live.** The refresh token exposed in the previous
phase remains valid until the user revokes it. Any authenticated deployment
attempt should wait for that revocation.

---

## 2. The two defects

The first deployment failed with:

```
Error: Invalid vercel.json - Invalid request: `functions.app.py.excludeFiles`
should NOT be longer than 256 characters.
```

### Defect 1 — pattern too long (269 / 256 chars)

### Defect 2 — the pattern deleted required runtime code

`excludeFiles` contained `ml/experiments/**`. But `ml/explain.py` and
`ml/inference.py` both import `ml.experiments.phase3`, which imports `phase2`,
`ml.training.train`, and more. Proven empirically by pruning a copy and
importing it:

```
File ".../ml/explain.py", line 26, in <module>
    from .experiments.phase3 import FINAL_DIR, load_json
ModuleNotFoundError: No module named 'ml.experiments'
```

**Had the length been legal, the function would have built and then failed on
every request.** Vercel's length check turned a silent total outage into a safe
rejection.

This is the most important finding of the phase: the Phase 6D bundle-content
verification read the pattern as *text* and passed, without ever applying it to
the import closure.

---

## 3. The repaired pattern

```
{docs/**,tests/**,scripts/**,logs/**,frontend/src/**,frontend/node_modules/**,frontend/.astro/**,ml/models/segments-v1/**,ml/models/baseline-v1/**,**/__pycache__/**,**/*.pyc,**/.pytest_cache/**}
```

**Length: 194 / 256 characters** (62 to spare).

Removed:

| Removed | Reason |
| ------- | ------ |
| `ml/experiments/**` | **Broke the import closure** — the defect |
| `ml/data/raw/**`, `ml/data/processed/**`, `ml/data/manifests/**` | Never uploaded anyway (`.vercelignore`); repeating them was what pushed the pattern over 256 |

Kept (genuinely unnecessary at runtime): docs, tests, scripts, logs, frontend
build inputs, the two research-only models, and Python caches.

---

## 4. Division of labour: `.vercelignore` vs `excludeFiles`

These are **not** interchangeable and were treated separately:

| Mechanism | Stage | Scope | Handles here |
| --------- | ----- | ----- | ------------ |
| `.vercelignore` + `.gitignore` | **CLI upload** | what leaves the machine | datasets, `segments-v1`, `baseline-v1`, caches, `node_modules`, `dist`, `.env*`, `.vercel/`, logs |
| `vercel.json` `excludeFiles` | **function bundle** | what stays in the bundle | docs, tests, scripts, frontend build inputs, caches, research models |

Because `segments-v1` / `baseline-v1` / datasets are already absent from the
upload, listing them in `excludeFiles` is defence-in-depth rather than load-bearing.

**`.vercelignore` gained `.vercel/`** this phase — the CLI's local link
directory was not previously covered by name.

---

## 5. Required runtime closure (computed, not assumed)

Method: AST closure from root `app.py`, **including imports inside function
bodies** (spaCy, scipy and torch are lazily imported), plus package
`__init__.py` files. Stored at `tests/fixtures/runtime-closure.json`, which is
regenerated from live source by the phase script.

**29 local modules**, including:

```
backend/app.py, backend/schemas.py
ml/inference.py, ml/explain.py, ml/segmentation.py
ml/experiments/phase2.py, ml/experiments/phase3.py     <- the previously-excluded ones
ml/training/train.py, ml/data/download.py
ml/features/{extract,syntactic,lexicons}.py
ml/preprocessing/{normalize,build_dataset,dedup,minhash,split}.py
ml/evaluation/{calibration,metrics}.py
ml/configs/phase1.py
```

Directories that must **never** be excluded wholesale:
`backend`, `ml`, `ml/configs`, `ml/data`, `ml/evaluation`, **`ml/experiments`**,
`ml/features`, `ml/preprocessing`, **`ml/training`**.

Runtime data: **10 `final-v1` artifacts** + 5 `ml/configs/*.json`, all present
and all not excluded.

---

## 6. Regression tests — `tests/test_phase6d2_packaging.py` (23 tests)

These **apply the exclusion logic to real file paths**, not just assert glob
text — the exact blind spot that let the defect through.

| # | Requirement | Test |
| - | ----------- | ---- |
| 1 | `excludeFiles` ≤ 256 chars | `test_exclude_files_within_documented_limit` |
| 2 | Valid schema keys | `test_only_documented_top_level_keys`, `test_only_documented_function_keys` |
| 3 | Required modules not excluded | `test_no_required_module_is_excluded`, `test_no_required_data_file_is_excluded`, `test_required_directories_not_wholly_excluded` (parametrised over all 9 required dirs) |
| 4 | Closure present | `test_closure_fixture_matches_source`, `test_closure_covers_entrypoint` |
| 5 | `final-v1` complete | `test_final_v1_present_and_complete` (names all 10 artifacts) |
| 6 | Datasets/research models/secrets absent | `test_datasets_and_research_models_not_in_function_bundle`, `test_env_and_credentials_never_in_upload`, `test_vercelignore_excludes_datasets_and_caches` |
| 7 | Size/file budget | `test_bundle_within_documented_500mb_limit`, `test_project_file_count_reasonable` |

**Proof the tests catch the defect:** reverting to the broken pattern fails
**3 tests** —

```
FAILED test_exclude_files_within_documented_limit      (269 chars)
FAILED test_no_required_module_is_excluded
FAILED test_required_directories_not_wholly_excluded[ml/experiments]
```

---

## 7. Real pruned-tree verification

A tree was built by applying `.gitignore` + `.vercelignore`, then
`excludeFiles`, then exercised in a **clean process** (`uvicorn app:app`,
`cwd` = tree, nothing from the repo on `sys.path`).

```
uploaded files : 156
bundle files   : 82
bundle bytes   : 7,604,528 (7.60 MB) project source
required closure files missing from tree : NONE
```

**13/13 checks passed:**

| Check | Result |
| ----- | ------ |
| Clean process imports `app:app` from the pruned tree | PASS |
| **`final-v1` loads** (`hgb_syn`) | PASS |
| `/api/v1/info` 200 | PASS |
| Normal analysis 200, 8 signals | PASS |
| `segments[]` empty | PASS |
| Disclaimer present | PASS |
| Short input abstains (`confidence: null`) | PASS |
| Invalid input → 422 | PASS |
| Unknown route → **404** | PASS |
| Frontend served (`/`) | PASS |
| Every required module imports (`ml.inference`, `ml.explain`, `ml.experiments.phase3`, `ml.training.train`, …) | PASS |
| No required module removed | PASS |

`.py` files removed by `excludeFiles` were **only** `scripts/*` (14) and
`tests/*` (16) — all confirmed outside the import closure.

### Post-build simulation

The first run failed "unknown route → 404". Cause: `frontend/dist` is a build
artifact (`.gitignore` excludes it from upload; `buildCommand` regenerates it
on Vercel), so a local pruned tree legitimately lacks it. Per FastAPI's
documented behaviour — *"If the configured directory is still missing when a
request is handled, FastAPI will raise an error then"* — the route check
failed. The tree was then populated with the built `frontend/dist` to model the
real post-build bundle, after which routing behaved correctly.

---

## 8. Test results

| Suite | Result |
| ----- | ------ |
| `python -m pytest tests/ -q` | **226 passed**, 2 warnings |
| `npm test` | **133 passed** |
| `tsc --noEmit` | no errors |
| `astro check` | 0 errors / 0 warnings / 0 hints |
| `npm run build` | 1 page built |
| Pruned-tree clean-process | **13/13** |

Two tests in earlier phases (`test_phase6b_deployment_config`,
`test_phase6d1_frontend_routing`) **correctly failed** because they asserted the
superseded pattern. They were updated to assert the *correct division of
labour* (research models in `excludeFiles`, datasets in `.vercelignore`) and to
assert `ml/experiments/**` is **absent** — strengthened, not weakened.

---

## 9. Integrity

| Check | Result |
| ----- | ------ |
| `ml/models/` dirty files | **0** |
| `final-v1/model.joblib` SHA-256 | `55a10a39…6d13800` — **matches** Phase 6C |
| Checkpoint tags | all 4 intact |
| HEAD | `889d4db`, unchanged; no commit/tag/push |
| Backend inference, API contract, thresholds | untouched (`backend/app.py` is the pre-existing Phase 6A CORS change) |
| `.gitignore` CLI changes | **intentional** — added `.vercel` and `.env*`; both correct and required |

No deployment exists: the attempt failed at config validation, before any
deployment was created. The Hobby project `writetrace` exists but has no
deployment and no production URL.

---

## 10. Size measurement method (stated explicitly)

**This is an UPLOAD-side estimate, not a measured Vercel bundle.**

```
estimated upload = real Linux wheel bytes (real `pip download` of every pin,
                   manylinux_2_28_x86_64 / py3.13: 121,081,817 B)
                 + on-disk bytes of closure + final-v1 + configs (7,604,528 B)
                 ≈ 128.7 MB  vs  the documented 500 MB Python bundle limit
```

Fixture: `tests/fixtures/linux-wheel-bytes.json`. Regenerating it is part of the
phase script, so the figure cannot silently drift.

---

## 11. Remaining risks and known limitations

1. **Vercel's bundler is not reproduced locally.** The pruned tree models
   `.gitignore`/`.vercelignore`/`excludeFiles` semantics, but Vercel's actual
   bundle assembly, `.pyc` compilation and `[tool.vercel.fastapi.static]`
   handling are **not** exercised.
2. **Linux runtime unverified.** All execution was Windows/Python 3.13.15.
   Wheel availability is proven; behaviour is not.
3. **CDN promotion untested.** Vercel's build-time promotion analysis was not
   run. Static assets currently pass through the function (top-level middleware
   disables promotion) — correct but slower.
4. **Build ordering untested.** `check_dir=False` removes the failure mode, but
   Vercel's import timing during build was not observed.
5. **Credential still live** until the user revokes the exposed refresh token.
6. **Blocker B2** — Hobby payment-method requirement remains unestablished.
7. **Research models are defence-in-depth only.** `segments-v1` /
   `baseline-v1` are excluded in both mechanisms; the bundle exclusion is a
   backstop, not the primary control.

---

## 12. Recommendation

**The project is safe to attempt a preview deployment**, subject to:

1. **Revoke the exposed refresh token first** (Account Settings → Tokens), then
   re-authenticate.
2. Expect a first build to take several minutes (pip install of ~121 MB of
   wheels, then `npm ci` + Astro build).
3. Verify on the preview URL that `/` serves the page and `/api/v1/health`
   returns **JSON, not HTML or 404** — that is the Phase 6D.0 question finally
   answerable.

If B1 passes, re-measure cold start and memory on the real Linux runtime and
update the Phase 6C report with measured figures rather than estimates.

---

*Local repair only. `vercel` was not invoked; nothing deployed, committed,
tagged or pushed. No credential was read, printed or modified.*