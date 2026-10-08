# WriteTrace — Phase 4 Report: Inference Engine + FastAPI Backend

Generated: `2026-10-08T11:15:30.454459+00:00` · artifact `hgb_syn` in `ml/models/final-v1/` · pipeline `phase3_v1` · every number below is read from executed logs (`logs/phase4_*.json`, `logs/pytest.log`) and the locked artifact at render time.

> **Protocol statement.** Phase 4 performs **no training, tuning, calibration, or threshold selection**. It consumes the Phase-3 artifact read-only, reproduces the validated preprocessing → features → model → calibrator → gate path in exactly one module (`ml/inference.py`), and adds only transport, validation, and presentation mapping. The Phase-1/2/3 artifacts and reports are unchanged (verified in Section 3).

## 1. Phase 4 objective

Turn the locked Phase-3 artifact into a reliable, reusable inference system with one authoritative path:

```text
raw user text
  → validation (Pydantic, safe 422s)
  → training-compatible preprocessing (ml/preprocessing/normalize)
  → training-compatible feature extraction (73 v1 + 16 syn features)
  → Phase-3 final model (hgb_syn)
  → Phase-3 calibrator (platt)
  → machine-generated probability p ∈ [0, 1]
  → evidence / abstention gate (min_words = 50)
  → 5-level presentation label + warnings
  → explainability signals (local reference perturbations)
  → structured, versioned API response
```

In scope and shipped in this phase:

| Component | File | Lines | Role |
|---|---|---|---|
| Inference engine | `ml/inference.py` | 438 | the ONE authoritative path; dataclasses; load-once predictor |
| API schemas | `backend/schemas.py` | 161 | Pydantic request/response/error models |
| FastAPI app | `backend/app.py` | 229 | endpoints, lifespan load-once, safe exception handlers |
| Measurement | `ml/experiments/phase4.py` | 235 | reference check + latency benchmark helpers |
| Scripts | `scripts/{verify,bench,smoke}_phase4.py` | 297 | produce logs/phase4_{reference,bench,smoke}.json |
| Tests | `tests/test_phase4_{inference,api}.py` | 682 | inference + API coverage (existing tests untouched) |
| API docs | `docs/api.md` | 305 | endpoint/schema/label/limitation documentation |

Explicitly NOT done (Phase-4 stop condition): Astro frontend, paper/editorial UI, animations, audio, deployment, segment-level model training, authentication, database, external/paid APIs.

## 2. Phase 3 model selected

- **Winner:** `hgb_syn` — hist_gradient_boosting (learning_rate=0.1, max_leaf_nodes=63, max_iter=300, random_state=42), trained on 155988 MAGE-train rows (seed 42).
- **Features:** 89 (73 base + 16 syntactic/spaCy); selected/calibrated on val_tune 31323 / val_cal 20673 rows.
- **Calibrator:** platt · **threshold:** 0.5 (Phase-1 fixed, never tuned) · **label policy:** canonical y=1 MACHINE (MAGE label complemented); calibrators fit in canonical space; p = cal(model(x)).
- **Data policy:** {'trained_on': 'MAGE-train only', 'selected_on': 'MAGE-val_tune (+ E4 cal_fit/cal_eval inside val_cal)', 'calibrated_on': 'MAGE-val_cal', 'eval_only': ['MAGE-test', 'OOD-GPT', 'OOD-GPT-para', 'HC3', 'RAID'], 'threshold': '0.5 fixed from Phase 1, never tuned'}
- **Selection order:** robustness_generalization → calibration → false_positive_behavior → pr_auc → interpretability → computational_practicality → in_distribution_performance; LODO override applied = False.

Held-out evaluation recorded in the artifact (never used for training/tuning):

| Set | n | ROC-AUC | PR-AUC | F1-macro | Acc | Brier | ECE-10 | human FPR |
|---|---|---|---|---|---|---|---|---|
| test | 51995 | 0.9618 | 0.9799 | 0.8856 | 0.8965 | 0.0739 | 0.0058 | 0.1523 |
| ood_gpt | 1562 | 0.6958 | 0.6416 | 0.6281 | 0.6287 | 0.2601 | 0.1950 | 0.3136 |
| ood_gpt_para | 2362 | 0.6165 | 0.7223 | 0.5111 | 0.5135 | 0.3450 | 0.3339 | 0.3136 |
| hc3 | 64952 | 0.8633 | 0.8271 | 0.7222 | 0.7223 | 0.1886 | 0.1636 | 0.3774 |
| raid_clean | 15843 | 0.8490 | 0.8678 | 0.7616 | 0.7618 | 0.1694 | 0.0941 | 0.2659 |

## 3. Artifact/version provenance

| Property | Value |
|---|---|
| artifact directory | `ml/models/final-v1/` (read-only) |
| artifact_complete | True |
| pipeline_version | phase3_v1 |
| artifact written | 2026-10-08T09:36:17.620839+00:00 |
| hashed files | 9 |
| API identity | `model_name=hgb_syn` served as `model_version=final-v1` (directory name) |
| Git tag | `phase-3-complete` @ `c92f59e20538` |
| baseline-v1 / phase1-3 outputs vs tag | unchanged (empty git diff on protected paths) |

Manifest file hashes (sha256, first 16 chars):

| File | sha256[:16] | bytes |
|---|---|---|
| `model.joblib` | `55a10a397de29698` | 2283992 |
| `calibrator.joblib` | `e885c27bd79e4f9e` | 929 |
| `feature_state.joblib` | `2500fbc1edcece4c` | 3469467 |
| `features.json` | `0c559a7a809986cf` | 4930 |
| `reference.json` | `2a388328bec44554` | 9565 |
| `importance.json` | `22b2b675bdc050cd` | 3403 |
| `config.json` | `f5857be987fb198d` | 591 |
| `calibration.json` | `5347625dc1e94721` | 3448 |
| `gate.json` | `7933e235136ef6cf` | 1505 |

Working-tree changes in this phase (18 paths):

```text
README.md
backend/__init__.py
backend/__main__.py
backend/app.py
backend/schemas.py
docs/api.md
docs/experiments/phase4-report.md
ml/experiments/phase4.py
ml/experiments/phase4_report_generator.py
ml/explain.py
ml/features/syntactic.py
ml/inference.py
ml/requirements.txt
scripts/bench_phase4.py
scripts/smoke_phase4.py
scripts/verify_phase4.py
tests/test_phase4_api.py
tests/test_phase4_inference.py
```

## 4. Inference pipeline

One implementation, used by every entry point (API, tests, scripts):

| Step | Code | Notes |
|---|---|---|
| validate | `backend/schemas.py` `AnalyzeRequest` | 1–30000 chars, ≥1 non-whitespace; wrong type/malformed JSON → 422 |
| normalize | `ml/inference.Predictor._normalize` | `normalize_text` — byte-identical to the training normalizer |
| extract | `Predictor._extract_row` | v1 feature extractor + cached spaCy syntactic family; deterministic |
| score | `Predictor._score_row` | `p = calibrator(model.predict_proba(x)[:, 1])` — canonical, no complement |
| gate | `Predictor.analyze` | min_words=50 → abstain; mixed band [0.4, 0.6] → warning |
| label | `presentation_label` | documented presentation policy, separate from the binary decision |
| explain | `ml/explain.local_contributions` | one-at-a-time reference perturbations (k signals) |
| map | `backend/app._build_response` | dataclass → Pydantic; probabilities rounded to 6 dp only here |

Result types (`ml/inference.py`): `TextAnalysis`, `FeatureEvidence`, `SegmentAnalysis` (reserved), `PredictionResult` — framework-free, so the API layer adds no ML logic.

- **Load once:** `get_predictor()` returns a process-wide singleton; the FastAPI lifespan loads it and runs `warmup()` before serving (startup cost in Section 11). A test asserts `load_final` runs exactly once across requests.
- **Deterministic:** same text → same features → same probability; verified in-process, across requests, and across OS processes (`test_clean_process_loading_reproduces_probability`).
- **Legacy contract preserved:** `predict()` (4 dp, legacy labels) and `explain()` still behave exactly as the Phase-3 tests expect.

## 5. Preprocessing parity

Phase 4 does not re-derive features independently; it calls the same modules the experiments called (`ml/preprocessing/normalize.py`, `ml/features/extract.py`, `ml/features/syntactic.py`). Parity is then proven numerically against the **Phase-3 cached feature path** (the oracle that produced the locked metrics):

| Check | Result |
|---|---|
| rows compared | 10 (seed 0, 5 human + 5 machine from MAGE-test) |
| feature source | Phase-3 cached parquet + syn cache vs live `Predictor.analyze` extraction |
| max abs(Δp) | `0.000e+00` (tolerance 1e-09) |
| within tolerance | True |
| normalization | same `normalize_text`; idempotent on `text_norm` (asserted in tests) |
| E7 cross-check | recorded / recomputed / Phase-4 p all 0.9998 → match = True |

A max delta of 0.0 on every compared row means preprocessing, feature order, and dtype handling are byte-for-byte equivalent; per-row evidence is in Section 13.

## 6. Calibration integration

- Calibrator `platt` from the artifact (`calibrator.joblib`), applied as `cal.predict(model_proba[:, 1])` — the canonical `p = cal(model(x))` from `config.json` (`label_policy`).
- Calibrator *type* was chosen in Phase-3 E4 on a group-disjoint val_cal split: Brier platt 0.0738 vs isotonic 0.0741 (chosen = platt). Phase 4 re-uses it unchanged.
- Test-set calibration as recorded in the artifact: Brier 0.0739, ECE-10 0.0058.
- **No orientation complement:** labels are canonical (y=1 MACHINE), so `human_probability = 1 − machine_probability` is exact; tests assert `machine + human == 1` within 1e-9.
- **Rounding policy:** decisions always use the full-precision p; the API rounds to 6 dp for presentation; the legacy `predict()` path rounds to 4 dp (parity deltas therefore ≤ 4e-5 by construction; the raw-vs-raw reference check in Section 13 is exactly 0).
- Tests: probability bounds, complement consistency, no NaN/∞, monotone bounded calibrator (Phase-2 suite), determinism.

## 7. Evidence/abstention implementation

| Gate property | Value (from `gate.json`) |
|---|---|
| status | active |
| min_words | 50 |
| threshold | 0.5 |
| abstain fraction (test) | 0.0933 |
| abstain fraction cap | 0.15 |
| mixed band | [0.4, 0.6] |
| mixed-band accuracy (val_cal) | 0.5342 |

Behavior for `word_count < 50`:

- HTTP **200** (not a rejection): `status=insufficient_evidence`, `label=insufficient_evidence`, `decision=abstain`, `confidence=null`, `evidence_sufficient=false`.
- `word_count`, `text_length`, gate metadata, and warnings are still returned; the probability is shown for transparency **with a warning that it must not support an authorship conclusion**.
- The binary decision threshold (0.5) and the mixed band are read from the artifact, never re-derived here.

Observed through the live API (from `logs/phase4_smoke.json`):

| Case | words | label | decision | confidence | warnings |
|---|---|---|---|---|---|
| short_under_gate | 42 | `insufficient_evidence` | abstain | None | 1 |
| normal_human_text | 118 | `likely_human` | human | 0.9381 | 0 |

Tests: `test_short_text_abstains_with_full_metadata`, `test_short_text_returns_insufficient_evidence`, `test_gate_values_match_phase3_report`.

## 8. Explainability implementation

- **Global importance:** permutation importance (mean PR-AUC drop over n_repeats shuffles on a seeded val_tune sample); SHAP/xgboost/lightgbm not installed - documented deviation, no fabricated values (n_rows=8000, n_repeats=5, baseline PR-AUC 0.979). Top features by |Δ PR-AUC|:

| Feature | Δ PR-AUC |
|---|---|
| `log_ttr` | 0.06135 |
| `root_ttr` | 0.05207 |
| `trigram_repeat_rate` | 0.01109 |
| `mean_log_df` | 0.01076 |
| `unique_bigram_ratio` | 0.00886 |
| `n_sentences` | 0.00682 |
| `std_sent_len` | 0.00269 |
| `syn_pos_adv_ratio` | 0.00261 |

- **Local explanation:** one-at-a-time reference perturbation (`ml/explain.local_contributions`) against train reference stats — batched for speed and verified equal to the sequential definition (`test_local_contributions_batched_equals_reference`).

| Signal field | Meaning |
|---|---|
| signal / label / family | feature id, human label, family |
| value | observed value in this text |
| reference_median | train-corpus median anchor |
| contribution | signed local contribution to p(machine) |
| direction | `machine` / `human` / `neutral` |
| interpretation | plain-language value vs reference sentence |
| caveat | `Statistical stylistic signal measured by the model; not proof of authorship.` |

Forbidden phrasing is enforced by a test (`test_signals_schema_and_no_causal_language`): no 'proves', no authorship claims, no causal attribution. Global explanations always carry the fixed disclaimer:

> WriteTrace does not establish authorship. It estimates how strongly observed writing patterns resemble machine-generated or human-written text under the model's training and evaluation conditions.

## 9. API design

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/analyze` | analyse one text |
| GET | `/api/v1/health` | liveness + loaded model identity |
| GET | `/api/v1/info` | static metadata (gate, labels, eval, data policy, top features) |

Request:

| Field | Type | Default | Bounds / notes |
|---|---|---|---|
| `text` | string | required | 1–30000 chars; ≥1 non-whitespace |
| `include_signals` | bool | true | local signals on/off |
| `max_signals` | int | 8 | 1–50 |

Response groups: `status`, `label`, `decision`, `machine_probability`, `human_probability`, `confidence`, `word_count`, `text_length`, `evidence_sufficient`, `mixed_low_confidence`, `signals[]`, `segments[]` (always empty in v1), `gate`, `model_version`, `model_name`, `calibrator`, `threshold`, `warnings[]`, `disclaimer`.

Label vocabulary generated from the implemented policy (`presentation_label`, outer cutoff 0.2):

| Probe | label |
|---|---|
| p=0.05, suffices=True | `likely_human` |
| p=0.3, suffices=True | `mostly_human` |
| p=0.5, suffices=True | `mixed` |
| p=0.7, suffices=True | `mostly_machine` |
| p=0.95, suffices=True | `likely_machine` |
| p=0.95, suffices=False | `insufficient_evidence` |

Design rules: versioned namespace `/api/v1`; probabilities rounded to 6 dp after full-precision decisions; `human_probability` derived as `1 − machine` so the pair is always consistent; no filesystem paths, stack traces, model objects, or debug detail in any response; submitted text is never logged, persisted, or transmitted (`test_submitted_text_is_never_logged`). Full contract: `docs/api.md`.

Startup:

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

## 10. Validation and error handling

Real-server smoke cases (from `logs/phase4_smoke.json`, server `127.0.0.1:8017`, ready = True):

| Case | Method | Expected | Got | OK | ms |
|---|---|---|---|---|---|
| short_under_gate | POST | 200 | 200 | True | 59.5 |
| normal_human_text | POST | 200 | 200 | True | 50.8 |
| machine_reference_text | POST | 200 | 200 | True | 36.2 |
| unicode_mixed_language | POST | 200 | 200 | True | 38.0 |
| repeated_text | POST | 200 | 200 | True | 40.8 |
| malformed_json | POST | 422 | 422 | True | 1.3 |
| empty_text | POST | 422 | 422 | True | 1.2 |
| whitespace_only | POST | 422 | 422 | True | 1.2 |
| wrong_type | POST | 422 | 422 | True | 1.5 |
| missing_field | POST | 422 | 422 | True | 1.5 |
| oversized_text | POST | 422 | 422 | True | 1.2 |
| analyze_via_get | GET | 405 | 405 | True | 1.1 |
| unknown_route | POST | 404 | 404 | True | 1.8 |
| health | GET | 200 | 200 | True | 1.8 |

Error contract (no input echo, no paths, no traces):

```json
{"status": "error", "error": {"code": "...", "message": "...",
 "details": [{"field": "...", "message": "...", "type": "..."}]}}
```

| Status | code | Trigger |
|---|---|---|
| 404 | `not_found` | unknown path |
| 405 | `method_not_allowed` | wrong HTTP method |
| 422 | `validation_error` | empty / whitespace-only / >30,000 chars / wrong type / missing field / malformed JSON / out-of-range `max_signals` |
| 500 | `internal_error` | unexpected failure - details never exposed |
| 200 | — | short text returns abstention, not an error |

Handlers: `RequestValidationError` → sanitized 422 (field + message only, offending value dropped), `StarletteHTTPException` → mapped codes, catch-all `Exception` → generic 500. Warnings (HTTP 200): insufficient evidence · mixed band (with the 0.5342 accuracy note) · non-English heuristic. Tests cover every row above plus the no-echo and no-path-leak guarantees.

## 11. Performance and latency

| Environment | Value |
|---|---|
| python | 3.13.15 |
| scikit-learn | 1.9.0 |
| platform | Windows-11-10.0.26200-SP0 |
| CPU count | 16 |
| device | cpu |
| artifact | final-v1 |
| runs per bucket | 20 |

| Phase | Milliseconds |
|---|---|
| cold load (model + calibrator + feature state) | 646.8 |
| startup warm-up incl. spaCy pipeline load (once per process) | 4232.3 |

Warm in-process latency (median / p95 ms):

| Bucket | words | chars | analyze + signals med/p95 | analyze no-signals med/p95 |
|---|---|---|---|---|
| short_under_gate | 42 | 227 | 26.2 / 29.0 | 19.4 / 21.5 |
| typical_human | 118 | 718 | 43.7 / 46.4 | 37.2 / 38.9 |
| typical_machine | 51 | 278 | 30.5 / 32.6 | 22.9 / 24.2 |
| long | 464 | 2644 | 106.9 / 109.3 | 99.6 / 105.0 |

End-to-end HTTP (smoke, 5 successful analyze requests): median `40.8 ms`, max `59.5 ms`.

Notes: CPU-only (₹0 budget, no paid/external APIs); the model is never reloaded per request (asserted by `test_model_not_reloaded_per_request`); signals cost a few milliseconds; the one-time spaCy load is paid at server startup, not by the first user request.

## 12. Test results

Executed: `python -m pytest tests/ -q` → **128 passed, 2 warnings** (`logs/pytest.log`).

| Suite | Test functions | Note |
|---|---|---|
| Phase 1–3 (existing) | 83 test cases | untouched; baseline was 83 passed / 2 warnings |
| `tests/test_phase4_inference.py` | 19 | artifact load, determinism, gate, labels, signals, parity, clean-process load |
| `tests/test_phase4_api.py` | 17 | schema, validation matrix, safety, load-once, privacy |
| total (executed) | 128 passed | new Phase-4 cases = 45 |

Requirement → test mapping (Phase-4 spec §12):

| Requirement | Test(s) |
|---|---|
| model / calibrator / feature-state loading | `test_all_artifact_components_load` |
| deterministic inference | `test_deterministic_inference_full_precision`, `test_deterministic_across_requests`, `test_clean_process_loading_reproduces_probability` |
| probability bounds / complement / no NaN | `test_probability_bounds_and_human_complement` |
| abstention + short text | `test_short_text_abstains_with_full_metadata`, `test_short_text_returns_insufficient_evidence` |
| empty / wrong type / malformed | `test_empty_and_invalid_text_raise`, `test_validation_errors_are_safe_422`, `test_malformed_json` |
| Unicode / multiline / punctuation | `test_unicode_and_multiline_inputs` |
| API + explanation schema | `test_analyze_response_schema`, `test_signals_in_response`, `test_signals_schema_and_no_causal_language` |
| model version reporting | `test_info_endpoint_exposes_metadata_without_paths`, `test_health_endpoint` |
| model loaded only once | `test_get_predictor_is_singleton_and_loads_once`, `test_model_not_reloaded_per_request` |
| reference predictions vs Phase 3 | `test_reference_parity_vs_phase3_cached_path`, `test_report_worked_example_reproduced` |
| privacy (no text logging) | `test_submitted_text_is_never_logged` |

Existing tests were not modified or weakened; `git status` in Section 3 shows no Phase-1/2/3 test file changed. The 2 warnings are the pre-existing sklearn imputer warnings from `test_pipeline_smoke`.

## 13. Reference prediction checks

Oracle = Phase-3 cached-feature predictions (the numbers the locked metrics were computed from); Phase-4 = the inference engine's own extraction + scoring on raw text.

| Row | domain | truth | words | p (Phase 3 cached) | p (Phase 4) | Δ |
|---|---|---|---|---|---|---|
| 193525 | sci_gen | human | 99 | 0.162009 | 0.162009 | 0.0e+00 |
| 182232 | squad | human | 156 | 0.074686 | 0.074686 | 0.0e+00 |
| 160384 | wp | human | 168 | 0.559102 | 0.559102 | 0.0e+00 |
| 163777 | yelp | human | 221 | 0.044429 | 0.044429 | 0.0e+00 |
| 246471 | eli5 | human | 132 | 0.014461 | 0.014461 | 0.0e+00 |
| 196225 | eli5 | machine | 124 | 0.657595 | 0.657595 | 0.0e+00 |
| 90378 | roct | machine | 295 | 0.999930 | 0.999930 | 0.0e+00 |
| 115025 | yelp | machine | 125 | 0.995691 | 0.995691 | 0.0e+00 |
| 229544 | eli5 | machine | 83 | 0.971025 | 0.971025 | 0.0e+00 |
| 107788 | yelp | machine | 150 | 0.998470 | 0.998470 | 0.0e+00 |

**max |Δ| = `0.000e+00` across 10 rows (tolerance 1e-09) → all_within_tolerance = True.**

Orientation check over the full locked test split (canonical y=1 MACHINE):

| Statistic | Human rows | Machine rows |
|---|---|---|
| n | 18046 | 33949 |
| mean p | 0.2048 | 0.8854 |
| median p | 0.0839 | 0.9857 |

separated (mean machine − mean human > 0.3): True.

E7 worked example (highest-confidence human FP on MAGE-test):

| Quantity | E7 recorded | Recomputed | Phase 4 |
|---|---|---|---|
| p(machine) | 0.9998 | 0.9998 | 0.9998 |
| words | 78 | 78 | — |
| domain | eli5 | eli5 | — |

match = **True** — this also documents an honest false positive (a human text the model scores as machine-like); it is reported, not hidden.

## 14. Known limitations

| Limitation | Evidence (from artifacts / live runs) |
|---|---|
| OOD degradation | OOD-GPT ROC 0.6958, OOD-GPT-para 0.6165, HC3 human FPR 0.3774, RAID-clean ROC 0.8490 |
| mixed band is near chance | accuracy 0.5342 inside [0.4, 0.6] → flagged, never silently classified |
| short text under-determined | gate abstains below 50 words (0.0933 of test) |
| repetition-heavy / formulaic text | smoke: repeated filler text scored 0.99972 → machine (a known feature-style failure mode) |
| non-English input | warned, not rejected; model was trained/evaluated on English only |
| confidence is derived | `confidence = max(p, 1-p)` — not a separately trained confidence model; null when abstaining |
| presentation cutoffs | outer 0.2/1-0.2 label split is a documented convention, never fitted; decision stays at 0.5 |
| service surface | no auth, no rate limit, no persistence — local inference only; put your own controls in front before exposure |
| statistics | single seed, no confidence intervals; CPU-only measurements on one machine |

Not an authorship test in any legal or forensic sense — the fixed disclaimer on every response states this.

## 15. Segment-analysis readiness

No segment model was trained in Phase 4 (per the stop condition), but the pipeline is already shaped for it:

- `SegmentAnalysis` dataclass exists with `status='not_available'`, `count=0`, and an explanatory note.
- `PredictionResult.segments` and the API `segments[]` field are already in the contract (always `[]` in v1) — Phase 5 can populate them without breaking clients.
- Normalization, feature extraction, scoring, gating, and explanation live in private `Predictor` methods (`_normalize` / `_extract_row` / `_score_row`) that operate on a single text — segment scoring can reuse them verbatim instead of re-implementing the pipeline.
- `TextAnalysis` already measures document-level facts (characters/words/sentences/paragraphs) that windowing will need.
- The gate is document-level today; Phase 5 must decide and **document** whether segment-level outputs carry the same 50-word gate or a validated alternative (a new artifact, not a silent change).

## 16. Phase 5 handoff

What Phase 5 inherits:

| Item | State |
|---|---|
| API contract | `/api/v1/*` stable and documented in `docs/api.md`; keep `/api/v1` frozen, extend additively |
| Artifact | `ml/models/final-v1/` read-only, hash-verified; never retrain/edit in place |
| Inference | `ml/inference.Predictor` is the single path — the frontend must not re-implement scoring |
| Tests | 128 passing; keep green; never weaken |
| Logs | `logs/phase4_{reference,bench,smoke}.json` + `logs/pytest.log` are the provenance for this report |
| Privacy | local-first/stateless — no DB, auth, telemetry may be added without an explicit decision |

Recommended Phase-5 order:

1. **Segment-level model** (the remaining ML work): reuse `_normalize`/`_extract_row`/`_score_row` on paragraph windows, train as a *new* versioned artifact (e.g. `ml/models/segments-v1/`), document the gate policy, then populate `segments[]`.
2. **Astro frontend** consuming `POST /api/v1/analyze` exactly as documented (label vocabulary, abstention, mixed-band and disclaimer semantics must be rendered honestly).
3. **Paper/editorial UI + deployment** last, behind explicit consent about false positives (Section 14).

```bash
# reproduce everything in this report:
python scripts/verify_phase4.py && python scripts/bench_phase4.py
python scripts/smoke_phase4.py --port 8017
python -m pytest tests/ -q
python ml/experiments/phase4_report_generator.py
```

---

## PHASE 4 FINAL REPORT

- **Status:** Phase 4 complete — inference engine + FastAPI backend live; frontend/deployment/segment model deliberately NOT started.
- **Model:** `hgb_syn` (hist_gradient_boosting, lr=0.1, leaves=63, iters=300, seed=42) on 89 features, calibrator platt, threshold 0.5 — consumed read-only from `ml/models/final-v1/` (9 hashed files, artifact_complete=True).
- **Architecture:** `backend/app.py` → `ml/inference.Predictor` (single path: normalize → extract → model → calibrator → gate → label → signals); process-wide singleton with startup warm-up (4232.3 ms) so requests never reload anything (cold load 646.8 ms).
- **API:** `POST /api/v1/analyze` (+ `/health`, `/info`), versioned v1, Pydantic-validated, safe 422/404/405/500 envelopes, no input echo / paths / traces; documented in `docs/api.md`.
- **Abstention:** gate min_words=50 (abstain 0.0933 of test; excluded F1m worse than kept), 200 + `insufficient_evidence` below the gate, mixed band [0.4, 0.6] (accuracy 0.5342) flagged; `confidence=null` on abstain.
- **Explainability:** permutation importance (Δ PR-AUC) + local reference perturbations; every signal carries the non-causal caveat; no causal/authorship language (tested).
- **Latency (CPU):** cold load 646.8 ms, startup warm-up 4232.3 ms, warm analyze medians short_under_gate 26.23 ms, typical_human 43.72 ms, typical_machine 30.52 ms, long 106.94 ms.
- **Tests:** 128 passed, 2 warnings (baseline 83 all still green; +45 Phase-4 cases incl. clean-process loading and privacy).
- **Smoke (real server):** 14/14 cases passed; end-to-end median 40.8 ms over successful analyze requests.
- **Reference checks:** max |Δp| = `0.000e+00` vs Phase-3 cached path over 10 rows (tolerance 1e-09); E7 worked example matches (p=0.9998, 78 words); orientation separated (mean p machine 0.8854 vs human 0.2048).
- **Integrity:** `phase-3-complete` tag intact (`c92f59e20538`); git diff on baseline-v1/final-v1/phase1-3 reports = empty (unchanged).
- **Files changed:** 18 (see Section 3); temporary probe files removed; nothing committed.
- **Known limitations:** OOD ROC 0.6958/0.6165, HC3 human FPR 0.3774, mixed band ≈ chance, non-English unvalidated, no auth/rate limits, single-seed CPU measurements.
- **Phase 5 recommendation:** (1) segment-level model as a new versioned artifact reusing `Predictor` internals, then populate `segments[]`; (2) Astro frontend against the frozen `/api/v1` contract in `docs/api.md`; (3) deployment last. Do not retrain or edit `final-v1` in place.
