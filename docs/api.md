# WriteTrace API (v1)

Statistical estimate of how strongly a text's writing patterns resemble
machine-generated or human-written text. **WriteTrace does not establish
authorship.** It estimates how strongly observed writing patterns resemble
machine-generated or human-written text under the model's training and
evaluation conditions.

- Local-first and stateless: submitted text is never logged, persisted,
  cached, or sent anywhere. No database, no auth, no telemetry, no external
  API calls.
- One authoritative inference path: every request goes through
  `ml/inference.py` (`Predictor`), the same preprocessing → features → model →
  calibrator → gate code that produced the Phase-3 results
  (`docs/experiments/phase3-report.md`). The FastAPI layer adds only
  transport, validation, and presentation mapping.
- Model artifact: `ml/models/final-v1/` (`hgb_syn`, 89 features, Platt
  calibrator, threshold 0.5, gate 50 words).

## Quick start

```bash
# from the repository root (CPU, no external services)
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

- Interactive docs: `http://127.0.0.1:8000/docs`
- Startup loads everything expensive once (model, calibrator, feature
  state, spaCy pipeline, reference stats) — about **0.6 s + 4.2 s warm-up**
  (`logs/phase4_bench.json`); the first user request never pays that cost.

Smoke test against a real server:

```bash
python scripts/smoke_phase4.py --port 8017   # -> logs/phase4_smoke.json
```

## Endpoints

| Method | Path               | Purpose |
| ------ | ------------------ | ------- |
| POST   | `/api/v1/analyze`  | Analyse one text |
| GET    | `/api/v1/health`   | Liveness + loaded model info |
| GET    | `/api/v1/info`     | Static model/gate/label/version metadata |

### POST `/api/v1/analyze`

Request body (JSON):

| Field            | Type    | Default | Bounds | Meaning |
| ---------------- | ------- | ------- | ------ | ------- |
| `text`           | string  | required | 1–30,000 chars, ≥1 non-whitespace | Raw text to analyse (English) |
| `include_signals`| boolean | `true`  | — | Return local statistical signals (slower when true) |
| `max_signals`    | integer | `8`     | 1–50 | Maximum number of signals returned |

Unknown fields are ignored.

#### Response fields (200 OK)

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `status` | `"ok"` \| `"insufficient_evidence"` | `insufficient_evidence` when the 50-word evidence gate is not met (the response itself is still a normal 200) |
| `label` | string | Presentation label — see [Label vocabulary](#label-vocabulary) |
| `decision` | `"machine"` \| `"human"` \| `"abstain"` | Binary decision at threshold 0.5, or `abstain` below the gate |
| `machine_probability` | 0–1, 6 dp | Calibrated p(machine), `calibrator(model(x))` with y=1 = machine |
| `human_probability` | 0–1, 6 dp | Always `1 - machine_probability` |
| `confidence` | 0–1, 4 dp \| null | `max(machine, human)`; **null when evidence is insufficient**. Not a separately trained confidence model |
| `word_count` | int | Gate-relevant word count |
| `text_length` | object | `{characters, words, sentences, paragraphs}` |
| `evidence_sufficient` | bool | `word_count >= 50` |
| `mixed_low_confidence` | bool | Probability inside the Phase-3 mixed band `[0.4, 0.6]` |
| `signals` | list | Local statistical signals (only when `include_signals=true`) |
| `segments` | list | Always `[]` — reserved for Phase 5 |
| `gate` | object | `{min_words: 50, mixed_band: [0.4, 0.6]}` |
| `model_version` / `model_name` / `calibrator` / `threshold` | string/float | `final-v1` / `hgb_syn` / `platt` / `0.5` |
| `warnings` | list of string | Empty normally; see [Warnings](#warnings) |
| `disclaimer` | string | Fixed authorship disclaimer (below) |

### Label vocabulary

The model is binary (`threshold = 0.5`, fixed in Phase 1, never tuned).
Labels are an inference-time **presentation** vocabulary derived from the
probability — they do not add model capacity:

| Condition | `label` |
| --------- | ------- |
| `word_count < 50` | `insufficient_evidence` (regardless of p) |
| `p ∈ [0.4, 0.6]` (mixed band, from Phase 3) | `mixed` |
| `p < 0.2` | `likely_human` |
| `0.2 ≤ p < 0.4` | `mostly_human` |
| `0.6 < p ≤ 0.8` | `mostly_machine` |
| `p > 0.8` | `likely_machine` |

The outer 0.2/0.8 cut-offs are a documented, symmetric presentation
convention only — they were never fitted or evaluated as thresholds. The
binary `decision` field always uses 0.5; `mixed`/`insufficient_evidence`
never override it (they only set `decision: "abstain"` below the gate).
`confidence` is rounded to 4 dp; probabilities to 6 dp; all decisions are
computed at full internal precision before rounding.

### Gate, abstention, and short text

- **50 words** (`gate.min_words`) was selected in Phase 3 because excluded
  text (9.33% of val_cal) scored F1(macro) 0.7956 vs 0.8924 for kept text —
  abstaining is more honest than guessing.
- Text below the gate returns **200 with `status:
  "insufficient_evidence"`, `decision: "abstain"`, `confidence: null`** —
  it is *not* rejected. The probability is still shown for transparency and
  carries an explicit warning that it must not support an authorship
  conclusion.
- Empty / whitespace-only text, text over 30,000 chars, wrong types, and
  malformed JSON are **422** errors (below).

### Warnings

| Warning contains | Meaning |
| ---------------- | ------- |
| `insufficient evidence…` | Below the 50-word gate |
| `mixed band… 0.5342…` | p inside `[0.4, 0.6]`, where validation accuracy was 0.5342 — essentially a coin flip |
| `English` | `looks_english` heuristic says the text may not be English; the model was trained and evaluated on English text |

### Signals (explainability semantics)

Each signal is a **local statistical association**, not a causal or
forensic fact:

- `signal` / `label` / `family` — feature id, human label, and feature
  family (lexical, stylometric, structural, repetition, syntactic_spacy, …).
- `value` — the text's observed feature value.
- `reference_median` — train-corpus median used as the comparison anchor.
- `contribution` — signed local contribution to p(machine) under the
  reference-perturbation explanation (`ml/explain.py`); the listed signals
  approximately explain the score.
- `direction` — `machine` / `human` / `neutral` (sign of contribution).
- `interpretation` — plain-language sentence tying value, reference, and
  direction together.
- `caveat` — always *"Statistical stylistic signal measured by the model;
  not proof of authorship."*

Global top features are available on `/api/v1/info`
(`global_importance`, permutation importance = mean PR-AUC drop; method is
stated verbatim there).

### Errors

All errors share one shape — no input echo, no filesystem paths, no stack
traces:

```json
{
  "status": "error",
  "error": {
    "code": "validation_error",
    "message": "Request validation failed. See details.",
    "details": [
      {"field": "text", "message": "String should have at least 1 character",
       "type": "string_too_short"}
    ]
  }
}
```

| Status | `code` | Trigger |
| ------ | ------ | ------- |
| 404 | `not_found` | Unknown path |
| 405 | `method_not_allowed` | Wrong method (e.g. GET on `/api/v1/analyze`) |
| 422 | `validation_error` | Empty/whitespace text, >30,000 chars, wrong type, missing field, malformed JSON, out-of-range `max_signals` |
| 500 | `internal_error` | Unexpected failure — details are never exposed |

The offending input value is deliberately **not** echoed back (it may be the
user's whole document).

### GET `/api/v1/health`

```json
{"status": "ok", "api_version": "v1", "model_version": "final-v1",
 "model_name": "hgb_syn", "loaded": true}
```

### GET `/api/v1/info`

Static metadata: version, model/calibrator/threshold, `n_features` (89),
gate, label vocabulary, data policy (trained/selected/calibrated/eval-only
splits), top global features, eval summary for MAGE-test + OOD sets, the
importance method string, and the disclaimer. Contains no filesystem paths.

## Examples

### Likely human

```bash
curl -s http://127.0.0.1:8000/api/v1/analyze \
  -H 'content-type: application/json' \
  -d '{"text": "so we ended up going to that new cafe near the station, ...", "max_signals": 3}'
```

```json
{
  "status": "ok",
  "label": "likely_human",
  "decision": "human",
  "machine_probability": 0.131255,
  "human_probability": 0.868745,
  "confidence": 0.8687,
  "word_count": 101,
  "evidence_sufficient": true,
  "mixed_low_confidence": false,
  "signals": [
    {"signal": "pron_i", "label": "first-person singular pronouns",
     "family": "stylometric", "value": 29.703, "reference_median": 0.0,
     "contribution": -0.0984, "direction": "human",
     "interpretation": "observed value 29.703 vs train median 0 moved the estimate toward human by -0.0984",
     "caveat": "Statistical stylistic signal measured by the model; not proof of authorship."}
  ],
  "segments": [],
  "gate": {"min_words": 50, "mixed_band": [0.4, 0.6]},
  "model_version": "final-v1", "model_name": "hgb_syn",
  "calibrator": "platt", "threshold": 0.5,
  "warnings": [],
  "disclaimer": "WriteTrace does not establish authorship. It estimates how strongly observed writing patterns resemble machine-generated or human-written text under the model's training and evaluation conditions."
}
```

### Mostly machine

```json
{
  "label": "mostly_machine",
  "decision": "machine",
  "machine_probability": 0.620904,
  "human_probability": 0.379096,
  "confidence": 0.6209,
  "word_count": 68,
  "signals": [
    {"signal": "log_ttr", "label": "log type-token ratio", "family": "lexical",
     "value": 0.9743, "reference_median": 0.920206,
     "contribution": 0.5062, "direction": "machine", "…": "…"}
  ]
}
```

### Short text (200, abstain)

```json
{
  "status": "insufficient_evidence",
  "label": "insufficient_evidence",
  "decision": "abstain",
  "machine_probability": 0.239561,
  "human_probability": 0.760439,
  "confidence": null,
  "word_count": 4,
  "evidence_sufficient": false,
  "warnings": ["insufficient evidence: the input has 4 words, below the 50-word threshold selected in Phase 3; the probability is shown for transparency and must not support an authorship conclusion"]
}
```

### Validation error (422)

```json
{
  "status": "error",
  "error": {"code": "validation_error",
            "message": "Request validation failed. See details.",
            "details": [{"field": "text",
                         "message": "String should have at least 1 character",
                         "type": "string_too_short"}]}
}
```

## Limitations (read before using any output)

- **Not authorship, not proof, not a plagiarism detector.** The output is a
  statistical resemblance estimate under the model's training and evaluation
  conditions.
- **In-distribution performance only is strong** (MAGE-test, n=51,995:
  ROC-AUC 0.9618, PR-AUC 0.9799, F1(macro) 0.8856, human FPR 0.1523).
- **Out-of-distribution degradation is real** (from the locked artifact's
  eval summary): OOD-GPT ROC-AUC 0.6958, OOD-GPT-para 0.6165, HC3 human FPR
  0.3774, RAID-clean 0.8490. Paraphrased or conversational text is the
  known weak spot — treat any OOD/paraphrase/conversation score as
  unreliable, especially inside the mixed band.
- **Mixed band `[0.4, 0.6]`** had validation accuracy 0.5342 — essentially a
  coin flip; the API flags it (`mixed_low_confidence`, warning).
- **Non-English / mixed-language input** is only warned about, never
  rejected; the model was trained and evaluated on English.
- **Calibration** was fitted on MAGE-val_cal (Platt); probabilities are
  well calibrated in-distribution (ECE-10 0.0058) and degrade with domain
  shift (ECE-10 0.195–0.334 OOD).
- **Short texts** (<50 words) are abstained by design rather than guessed.
- The API is a local inference service: no authentication, no rate limit, no
  persistence — put it behind your own controls before exposing it.

## Versioning

- URL namespace: `/api/v1/…`; breaking changes require `/api/v2`.
- Model artifact version (`final-v1`, winner `hgb_syn`) is reported in every
  response and in `/api/v1/health`, and is pinned by
  `ml/models/final-v1/manifest.json` hashes.

## Disclaimer

> WriteTrace does not establish authorship. It estimates how strongly
> observed writing patterns resemble machine-generated or human-written text
> under the model's training and evaluation conditions.
