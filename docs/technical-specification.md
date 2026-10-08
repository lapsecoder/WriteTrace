# WriteTrace — Technical Specification (Phase 0)

**Status:** Draft v0.1 — research & specification only. No detector implementation in this phase.
**Date:** 2026-06-10

---

## 1. Problem Statement

Given a piece of English text, WriteTrace estimates:

- Machine-generated likelihood `p(machine)` and human likelihood (complement)
- A 5-level classification: `likely_human` / `mostly_human` / `mixed` / `mostly_machine` / `likely_machine`
- Confidence + an explicit *insufficient evidence* state
- Influential linguistic/stylometric signal families (model-level contribution, not per-sentence verdicts)
- (Later phase) per-paragraph machine-like likelihood

**The prediction is a statistical estimate. It is never presented as proof of authorship.**

### 1.1 Problem characteristics that shape the design

1. **Distributional problem.** No single cue (em dash, "delve", low perplexity) proves authorship. The model must learn *relationships between* features.
2. **Generator drift.** Detectors age quickly; detectors trained on 2023 ChatGPT fail on newer models. Multi-generator training + **held-out-generator evaluation is mandatory**.
3. **Domain shift** (essays vs. news vs. chat vs. abstracts) is the dominant documented failure mode (M4, RAID, MAGE).
4. **Short texts are under-determined.** The system must say so rather than fabricate confidence.
5. **Human-edited / paraphrased machine text** intentionally erodes stylometric cues; robustness experiments must quantify this, not hide it.
6. **Non-native writer bias** is a documented risk of AI-text detectors; evaluation slices and UI disclaimers must address it.

## 2. Target Architecture

```
text → preprocessing → feature extraction (v1 feature config)
     → trained classical model(s) → probability calibration
     → classification + confidence + reliability gate
     → explainability layer (signal families)
     → segment scorer (future: same model on paragraph windows)
```

Hard separation of concerns:

- `ml/` produces **versioned artifacts**: feature config, model, calibrator, metrics, model card.
- `backend/` (FastAPI) serves inference only; it does not re-implement feature logic beyond calling `ml` code.
- `frontend/` (Astro) is a pure consumer of the API.

## 3. Output Contract (API sketch)

`POST /api/v1/analyze` → `{"text": "..."}`

```json
{
  "classification": "mostly_machine",
  "machine_probability": 0.73,
  "human_probability": 0.27,
  "confidence": 0.84,
  "evidence_sufficiency": "sufficient",
  "text_length": {"characters": 2310, "words": 387, "sentences": 19, "paragraphs": 4},
  "signals": [
    {"family": "sentence_length_uniformity", "contribution": "high", "direction": "machine"},
    {"family": "syntactic_regularity", "contribution": "moderate", "direction": "machine"}
  ],
  "segments": [],
  "disclaimer": "Statistical estimate from a trained classifier; not proof of authorship."
}
```

Validation: min/max length bounds (configurable; default 50–30,000 chars), 422 on malformed input. Texts below the evidence threshold return `evidence_sufficiency: "insufficient"` with a wide probability band — never a confident label.

## 4. Feature Set v1 (~140 features, all computed locally)

| Family | Examples |
|---|---|
| Length/structure | chars, words, sentences, paragraphs, avg/median word & sentence length, sentence-length CV, position-wise first/last sentence length |
| Lexical | raw & root TTR, hapax ratio, rare-word ratio (vs. built-in frequency list), avg log word frequency, stopword ratio, contraction rate |
| Punctuation (per 1k chars) | comma, semicolon, colon, em/en dash, ellipsis, quotes, parens, ? ! %, digits, list markers |
| Stylometric lexicons | pronoun rates (I/you/we/they/it), 1st-person rate, hedging ("may, perhaps, it seems"), transitions ("furthermore, in conclusion"), boilerplate ("it's important to note"), enumeration patterns |
| Repetition | repeated 3-gram rate, unique-bigram ratio, self-overlap |
| Syntactic (optional spaCy `en_core_web_sm`) | POS unigram/bigram distributions, POS entropy, subordinating-conjunction (clause-marker) rate, parse-depth stats |
| Readability | Flesch reading ease, Flesch–Kincaid grade |

Full definitions in `docs/methodology.md`. **No feature is assumed to prove machine authorship; the classifier learns combinations.**

## 5. Models to Train (evidence-driven selection)

1. **Logistic Regression** (L2, standardized, class-weighted) — interpretable coefficients, well-behaved probabilities.
2. **HistGradientBoostingClassifier** — nonlinear feature interactions.
3. **Random Forest / ExtraTrees** — variance baseline.
4. **Linear SVM + `CalibratedClassifierCV`** — discriminative comparison.
5. **Complement/Bernoulli Naive Bayes on word+char n-gram TF-IDF** — classic text baseline.
6. **Optional:** sentence-transformers MiniLM embeddings + LR/GBD (free, local; RTX 5050 8 GB handles it easily).
7. **Neural transformer fine-tune:** allowed only after a documented decision gate showing classical models underfit; do not add for impressiveness.

**Selection criteria:** PR-AUC/ROC-AUC on a *locked* test set + calibration quality (Brier/ECE) + robustness on held-out generators/domains — **not** headline accuracy.

## 6. Evaluation Methodology

- **Dedup first:** normalized-text hash dedup across *all* sources before any split.
- **Grouped, stratified splits:** group by `source_id`/prompt family (machine text and its human reference stay together); stratify by label × domain; fixed seed; test set locked before modeling.
- **Metrics:** accuracy, precision/recall/F1 (macro + per class), ROC-AUC, PR-AUC, confusion matrix, Brier score, ECE + calibration curve.
- **Threshold selection on validation only**, rule documented, frozen for test.
- **Slices:** domain × generator × length bucket (<50, 50–150, 150–400, 400+ words).
- **Error analysis:** qualitative review of failures per bucket; check feature attributions for confounds (length, formatting artifacts).
- **Confidence model:** calibrated `p(machine)` + reliability gate (text length + cross-model agreement) → 5-class label + confidence + `insufficient` state. Short text never receives the confidence of a long sample.

## 7. Robustness Experiments

E1 unseen generator · E2 paraphrased machine text (RAID attacks + local paraphrase models) · E3 human-edited machine text (local LLM instructed to humanize; prompts/seeds documented) · E4 mixed authorship at document level (25/50/75 % machine paragraphs; also informs the future segment model) · E5 leave-one-domain-out transfer · E6 length sensitivity & calibration degradation · E7 temporal drift (HC3 Jan-2023 ChatGPT vs. newer generators). Full definitions in `docs/experiments/robustness-plan.md`. **Failures are documented, not hidden.**

## 8. Explainability

- Linear models: standardized coefficient × feature value → per-family contribution, aggregated into human-readable signal families.
- Tree models: permutation importance + TreeSHAP if available.
- UI language rules: explain *model-level statistical contribution*; never "this sentence is AI because it contains an em dash."

## 9. Repository Structure (initial)

```
WriteTrace/
├── frontend/            # Astro + TypeScript + Tailwind (Phase 4)
├── backend/             # FastAPI inference service (Phase 3)
├── ml/
│   ├── data/            # download scripts, manifests, license records, raw/processed (git-ignored)
│   ├── preprocessing/   # cleaning, dedup, language filter, split logic
│   ├── features/        # feature config + extraction code
│   ├── training/        # baseline training scripts
│   ├── models/          # artifacts + model card (artifacts git-ignored)
│   └── evaluation/      # metrics, slicing, calibration, reports
├── docs/
│   ├── technical-specification.md
│   ├── dataset.md
│   ├── methodology.md
│   └── experiments/robustness-plan.md
├── scripts/             # thin CLI entry points (download, train, evaluate)
├── tests/
├── .gitignore
└── README.md
```

## 10. Phase Roadmap & Exit Criteria

| Phase | Deliverable | Exit criteria |
|---|---|---|
| 0 (current) | Research & specification | Spec + dataset strategy reviewed |
| 1 | Data acquisition, preprocessing, features, baseline training | Reproducible run; baseline table on locked test; model card |
| 2 | Robustness experiments, calibration, explainability, final model selection | Robustness report; calibrated probabilities; documented selection decision |
| 3 | FastAPI service | API tests green; schema validated |
| 4 | Astro frontend (editorial/forensic paper design, animations, a11y) | UI matches design spec; `prefers-reduced-motion` respected |
| 5 | Segment-level model + polish | Paragraph probabilities from trained model, not keyword highlighting |

## 11. Non-Goals & Ethics

- No external LLM API or "is this AI?" prompt as the detector; no regex/keyword scoring.
- No authorship-proof claims anywhere in the product.
- Known biases (ESL/non-native writers, short texts, specific genres) disclosed in UI and docs.
- All text processed locally; nothing stored by default.
- ₹0 stack only: free datasets, open-source libraries, local inference/training, system/open-source fonts, procedural visuals.

