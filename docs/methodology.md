# WriteTrace — ML Methodology (Phase 0 proposal)

---

## 1. Preprocessing

1. Unicode NFC; strip control chars; collapse whitespace; URLs → `[URL]`; emails → `[EMAIL]`.
2. Language filter: English only for v1.
3. Length filter for training rows: 40–5,000 words (extremes destabilize classical models and are rare in real usage).
4. Two text views per row: `raw` and `artifact_stripped` (common LLM lead-ins/lead-outs removed) — primary results on `raw`, robustness slice on `artifact_stripped`.
5. Dedup + grouped split assignment happens **before** feature extraction; split manifest persisted as a file so every experiment uses identical rows.

## 2. Feature Extraction (v1, ~140 features)

All features are computed locally with stdlib + numpy/pandas (+ optional spaCy small model). Features are **per-document scalars**; per-paragraph extraction reuses the same code (enables future segment analysis honestly).

- **Structure:** char/word/sentence/paragraph counts; avg & median word/sentence length; sentence-length std & coefficient of variation; first/last sentence length; paragraph-length CV.
- **Lexical:** type-token ratio + root-TTR (length-robust); hapax ratio; rare-word ratio vs. a bundled open frequency list; mean log word frequency; stopword ratio; contraction rate (`don't`, `it's`, …).
- **Punctuation per 1,000 chars:** `, ; : — – … " ' ( ) ? ! % $ # * -` and digit ratio; list/enumeration marker rate.
- **Stylometric lexicons (rates per 1k words):** pronouns (I/you/we/he/she/they/it), 1st-person singular/plural, hedges (may, might, perhaps, seems, arguably), certainty markers (clearly, obviously), transitions (furthermore, moreover, however, in conclusion), boilerplate LLM phrases (`it's important to note`, `in today's world`, `delve`), discourse connectives.
- **Repetition:** repeated 3-gram rate; unique-bigram ratio; max n-gram repeat count.
- **Syntactic (optional, spaCy `en_core_web_sm`):** POS unigram & selected bigram distributions (normalized), POS Shannon entropy, subordinating-conjunction rate (clause density proxy), mean/max parse-tree depth; the syntactic family is feature-flagged so the pipeline runs without spaCy if needed.
- **Readability:** Flesch reading ease; Flesch–Kincaid grade level.
- **Feature hygiene:** near-zero-variance features dropped; highly correlated pairs (>0.98) pruned on the train fold only; standardization fit on train only.

## 3. Baselines & Training

| Model | Why |
|---|---|
| Logistic Regression (L2, standardized, balanced) | Interpretable; coefficients feed explainability; naturally reasonable probabilities |
| HistGradientBoostingClassifier | Captures nonlinear interactions among stylometric features |
| Random Forest / ExtraTrees | Robustness/variance baseline |
| Linear SVM + CalibratedClassifierCV | Strong linear separator; calibrated post-hoc |
| Complement/Bernoulli NB on word+char TF-IDF (1–2 / 3–5 grams, capped vocab) | Classic authorship-style text baseline; cheap |
| (Optional) MiniLM embeddings + LR/GBD | Tests whether semantic embeddings add signal beyond stylometry |
| (Gate) Transformer fine-tune | Only with written justification after classical results; RTX 5050 8 GB is sufficient if needed |

Training protocol: fixed seeds; `class_weight` or stratified sampling for balance; small seeded hyperparameter grid on validation (never on test); single-command reproducibility via `scripts/train.py` writing metrics + artifacts + run manifest (config hash, data manifest hash, seed).

## 4. Probability Calibration & Confidence

1. Fit calibrator (**isotonic** and **Platt** compared) on a validation fold; select by Brier score/ECE.
2. Report **reliability diagrams** and ECE (10 equal-width bins) for every candidate model on validation and locked test.
3. **Reliability gate** at inference: combine calibrated `p(machine)` with evidence indicators — text length bucket, feature completeness (spaCy present?), and cross-model disagreement when an ensemble is served. Outcomes:
   - `sufficient` + confident label (bands)
   - `sufficient` + `mixed` (p near 0.5 or high disagreement)
   - `insufficient` (below ~50 words, or extreme disagreement) → UI must display "insufficient evidence", never a bare percentage.
4. **Classification bands** (5 classes) derived from calibrated probability; boundaries fitted on validation (e.g., optimize per-band purity), then frozen. Bands are a *presentation layer over calibrated probabilities*, not a second model.

## 5. Explainability

- **Linear models:** contribution = standardized feature value × coefficient; aggregated into ~12 human-readable **signal families** (e.g., "sentence-length uniformity" ← sentence-length CV, POS entropy; "punctuation distribution" ← punctuation rates). High/Moderate/Low thresholds set from validation-set contribution percentiles.
- **Tree models:** impurity/permutation importance for families + TreeSHAP (if `shap` installs) for local explanations.
- **Guardrails:** explanations describe model-level statistical association only. No UI string may assert authorship from a single feature.

## 6. Segment-Level Analysis (design now, implement Phase 5)

- Architecture already supports it: features are computed per text unit; the trained model scores any unit above a minimum evidence length (~50 words).
- v1 API returns `segments: []`; Phase 5 emits per-paragraph `p(machine)` with per-segment `evidence_sufficiency`.
- A dedicated segment-level model (trained on paragraph slices with document-level labels as weak supervision) is an *option*, evaluated against the "same model, windowed" baseline — no keyword highlighting, ever.

## 7. Environment & Cost (₹0)

- Python 3.11+, scikit-learn, pandas, numpy, optional spaCy/sentence-transformers, `datasets`/`huggingface_hub`; all free/open.
- Training runs on CPU for classical models; GPU (RTX 5050, 8 GB) reserved for optional embedding/neural experiments with small batches + fp16.
- No paid APIs, datasets, fonts, or hosting anywhere in the pipeline.

