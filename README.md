# WriteTrace — Machine-Generated Writing Forensics

> **Statistical estimate, not proof.** WriteTrace runs a *trained statistical classifier* over linguistic and stylometric features. AI-text detection is inherently imperfect: every prediction is a probabilistic estimate that can be wrong — on short texts, on paraphrased or human-edited AI text, on non-native writing, and on generators newer than the training data. Nothing here should be used as sole evidence of authorship.

---

## 1. What WriteTrace is

WriteTrace estimates whether a piece of writing shows stylistic patterns *associated with* machine-generated text and reports:

- Machine/human likelihood (calibrated probabilities)
- 5-level classification: Likely human · Mostly human · Mixed/uncertain · Mostly machine · Likely machine
- Confidence + an explicit **insufficient evidence** state for short texts
- Stylometric signal families contributing to the estimate (model-level, not per-sentence verdicts)
- (Roadmap) per-paragraph likelihood analysis

## 2. Why this project exists

Generative models now produce fluent prose everywhere. People need tools that help *investigate* text origin — but published research shows detectors are fragile under paraphrasing, editing, new generators, and domain shifts. WriteTrace is built to make that fragility **visible**: honest evaluation, calibrated uncertainty, and documented failure modes are features, not embarrassments.

## 3. How the model works (status: Phase 0 — specified, not yet trained)

Handcrafted lexical, sentence-level, syntactic, and stylometric features (~140 per document) feed **classical ML models** (logistic regression, gradient boosting, forests, SVM, TF-IDF baselines). Predictions come from the trained model — never from keywords, regex, or an external LLM API. Probabilities are calibrated (Platt/isotonic), and a reliability gate downgrades confidence on short or ambiguous inputs.

See `docs/technical-specification.md`, `docs/methodology.md`.

## 4. Dataset methodology (details in `docs/dataset.md`)

Primary training corpus **MAGE** (Apache-2.0; multi-domain, multi-generator); **RAID** (MIT; adversarial/paraphrase attacks, held-out robustness); **HC3** (CC-BY-SA-4.0; unseen-generator/domain stress test). Deduplication, grouped splits, and a locked test set prevent leakage. Licensing verified; ambiguous-license datasets (M4) are excluded pending review. All data is free and downloaded locally — ₹0.

## 5. Feature engineering

Lexical diversity & word-frequency structure · sentence-length statistics & uniformity · punctuation distribution · pronoun/hedge/transition/boilerplate lexicons · repetition · clause density & POS distributions (optional spaCy) · readability. No single feature proves authorship; the classifier learns combinations.

## 6. Models tested & 7. Evaluation results

**Not yet available — Phase 1.** Plan: multiple classical baselines, selected on PR-AUC/ROC-AUC **plus** calibration (Brier/ECE) plus robustness on held-out generators — not on headline accuracy. Full metrics (precision, recall, F1, confusion matrix, calibration curves, per-domain/per-length slices) will be published in `docs/experiments/` when training runs exist. This README will not claim numbers before they are honestly measured.

## 8. Robustness experiments (plan in `docs/experiments/robustness-plan.md`)

Unseen generators · paraphrased machine text · human-edited machine text · mixed authorship documents · leave-one-domain-out transfer · length sensitivity · temporal drift. Failures will be documented.

## 9. Limitations (current, pre-training)

- Detection is probabilistic and ages quickly; new generators will degrade performance.
- Paraphrasing and human editing substantially erode stylometric cues (expected finding, to be quantified).
- Short texts (< ~50 words) are fundamentally under-determined → "insufficient evidence."
- Known bias risk: non-native/ESL writing can resemble machine regularity; UI must warn against punitive use.
- Training corpora skew to forums/news/academic text; other genres are less validated.

## 10. Local setup (Phase 1+)

```bash
# Python 3.11+
python -m venv .venv && .venv\Scripts\activate        # Windows
pip install -r ml/requirements.txt
python scripts/download_data.py                        # free HF/git downloads
python scripts/train.py --config ml/training/configs/baselines.yaml
python scripts/evaluate.py --run latest                # writes docs/experiments reports
```

Frontend/backend setup arrives with Phases 3–4 (Astro + FastAPI). No API keys, no paid services.

## 11. Training / 12. Inference

Training is fully local and reproducible (fixed seeds, config + data-manifest hashes, artifacts in `ml/models/`). Inference will be exposed at `POST /api/v1/analyze` returning calibrated probabilities, classification, confidence, signals, and evidence sufficiency — see §16 of the technical specification.

## 13. Architecture

```
frontend/ (Astro, editorial-paper UI)  →  backend/ (FastAPI)  →  ml/ (features + trained model)
docs/ (spec, dataset, methodology, experiments)   tests/   scripts/
```

## 14. Ethical considerations

- **Not an accusation engine.** Results are statistical estimates; false positives have real consequences (students, employees, ESL writers).
- **Local processing:** submitted text is analyzed on your machine and not stored by default.
- **No surveillance design:** no accounts, no logging of submitted text in the product vision.
- **Transparency:** dataset provenance, licenses, evaluation results, and failure modes are public in `docs/`.
- **Honest uncertainty:** the UI must never fabricate confidence; insufficient evidence is reported as such.

## Repository status

Phase 0 complete: research, specification, and dataset strategy. Implementation begins only after this plan is reviewed.

**License:** project code will be MIT (decided at Phase 1); dataset licenses apply to data, not to this code.
