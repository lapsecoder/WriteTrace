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

## 3. How the model works (status: trained — locked Phase-3 artifact)

Handcrafted lexical, sentence-level, syntactic, and stylometric features feed **classical ML models** (logistic regression, gradient boosting, forests, SVM, TF-IDF baselines); the selected winner uses **89 features** (73 base + 16 spaCy syntactic). Predictions come from the trained model — never from keywords, regex, or an external LLM API. Probabilities are calibrated (Platt), and a reliability gate (≥50 words) abstains on short inputs. Artifact: `ml/models/final-v1/` (winner `hgb_syn`, threshold 0.5).

See `docs/technical-specification.md`, `docs/methodology.md`.

## 4. Dataset methodology (details in `docs/dataset.md`)

Primary training corpus **MAGE** (Apache-2.0; multi-domain, multi-generator); **RAID** (MIT; adversarial/paraphrase attacks, held-out robustness); **HC3** (CC-BY-SA-4.0; unseen-generator/domain stress test). Deduplication, grouped splits, and a locked test set prevent leakage. Licensing verified; ambiguous-license datasets (M4) are excluded pending review. All data is free and downloaded locally — ₹0.

## 5. Feature engineering

Lexical diversity & word-frequency structure · sentence-length statistics & uniformity · punctuation distribution · pronoun/hedge/transition/boilerplate lexicons · repetition · clause density & POS distributions (optional spaCy) · readability. No single feature proves authorship; the classifier learns combinations.

## 6. Models tested & 7. Evaluation results

Measured and published: baselines, model selection, calibration, and the locked test evaluation live in `docs/experiments/` (`phase2-report.md`, `phase3-report.md`). Headline (MAGE-test, n=51,995): **ROC-AUC 0.9618, PR-AUC 0.9799, F1(macro) 0.8856, Brier 0.0739, ECE-10 0.0058**; OOD results (GPT/GPT-para/HC3/RAID) and per-length/abstention slices are reported there with their failure modes.

## 8. Robustness experiments (plan in `docs/experiments/robustness-plan.md`)

Unseen generators · paraphrased machine text · human-edited machine text · mixed authorship documents · leave-one-domain-out transfer · length sensitivity · temporal drift. Failures will be documented.

## 9. Limitations (current, measured — see `docs/experiments/`)

- Detection is probabilistic and ages quickly; new generators degrade performance (OOD-GPT ROC-AUC 0.6958, OOD-GPT-para 0.6165).
- Paraphrasing and human editing substantially erode stylometric cues (quantified in the phase reports).
- Short texts (< 50 words) are under-determined → the API reports **insufficient evidence** and abstains (9.33% of validation text).
- Known bias risk: conversational/non-native writing can resemble machine regularity (HC3 human FPR 0.3774); never use output as sole evidence of authorship.
- Training corpora skew to forums/news/academic text; other genres are less validated.

## 10. Local setup (training + API)

```bash
# Python 3.11+
python -m venv .venv && .venv\Scripts\activate        # Windows
pip install -r ml/requirements.txt
python -m spacy download en_core_web_sm               # syntactic features
python scripts/download_data.py                        # free HF/git downloads
python scripts/train.py --config ml/training/configs/baselines.yaml
python scripts/evaluate.py --run latest                # writes docs/experiments reports
python -m pytest tests/ -q                             # full suite (127 tests)
```

Run the inference API (Phase 4; no API keys, no paid services, no DB):

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
# interactive docs: http://127.0.0.1:8000/docs   (API contract: docs/api.md)
```

Frontend arrives with Phase 5 (Astro).

## 11. Training / 12. Inference

Training is fully local and reproducible (fixed seeds, config + data-manifest hashes, artifacts in `ml/models/`). Inference is exposed at `POST /api/v1/analyze`, returning calibrated probabilities, the 5-level label, confidence, gate/abstention state, signals, and the authorship disclaimer — see `docs/api.md`.

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

**Phases 0–5A complete** (research/spec/datasets → features → training → locked model selection → inference engine + FastAPI backend → segment-level research + validation). Test suite: 157 passing. Locked artifacts: `ml/models/baseline-v1/` (Phase 1), `ml/models/final-v1/` (Phase 3, winner `hgb_syn`), and a research-only `ml/models/segments-v1/` (Phase 5A). Reports: `docs/experiments/phase2-report.md`, `phase3-report.md`, `phase4-report.md`, `phase5a-report.md`.

**Segment-level analysis (Phase 5A):** trained and evaluated on real corpora; the honest verdict is that it is **not defensible for user-facing display** (out-of-distribution segment AUC stays near chance; short segments are over-confident). The API therefore keeps `segments[]` empty and the document-level prediction authoritative.

**Next:** Phase 5B (frontend) — build on the document-level contract in `docs/api.md`.

**License:** project code will be MIT (decided at Phase 1); dataset licenses apply to data, not to this code.
