# WriteTrace — Baseline Report: `baseline-v1`

Generated: 2026-10-07T06:55:22.313442+00:00 · seed=42 · feature_version=features_v1

> **Non-additivity note.** Stages report each filtering step in isolation. HC3 rows dropped during exact-dedup/cross-corpus filtering never entered the split pools. Split totals: train 155,988 + val 51,996 + test 51,995 = 259,979 (target 260,000 − 21 MAGE exact duplicates).
> **Honest-evaluation note.** This is a first baseline, not a product claim. Numbers below are measured on a locked test split with no test-set tuning. AI-text detection is probabilistic and these results *understate* real-world difficulty (paraphrase, human editing, new generators are Phase 2).

## 1. Dataset sizes after every filtering stage

| Stage | Rows |
|---|---|
| mage_raw_rows | 432682 |
| mage_dropped_invalid_string | 0 |
| mage_dropped_not_english | 2280 |
| mage_dropped_out_of_length_range | 59224 |
| mage_after_filters | 371178 |
| hc3_raw_rows | 85449 |
| hc3_dropped_invalid_string | 18 |
| hc3_dropped_not_english | 1998 |
| hc3_dropped_out_of_length_range | 14147 |
| hc3_after_filters | 69286 |
| mage_before_stratified_sample | 371178 |
| mage_after_stratified_sample | 260000 |
| exact_duplicates_dropped_total | 4341 |
| exact_duplicates_dropped_by_dataset | {'hc3': 4320, 'mage': 21} |
| cross_corpus_rows_removed_from_eval | 14 |

Processing duration: 375.0s · RAID status: test.csv:already-present (downloaded; normalization/features deferred to Phase 2 - eval-only)

## 2. Class distribution, groups, splits

| Split | Rows | Human(0) | Machine(1) | Groups | Median words | Domains |
|---|---|---|---|---|---|---|
| eval_hc3 | 64952 | 38976 | 25976 | 64931 | 141 | 5 |
| test | 51995 | 33949 | 18046 | 51923 | 143 | 10 |
| train | 155988 | 101847 | 54141 | 155769 | 144 | 10 |
| val | 51996 | 33948 | 18048 | 51923 | 143 | 10 |

## 3. Duplicate removal statistics

| Statistic | Value |
|---|---|
| Exact duplicates dropped (total) | 4341 |
| Exact drops by dataset | {'hc3': 4320, 'mage': 21} |
| Near-dup candidate pairs (LSH) | 97253 |
| Near-dup pairs verified (est. J ≥ 0.75) | 447 |
| Exact pairs unioned | 0 |
| Duplicate clusters (exact+near) | 324546 |
| Rows in multi-row clusters | 731 |
| Cross-corpus eval rows removed | 14 |
| Bucket overruns (capped) | 2 |

## 4. Features

| Item | Value |
|---|---|
| Feature count | 73 |
| Families | structure (18), lexical (10), punctuation (15), stylometric (16), repetition (3), syntactic (5), readability (4), positional (2) |
| Feature version | features_v1 |
| State fitted on | train_split_only (155988 docs) |
| Vocabulary (train DF) | 266728 |
| Rare threshold (DF < | 156) |
| Column-order sha256 | fc1539254b5abab6… |
| All-NaN feature rows (invalid input) | 0 |

## 5. Model configuration

| Model | Params | Val-tune PR-AUC |
|---|---|---|
| dummy_prior | {'strategy': 'prior'} | 0.3448 |
| logreg | {'C': 0.1} | 0.7447 |
| logreg | {'C': 1.0} | 0.7479 |
| logreg | {'C': 10.0} | 0.7481 |
| hist_gbdt | {'learning_rate': 0.06, 'max_leaf_nodes': 31} | 0.9138 |
| hist_gbdt | {'learning_rate': 0.06, 'max_leaf_nodes': 63} | 0.9202 |
| hist_gbdt | {'learning_rate': 0.1, 'max_leaf_nodes': 31} | 0.9177 |
| hist_gbdt | {'learning_rate': 0.1, 'max_leaf_nodes': 63} | 0.9222 |

Selection rule: max PR-AUC on `val_tune` → **hist_gradient_boosting**. val_tune=31323, val_cal=20673 (group-disjoint). Calibrator fitted only on val_cal; test never used for tuning.

## 6. Validation calibration record

| Item | Value |
|---|---|
| Calibrator chosen | isotonic |
| Platt Brier (val_cal) | 0.0764 |
| Isotonic Brier (val_cal) | 0.0757 |
| Val_cal raw Brier / ECE | 0.0765 / 0.0131 |
| Val_cal calibrated Brier / ECE | 0.0757 / 0.0000 |

## 7. Test results (locked test split)

| Model | accuracy | precision_machine | recall_machine | f1_machine | f1_macro | roc_auc | pr_auc | brier | ece_10 |
|---|---|---|---|---|---|---|---|---|---|
| dummy_prior (raw) | 0.6529 | 0.0000 | 0.0000 | 0.0000 | 0.3950 | 0.5000 | 0.3471 | 0.2266 | 0.0000 |
| logistic_regression (raw) | 0.7730 | 0.6295 | 0.8406 | 0.7199 | 0.7645 | 0.8737 | 0.7615 | 0.1488 | 0.0905 |
| hist_gradient_boosting (raw) | 0.8923 | 0.8528 | 0.8338 | 0.8431 | 0.8806 | 0.9588 | 0.9270 | 0.0770 | 0.0089 |
| hist_gradient_boosting (calibrated) | 0.8919 | 0.8422 | 0.8472 | 0.8447 | 0.8809 | 0.9586 | 0.9231 | 0.0772 | 0.0067 |

Confusion matrix (selected model, calibrated, threshold 0.5):

|  | pred human | pred machine |
|---|---|---|
| true human | 31084 | 2865 |
| true machine | 2757 | 15289 |

## 8. Calibration & reliability (selected model)

Raw → calibrated on test:

| Metric | Raw | Calibrated |
|---|---|---|
| Brier | 0.0770 | 0.0772 |
| ECE(10) | 0.0089 | 0.0067 |
| ROC-AUC | 0.9588 | 0.9586 |

Reliability bins (calibrated):

| Bin | n | mean p(machine) | observed frac machine |
|---|---|---|---|
| 0.0-0.1 | 24989 | 0.0147 | 0.0172 |
| 0.1-0.2 | 2372 | 0.1385 | 0.1400 |
| 0.2-0.3 | 3131 | 0.2436 | 0.2373 |
| 0.3-0.4 | 2173 | 0.3793 | 0.3576 |
| 0.4-0.5 | 1176 | 0.4289 | 0.4039 |
| 0.5-0.6 | 1599 | 0.5623 | 0.5222 |
| 0.6-0.7 | 1702 | 0.6201 | 0.6210 |
| 0.7-0.8 | 2048 | 0.7425 | 0.7178 |
| 0.8-0.9 | 4416 | 0.8536 | 0.8546 |
| 0.9-1.0 | 8389 | 0.9797 | 0.9719 |

## 9. Slices — where the model works and fails

By text length (calibrated):

| Length | n | ROC-AUC | PR-AUC | F1-macro | ECE | note |
|---|---|---|---|---|---|---|
| <50w | 4852 | 0.8907 | 0.7953 | 0.7983 | 0.0192 |  |
| 50-150w | 22052 | 0.9275 | 0.8718 | 0.8400 | 0.0083 |  |
| 150-400w | 15478 | 0.9720 | 0.9520 | 0.9072 | 0.0102 |  |
| 400w+ | 9613 | 0.9960 | 0.9902 | 0.9696 | 0.0029 |  |

By domain (calibrated; domains with n≥20):

| Domain | n | ROC-AUC | PR-AUC | F1-macro | ECE | note |
|---|---|---|---|---|---|---|
| cmv | 4347 | 0.9793 | 0.9567 | 0.9179 | 0.0221 |  |
| eli5 | 6654 | 0.9490 | 0.9381 | 0.8692 | 0.0250 |  |
| hswag | 3274 | 0.8751 | 0.2109 | 0.6048 | 0.0482 |  |
| roct | 4689 | 0.9899 | 0.9538 | 0.9357 | 0.0322 |  |
| sci_gen | 4315 | 0.9506 | 0.8810 | 0.8665 | 0.0207 |  |
| squad | 5911 | 0.9518 | 0.9356 | 0.8792 | 0.0355 |  |
| tldr | 4162 | 0.9140 | 0.7265 | 0.7920 | 0.0203 |  |
| wp | 5494 | 0.9779 | 0.9607 | 0.9208 | 0.0240 |  |
| xsum | 5893 | 0.9328 | 0.8342 | 0.8329 | 0.0341 |  |
| yelp | 7256 | 0.9375 | 0.9488 | 0.8582 | 0.0413 |  |

## 10. Errors & limitations (honest assessment)

- **Scope:** trained and tested on MAGE only. HC3 (unseen generator, conversational) and RAID (adversarial attacks) are downloaded and dedup-checked but **not yet evaluated** — Phase 2 robustness experiments may lower these numbers.
- **Calibrator optimism:** Platt/isotonic were selected on val_cal, so val_cal calibration scores are slightly optimistic; test calibration is the honest figure.
- **Threshold:** fixed at 0.5; no threshold optimization was performed (avoids test-set tuning).
- **Sentence segmentation and clause markers are heuristic** (regex/lexicon, no parser) — spaCy syntactic features are Phase-2 additions.
- **Short texts:** see length slice; reliability degrades below ~150 words, which will drive the inference-time 'insufficient evidence' gate.
- **MAGE prompt IDs are unavailable in the public release**, so grouping uses exact + near-duplicate clusters as a proxy for source families (documented in docs/dataset.md §5 risk register).
- **Never authorship proof:** all figures are statistical associations in one corpus, not guarantees about any individual document.

## 11. Reproduction

```bash
python scripts/download_data.py     # idempotent, resumes, writes manifest
python scripts/build_dataset.py     # filters, dedup, grouped split (seed 42)
python scripts/extract_features.py  # feature state fit on train only
python scripts/train.py             # run baseline-v1, seed 42
python scripts/evaluate.py          # writes this report + test_metrics.json
```

Data manifest sha256: `d2cb15f255c38dde1e74a2492b5806d80de6a1c644cb13edc45ac66054f33b22`  
Processing report sha256: `7d29e1b7ffa778e4b661773b57473e6197c52a186dec28e2c6ae06969029e14e`  
Feature names sha256: `fc1539254b5abab65ed10c289eb2e849307c70eafd7cc8582b8545d046f8920f`  
Versions: {'python': '3.13.15', 'numpy': '2.5.3', 'pandas': '3.0.5', 'scikit_learn': '1.9.0'}
