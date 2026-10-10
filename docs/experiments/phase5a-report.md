# WriteTrace — Phase 5A Report: Segment-Level Analysis (research + validation)

Generated: `2026-10-10T08:03:49.657058+00:00` · study `phase5a_v1` · segment artifact `ml\models\segments-v1` · every number below is read from executed artifacts (`phase5a-results.json`, `segments-v1-results.json`, `calibration-sensitivity.json`, `logs/pytest.log`, the locked `final-v1`) at render time.

> **Protocol statement.** Phase 5A is RESEARCH + MODEL/INFERENCE VALIDATION. The document-level path is authoritative and unchanged. No new feature, no retraining of `final-v1`, no change to the public API contract (`AnalyzeResponse.segments` stays empty). Whether segment-level inference is exposed to users is decided by measured evidence, not assumption.

## 1. Phase 5A objective

Decide, on evidence, whether WriteTrace may ever say anything about a *part* of a document, and on what grounds. Concretely:

1. build a deterministic, lossless segmentation of arbitrary text;
2. evaluate the locked `final-v1` (document model) **at segment level**;
3. study segment-length / evidence policy, calibration, segment↔document consistency and aggregation;
4. decide whether reuse is sufficient or a separate `segments-v1` model is scientifically justified, and if so train it in its own artifact;
5. state plainly whether segment-level output is defensible for production display.

| Outcome | Verdict |
|---|---|
| final-v1 reused at segment level behind a gate | no |
| separate segments-v1 model justified | yes |
| segments-v1 trained | yes |
| segments-v1 status | research_only_not_adopted_for_production |
| segment-level DISPLAY cleared for production | no |

**Bottom line:** NO artifact clears the declared segment-display criteria; segment-level verdicts must NOT be shown to users. Keep document-level prediction authoritative and, if segments are ever surfaced, expose them as non-decisional indicators with explicit abstention and warnings.

## 2. Segment definition

A segment is one contiguous, non-overlapping span of the ORIGINAL text (`ml/segmentation.py::Segment`):

| Field | Meaning |
|---|---|
| `segment_id` | 0-based position in reading order |
| `text` | exact substring `original[start_char:end_char]` |
| `start_char` / `end_char` | offsets into the ORIGINAL text (end exclusive) — lets a future UI map a segment back to source |
| `word_count` | training-normalizer word count (same counter as the Phase-4 gate) |
| `paragraph_index` | index of the source block |
| `kind` | `paragraph` or `window` |
| `window_index` | position within the windowed block (windows only) |
| `n_sentences` / `hard_split` | sentence count; flag when a single over-long sentence had to be cut at word boundaries |

Contracts asserted by tests: `text[start:end] == text` for every segment, spans strictly ordered and never overlapping, zero non-whitespace characters lost, and empty/whitespace input → no segments.

## 3. Segmentation algorithm

| Step | Rule |
|---|---|
| paragraph blocks | split on blank lines; each block trimmed to its first/last non-whitespace char, exact offsets kept |
| heading merge | a block ≤12 words with no sentence-final punctuation and not the last block merges forward into the next body block |
| windowing | block > 150 words is packed into non-overlapping sentence-aligned windows using the project's own splitter (`split_sentences`) |
| over-long sentence | hard-split at word boundaries, flagged `hard_split=True`, counted in the study |
| short tail | a trailing window < 30 words merges backwards (no micro-segment emitted) |
| overlap | none (by construction; asserted in tests) |

Observed on the study corpora: 16904 window segments, 9903 paragraph segments, 284 hard-split segments, mean 103.48 words/segment. Text preservation across every segmented document is exact (coverage report `non_whitespace_missing = 0`).

## 4. Segment length analysis

Bucket distribution and mean probability (all scored segments):

| Length bucket | Share of segments | mean p(machine) | share called machine |
|---|---|---|---|
| <30w | 0.0646 | 0.8351 | 0.9405 |
| 30-49w | 0.1051 | 0.8090 | 0.9113 |
| 50-79w | 0.1453 | 0.6947 | 0.7526 |
| 80-149w | 0.6155 | 0.4995 | 0.4748 |
| 150-299w | 0.0695 | 0.5257 | 0.4944 |
| 300w+ | 0.0000 | n/a | n/a |

Segment length strongly drives behaviour — this is the core evidence of the study:

| Length bucket (MAGE-test, inherited labels) | n | ROC-AUC | PR-AUC | Acc | F1-macro | human FPR | Brier | ECE-10 |
|---|---|---|---|---|---|---|---|---|
| <30w | 10 | 0.3750 | 0.7481 | 0.8000 | 0.4444 | 1.0000 | 0.1847 | 0.1817 |
| 30-49w | 704 | 0.7537 | 0.8404 | 0.7358 | 0.6422 | 0.6722 | 0.1972 | 0.1344 |
| 50-79w | 1105 | 0.8520 | 0.9260 | 0.7846 | 0.7343 | 0.4406 | 0.1459 | 0.0754 |
| 80-149w | 5885 | 0.9250 | 0.9635 | 0.8445 | 0.8286 | 0.2055 | 0.1065 | 0.0296 |
| 150-299w | 714 | 0.9624 | 0.9841 | 0.8866 | 0.8700 | 0.1525 | 0.0739 | 0.0301 |
| 300w+ | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

Read: short segments (30–79 words) are weak and over-predict machine (human FPR 0.44–0.67); 150–299 word segments are close to document quality; <30-word segments are essentially unreliable. Segment quality grows with length, exactly as the document gate implies but with much lower usable coverage.

## 5. Evidence/abstention policy

The document gate (50 words) was NOT assumed to transfer. Candidate segment gates were scored on the kept slice, the way Phase 3 scored its document gate:

| Candidate gate (words) | Kept share | Kept acc | Kept human FPR | Kept ECE | Kept AUC | Genuine composite AUC | Criteria met |
|---|---|---|---|---|---|---|---|
| 30 | 0.9988 | 0.8311 | 0.2702 | 0.0399 | 0.9100 | 0.9386 | no |
| 40 | 0.9784 | 0.8361 | 0.2541 | 0.0356 | 0.9153 | 0.9386 | no |
| 50 | 0.9152 | 0.8398 | 0.2325 | 0.0323 | 0.9210 | 0.9386 | no |
| 60 | 0.8658 | 0.8415 | 0.2206 | 0.0321 | 0.9229 | 0.9386 | no |
| 80 | 0.7839 | 0.8491 | 0.2002 | 0.0285 | 0.9296 | 0.9519 | no |
| 100 | 0.6973 | 0.8487 | 0.1964 | 0.0298 | 0.9296 | 0.9461 | no |
| 120 | 0.5997 | 0.8477 | 0.1971 | 0.0316 | 0.9285 | 0.9536 | no |
| 150 | 0.0848 | 0.8866 | 0.1525 | 0.0301 | 0.9624 | 1.0000 | no |

Declared rule: smallest candidate gate whose kept slice keeps >= 50% of MAGE-test segments with accuracy >= 0.85, human FPR <= 0.3, ECE-10 <= 0.05, kept AUC within 0.1 of the document AUC, genuine-label composite AUC >= 0.85, and BOTH OOD corpora kept AUC >= 0.65.

**final-v1 result:** no candidate gate clears every bar (chosen = None). Failures per gate are recorded in Section 13. The binding failures are kept-accuracy (0.83–0.85 vs ≥0.85) and OOD segment AUC (~0.55–0.59 vs ≥0.65).

`segments-v1` therefore ships its OWN segment evidence gate: **min_words = 120** (recorded in `ml/models/segments-v1/gate.json`), chosen by the gate sweep as the largest coverage with the lowest kept human-FPR while genuine composite AUC stays healthy.

## 6. final-v1 segment evaluation

final-v1 (the shipped document model) applied to segments, by corpus. Inherited-label corpora use the document's label for its segments; composite uses genuine whole-text segment labels. Document-level numbers are the reference:

| Set / unit | n | ROC-AUC | PR-AUC | Acc | F1-macro | human FPR | Brier | ECE-10 |
|---|---|---|---|---|---|---|---|---|
| segments: composite | 1110 | 0.9386 | 0.9493 | 0.8550 | 0.8548 | 0.1730 | 0.1010 | 0.0432 |
| segments: hc3 | 3506 | 0.8369 | 0.8222 | 0.7444 | 0.7402 | 0.3915 | 0.1805 | 0.1126 |
| segments: mage_test | 8418 | 0.9097 | 0.9545 | 0.8311 | 0.8084 | 0.2707 | 0.1166 | 0.0399 |
| segments: ood_gpt | 3045 | 0.5868 | 0.5543 | 0.5635 | 0.5621 | 0.3661 | 0.3090 | 0.2255 |
| segments: ood_gpt_para | 4902 | 0.5501 | 0.7039 | 0.4833 | 0.4784 | 0.3661 | 0.3565 | 0.3385 |
| segments: raid | 5826 | 0.7761 | 0.8252 | 0.7455 | 0.7178 | 0.4417 | 0.1894 | 0.0912 |
| documents: hc3 | 2000 | 0.8555 | 0.8161 | 0.7200 | 0.7199 | 0.3780 | 0.1929 | 0.1643 |
| documents: mage_test | 4000 | 0.9638 | 0.9805 | 0.8992 | 0.8903 | 0.1421 | 0.0729 | 0.0123 |
| documents: ood_gpt | 1562 | 0.6958 | 0.6416 | 0.6287 | 0.6281 | 0.3136 | 0.2601 | 0.1950 |
| documents: ood_gpt_para | 2362 | 0.6165 | 0.7223 | 0.5135 | 0.5111 | 0.3136 | 0.3450 | 0.3339 |
| documents: raid | 1494 | 0.8259 | 0.8740 | 0.7463 | 0.7442 | 0.2379 | 0.1804 | 0.0983 |
| documents: MAGE-test (locked, n=51995) | 51995 | 0.9618 | 0.9799 | 0.8965 | 0.8856 | 0.1523 | 0.0739 | 0.0058 |

**Verdict for final-v1 at segment level:** usable for genuinely long segments (≥150 words, AUC 0.962 ≈ document level), unreliable for the 50–150 word segments that make up most of a document, and near chance out of distribution (OOD-GPT AUC 0.587, OOD-GPT-para 0.550). Because no gate can both keep useful coverage and clear the OOD bar, reuse alone is not defensible for display.

## 7. Calibration analysis

Calibration is the make-or-break property for per-segment numbers: a user reading "this paragraph is 78% machine-like" needs the number to mean something. Per-bucket (MAGE-test segments):

| Length bucket | n | Brier | mean abs(p-y) | confident-subset accuracy | confident human FPR |
|---|---|---|---|---|---|
| <30w | 10 | n/a | n/a | n/a | n/a |
| 30-49w | 704 | 0.1972 | 0.3000 | 0.7992 | 0.6806 |
| 50-79w | 1105 | 0.1459 | 0.2571 | 0.8931 | 0.3865 |
| 80-149w | 5885 | 0.1065 | 0.1957 | 0.9269 | 0.0942 |
| 150-299w | 714 | 0.0739 | 0.1403 | 0.9615 | 0.0714 |
| 300w+ | 0 | n/a | n/a | n/a | n/a |

Short segments are systematically over-confident (30–49 words: ECE 0.13, confident-subset human FPR 0.68) because the calibrator was fit on document-length inputs. Long segments are well calibrated (150–299: ECE 0.030, confident accuracy 0.96).

Calibrator sensitivity for the new segment model (same fit data, val_cal segments):

| Calibrator | MAGE-test seg ECE | MAGE-test seg saturated | Composite (genuine) ECE | Composite accuracy | Composite saturated |
|---|---|---|---|---|---|
| platt | 0.0102 | 0.1650 | 0.0761 | 0.7900 | 0.0656 |
| isotonic | 0.0102 | 0.1993 | 0.0773 | 0.7850 | 0.0838 |

Both calibrators behave similarly and both saturate near 0/1 on a noticeable fraction of segments; on the genuine composite distribution the segment calibrator's ECE (~0.08) is far worse than on inherited windows (~0.01), i.e. segment calibration does NOT transfer to complete short texts. Segment probabilities must not be presented as precise.

## 8. Segment/document consistency

How segment probabilities relate to the document prediction (MAGE-test, 1791 multi-segment documents):

| Aggregation | Pearson vs doc | Spearman vs doc | mean abs diff | decision agreement |
|---|---|---|---|---|
| mean | 0.8532 | 0.8644 | 0.1641 | 0.8855 |
| median | 0.8515 | 0.8902 | 0.1450 | 0.8800 |
| weighted_mean | 0.8734 | 0.8779 | 0.1485 | 0.9012 |
| max | 0.7137 | 0.8783 | 0.1958 | 0.7767 |
| min | 0.7635 | 0.7909 | 0.2091 | 0.8063 |
| std | -0.3400 | -0.4271 | 0.6049 | 0.3523 |

Mean/median/weighted aggregations track the document reasonably (Spearman ≈ 0.86–0.89) but disagree with the document verdict on 10–12% of documents; max/min (extreme segments) agree less. No aggregation is treated as authoritative — the document prediction stays the verdict. Within genuine composites the mean within-document segment spread is 0.6293, i.e. real local variation exists, but so does document-level dilution (Section 9).

## 9. Aggregation experiments

Each aggregation used as a document-level predictor (MAGE-test, inherited document labels):

| Predictor | Accuracy | ROC-AUC | Brier | F1-macro |
|---|---|---|---|---|
| mean | 0.8660 | 0.9373 | 0.0968 | 0.8528 |
| median | 0.8635 | 0.9388 | 0.0956 | 0.8501 |
| weighted_mean | 0.8715 | 0.9405 | 0.0936 | 0.8603 |
| max | 0.8207 | 0.9120 | 0.1253 | 0.7914 |
| min | 0.8260 | 0.9178 | 0.1210 | 0.8177 |
| std | 0.3573 | 0.4758 | 0.5921 | 0.2632 |
| p_doc | 0.8992 | 0.9638 | 0.0729 | 0.8903 |

Extreme-segment influence:

| Question | Answer |
|---|---|
| share of machine docs whose EVERY segment is ≥0.5 | 0.8082 |
| share of machine docs whose segment MEAN is <0.5 | 0.0937 |
| share of human docs with ANY segment ≥0.5 | 0.3765 |
| mean max-segment p of human docs | 0.3928 |

Interpretation: segments carry real local signal (human documents do contain machine-scored segments), but neither a single machine-like paragraph nor a single human-like one should flip the document — which is exactly why the document verdict stays authoritative and segments, if ever shown, are non-decisional.

Mixed-composition documents (genuine segment labels, document label intentionally absent):

| Measure | Value |
|---|---|
| mean p, human segment / machine segment | 0.2432 / 0.8424 |
| mean margin (machine − human segment) | 0.5992 |
| share machine segment > human segment (paired) | 0.9405 |
| mean within-document max−min spread | 0.6293 |
| share of composites whose DOCUMENT verdict is <0.5 (i.e. a machine paragraph hidden inside a 'human' document) | 0.5153 |
| document order effect (machine-first vs human-first doc p) | human_first: 0.5095, machine_first: 0.4990 |

A machine paragraph and a human paragraph inside one document are, on average, separated by 0.5992 in probability and the machine segment is scored higher in ~94% of pairs — strong evidence that segment scoring carries genuine local information. But document-level prediction lands near the decision boundary on exactly these mixed documents (0.1189 land in the mixed band), which is the practical case where per-segment evidence would help — and the case where it is currently least trustworthy.

## 10. Explainability

Segment explanations reuse the Phase-4 perturbation explainer verbatim (`ml/explain.local_contributions` via `Predictor._signals`). Behavior measured on sampled segments:

| Check | Result |
|---|---|
| sampled segments | 30 |
| segments that produced signals | 30 |
| all contributions finite | yes |
| high-p segments with machine-directed top signals | 9 |
| causal-language hits (proves/AI) | none |

Each valid segment carries feature name, observed value, train reference median, signed contribution, direction, plain-language interpretation and the non-causal caveat — identical wording rules to Phase 4. Segments below the evidence gate never receive an explanation.

## 11. Robustness/error analysis

Synthetic probes (final-v1, whole text):

| Probe | p(machine) | repetition rate |
|---|---|---|
| normal_human | 0.2710 | 0.0000 |
| repeated_sentence | 0.9998 | 0.9143 |
| repeated_paragraph | 0.9999 | 0.8696 |
| punctuation_heavy | 0.8524 | 0.8269 |
| list_like | 0.5985 | 0.0000 |
| very_short | 0.2396 | 0.0000 |
| heading_only | 0.5093 | 0.0000 |
| unicode_multilingual | 0.3554 | 0.0000 |
| one_long_sentence | 1.0000 | 0.9968 |
| formally_machine | 0.8088 | 0.0000 |

The documented Phase-4 failure mode holds and is amplified at segment scale: a single repeated sentence scores 0.9998, a repeated paragraph 0.9999, and a 320-word single sentence 1.0 — these are repetition artifacts, not authorship evidence. On real data, the top-decile of repetition-rate MAGE-test segments has mean p 0.93 and human FPR 0.60, versus 0.64/0.26 for the rest.

Other OOD/attack segment slices (inherited labels):

| Slice | n | ROC-AUC | PR-AUC | Acc | F1-macro | human FPR | Brier | ECE-10 |
|---|---|---|---|---|---|---|---|---|
| ood_gpt | 3045 | 0.5868 | 0.5543 | 0.5635 | 0.5621 | 0.3661 | 0.3090 | 0.2255 |
| ood_gpt_para | 4902 | 0.5501 | 0.7039 | 0.4833 | 0.4784 | 0.3661 | 0.3565 | 0.3385 |
| hc3 | 3506 | 0.8369 | 0.8222 | 0.7444 | 0.7402 | 0.3915 | 0.1805 | 0.1126 |

RAID attack slices (inherited labels):

| Attack | n | ROC-AUC | PR-AUC | Acc | F1-macro | human FPR | Brier | ECE-10 |
|---|---|---|---|---|---|---|---|---|
| article_deletion | 538 | 0.8449 | 0.8936 | 0.7844 | 0.7635 | 0.3105 | 0.1512 | 0.0673 |
| homoglyph | 668 | 0.8564 | 0.8802 | 0.7889 | 0.7856 | 0.2575 | 0.1541 | 0.0580 |
| insert_paragraphs | 1166 | 0.5865 | 0.5777 | 0.5137 | 0.4576 | 0.8228 | 0.3741 | 0.3403 |
| none | 689 | 0.8461 | 0.8718 | 0.7896 | 0.7802 | 0.3140 | 0.1566 | 0.0633 |
| number | 640 | 0.9109 | 0.9267 | 0.8531 | 0.8436 | 0.2422 | 0.1119 | 0.0276 |
| paraphrase | 279 | 0.6986 | 0.9716 | 0.6308 | 0.4750 | 0.3333 | 0.2448 | 0.3324 |
| synonym | 580 | 0.8522 | 0.8967 | 0.8172 | 0.7920 | 0.3234 | 0.1392 | 0.0494 |
| upper_lower | 677 | 0.8614 | 0.9351 | 0.8449 | 0.7865 | 0.3434 | 0.1157 | 0.0581 |
| whitespace | 589 | 0.8870 | 0.9374 | 0.8200 | 0.7974 | 0.2921 | 0.1288 | 0.0288 |

## 12. Label validity and leakage risks

This is the central honesty constraint of Phase 5A.

| Label kind | Meaning | Where used |
|---|---|---|
| `inherited_from_document` | a segment (window/paragraph) carries its source document's label; NOT an independent authorship determination | all MAGE/HC3/RAID/OOD segment metrics |
| `genuine_whole_text` | a segment that IS a complete labeled corpus text (built by joining two complete human+machine texts); its label applies to the segment itself | composite separation + the fair upper bound |

Leakage risks actively considered:

- Splitting a machine document does **not** prove each paragraph was independently generated; inherited metrics are always tagged as such.
- The composites give the model no new authorship signal (both segment texts exist in the evaluation split as whole documents); they only control *composition*.
- No segment model was ever fit on composite labels or on any OOD/HC3/RAID data; training used MAGE-train segments only, selection val_tune, calibration val_cal (same group-disjoint split as Phase 3).
- The segments-v1 training labels are inherited: 127430 training segments inherit their document's label. This is the single biggest threat to segment validity and the reason segment conclusions are framed as bounded estimates, not proof.

Paraphrase/edited text is only covered through RAID attacks (inherited labels), never through a genuine segment-level paraphrase label — a known gap.

## 13. Decision: reuse final-v1 vs segments-v1

Criteria were declared before results. Measured:

| Criterion (declared) | final-v1 kept | segments-v1 (hgb_seg) kept |
|---|---|---|
| kept accuracy ≥0.85 | 0.8311 | 0.8664 |
| kept human FPR ≤0.30 | 0.2707 | 0.2234 |
| kept ECE ≤0.05 | 0.0399 | 0.0102 |
| genuine composite AUC ≥0.85 | 0.9386 | 0.9024 |
| OOD-GPT segment AUC ≥0.65 | 0.5868 | 0.6181 |
| OOD-GPT-para segment AUC ≥0.65 | 0.5501 | 0.5918 |

Interpretation:

- `final-v1` cannot clear the bar at any useful gate (Section 5).
- A dedicated `segments-v1` is scientifically justified and was trained (winner `hgb_seg`, selected lexicographically robustness-first with tie bands, Section 14). It **materially improves** in-distribution segment scoring over final-v1: MAGE-test segment AUC 0.9370 vs 0.9063, human FPR 0.2234 vs 0.2331, ECE 0.0102 vs 0.0442.
- **But** segments-v1 also fails the OOD bar (0.6181 / 0.5918), and its genuine composite AUC is slightly below final-v1's. Training on inherited segment labels improved fit to that label distribution but did not transfer.

**Decision: segment-level verdicts are NOT cleared for production display.** Neither artifact clears every declared criterion (display checks: {"hgb_seg": {"accuracy": true, "human_fpr": true, "ece": true, "auc_vs_document": true, "genuine_auc": true, "ood_auc": false}, "lr_seg": {"accuracy": false, "human_fpr": false, "ece": true, "auc_vs_document": true, "genuine_auc": false, "ood_auc": false}, "final_v1": {"accuracy": false, "human_fpr": true, "ece": true, "auc_vs_document": true, "genuine_auc": true, "ood_auc": false}}). segments-v1 is kept as a documented research artifact with its own manifest/gate/calibration for future work, and the document-level prediction remains the only user-facing verdict.

## 14. If segments-v1 exists: artifact and provenance

| Property | Value |
|---|---|
| artifact directory | `ml/models/segments-v1/` (separate from `final-v1` and `baseline-v1`, which are untouched) |
| winner / type | `hgb_seg` (hist gradient boosting trained on MAGE-train segments) |
| calibrator | isotonic (fit on val_cal SEGMENTS) |
| segment gate | min_words = 120 |
| pipeline | phase5a_segments_v1 |
| status | research_only_not_adopted_for_production |
| selection order | ood_auc_mean → calibration_brier → false_positive_behavior → pr_auc → in_distribution_performance |
| hashed files | 7 |

Head-to-head on IDENTICAL segment frames (final-v1 uses its own feature state/model/calibrator):

| Artifact | mage_test AUC | hc3 AUC | raid AUC | ood_gpt AUC | ood_gpt_para AUC | composite AUC |
|---|---|---|---|---|---|---|
| hgb_seg | 0.9370 | 0.8531 | 0.7675 | 0.6181 | 0.5918 | 0.9024 |
| lr_seg | 0.8645 | 0.8510 | 0.7135 | 0.6308 | 0.6072 | 0.8209 |
| final_v1_reference | 0.9063 | 0.8376 | 0.8024 | 0.5871 | 0.5503 | 0.9330 |

Human-FPR on the same frames:

| Artifact | mage_test FPR | hc3 FPR | raid FPR | ood_gpt FPR | ood_gpt_para FPR | composite FPR |
|---|---|---|---|---|---|---|
| hgb_seg | 0.2234 | 0.3899 | 0.4364 | 0.4023 | 0.4023 | 0.2938 |
| lr_seg | 0.3081 | 0.3119 | 0.3515 | 0.3594 | 0.3594 | 0.3450 |
| final_v1_reference | 0.2331 | 0.4054 | 0.3703 | 0.3661 | 0.3661 | 0.1650 |

Data policy (identical in spirit to Phase 3, never overlapping evaluation sets):

| Role | Source |
|---|---|
| train | MAGE-train windows/paragraphs (INHERITED document label) |
| selection | val_tune (group-disjoint 60% of MAGE-val) |
| calibration | val_cal (group-disjoint 40% of MAGE-val) |
| eval only | ['MAGE-test', 'HC3', 'RAID', 'OOD-GPT', 'OOD-GPT-para', 'composite (genuine)'] |
| feature state | refit on TRAINING segments only |

Label semantics recorded in the manifest: training segments carry **inherited** document labels; genuine composite labels are evaluation-only. The artifact is a research record, not an adoption.

## 15. Tests

Executed: `python -m pytest tests/ -q` → **157 passed** (logs/pytest.log). Phase-4 baseline 128 all still green; Phase-5A adds 29 tests. Existing tests were not modified or weakened.

| Test file | Cases | Covers |
|---|---|---|
| `tests/test_phase5a_segmentation.py` | 14 | determinism · paragraph boundaries · windowing · short paragraphs · Unicode · offsets/spans · ordering · no text loss · no overlap · heading merge · hard split · empty input |
| `tests/test_phase5a_inference.py` | 15 | segment scoring via the final-v1 path · determinism · separation · segment evidence gate (abstain, no confidence/explanations) · aggregation math · final-v1 manifest hash integrity · segments-v1 separate artifact · repetition failure mode · API backward compatibility (segments stays []) |

## 16. Known limitations

- **Segment labels are inherited**, not independent; every inherited-label metric carries that caveat.
- **No genuine segment-level paraphrase/edit labels exist**; RAID attacks only approximate them.
- **Calibration does not transfer** to complete short texts (composite ECE ~0.08 vs window ECE ~0.01), and saturates near 0/1 on many segments.
- **Repetition inflates segment scores** (human FPR 0.60 on the top repetition decile); a segment-level view is MORE exposed to this than a document-level view.
- **OOD remains the ceiling**: no segment model reached the 0.65 OOD bar, so segment output is unsafe for paraphrased/newer-generator text.
- Corpus structure limits paragraph-based validation: MAGE has almost no blank-line paragraphs, so windows carry most of the segment load.
- Single seed, no CIs; segments-v1 not cross-validated across seeds.

## 17. Phase 5B frontend handoff

**Recommendation: do NOT build a segment-level verdict UI on current evidence.** The safest alternative is to keep Phase-5B document-level and, if segment insight is desired, add an explicitly non-decisional layer:

| Option | Recommendation |
|---|---|
| Per-segment human/machine verdicts | NO — neither artifact clears the display criteria |
| Segment probabilities as decorative 'style signal' bars with abstention under 120 words and 'indicative only' labels | MAYBE, clearly caveated, document verdict stays authoritative |
| Document-level only (status quo) | SAFE — keep for Phase 5B main UI |

If segments are ever surfaced, Phase 5B must:

1. use `Predictor.analyze_segments` (never re-implement scoring);
2. apply the segments-v1 gate (min_words = 120) and abstain below it;
3. render segments as non-authoritative indicators next to the document verdict, with the mixed-band and disclaimer semantics from Phase 4;
4. never imply authorship; keep the OOD caveat visible.

Next research steps if segment display is revisited: genuine segment-level labels (annotated paraphrase/mixed authorship), a segment-calibrated calibrator on complete short texts, and an OOD segment benchmark above 0.65 AUC.
