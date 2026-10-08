# WriteTrace — Phase 2 Report: Robustness & Generalization (`baseline-v1`)

Generated: `2026-10-07T13:47:51.721898+00:00` · run id `phase2-0723c78104b4` · config sha256 `0723c78104b4bdca54398d2940056ea484793a2bcc49a98b69805d0ff37bd6fa` · model `hist_gradient_boosting`

> **Lock statement.** The Phase-1 baseline artifacts (model, isotonic calibrator, feature state, threshold 0.5) were evaluated as-is. No retraining, no tuning, no threshold changes were made on any evaluation set. Experimental models (E1 single-generator, E5 LODO) are trained only on MAGE train/val, use the Phase-1 hyperparameters, and are reported separately from the baseline. Canonical label convention throughout: **y=1 machine, y=0 human**; scores are always `1 - calibrated stored score` (see Section 2).

## 1. Summary & key findings

| Setting | n | ROC-AUC | PR-AUC | F1-macro | ECE-10 | mean|p-y| |
|---|---|---|---|---|---|---|
| MAGE test (in-domain, locked) | 51995 | 0.9586 | 0.9770 | 0.8809 | 0.0067 | 0.1521 |
| MAGE OOD GPT-4 set | 1562 | 0.6907 | 0.6278 | 0.6291 | 0.1772 | 0.3972 |
| MAGE OOD GPT-4 + paraphrase | 2362 | 0.6026 | 0.7114 | 0.5106 | 0.3350 | 0.4882 |
| HC3 (ChatGPT Jan-2023) | 64952 | 0.8277 | 0.7779 | 0.6917 | 0.1762 | 0.3392 |
| RAID clean (labelled subset) | 15843 | 0.8386 | 0.8520 | 0.7499 | 0.0948 | 0.2867 |

**Findings (all measured, nothing tuned):**

1. **In-domain performance holds and reproduces bit-identically** (ROC-AUC 0.9586, F1-macro 0.8809, ECE 0.0067, n=51995); repeated scoring is bit-identical (True).
2. **Unseen generators degrade gracefully-to-badly depending on the corpus.** Supplied GPT-4 OOD sets fall to ROC-AUC 0.6907 (paraphrase variant 0.6026); HC3 to 0.8277 with human false-positive rate 0.408; RAID clean to 0.8386.
3. **Single-generator training collapses cross-generator generalization**: a ChatGPT-only model reaches ROC-AUC 0.5847 vs 0.9606 for multi-generator training on the same evaluation rows (ΔPR-AUC gap 0.2576).
4. **Paraphrase (E2) lowers machine recall by 0.0904 on RAID and 0.1892 mean p on the local rewrite — but the human control also moves (+0.2195 mean p, FPR 0.094→0.375), so paraphrasing *compresses scores toward the middle* rather than cleanly beating the detector.
5. **Human-style editing by a small local LLM (E3) barely moves detection**: recall 1.000 → 0.906 (mean Δp -0.0530); the 0.5B editor mostly performs light edits.
6. **Mixed authorship (E4) is only weakly monotonic**: mean p by machine share 0.25: 0.350 → 0.5: 0.353 → 0.75: 0.526; Spearman ρ=0.196 (p=0.0083) — a stated failure.
7. **Domain transfer (E5)**: leave-one-domain-out mean PR-AUC 0.9388 vs locked baseline 0.9732 on identical rows; worst held-out domains yelp, xsum, squad. The hswag weakness is a *length/structure confound*, diagnosed in Section 9.
8. **Short text is the reliability weak point (E6)**: <50-word ROC-AUC 0.8907 vs 0.9586 overall, with mean|p-y| 0.2086 vs 0.1521 — the reliability gate in Section 10 abstains below 50 words (0.093 of test).
9. **Temporal drift (E7)** shows large class-conditional feature shifts (e.g. hapax_ratio Cohen-d -1.07 for machine text), and HC3-domain ROC ranges from 0.472 (open_qa) to 0.942 (medicine).

## 2. Scope, locks, and label orientation

**What is locked:** model `HistGradientBoostingClassifier` (learning_rate=0.1, max_leaf_nodes=63, max_iter=300, random_state=42), isotonic calibrator fitted on validation only, 73 v1 features, rare-token threshold rare_frac=0.001, decision threshold 0.5 (never tuned).

**Label orientation (the load-bearing decision of Phase 2).** The locked model was trained on MAGE stored labels in which **stored-1 == HUMAN**; therefore its raw output estimates `p(human-like)` — a *corpus-agnostic* model semantic. The canonical probability is `p(machine) = 1 - calibrated_stored_score` **for every corpus** (MAGE, HC3, RAID, OOD). Labels are mapped separately per corpus:

| Corpus | Stored label meaning | Canonical mapping | Score mapping |
|---|---|---|---|
| MAGE / OOD sets | stored-1 == human (raw CSV) | y = 1 - raw | p(machine) = 1 - p_stored_cal |
| HC3 | human_answers=0, chatgpt_answers=1 (already canonical) | y = raw | p(machine) = 1 - p_stored_cal |
| RAID | model != 'human' means machine | y = (model != 'human') | p(machine) = 1 - p_stored_cal |

This differs from the raw-file encodings but is verified empirically in the E0 audit (Section 4): on MAGE, stored-1 rate is 1.000 for human rows and 0.000 for machine rows, and the calibrated scores separate canonical classes with mean p(machine|machine)=0.880 vs mean p(machine|human)=0.213.

**Protocol deviations from the Phase-0 plan (all disclosed, none silent):**

- E1's plan text says "train on MAGE + RAID except one held-out family". RAID is **evaluation-only** in this project (never trained/calibrated/tuned on), so the single-vs-multi experiment trains on MAGE-train only; RAID appears strictly as held-out evaluation. The plan's within-MAGE held-out family is impossible because MAGE train covers every MAGE test generator family (test-only families: []); unseen-generator evidence therefore comes from the supplied GPT-4 OOD sets, HC3 and RAID.
- E7 originally conflated reproducibility with drift; reproducibility is reported as a sub-block of E7 (bit-identical scores + config/feature hashes).
- E4 mixed-authorship documents are synthetic; no claim is made about measuring the true AI share of any real document.
- E3's "human-edited" condition uses a local instruct LLM as the editor (no human editors were available); both label views coincide by construction and the limitation is stated in Section 7.

## 3. Reproducibility & run manifest

| Item | Value |
|---|---|
| Run id | `phase2-0723c78104b4` |
| Results generated at (UTC) | 2026-10-07T13:47:51.721898+00:00 |
| Config sha256 | `0723c78104b4bdca54398d2940056ea484793a2bcc49a98b69805d0ff37bd6fa` |
| Feature-names sha256 | `0fffe6bf4f79ae82232b06169c8e95a68ca5fb296175a8d43429a3f4990fbd12` |
| Model file sha256 | `25642d7945da76ddd5ecdf2a70730f66176549d24e739f183e5bde3d1b4563cb` |
| Calibrator file sha256 | `4efc07c15a54b79b235fb691942242daecd9109d91995b879dfea4aafa36309a` |
| Test meta sha256 | `e680a49607caf50ffd69fc5a673e69efc95295bde7b5500fb03724e642e9cbf4` |
| Features-test sha256 | `532116360e84e391b925149d4bf107f569b07d3300e9f5868bd2c46bdc2f133f` |
| Re-scored predictions bit-identical | True |
| RAID acquisition manifest | `ml/data/manifests/raid_phase2_acquisition.json` |
| Python / platform | 3.13.15 / Windows-11-10.0.26200-SP0 |
| Seeds | master seed 42; E2 seed 42; E3 seed 42 (all recorded in config) |
| Local models | `Qwen/Qwen2.5-0.5B-Instruct` (E2 rewrite + E3 editor) — local, free, CPU/GPU via transformers; no paid API |
| Caching | each experiment cached under `ml/experiments/phase2/cache/`; re-running resumes from cache (`--force` recomputes) |

Experiment durations of this run (seconds): E0=6.1, E1=48.5, E2=1636.1, E3=536.3, E4=29.5, E5=406.9, E6=0.3, E7=17.0, error_examples=46.9.

Reproduce:

```bash
python scripts/run_phase2.py            # all E0-E7 (+ reference, error examples)
python scripts/run_phase2.py --only E5  # single experiment
python ml/experiments/report_generator.py  # regenerate this report
python -m pytest tests/ -q
```

## 4. Label-orientation audit (E0) & locked-baseline reproduction

| Check | Evidence | Conclusion |
|---|---|---|
| MAGE stored labels | human rows n=18046, stored-1 rate 1.000; machine rows n=33949, stored-1 rate 0.000 | stored-1 == HUMAN → complement to canonical |
| HC3 stored labels | n=64952, build assigned human=0/chatgpt=1 | labels already canonical; scores still complemented |
| RAID test.csv schema | columns=['id', 'generation'] | UNLABELED (id+generation) → distribution diagnostics only |
| RAID labelled subset | rows=31943, balance {'1': 20000, '0': 11943}, policy rows 31601 | EVAL-ONLY; never trained/calibrated on |
| Score orientation | mean p(machine|machine)=0.880 > mean p(machine|human)=0.213 | sanity passed |
| Determinism | bit-identical=True | scoring is deterministic |

### Locked-baseline reproduction (canonical space, this run)

Test n=51995, machine share 0.6529, threshold 0.5 (0.5 fixed from Phase 1; never tuned on evaluation data).

| Slice / set | n | Acc | P(mach) | R(mach) | F1(mach) | F1-macro | ROC-AUC | PR-AUC | Brier | ECE-10 | mean|p-y| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hist_gradient_boosting + isotonic (locked) | 51995 | 0.8919 | 0.9185 | 0.9156 | 0.9171 | 0.8809 | 0.9586 | 0.9770 | 0.0772 | 0.0067 | 0.1521 |

|  | pred human | pred machine |
|---|---|---|
| true human | 15289 | 2757 |
| true machine | 2865 | 31084 |

Phase-1 reported the same rows with the positive class stored-1 (i.e. HUMAN) and headers that said "machine". Reproduced in that stored orientation from `test_metrics.json` for correction: Acc 0.8919, F1-macro 0.8809, ROC-AUC 0.9586, PR-AUC 0.9231 (positive = HUMAN, share 0.653), Brier 0.0772, ECE 0.0067. The confusion matrix mirrors exactly (stored tp=15289 = canonical tn=15289). All Phase-2 tables use the canonical headers above.

## 5. E1 — Unseen generators

### 5.1 Held-out generator families inside MAGE test (protocol note)

MAGE-train covers 27 machine generator families; MAGE-test contains **no** held-out families (no test-only generator families in MAGE test (train covers all test families); unseen-generator evidence comes from OOD/HC3/RAID parts below). The per-family table below is therefore *seen-family* behaviour (all rows machine):

| Generator family | n | recall@0.5 | mean p(machine) |
|---|---|---|---|
| text-davinci-002 | 2594 | 0.805 | 0.764 |
| 65B | 1291 | 0.827 | 0.812 |
| 30B | 1289 | 0.849 | 0.825 |
| text-davinci-003 | 2863 | 0.856 | 0.797 |
| 13B | 1284 | 0.868 | 0.855 |
| gpt-3.5-trubo | 3003 | 0.874 | 0.818 |
| flan_t5_large | 692 | 0.883 | 0.809 |
| flan_t5_base | 593 | 0.894 | 0.827 |
| flan_t5_xl | 819 | 0.907 | 0.817 |
| 7B | 1289 | 0.911 | 0.890 |
| GLM130B | 1242 | 0.912 | 0.897 |
| opt_125m | 1205 | 0.915 | 0.873 |
| opt_iml_max_1.3b | 1249 | 0.944 | 0.899 |
| t0_11b | 645 | 0.944 | 0.879 |
| opt_350m | 1194 | 0.946 | 0.922 |
| t0_3b | 740 | 0.946 | 0.880 |
| flan_t5_small | 504 | 0.952 | 0.876 |
| opt_6.7b | 1210 | 0.964 | 0.937 |
| flan_t5_xxl | 1002 | 0.964 | 0.939 |
| opt_1.3b | 1257 | 0.969 | 0.938 |
| opt_iml_30b | 1241 | 0.976 | 0.953 |
| opt_30b | 1218 | 0.988 | 0.976 |
| opt_2.7b | 1244 | 0.989 | 0.972 |
| gpt_j | 1055 | 0.992 | 0.986 |
| bloom_7b | 1216 | 0.993 | 0.986 |
| opt_13b | 1091 | 0.996 | 0.982 |
| gpt_neox | 919 | 0.999 | 0.982 |

Worst: text-davinci-002 (recall 0.805); best: gpt_neox (recall 0.999). The OpenAI-style families sit at the bottom of the recall range — polished human-written MAGE rows and polished OpenAI generations are closest in feature space.

### 5.2 Supplied GPT-4 OOD sets (unseen generators + unseen domains)

| Slice / set | n | Acc | P(mach) | R(mach) | F1(mach) | F1-macro | ROC-AUC | PR-AUC | Brier | ECE-10 | mean|p-y| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| test_ood_set_gpt (GPT-4 vs human) | 1562 | 0.6300 | 0.6613 | 0.5687 | 0.6116 | 0.6291 | 0.6907 | 0.6278 | 0.2561 | 0.1772 | 0.3972 |
| test_ood_set_gpt_para (+paraphrase) | 2362 | 0.5127 | 0.7454 | 0.4263 | 0.5423 | 0.5106 | 0.6026 | 0.7114 | 0.3453 | 0.3350 | 0.4882 |

The plain OOD set's confusion matrix {'tn': 529, 'fp': 233, 'fn': 345, 'tp': 455} means recall on GPT-4 text 0.569 and FPR on the OOD human rows 0.306 (those humans are also from unseen domains: cnn/imdb/pubmed/dialogsum). Source breakdown: cnn_human=200, cnn_gpt4=200, dialogsum_gpt4=200, imdb_human=200, imdb_gpt4=200, pubmed_human=200, pubmed_gpt4=200, dialogsum_human=162.

### 5.3 HC3 (unseen ChatGPT generator, conversational register)

| Slice / set | n | Acc | P(mach) | R(mach) | F1(mach) | F1-macro | ROC-AUC | PR-AUC | Brier | ECE-10 | mean|p-y| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HC3 overall | 64952 | 0.6918 | 0.5789 | 0.8416 | 0.6859 | 0.6917 | 0.8277 | 0.7779 | 0.2085 | 0.1762 | 0.3392 |

HC3 human FPR = 0.408 (n=38976 human rows) — the single largest false-positive regime found in Phase 2. Per-domain breakdown is in Section 11.

### 5.4 RAID labelled subset (11 unseen machine generators)

| Slice / set | n | Acc | P(mach) | R(mach) | F1(mach) | F1-macro | ROC-AUC | PR-AUC | Brier | ECE-10 | mean|p-y| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RAID clean, in 40–5000-word policy | 15843 | 0.7501 | 0.7386 | 0.7750 | 0.7564 | 0.7499 | 0.8386 | 0.8520 | 0.1747 | 0.0948 | 0.2867 |

Human FPR at 0.5 = 0.275 (mean p on humans 0.327); policy rows 15843 (human 7913, machine 7930).

| RAID machine model | n | recall@0.5 | mean p(machine) |
|---|---|---|---|
| cohere | 446 | 0.498 | 0.500 |
| gpt4 | 458 | 0.594 | 0.588 |
| gpt3 | 429 | 0.688 | 0.678 |
| cohere-chat | 479 | 0.706 | 0.688 |
| llama-chat | 913 | 0.779 | 0.735 |
| mistral | 995 | 0.785 | 0.792 |
| chatgpt | 487 | 0.786 | 0.744 |
| gpt2 | 909 | 0.821 | 0.807 |
| mistral-chat | 945 | 0.824 | 0.772 |
| mpt-chat | 960 | 0.849 | 0.811 |
| mpt | 909 | 0.884 | 0.876 |

**Unlabeled `test.csv` distribution diagnostics** (n=5000 reservoir-sampled of 672000 rows; no labels exist → no supervised claims): mean p 0.7317, sd 0.3195, frac ≥0.5 0.7474, frac ≥0.9 0.5048, frac ≤0.1 0.0488.

### 5.5 Single- vs multi-generator training (experimental models)

Protocol: MAGE-train: all human rows + machine rows of the named family → MAGE-test: machine rows of OTHER families + all human rows; calibration isotonic fit on MAGE-validation (all rows); no eval rows touched. experimental models: fitted on MAGE-train only, calibrated on MAGE-validation; evaluated on the same held-out rows as the locked model. NOT the baseline; baseline untouched.

| Single-generator family | train machine n | single ROC | single PR | single F1m | multi ROC | multi PR | multi F1m | ΔROC | ΔPR |
|---|---|---|---|---|---|---|---|---|---|
| gpt-3.5-trubo | 9017 | 0.5847 | 0.7188 | 0.3871 | 0.9606 | 0.9764 | 0.8847 | 0.3759 | 0.2576 |
| text-davinci-003 | 8591 | 0.6414 | 0.7618 | 0.4506 | 0.9612 | 0.9768 | 0.8855 | 0.3198 | 0.2150 |

The single-generator models separate *other* families' machine text from human text barely above chance — a one-generator detector does not generalize. This is the measured generalization gap the plan asked for.

## 6. E2 — Paraphrased machine text

### 6.1 RAID adversarial subsets (labelled, attack column = paraphrase family)

| Slice / set | n | Acc | P(mach) | R(mach) | F1(mach) | F1-macro | ROC-AUC | PR-AUC | Brier | ECE-10 | mean|p-y| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| policy rows, clean (machine+human) | 15843 | 0.7501 | 0.7386 | 0.7750 | 0.7564 | 0.7499 | 0.8386 | 0.8520 | 0.1747 | 0.0948 | 0.2867 |
| policy rows, paraphrase attack | 15739 | 0.7050 | 0.7113 | 0.6846 | 0.6977 | 0.7048 | 0.7821 | 0.7761 | 0.2031 | 0.1092 | 0.3318 |
| policy rows, other attacks (context) | 11878 | 0.7425 | 0.5862 | 0.7770 | 0.6683 | 0.7289 | 0.8409 | 0.7722 | 0.1813 | 0.1503 | 0.2990 |

RAID human control (clean human rows vs the paraphrase-attacked human rows):

| Group | n | FPR @0.5 | mean p(machine) |
|---|---|---|---|
| human, clean | 7913 | 0.275 | 0.327 |
| human, paraphrase attack | 351 | 0.396 | 0.421 |

(Note: the paraphrased-human slice is small — n=351 — so treat its FPR as indicative; the machine-side deltas above rest on n≈15739. The attack subsets contain both classes (positive rate ≈ 0.5); recall_machine is the machine-class signal.)

Deltas (paraphrase − clean, same policy rows incl. shared human rows): roc_auc=-0.0565, pr_auc=-0.0759, recall_machine=-0.0904, f1_macro=-0.0451, accuracy=-0.0451, brier=0.0284, ece_10=0.0144.

| RAID machine model | n clean | n para | recall clean | recall para | Δ recall |
|---|---|---|---|---|---|
| mpt | 909 | 883 | 0.884 | 0.651 | -0.233 |
| mistral | 995 | 933 | 0.785 | 0.570 | -0.215 |
| gpt2 | 909 | 937 | 0.821 | 0.620 | -0.201 |
| mpt-chat | 960 | 928 | 0.849 | 0.665 | -0.184 |
| mistral-chat | 945 | 875 | 0.824 | 0.752 | -0.072 |
| llama-chat | 913 | 950 | 0.779 | 0.750 | -0.028 |
| cohere-chat | 479 | 487 | 0.706 | 0.721 | 0.015 |
| chatgpt | 487 | 459 | 0.786 | 0.804 | 0.018 |
| gpt3 | 429 | 451 | 0.688 | 0.738 | 0.051 |
| cohere | 446 | 461 | 0.498 | 0.590 | 0.092 |
| gpt4 | 458 | 462 | 0.594 | 0.773 | 0.179 |

Method: RAID-train derived stratified subset (eval-only); RAID clean human rows held unchanged for both tasks; invalid normalized texts dropped: 0.

### 6.2 Local rewrite (free instruct model) + human control

Model `Qwen/Qwen2.5-0.5B-Instruct`, prompting: native chat template (system + user), instruction at the end of a bounded input (<=1500 chars, sentence-boundary cut); generation: greedy (do_sample=False); new-token budget = max(200, 1.4 x input tokens) capped at 800; seed 42; device cpu.

| Group | n | recall/FPR pre | recall/FPR post | mean p pre | mean p post | mean Δp | frac post <0.5 |
|---|---|---|---|---|---|---|---|
| machine (true origin) | 64 | 0.922 | 0.766 | 0.876 | 0.687 | -0.1892 | 0.234 |
| human control (same paraphraser) | 64 | 0.094 | 0.375 | 0.176 | 0.395 | 0.2195 | — |

Edit strength (disclosed limitation — the 0.5B paraphraser shortens text):

| Group | mean words pre | mean words post | words ratio | mean similarity (600 chars) |
|---|---|---|---|---|
| machine | 282.7 | 114.9 | 0.657 | 0.297 |
| human control | 240.4 | 124.4 | 0.730 | 0.295 |

Qualitative examples (largest machine evasions, p_pre ≥ 0.5 → p_post < 0.5):

- machine 0.877 → 0.014 · orig: "Insulting, dishonest and bottom line an absolute joke! We made reservations over a month ago, arrived and waited over 40 minutes but that wasn't the problem, the problem was the s…" · paraphrase: "Insulting, dishonest, and bottom-line jokes! We reserved over a month ago, arrived late but didn’t mind; the issue was the service. Service was slow, causing us to wait 40 minutes…"
- machine 0.978 → 0.141 · orig: "Notts' 39-run win against Essex at Trent Bridge on Monday sees them return to the showpiece event at Edgbaston for the first time since 2010. Read, who has won a Test cap and an O…" · paraphrase: "Notts' 39-run win over Essex at Trent Bridge marks the club's first appearance at Edgbaston since 2010. Nottinghamshire's men in white are set to face off against Lancashire in th…"
- machine 0.965 → 0.146 · orig: "In recent times, using small data to train networks has become a hot topic in the field of deep learning. Reusing pre-trained parameters is one of the most important strategies fo…" · paraphrase: "In recent years, utilizing small datasets to train neural networks has gained prominence in the field of deep learning. One crucial strategy for achieving state-of-the-art models …"
- **human control flip** 0.000 → 0.907 · orig: "Standing on a escalator - even if it's to one side so as to let people pass - completely undermines the point of the device. The point of escalators is to increase the flow of hum…" · paraphrase: "Standing on an escalator, even if it's to one side to allow passage, completely undermines the purpose of the device. Escalators are designed to increase traffic flow in congested…"

**Reading:** paraphrasing genuinely lowers machine scores (6 of 64 sampled machine rows crossed below 0.5; RAID Δrecall -0.0904) — but it also inflates human scores through the same pipeline (local control FPR 0.094→0.375; RAID paraphrased-human FPR 0.275→0.396). The honest claim is "paraphrase compresses the score distribution and causes real evasion + real false positives", not "paraphrase beats the detector".

## 7. E3 — Human-edited machine text (local instruct editor)

Model `Qwen/Qwen2.5-0.5B-Instruct` (local, free, CPU/GPU via transformers; no paid API); prompting: native chat template (system + user); inputs cut at a sentence boundary <=1500 chars; decode sliced by token count; temperature 0.3, top_p 0.9, seed 42, max_new_tokens 400. Prompt template (verbatim): `Edit this so it reads like it was written by a person: vary sentence lengths, add contractions and personal asides, keep all the original information, and avoid list-like parallelism. Output only the edited text.  Text:…`

| View | n | recall pre | recall post | mean p pre | mean p post | mean Δp | frac post <0.5 |
|---|---|---|---|---|---|---|---|
| machine-origin (y=1) | 32 | 1.000 | 0.906 | 0.923 | 0.870 | -0.0530 | 0.094 |

Non-empty edits: 32/32. Post metrics: positive_rate=1.0000, threshold=0.5000, accuracy=0.9062, precision_machine=1.0000, recall_machine=0.9062, f1_machine=0.9508, precision_macro=0.5000, recall_macro=0.4531, f1_macro=0.4754, roc_auc=n/a, pr_auc=n/a, brier=0.0608, ece_10=0.1304, mean_abs_p_minus_y=0.1304.

Both label views (plan requires both):

- **machine_origin**: y=1, recall 0.906 — ground truth = authorship lineage (originally machine-written)
- **predominant_authorship**: y=1, recall 0.906 — under LLM editing the predominant author of the final text is the editing model, so this view is ALSO machine; the views coincide by construction here (no human editors were available for this study) - stated, not hidden

Qualitative examples:

- p 1.000 → 0.251 · orig: "Consider if the kids you are inviting to the wedding will be okay on their own without their parents at the same table. If so, a kids table may be the best solution. Kids' Menus A…" · edited: "Consider inviting kids to the wedding if they are okay without their parents at the same table. If so, a kids table might be the best solution. Kids' menus should be light and kid…"
- p 1.000 → 0.608 · orig: "Both feature selection and hyperparameter tuning are key tasks in machine learning. Hyperparameter tuning is often useful to increase model performance, while feature selection is…" · edited: "Both feature selection and hyperparameter tuning are crucial steps in machine learning. Hyperparameter tuning is particularly beneficial for enhancing model performance, whereas f…"
- p 0.608 → 0.251 · orig: "This boutique was one of my absolute favorites! I used to always shop here when I was trying to find a cute outfit for a special night out. I was very excited when I walked the bl…" · edited: "I absolutely loved this boutique! I used to go there every time I needed a cute outfit for a special night out. Walking from Barrow to this place was just right. It was located cl…"

ground truth stays MACHINE (authorship, not final form); recall drop = humanization gain. The editor performs mostly light rewrites, so the recall drop is small; a stronger editor would likely evade more — an open risk, not a pass.

## 8. E4 — Mixed authorship

| Mixture | docs | machine paras | human paras | mean p(machine) | sd | frac ≥0.5 | domains |
|---|---|---|---|---|---|---|---|
| 0.25 machine | 60 | 1 | 3 | 0.3499 | 0.3868 | 0.350 | 10 |
| 0.5 machine | 60 | 2 | 2 | 0.3531 | 0.3974 | 0.317 | 10 |
| 0.75 machine | 60 | 3 | 1 | 0.5263 | 0.4164 | 0.517 | 10 |

Monotonic in the means: **True**. Rank correlation (machine share vs document p): Spearman ρ=0.1962, p=0.0083, n=180 — rank correlation between machine share and document p(machine); failure to be monotonic is itself a finding.

### Segment-level benchmark (seeds Phase 5): 720 paragraphs (360 human / 360 machine, ground truth known by construction) scored by the locked *document-level* model:

| Slice / set | n | Acc | P(mach) | R(mach) | F1(mach) | F1-macro | ROC-AUC | PR-AUC | Brier | ECE-10 | mean|p-y| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paragraph-level | 720 | 0.8458 | 0.8136 | 0.8972 | 0.8534 | 0.8454 | 0.9380 | 0.9437 | 0.1074 | 0.0655 | 0.2012 |

mean p(machine|machine para)=0.863, mean p(machine|human para)=0.265. paragraph-level predictions on E4 documents; ground truth known by construction - an honest first segment-analysis benchmark (no segment model trained; locked document-level model applied to paragraphs). Controlled synthetic mixtures. The detector is NOT claimed to measure exact AI authorship percentages; this probes whether document-level p(machine) responds monotonically to evidence.

## 9. E5 — Domain transfer (leave-one-domain-out) & cross-corpus

Protocol: MAGE-train rows of all domains except d (stored labels); document-frequency stats refit on train-minus-d text (rare_token_ratio/mean_log_df recomputed; other 71 features are state-independent); isotonic fit on MAGE-val rows of domains != d; MAGE-test rows of domain d only. experimental models - reported separately from the locked baseline

| Held-out domain | n test | PR (LODO) | PR (locked, same rows) | ΔPR | ROC (LODO) | ROC (locked) | F1m (LODO) | F1m (locked) |
|---|---|---|---|---|---|---|---|---|
| cmv | 4347 | 0.9849 | 0.9908 | -0.0059 | 0.9660 | 0.9793 | 0.8914 | 0.9179 |
| eli5 | 6654 | 0.9270 | 0.9572 | -0.0302 | 0.9051 | 0.9490 | 0.7770 | 0.8692 |
| hswag | 3274 | 0.9813 | 0.9917 | -0.0104 | 0.7279 | 0.8751 | 0.4733 | 0.6048 |
| roct | 4689 | 0.9385 | 0.9974 | -0.0589 | 0.7647 | 0.9899 | 0.5151 | 0.9357 |
| sci_gen | 4315 | 0.9514 | 0.9779 | -0.0265 | 0.8876 | 0.9506 | 0.7832 | 0.8665 |
| squad | 5911 | 0.9243 | 0.9598 | -0.0355 | 0.9035 | 0.9518 | 0.6809 | 0.8792 |
| tldr | 4162 | 0.9509 | 0.9718 | -0.0209 | 0.8481 | 0.9140 | 0.6913 | 0.7920 |
| wp | 5494 | 0.9446 | 0.9874 | -0.0428 | 0.8965 | 0.9779 | 0.7803 | 0.9208 |
| xsum | 5893 | 0.9145 | 0.9737 | -0.0592 | 0.8032 | 0.9328 | 0.6935 | 0.8329 |
| yelp | 7256 | 0.8709 | 0.9244 | -0.0535 | 0.8913 | 0.9375 | 0.6727 | 0.8582 |

Summary: mean PR-AUC LODO 0.9388 vs locked 0.9732; min LODO PR-AUC 0.8709; worst held-out domains: yelp, xsum, squad. All 10 fold models are saved under `ml/experiments/phase2/models/` for audit.

### 9.1 hswag forensics (Phase-1 weakness, investigated not fixed)

| Slice / set | n | Acc | P(mach) | R(mach) | F1(mach) | F1-macro | ROC-AUC | PR-AUC | Brier | ECE-10 | mean|p-y| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hswag slice, canonical | 3274 | 0.9172 | 0.9576 | 0.9548 | 0.9562 | 0.6048 | 0.8751 | 0.9917 | 0.0587 | 0.0482 | 0.1186 |

Stored-space reproduction of the Phase-1 numbers: PR-AUC 0.2109 (calibrated) / 0.2114 (raw), F1-macro 0.6048 — matching Phase 1's reported 0.21 / 0.60. Why: stored-1 == HUMAN (Phase-1 header said 'machine'), human share only 0.054 of the slice.

Root cause is a **length/structure confound**: hswag human rows are 40–62 words (median 44), machine rows median 130 (mean 244.4). Human behaviour: mean p(machine)=0.651, 0.740 flagged as machine. Feature contrasts machine−human (hswag vs rest-of-test):

| Feature | hswag Δ | rest-of-test Δ |
|---|---|---|
| n_chars | 1131.32 | 150.98 |
| n_chars_no_space | 932.95 | 132.59 |
| n_words | 199.36 | 18.96 |
| avg_para_len | 199.36 | 18.96 |
| exclaim_rate_per_sent | 60.38 | -1.99 |
| second_person_rate | -26.74 | -0.63 |
| pron_you | -26.74 | -0.63 |
| max_sent_len | 14.51 | 4.61 |

The model catches hswag machine text well (recall@0.5 range 0.830–1.000) but fails on the short templated human continuations — a data-construction artifact, not a generator effect. It is documented, not patched.

### 9.2 RAID cross-corpus (hardest realistic setting)

cross-corpus supervised view (locked MAGE-trained model); policy 40-5000 words (Phase-1 inclusion range).

| Slice / set | n | Acc | P(mach) | R(mach) | F1(mach) | F1-macro | ROC-AUC | PR-AUC | Brier | ECE-10 | mean|p-y| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| abstracts | 3982 | 0.6251 | 0.7003 | 0.6966 | 0.6984 | 0.6015 | 0.6735 | 0.7876 | 0.2513 | 0.1728 | 0.3895 |
| recipes | 3956 | 0.6775 | 0.7132 | 0.8037 | 0.7557 | 0.6405 | 0.7204 | 0.8087 | 0.2258 | 0.1322 | 0.3555 |
| poetry | 3912 | 0.6887 | 0.7644 | 0.7189 | 0.7410 | 0.6754 | 0.7459 | 0.8089 | 0.2244 | 0.1536 | 0.3359 |
| news | 3972 | 0.7560 | 0.8272 | 0.7686 | 0.7968 | 0.7458 | 0.8267 | 0.8753 | 0.1708 | 0.0804 | 0.2845 |
| reddit | 3938 | 0.7542 | 0.8870 | 0.6915 | 0.7772 | 0.7515 | 0.8378 | 0.8826 | 0.1815 | 0.1394 | 0.2798 |
| wiki | 3980 | 0.7651 | 0.8507 | 0.7556 | 0.8003 | 0.7575 | 0.8472 | 0.8897 | 0.1633 | 0.0941 | 0.2727 |
| books | 3988 | 0.7766 | 0.8494 | 0.7809 | 0.8137 | 0.7673 | 0.8505 | 0.8936 | 0.1573 | 0.0783 | 0.2654 |
| reviews | 3873 | 0.7563 | 0.8960 | 0.6998 | 0.7858 | 0.7515 | 0.8648 | 0.8997 | 0.1754 | 0.1636 | 0.2788 |
| RAID overall (policy) | 31601 | 0.7249 | 0.8039 | 0.7396 | 0.7704 | 0.7136 | 0.7952 | 0.8540 | 0.1937 | 0.1138 | 0.3078 |

## 10. E6 — Length sensitivity & the reliability gate

| Length bucket | n | Acc | P(mach) | R(mach) | F1(mach) | F1-macro | ROC-AUC | PR-AUC | Brier | ECE-10 | mean|p-y| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <50w | 4852 | 0.8502 | 0.8677 | 0.9360 | 0.9006 | 0.7983 | 0.8907 | 0.9473 | 0.1087 | 0.0192 | 0.2086 |
| 50-150w | 22052 | 0.8508 | 0.8838 | 0.8795 | 0.8816 | 0.8400 | 0.9275 | 0.9564 | 0.1048 | 0.0083 | 0.2042 |
| 150-400w | 15478 | 0.9114 | 0.9404 | 0.9140 | 0.9270 | 0.9072 | 0.9720 | 0.9824 | 0.0643 | 0.0102 | 0.1311 |
| 400w+ | 9613 | 0.9757 | 0.9867 | 0.9797 | 0.9832 | 0.9696 | 0.9960 | 0.9981 | 0.0186 | 0.0029 | 0.0378 |

Fine buckets:

| Words | n | ROC-AUC | F1-macro | ECE-10 | mean|p-y| | Acc |
|---|---|---|---|---|---|---|
| 0-25w | 0 | n/a | n/a | n/a | n/a | n/a |
| 25-50w | 4852 | 0.8907 | 0.7983 | 0.0192 | 0.2086 | 0.8502 |
| 50-75w | 7150 | 0.8980 | 0.7994 | 0.0131 | 0.2206 | 0.8413 |
| 75-100w | 6155 | 0.9250 | 0.8349 | 0.0143 | 0.2098 | 0.8434 |
| 100-150w | 8747 | 0.9411 | 0.8617 | 0.0094 | 0.1869 | 0.8638 |
| 150-200w | 5301 | 0.9572 | 0.8829 | 0.0164 | 0.1619 | 0.8847 |
| 200-300w | 6733 | 0.9756 | 0.9136 | 0.0108 | 0.1248 | 0.9188 |
| 300-400w | 3444 | 0.9840 | 0.9326 | 0.0122 | 0.0958 | 0.9382 |
| 400-infw | 9613 | 0.9960 | 0.9696 | 0.0029 | 0.0378 | 0.9757 |

Gate candidates (abstain below `min_words`; threshold otherwise unchanged at 0.5):

| min_words | kept | abstained | frac abstained | ROC-AUC | PR-AUC | F1-macro | ECE-10 | Brier |
|---|---|---|---|---|---|---|---|---|
| 25 | 51995 | 0 | 0.000 | 0.9586 | 0.9770 | 0.8809 | 0.0067 | 0.0772 |
| 50 | 47143 | 4852 | 0.093 | 0.9626 | 0.9789 | 0.8871 | 0.0057 | 0.0739 |
| 75 | 39993 | 12002 | 0.231 | 0.9694 | 0.9822 | 0.8996 | 0.0059 | 0.0669 |
| 100 | 33838 | 18157 | 0.349 | 0.9754 | 0.9857 | 0.9116 | 0.0064 | 0.0595 |
| 150 | 25091 | 26904 | 0.517 | 0.9839 | 0.9913 | 0.9296 | 0.0074 | 0.0468 |
| 200 | 19790 | 32205 | 0.619 | 0.9890 | 0.9944 | 0.9427 | 0.0065 | 0.0374 |

Short-text behaviour (<50 words): n=4852, machine share 0.725, mean |p−0.5| implies mean abs confidence 0.361, fraction uncertain (0.4–0.6) 0.068, fraction extreme (≥0.9 or ≤0.1) 0.542. the gate RECOMMENDATION is derived in the report from these tables; no threshold was tuned against them

**Gate recommendation (derived from the tables above, not hand-waved):**

- **< 50 words → "insufficient evidence" (abstain).** This bucket is the clear outlier on every axis: ROC-AUC 0.8907 vs 0.9586 overall, F1-macro 0.7983 vs 0.8809, mean|p−y| 0.2086 vs 0.1521, ECE 0.0192 vs 0.0067 — and 0.542 of these rows still get *extreme* scores (overconfidence, not honest uncertainty). Cost: abstaining on 50 words abstains on only 0.093 of the test split and lifts F1-macro to 0.8871.
- **50–100 words → allowed, but low-confidence band.** ROC-AUC 0.8980–0.9250 is acceptable and ECE stays small, but accuracy/F1 sit well below the long-text regime; pair predictions with an explicit confidence band.
- **≥ 100 words → full reporting.** Metrics rise monotonically (mean|p−y| down to 0.0378 at 400+ words).
- **Threshold stays 0.5.** The gate is a *coverage* rule; no threshold was tuned against any of these tables.

## 11. E7 — Temporal / generator drift

| Slice / set | n | Acc | P(mach) | R(mach) | F1(mach) | F1-macro | ROC-AUC | PR-AUC | Brier | ECE-10 | mean|p-y| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mage_test_in_domain | 51995 | 0.8919 | 0.9185 | 0.9156 | 0.9171 | 0.8809 | 0.9586 | 0.9770 | 0.0772 | 0.0067 | 0.1521 |
| hc3_overall | 64952 | 0.6918 | 0.5789 | 0.8416 | 0.6859 | 0.6917 | 0.8277 | 0.7779 | 0.2085 | 0.1762 | 0.3392 |
| test_ood_set_gpt | 1562 | 0.6300 | 0.6613 | 0.5687 | 0.6116 | 0.6291 | 0.6907 | 0.6278 | 0.2561 | 0.1772 | 0.3972 |
| test_ood_set_gpt_para | 2362 | 0.5127 | 0.7454 | 0.4263 | 0.5423 | 0.5106 | 0.6026 | 0.7114 | 0.3453 | 0.3350 | 0.4882 |

HC3 per-domain:

| Slice / set | n | Acc | P(mach) | R(mach) | F1(mach) | F1-macro | ROC-AUC | PR-AUC | Brier | ECE-10 | mean|p-y| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| open_qa | 3585 | 0.5743 | 0.9224 | 0.5935 | 0.7222 | 0.4059 | 0.4718 | 0.9355 | 0.2738 | 0.3498 | 0.4319 |
| finance | 7950 | 0.7669 | 0.7725 | 0.8205 | 0.7958 | 0.7622 | 0.8484 | 0.8588 | 0.1597 | 0.0493 | 0.2867 |
| reddit_eli5 | 49451 | 0.6784 | 0.5035 | 0.8848 | 0.6418 | 0.6750 | 0.8582 | 0.7709 | 0.2177 | 0.2401 | 0.3484 |
| wiki_csai | 1570 | 0.8051 | 0.7588 | 0.9281 | 0.8350 | 0.7985 | 0.9041 | 0.9086 | 0.1433 | 0.1168 | 0.2550 |
| medicine | 2396 | 0.8197 | 0.7664 | 0.9608 | 0.8527 | 0.8102 | 0.9423 | 0.9448 | 0.1255 | 0.1339 | 0.2391 |

### Feature-space drift (class-matched Cohen-d, HC3 vs MAGE)

Machine text, top shifts:

| Feature | HC3 mean | MAGE mean | Cohen-d |
|---|---|---|---|
| hapax_ratio | 0.3442 | 0.5262 | -1.07 |
| ttr | 0.5267 | 0.6649 | -0.92 |
| log_ttr | 0.8731 | 0.9120 | -0.80 |
| first_person_rate | 3.6851 | 20.9454 | -0.73 |
| root_ttr | 6.7647 | 8.6815 | -0.68 |
| n_paragraphs | 1.7368 | 1.0000 | 0.68 |
| cv_para_len | 0.0661 | 0.0000 | 0.65 |
| pron_i | 1.1617 | 14.3597 | -0.65 |
| coordinator_rate | 48.4148 | 35.9971 | 0.64 |
| pron_heshe | 1.8361 | 15.4279 | -0.61 |

Human text, top shifts:

| Feature | HC3 mean | MAGE mean | Cohen-d |
|---|---|---|---|
| contraction_rate | 0.0012 | 0.0110 | -0.88 |
| pron_you | 23.1383 | 9.0183 | 0.61 |
| second_person_rate | 23.1383 | 9.0183 | 0.61 |
| root_ttr | 7.1691 | 8.2406 | -0.56 |
| subordinator_rate | 41.1469 | 30.4095 | 0.54 |
| first_person_rate | 13.5370 | 27.0050 | -0.49 |
| pron_heshe | 2.9745 | 11.8722 | -0.44 |
| pron_it | 18.8040 | 12.0687 | 0.43 |

Lexicon rates (per corpus-age narrative):

| Lexicon | MAGE human | MAGE machine | HC3 human | HC3 machine |
|---|---|---|---|---|
| hedge_rate | 4.7208 | 5.0800 | 6.6391 | 8.9290 |
| boilerplate_rate | 0.0824 | 0.1037 | 0.0484 | 0.6366 |
| transition_rate | 1.9640 | 2.3672 | 2.1779 | 2.6119 |
| first_person_rate | 27.0050 | 20.9454 | 13.5370 | 3.6851 |
| contraction_rate | 0.0110 | 0.0116 | 0.0012 | 0.0052 |

HC3's Jan-2023 ChatGPT text has notably higher boilerplate (0.6366 vs MAGE machine 0.1037) and far lower first-person rates — the lexical signature of an earlier assistant style. Drift in feature space is therefore measurable before it is measurable in metrics.

**Reproducibility sub-block:** score_bit_identical=True, config sha256 `0723c78104b4bdca…`, feature-names sha256 `0fffe6bf4f79ae82…`.

## 12. Calibration & reliability (cross-cutting)

Locked model on test: Brier 0.0772, ECE-10 0.0067 (calibrator: isotonic, fitted on validation only).

| p bin | count | mean p | observed frac machine |
|---|---|---|---|
| 0.0–0.1 | 8389 | 0.0203 | 0.0281 |
| 0.1–0.2 | 4416 | 0.1464 | 0.1454 |
| 0.2–0.3 | 2048 | 0.2575 | 0.2822 |
| 0.3–0.4 | 1702 | 0.3799 | 0.3790 |
| 0.4–0.5 | 1599 | 0.4377 | 0.4778 |
| 0.5–0.6 | 1176 | 0.5711 | 0.5961 |
| 0.6–0.7 | 2173 | 0.6207 | 0.6424 |
| 0.7–0.8 | 3131 | 0.7564 | 0.7627 |
| 0.8–0.9 | 2372 | 0.8615 | 0.8600 |
| 0.9–1.0 | 24989 | 0.9853 | 0.9828 |

Calibration does **not** transfer: Brier / ECE by regime —

| Regime | Brier | ECE-10 | mean|p-y| |
|---|---|---|---|
| MAGE test (in-domain) | 0.0772 | 0.0067 | 0.1521 |
| MAGE OOD GPT-4 | 0.2561 | 0.1772 | 0.3972 |
| MAGE OOD GPT-4+para | 0.3453 | 0.3350 | 0.4882 |
| HC3 | 0.2085 | 0.1762 | 0.3392 |
| RAID clean | 0.1747 | 0.0948 | 0.2867 |
| MAGE <50 words | 0.1087 | 0.0192 | 0.2086 |

In-domain ECE is excellent (isotonic did its job); out-of-domain the isotonic map is simply wrong for the new distribution — probabilities from any cross-corpus run should be treated as ordinal scores, not calibrated risks.

## 13. Error examples with feature attributions

Method: per-row z-scores of the 73 v1 features against MAGE-train mean/std; the top-|z| features are shown (standardized deviation, NOT SHAP). Policy: false positives of humans and false negatives of machine text, sorted by confidence in the wrong direction; threshold 0.5.

**HC3 human text flagged as machine (highest confidence)** (6 shown of 6)

- p=1.000, true=human, domain=reddit_eli5, src=reddit_eli5, words=137: "" I " is a subject and " me " is an object . The subject is the one performing the action and the object is the one on whom the action is being performed . So if you were to say "…"
  - deviations: quote_rate=28.4091 (z=6.254), pron_i=87.5912 (z=2.505), log_ttr=0.7992 (z=-2.122), first_person_rate=87.5912 (z=2.033)
- p=1.000, true=human, domain=finance, src=finance, words=72: "The T+3 settlement date only affects cash accounts. In a cash account, you need to wait until the T+3 settlement date for your funds to be available to make your next trade. But i…"
  - deviations: pron_you=125.0 (z=5.752), second_person_rate=125.0 (z=5.752), sent_len_trend=0.9982 (z=2.107), unique_bigram_ratio=0.7465 (z=-1.278)
- p=1.000, true=human, domain=reddit_eli5, src=reddit_eli5, words=90: "- If you own a share in a company this means that you are part owner of the company . The more shares you own , the more of the company you own - The stock market is just a market…"
  - deviations: bullet_rate=2.3095 (z=40.578), pron_you=100.0 (z=4.501), second_person_rate=100.0 (z=4.501), hyphen_rate=9.2379 (z=2.717)

**MAGE machine text missed (lowest confidence)** (6 shown of 6)

- p=0.000, true=machine, domain=roct, src=roct_machine_continuation_gpt-3.5-trubo, words=49: "Laney was chatting with some friends at the store. She mentioned that she hadn't gotten her Christmas tree yet. Then, her friend offered to give her her old tree. Laney was thrill…"
  - deviations: pron_heshe=183.6735 (z=5.753), hedge_rate=20.4082 (z=1.998), cv_sent_len=0.1354 (z=-1.312), std_word_len=1.8824 (z=-1.283)
- p=0.000, true=machine, domain=yelp, src=yelp_machine_continuation_GLM130B, words=175: "Off the beaten path...Sunset Road..but easy to find ad it is on the main drag...This place has home deco and finishing statement pieces which would add glamour and a touch of clas…"
  - deviations: ellipsis_rate=12.1681 (z=7.8), clause_density=5.3333 (z=2.113), min_sent_len=48.0 (z=1.953), exclaim_rate_per_sent=1333.3333 (z=1.868)
- p=0.000, true=machine, domain=squad, src=squad_machine_continuation_gpt-3.5-trubo, words=109: "The first European to travel the length of the Amazon River was Francisco de Orellana in 1542. The BBC's Unnatural Histories presents evidence that Orellana, rather than exaggerat…"
  - deviations: certainty_rate=9.1743 (z=3.185), polysyllable_ratio=0.3028 (z=2.254), avg_syllables_per_word=1.9817 (z=2.139), mean_log_df=3.5603 (z=-1.933)

**RAID machine text missed (in-policy)** (6 shown of 6)

- p=0.000, true=machine, domain=abstracts, model=gpt3, attack=none, words=142: "We report a novel class of integrated nanophotonic metamaterials for on-chip nonlinear optics with extraordinary second-order nonlinear optical responses. The metamaterials are co…"
  - deviations: hyphen_rate=11.8534 (z=3.635), polysyllable_ratio=0.331 (z=2.635), std_word_len=3.3035 (z=2.248), avg_syllables_per_word=2.0 (z=2.227)
- p=0.000, true=machine, domain=reddit, model=cohere, attack=none, words=151: "350hrs in and I still have no idea what I’m doing.. I feel like this is probably normal? Every day is completely different and I’m seeing a wide variety of patients. Absolutely lo…"
  - deviations: qmark_rate=5.1151 (z=2.657), certainty_rate=6.6225 (z=2.229), coordinator_rate=79.4702 (z=2.188), n_paragraphs=3.0 (z=2.0)
- p=0.000, true=machine, domain=reddit, model=gpt4, attack=none, words=281: "First of all, I want to thank you, guys, for taking the time to read and contribute to my post.  Background info: I am 25 years old, 5' 10'', currently weigh 175 lbs, and have bee…"
  - deviations: n_paragraphs=7.0 (z=6.0), pron_i=81.8505 (z=2.304), first_person_rate=81.8505 (z=1.853), sent_len_trend=-0.6658 (z=-1.538)

**GPT-4 OOD machine text missed** (6 shown of 6)

- p=0.000, true=machine, src=pubmed_gpt4: "Yes, comparative secretome analysis suggests low plant cell wall degrading capacity in Frankia symbionts. A study by Nguyen et al. (2018) showed that Frankia strains have a reduce…"
  - deviations: hedge_rate=26.087 (z=2.731), mean_log_df=3.49 (z=-2.233), polysyllable_ratio=0.2783 (z=1.924), rare_token_ratio=0.1565 (z=1.806)
- p=0.010, true=machine, src=pubmed_gpt4: "Yes, the L1-type cell adhesion molecule Neuroglian is necessary for maintenance of sensory axon advance in the Drosophila embryo. Studies have shown that neuroglian, a homolog of …"
  - deviations: rare_token_ratio=0.2326 (z=3.307), polysyllable_ratio=0.3372 (z=2.719), mean_log_df=3.4213 (z=-2.526), transition_rate=11.6279 (z=2.072)
- p=0.010, true=machine, src=pubmed_gpt4: "Yes, the primary metabolism of chickpea can be an initial target during the early sensing of Fusarium oxysporum f. sp. ciceri race I, which is a fungus that causes wilt disease in…"
  - deviations: median_word_len=5.5 (z=2.31), sent_len_trend=-0.9659 (z=-2.196), polysyllable_ratio=0.29 (z=2.082), rare_token_ratio=0.17 (z=2.072)

**OOD human text flagged as machine** (6 shown of 6)

- p=1.000, true=human, src=cnn_human: "A zoo in Ohio is celebrating the 'miracle' birth of three rare Amur tiger cubs after it was feared their mother's biological clock had stopped ticking. Keepers at the Columbus Zoo…"
  - deviations: trigram_repeat_rate=0.2311 (z=1.66), hapax_ratio=0.2143 (z=-1.56), hyphen_rate=5.9254 (z=1.554), unique_bigram_ratio=0.7117 (z=-1.542)
- p=1.000, true=human, src=cnn_human: "A Coast Guard cutter arrived in San Diego on Thursday with more than 14 tons of cocaine, part of what authorities has described as a surge of seizures near Central and South Ameri…"
  - deviations: trigram_repeat_rate=0.3577 (z=2.776), unique_bigram_ratio=0.5677 (z=-2.638), hapax_ratio=0.1039 (z=-2.132), ttr=0.3403 (z=-1.87)
- p=1.000, true=human, src=cnn_human: "Poundland has been been forced to pull decorative plastic Easter Eggs from their shelves over fears children may choke - because they look like Cadbury Mini Eggs. Trading Standard…"
  - deviations: last_sent_len=125.0 (z=2.582), cv_sent_len=0.9705 (z=2.217), std_sent_len=39.5466 (z=2.173), hedge_rate=21.4724 (z=2.135)

**MAGE human text flagged as machine** (6 shown of 6)

- p=1.000, true=human, domain=wp, src=wp_human, words=553: "Player 1 plays a Blue Charge Crystal Player 1 summons Dragon Whelp using 1 Blue Charge Player 2 plays a Yellow Charge Crystal Player 2 summons a Young Valiant using 1 Yellow Charg…"
  - deviations: mean_log_df=3.1724 (z=-3.588), rare_token_ratio=0.2441 (z=3.535), stopword_ratio=0.1863 (z=-3.405), cv_sent_len=1.2496 (z=3.396)
- p=1.000, true=human, domain=eli5, src=eli5_human, words=44: "What evidence do you have that they are crossing constitutional lines all the time? Additionally what evidence do you have that they are never penalized? If someone broke a law wh…"
  - deviations: subordinator_rate=159.0909 (z=6.331), enum_rate=22.7273 (z=5.229), transition_rate=22.7273 (z=4.518), qmark_rate=7.4906 (z=3.998)
- p=1.000, true=human, domain=eli5, src=eli5_human, words=78: "You will find a scorpion in places that scorpions are likely to be found. If you found one scorpion in a place that place is a place where scorpions are likely to be. Since that p…"
  - deviations: hedge_rate=76.9231 (z=9.293), subordinator_rate=128.2051 (z=4.78), aux_rate=166.6667 (z=3.387), log_ttr=0.7478 (z=-3.084)

Qualitative rewrite/edit examples for E2/E3 are embedded in Sections 6 and 7 (original + transformed text with p before/after).

## 14. Limitations, deviations, and failures

- **OOD & cross-corpus performance is the headline risk.** GPT-4 OOD ROC-AUC 0.6907, HC3 human FPR 0.408, RAID human FPR 0.275. These are measured, not projected; they bound any deployment claim.
- **hswag remains weak** (F1-macro 0.6048) due to a construction artifact (40–62-word human continuations vs hundreds-of-words machine rows); documented, not patched.
- **E2 local paraphrase shortens text** (words ratio 0.66–0.73), so the local-rewrite result confounds paraphrase with length; the labelled RAID attack subsets (same-length attack text) are the cleaner paraphrase evidence.
- **E3's editor is a 0.5B model** producing light edits; a stronger editor is an unmeasured evasion risk. No human editors were available, so the "human-edited" condition is really "LLM-edited" (both label views y=1 by construction).
- **E1 cannot hold out a MAGE family** (train covers all test families); RAID is eval-only so it never appears in training despite the plan's wording.
- **RAID `test.csv` is unlabeled** — only distribution diagnostics are reported for it; supervised RAID numbers come from the disclosed RAID-train-derived subset (31,943 rows, sha256 in the acquisition manifest).
- **E4 documents are synthetic mixtures**; Spearman ρ is weak (0.1962) — the model does not track machine share well at document level.
- **No threshold or calibrator was adapted** to any evaluation corpus, so every out-of-domain number above is a *worst-case-but-honest* transfer measurement.
- **Environment:** CPU-only run (torch CPython 3.13.15), local HF models only — no paid APIs, no GPU-dependent steps.

## 15. What this means for users

1. **Trust in-domain scores; distrust raw scores off-domain.** On MAGE-like text (acc 0.8919, ECE 0.0067) the calibrated probability is usable as stated. On GPT-4-style, conversational (HC3-like) or RAID-like text, treat the number as an ordinal suspicion score: human false-positive rates of 0.275–0.408 are realistic.
2. **Apply the reliability gate:** <50 words → report *insufficient evidence* instead of a label (covers 0.093 of test traffic, improves everything downstream); 50–100 words → label with a low-confidence band; ≥100 words → normal reporting. Threshold 0.5, never retuned.
3. **Paraphrasing changes the answer, not in a clean way.** Machine text loses ~0.09 recall under RAID paraphrase attacks, but human text through the same pipeline gains false positives — a detector facing paraphrased input should widen its uncertainty band, not silently flip labels.
4. **Single-generator or narrow training is not enough.** The measured generalization gap (Section 5.5) says training data must span many generators; monitor per-generator recall (Section 5.4 shows recall ranging 0.498–0.884 across RAID models).
5. **Expect drift to show up in features first** (Section 11): hedge/boilerplate/lexicon-rate shifts precede metric drops — alert on feature drift, not only on labelled accuracy (which needs labels you may not have).
6. **Segment-level claims are not yet supported** beyond the honest paragraph benchmark (ROC-AUC 0.9380 on constructed 4-paragraph documents); mixed-authorship documents are not reliably ranked by machine share (ρ=0.1962).
7. **Everything above is reproducible** from `phase2-results.json` with the config hash `0723c78104b4`; re-running yields bit-identical scores.

---

*All tables in this report are generated from `ml/experiments/phase2/phase2-results.json` (E0–E7) plus the locked Phase-1 artifacts. Failed or degraded results are included as measured; nothing was excluded, re-tuned, or rounded in favour of the model.*