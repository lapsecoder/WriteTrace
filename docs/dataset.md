# WriteTrace — Dataset Strategy (Phase 0)

**Status:** Research complete; acquisition not yet performed. All sources verified free/open as of 2026-06-10.

---

## 1. Candidate Datasets (verified)

| Dataset | License | Size | Domains | Generators | Role |
|---|---|---|---|---|---|
| **MAGE** (`yaful/MAGE`, HF) | **Apache-2.0** | 436,606 rows (train 319k / val 57k / test 61k), 554 MB | news, reviews, academic papers, forums (322 `src` categories) | ChatGPT 3.5/4, davinci-003, LLaMA, Alpaca, Koala, Vicuna, WizardLM, and others | **Primary training corpus** |
| **RAID** (`liamdugan/raid`, HF) | **MIT** | ~7.4M rows (train ~5.4M, extra 2.0M, test 672k) | abstracts, news, creative writing, tweets | 12 model classes + human, decoding/repetition-penalty variants; 12 adversarial attack types (incl. paraphrase family); EN/DE/CS | **Held-out robustness & adversarial evaluation** |
| **HC3** (`Hello-SimpleAI/HC3`, HF) | **CC-BY-SA-4.0**; stricter source licenses take precedence (per dataset card) | ~10k–100k Q&A pairs, EN + ZH | Reddit ELI5, OpenQA, Wiki, Medical, Yahoo (EN) | ChatGPT only (gpt-3.5-turbo, Jan 2023) | **Unseen-generator + unseen-conversational-domain test** |
| **M4** (GitHub `mbzuai-nlp/M4`) | ⚠️ **No LICENSE file in repo.** Underlying human sources vary: Wikipedia (CC-BY-SA), arXiv (CC-BY), WikiHow (CC BY-NC-SA), Reddit | jsonl per domain/generator | wikipedia, wikihow, reddit, arxiv + multilingual (AR, RU, UR, BG, ID, KZ, DE) | davinci, ChatGPT, Cohere, Dolly-v2, BLOOMZ, Flan-T5, LLaMA | Supplementary only; **license review required before use**; never redistribute raw files |
| **LLM-DetectAIve** (GitHub `mbzuai-nlp/LLM-DetectAIve`) | ⚠️ No LICENSE visible | — | essays / academic | multiple | Reference for *machine-humanized* & *machine-polished* classes (robustness design); include only if data license confirmed |

**Decision:** v0 corpus = **MAGE (train)** + **RAID (robustness test)** + **HC3 (unseen-generator/domain test)**. M4 stays out of the pipeline until licensing is confirmed; it is not needed for v0.

## 2. Label Strategy

- **Primary label (v0):** binary — `human = 0`, `machine = 1`, taken from dataset-provided labels (MAGE `label`, RAID `model="human"` vs model class, HC3 `human_answers` vs `chatgpt_answers`).
- **Metadata retained per row:** `dataset`, `domain`, `generator`, `attack`, `source_id`, `prompt_id`, language. Metadata powers slicing and grouped splits but is **never a model input** (avoids trivial leakage).
- **"Mixed / uncertain" is not a training class in v0.** It is an *inference-time outcome*: calibrated `p(machine)` near 0.5, or model disagreement, maps to the `mixed` band. LLM-DetectAIve-style humanized/polished classes are a Phase-2+ experiment, not part of v0 labels.
- HC3 human answers are informal/short Reddit text vs. polished ChatGPT answers → strong length/register confound; HC3 is therefore used as a *stress test*, never mixed into training without length stratification (see §5).

## 3. Collection Methodology (to execute in Phase 1)

1. Download via `huggingface_hub` / `git clone` (both free; these repos require no paid access).
2. Record in `ml/data/manifests/`: dataset name, revision hash, license, download date, row counts, checksums.
3. English-only filter for v1 (fast free language ID, e.g. `langdetect`/`langid`).
4. Normalize minimally: unicode NFC, whitespace collapse, URL → `[URL]` placeholder. **Punctuation is preserved** (features depend on it).
5. **Primary/secondary text views:** train on raw text *and* report an *artifact-stripped* evaluation slice (common ChatGPT lead-ins like "Sure, here's…" or "As an AI…" removed) to expose shortcut learning.
6. Keep the original text alongside all derived versions for auditability.

## 4. Split Design (contamination control)

- **Dedup before splitting:** normalized-text hash across *all* corpora; also near-dup via 5-gram shingle overlap (MinHash, free/local) to catch paraphrase-level reuse between HC3/MAGE-family Reddit overlap.
- **Grouped splits:** rows sharing a `source_id`/prompt family (human reference + its machine generations + attack variants) stay in one split. RAID attack variants must never straddle train/test.
- **Locked test sets:** RAID test (unseen generators + attacks), HC3 (unseen generator, unseen conversational domain), plus a held-out slice of MAGE generators (e.g., hold out one model family) and one domain.
- **Class balance:** training subsampled to ~150–200k rows, 50/50 by label, stratified by domain × generator (full MAGE is larger than classical models need; sampling documented and seeded).

## 5. Leakage & Confound Risk Register

| Risk | Example | Mitigation |
|---|---|---|
| Cross-dataset duplicates | Reddit Q&A appears in HC3 and M4/MAGE-family sets | Global dedup (exact + near-dup) before splitting |
| Prompt-pair leakage | Machine text generated *from* the human text's prompt | Group splits by prompt/source_id |
| Domain-label confound | Human = Reddit, Machine = formal essays → model learns domain | Within-domain human/machine pairs (MAGE/RAID provide them); domain-stratified evaluation |
| Length/format confound | ChatGPT answers longer, headed, bulleted | Length-bucket metrics; length-matched training option; formatting ablation |
| Generator-era memorization | "delve/tapestry/moreover" 2023-isms | Multi-generator training; held-out newer generator test |
| Attack-variant straddling | RAID paraphrase variants in both train and test | Group by `source_id`/`adv_source_id` |
| Shortcut features | "As an AI language model" phrases | Artifact-stripped slice reported alongside primary results |

## 6. Known Limitations (disclosed in README & UI)

- All machine text was *generated for datasets*; real-world AI text often includes human editing — hence robustness experiments E2–E4.
- Generator coverage ends where the datasets end; WriteTrace cannot claim performance on models released after the corpus.
- HC3-era ChatGPT (Jan 2023) is stylistically stale; used deliberately as an unseen/temporal test, not training data.
- Human corpora skew to Reddit/forums/news/abstracts; essay-native human corpora are limited → essay-domain claims must be cautious.
- Licenses: MAGE (Apache-2.0) and RAID (MIT) permit redistribution of derived models/metrics; HC3 CC-BY-SA requires attribution + share-alike for the dataset itself. M4 excluded until licensing is clarified.

