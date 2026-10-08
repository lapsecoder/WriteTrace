# WriteTrace — Robustness Experiment Plan (Phase 0)

Each experiment has a defined protocol, metric set, and reporting requirement. **All results — including failures — go into `docs/experiments/` with run manifests.**

Common protocol: use the frozen calibration + banding from the final model; report ROC-AUC, PR-AUC, F1, Brier, ECE, and mean |p − label| alongside qualitative error samples per experiment.

---

## E1 — Unseen Generator

- **Train:** MAGE + RAID machine text from all generators **except** one held-out family (e.g., hold out GPT-4-class or LLaMA-class), humans from all sources.
- **Test:** held-out generator's text only (+ human control set).
- **Also:** train on single-generator subsets (ChatGPT-only) vs. multi-generator to quantify the generalization gap.
- **Success looks like:** graceful degradation; report recall on machine text per generator. Expect notable drops — document them.

## E2 — Paraphrased Machine Text

- **Test-only sets:** RAID adversarial subsets whose `attack` column is paraphrase-family (human-feedback / incremental paraphrase variants), plus locally paraphrased MAGE machine samples using a free local model (e.g., T5 paraphrase or instruct model; prompts + seeds logged).
- **Compare:** clean machine text vs. paraphrased → ΔAUC, Δrecall.
- Human control set unchanged, to verify paraphrase doesn't also flip human text.

## E3 — Human-Edited Machine Text

- **Protocol:** take machine samples → instruct a **local** LLM (free, runs on RTX 5050 8 GB; e.g., Qwen2.5-7B-Instruct quantized or smaller) with a documented prompt: *"Edit this so it reads like it was written by a person: vary sentence lengths, add personal asides, break perfect parallelism."* Temperature/seed recorded.
- **Test:** edited samples labeled as **machine-origin** (ground truth = authorship, not final form) *and* a second view labeled by "predominant authorship" — both views reported.
- **Metric focus:** recall drop on edited text; calibration behavior (does p move toward 0.5 honestly?).

## E4 — Mixed Authorship

- **Synthesis:** documents built from human and machine paragraphs at ratios 25/75, 50/50, 75/25 (machine share), drawn from the same domain to avoid domain confounds.
- **Expectation:** document-level p(machine) should be *monotonic* in machine share — verify with rank correlation; failure is a finding.
- **Also seeds Phase 5:** paragraph-level predictions on these documents are the first honest segment-analysis benchmark (ground truth known by construction).

## E5 — Domain Transfer (Leave-One-Domain-Out)

- Rotate: train on {forums, news} → test {academic, reviews}; train on {academic} → test {forums}, etc., using MAGE domains.
- Report per-domain matrix; identify which writing domains break the model.
- Cross-corpus view: train on MAGE → test RAID abstracts/creative (different generators *and* domains simultaneously = hardest realistic setting).

## E6 — Length Sensitivity

- Slice all test results into <50, 50–150, 150–400, 400+ words.
- Measure metric decay + ECE growth as length shrinks; use results to **fit** the reliability gate (min-evidence thresholds), not to hand-wave.

## E7 — Temporal / Generator Drift

- HC3 (ChatGPT Jan-2023, conversational) as a time-shifted unseen-generator test; compare with newer-generator slices.
- Document what drift looks like in feature space (e.g., hedge/boilerplate lexicon rates shifting over corpus age) as part of the limitations narrative.

## Reporting Requirements

- One markdown report per experiment: setup, metrics table, calibration note, 5–10 error examples with feature attributions, and an honest "what this means for users" paragraph.
- Experiments run via `scripts/run_experiments.py`; every report links a run ID + config hash for reproducibility.
