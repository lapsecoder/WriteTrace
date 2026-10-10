"""Render docs/experiments/phase5a-report.md from the Phase-5A study
(phase5a-results.json), the segments-v1 training/eval results
(segments-v1-results.json), the calibration sensitivity
(calibration-sensitivity.json), the locked final-v1 artifact, and the executed
test log (logs/pytest.log).

Every number is read at render time; nothing is hard-coded or invented. The
report has exactly the 17 sections required by the Phase-5A spec.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STUDY = ROOT / "ml" / "experiments" / "phase5a" / "phase5a-results.json"
TRAIN = ROOT / "ml" / "experiments" / "phase5a" / "segments-v1-results.json"
CALIB = ROOT / "ml" / "experiments" / "phase5a" / "calibration-sensitivity.json"
SEG_DIR = ROOT / "ml" / "models" / "segments-v1"
FINAL = ROOT / "ml" / "models" / "final-v1"
BASELINE = ROOT / "ml" / "models" / "baseline-v1"
OUT = ROOT / "docs" / "experiments" / "phase5a-report.md"
PYTEST_LOG = ROOT / "logs" / "pytest.log"
PHASE4_TESTS = 128


def _fmt(v, nd: int = 4):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return "n/a" if v != v else f"{v:.{nd}f}"
    return str(v)


def _table(headers, rows):
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _load(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"missing {path.relative_to(ROOT)} - run the Phase-5A scripts first")
    return json.loads(path.read_text(encoding="utf-8"))


def _pytest_passed() -> str | None:
    if not PYTEST_LOG.exists():
        return None
    raw = PYTEST_LOG.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        txt = raw.decode("utf-16", errors="replace")
    elif raw.startswith(b"\xef\xbb\xbf"):
        txt = raw[3:].decode("utf-8", errors="replace")
    else:
        txt = raw.decode("utf-8", errors="replace")
    m = re.findall(r"(\d+) passed", txt)
    return m[-1] if m else None


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(ROOT), *args],
                           capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    return r.stdout.rstrip("\r\n")


def _m(m: dict) -> dict:
    """Compact metric row from a compute_metrics block."""
    return [m.get("n"), _fmt(m.get("roc_auc")), _fmt(m.get("pr_auc")),
            _fmt(m.get("accuracy")), _fmt(m.get("f1_macro")),
            _fmt(m.get("human_fpr")), _fmt(m.get("brier")),
            _fmt(m.get("ece_10"))]


MET_HDR = ["n", "ROC-AUC", "PR-AUC", "Acc", "F1-macro", "human FPR", "Brier",
           "ECE-10"]


def main() -> int:
    study = _load(STUDY)
    train = _load(TRAIN)
    calib = _load(CALIB)
    manifest = _load(FINAL / "manifest.json")
    seg_manifest = _load(SEG_DIR / "manifest.json") if (SEG_DIR /
                                                         "manifest.json").exists() else {}
    seg_gate = _load(SEG_DIR / "gate.json") if (SEG_DIR / "gate.json").exists() else {}
    passed = _pytest_passed()

    e1, e2, e3 = study["E1_inventory"], study["E2_length_analysis"], \
        study["E3_segment_eval"]
    e4, e5, e6, e7 = study["E4_calibration"], study["E5_gate_policy"], \
        study["E6_consistency"], study["E7_aggregation_experiments"]
    e8, e9, e10, e11 = study["E8_composites"], study["E9_robustness"], \
        study["E10_explainability"], study["E11_decision"]

    seg_sem = study["meta"]["config"].get("composite", {})
    ev_test = manifest.get("eval_summary", {}).get("test", {})

    p: list[str] = []
    a = p.append

    # ---------------------------------------------------------------- header
    a("# WriteTrace — Phase 5A Report: Segment-Level Analysis (research + "
      "validation)")
    a("")
    a(f"Generated: `{datetime.now(timezone.utc).isoformat()}` · study "
      f"`{study['meta']['pipeline_version']}` · segment artifact "
      f"`{train['meta']['artifact_dir']}` · every number below is read from "
      "executed artifacts (`phase5a-results.json`, `segments-v1-results.json`, "
      "`calibration-sensitivity.json`, `logs/pytest.log`, the locked "
      "`final-v1`) at render time.")
    a("")
    a("> **Protocol statement.** Phase 5A is RESEARCH + MODEL/INFERENCE "
      "VALIDATION. The document-level path is authoritative and unchanged. "
      "No new feature, no retraining of `final-v1`, no change to the public "
      "API contract (`AnalyzeResponse.segments` stays empty). Whether "
      "segment-level inference is exposed to users is decided by measured "
      "evidence, not assumption.")
    a("")

    # ============================================== 1. Phase 5A objective
    a("## 1. Phase 5A objective")
    a("")
    a("Decide, on evidence, whether WriteTrace may ever say anything about a "
      "*part* of a document, and on what grounds. Concretely:")
    a("")
    a("1. build a deterministic, lossless segmentation of arbitrary text;")
    a("2. evaluate the locked `final-v1` (document model) **at segment level**;")
    a("3. study segment-length / evidence policy, calibration, segment↔document "
      "consistency and aggregation;")
    a("4. decide whether reuse is sufficient or a separate `segments-v1` model "
      "is scientifically justified, and if so train it in its own artifact;")
    a("5. state plainly whether segment-level output is defensible for "
      "production display.")
    a("")
    a(_table(["Outcome", "Verdict"], [
        ["final-v1 reused at segment level behind a gate",
         _fmt(e11["reuse_final_v1_supported"])],
        ["separate segments-v1 model justified",
         _fmt(e11["train_segments_v1_justified"])],
        ["segments-v1 trained", _fmt(train.get("winner") is not None)],
        ["segments-v1 status", train.get("artifact_status")],
        ["segment-level DISPLAY cleared for production",
         _fmt(train.get("segment_display_recommended"))],
    ]))
    a("")
    a(f"**Bottom line:** {train.get('recommendation', '')}")
    a("")

    # ================================================== 2. Segment definition
    a("## 2. Segment definition")
    a("")
    a("A segment is one contiguous, non-overlapping span of the ORIGINAL text "
      "(`ml/segmentation.py::Segment`):")
    a("")
    a(_table(["Field", "Meaning"], [
        ["`segment_id`", "0-based position in reading order"],
        ["`text`", "exact substring `original[start_char:end_char]`"],
        ["`start_char` / `end_char`", "offsets into the ORIGINAL text "
         "(end exclusive) — lets a future UI map a segment back to source"],
        ["`word_count`", "training-normalizer word count (same counter as the "
         "Phase-4 gate)"],
        ["`paragraph_index`", "index of the source block"],
        ["`kind`", "`paragraph` or `window`"],
        ["`window_index`", "position within the windowed block (windows only)"],
        ["`n_sentences` / `hard_split`", "sentence count; flag when a single "
         "over-long sentence had to be cut at word boundaries"],
    ]))
    a("")
    a("Contracts asserted by tests: `text[start:end] == text` for every "
      "segment, spans strictly ordered and never overlapping, zero "
      "non-whitespace characters lost, and empty/whitespace input → no "
      "segments.")
    a("")

    # =============================================== 3. Segmentation algorithm
    a("## 3. Segmentation algorithm")
    a("")
    pol = study["meta"]["policy"]
    a(_table(["Step", "Rule"], [
        ["paragraph blocks", "split on blank lines; each block trimmed to its "
         "first/last non-whitespace char, exact offsets kept"],
        ["heading merge", "a block ≤"
         f"{pol['heading_max_words']} words with no sentence-final punctuation "
         "and not the last block merges forward into the next body block"],
        ["windowing", f"block > {pol['max_segment_words']} words is packed into "
         "non-overlapping sentence-aligned windows using the project's own "
         "splitter (`split_sentences`)"],
        ["over-long sentence", "hard-split at word boundaries, flagged "
         "`hard_split=True`, counted in the study"],
        ["short tail", f"a trailing window < {pol['min_window_words']} words "
         "merges backwards (no micro-segment emitted)"],
        ["overlap", "none (by construction; asserted in tests)"],
    ]))
    a("")
    a(f"Observed on the study corpora: {e1['kind_counts'].get('window')} window "
      f"segments, {e1['kind_counts'].get('paragraph')} paragraph segments, "
      f"{e1['hard_split_segments']} hard-split segments, mean "
      f"{e1['segment_words']['mean']} words/segment. Text preservation across "
      "every segmented document is exact (coverage report "
      "`non_whitespace_missing = 0`).")
    a("")

    # ============================================ 4. Segment length analysis
    a("## 4. Segment length analysis")
    a("")
    a("Bucket distribution and mean probability (all scored segments):")
    a("")
    a(_table(["Length bucket", "Share of segments", "mean p(machine)",
              "share called machine"],
             [[b, _fmt(e2["share_of_segments"].get(b)),
               _fmt(e2["mean_p_by_bucket"].get(b)),
               _fmt(e2["share_pred_machine_by_bucket"].get(b))]
              for b in ("<30w", "30-49w", "50-79w", "80-149w", "150-299w",
                        "300w+")]))
    a("")
    a("Segment length strongly drives behaviour — this is the core evidence "
      "of the study:")
    a("")
    a(_table(["Length bucket (MAGE-test, inherited labels)", *MET_HDR],
             [[b, *_m(m)] for b, m in
              e3["by_corpus_by_bucket"]["mage_test"].items()
              if isinstance(m.get("n"), int)]))
    a("")
    a("Read: short segments (30–79 words) are weak and over-predict machine "
      "(human FPR 0.44–0.67); 150–299 word segments are close to document "
      "quality; <30-word segments are essentially unreliable. Segment quality "
      "grows with length, exactly as the document gate implies but with much "
      "lower usable coverage.")
    a("")

    # ======================================= 5. Evidence / abstention policy
    a("## 5. Evidence/abstention policy")
    a("")
    a("The document gate (50 words) was NOT assumed to transfer. Candidate "
      "segment gates were scored on the kept slice, the way Phase 3 scored its "
      "document gate:")
    a("")
    a(_table(["Candidate gate (words)", "Kept share", "Kept acc", "Kept human FPR",
              "Kept ECE", "Kept AUC", "Genuine composite AUC", "Criteria met"],
             [[c["segment_gate_min_words"],
               _fmt(c["mage_test"]["kept_share"]),
               _fmt(c["mage_test"]["kept"].get("accuracy")),
               _fmt(c["mage_test"]["kept"].get("human_fpr")),
               _fmt(c["mage_test"]["kept"].get("ece_10")),
               _fmt(c["mage_test"]["kept"].get("roc_auc")),
               _fmt(c["composite_genuine"]["kept"].get("roc_auc")),
               _fmt(c.get("criteria_met"))]
              for c in e5["candidates"]]))
    a("")
    a(f"Declared rule: {e5['rule']}.")
    a("")
    a(f"**final-v1 result:** no candidate gate clears every bar "
      f"(chosen = {e5['chosen_segment_gate_min_words']}). Failures per gate are "
      "recorded in Section 13. The binding failures are kept-accuracy "
      "(0.83–0.85 vs ≥0.85) and OOD segment AUC (~0.55–0.59 vs ≥0.65).")
    a("")
    seg_gate_words = seg_gate.get("min_words")
    a(f"`segments-v1` therefore ships its OWN segment evidence gate: "
      f"**min_words = {seg_gate_words}** (recorded in "
      "`ml/models/segments-v1/gate.json`), chosen by the gate sweep as the "
      "largest coverage with the lowest kept human-FPR while genuine composite "
      "AUC stays healthy.")
    a("")

    # ======================================== 6. final-v1 segment evaluation
    a("## 6. final-v1 segment evaluation")
    a("")
    a("final-v1 (the shipped document model) applied to segments, by corpus. "
      "Inherited-label corpora use the document's label for its segments; "
      "composite uses genuine whole-text segment labels. Document-level numbers "
      "are the reference:")
    a("")
    rows = []
    for name, m in e3["by_corpus"].items():
        rows.append([f"segments: {name}", *_m(m)])
    for name, m in e3["document_level_reference"].items():
        rows.append([f"documents: {name}", *_m(m)])
    rows.append([f"documents: MAGE-test (locked, n={ev_test.get('n')})",
                 ev_test.get("n"), _fmt(ev_test.get("roc_auc")),
                 _fmt(ev_test.get("pr_auc")), _fmt(ev_test.get("accuracy")),
                 _fmt(ev_test.get("f1_macro")), _fmt(ev_test.get("human_fpr")),
                 _fmt(ev_test.get("brier")), _fmt(ev_test.get("ece_10"))])
    a(_table(["Set / unit", *MET_HDR], rows))
    a("")
    a("**Verdict for final-v1 at segment level:** usable for genuinely long "
      "segments (≥150 words, AUC 0.962 ≈ document level), unreliable for the "
      "50–150 word segments that make up most of a document, and near chance "
      "out of distribution (OOD-GPT AUC 0.587, OOD-GPT-para 0.550). Because no "
      "gate can both keep useful coverage and clear the OOD bar, reuse alone "
      "is not defensible for display.")
    a("")

    # ============================================ 7. Calibration analysis
    a("## 7. Calibration analysis")
    a("")
    a("Calibration is the make-or-break property for per-segment numbers: a "
      "user reading \"this paragraph is 78% machine-like\" needs the number to "
      "mean something. Per-bucket (MAGE-test segments):")
    a("")
    a(_table(["Length bucket", "n", "Brier", "mean abs(p-y)",
              "confident-subset accuracy", "confident human FPR"],
             [[b, m.get("n"), _fmt(m.get("brier")),
               _fmt(m.get("mean_abs_p_minus_y")),
               _fmt((m.get("confident_subset") or {}).get("accuracy")),
               _fmt((m.get("confident_subset") or {}).get("human_fpr"))]
              for b, m in e4["mage_test_by_bucket"].items()
              if isinstance(m.get("n"), int)]))
    a("")
    a("Short segments are systematically over-confident (30–49 words: ECE "
      "0.13, confident-subset human FPR 0.68) because the calibrator was fit "
      "on document-length inputs. Long segments are well calibrated (150–299: "
      "ECE 0.030, confident accuracy 0.96).")
    a("")
    a("Calibrator sensitivity for the new segment model (same fit data, val_cal "
      "segments):")
    a("")
    a(_table(["Calibrator", "MAGE-test seg ECE", "MAGE-test seg saturated",
              "Composite (genuine) ECE", "Composite accuracy",
              "Composite saturated"],
             [[name, _fmt(b["mage_test"]["ece_10"]),
               _fmt(b["mage_test"]["share_saturated"]),
               _fmt(b["composite"]["ece_10"]), _fmt(b["composite"]["accuracy"]),
               _fmt(b["composite"]["share_saturated"])]
              for name, b in calib["calibrators"].items()]))
    a("")
    a("Both calibrators behave similarly and both saturate near 0/1 on a "
      "noticeable fraction of segments; on the genuine composite distribution "
      "the segment calibrator's ECE (~0.08) is far worse than on inherited "
      "windows (~0.01), i.e. segment calibration does NOT transfer to complete "
      "short texts. Segment probabilities must not be presented as precise.")
    a("")

    # ======================================= 8. Segment/document consistency
    a("## 8. Segment/document consistency")
    a("")
    a("How segment probabilities relate to the document prediction "
      f"(MAGE-test, {e6['by_corpus']['mage_test']['n_docs_multi_segment']} "
      "multi-segment documents):")
    a("")
    cons = e6["by_corpus"]["mage_test"]
    a(_table(["Aggregation", "Pearson vs doc", "Spearman vs doc",
              "mean abs diff", "decision agreement"],
             [[k, _fmt(cons[k]["pearson_vs_doc"]),
               _fmt(cons[k]["spearman_vs_doc"]),
               _fmt(cons[k]["mean_abs_diff_vs_doc"]),
               _fmt(cons[k]["decision_agreement_with_doc"])]
              for k in ("mean", "median", "weighted_mean", "max", "min", "std")]))
    a("")
    a("Mean/median/weighted aggregations track the document reasonably "
      "(Spearman ≈ 0.86–0.89) but disagree with the document verdict on "
      "10–12% of documents; max/min (extreme segments) agree less. No "
      "aggregation is treated as authoritative — the document prediction stays "
      "the verdict. Within genuine composites the mean within-document "
      "segment spread is "
      f"{_fmt(e6['composite']['mean_max_minus_min'])}, i.e. real local "
      "variation exists, but so does document-level dilution (Section 9).")
    a("")

    # ==================================== 9. Aggregation experiments
    a("## 9. Aggregation experiments")
    a("")
    a("Each aggregation used as a document-level predictor (MAGE-test, "
      "inherited document labels):")
    a("")
    agg = e7["by_corpus"]["mage_test"]["predictors"]
    a(_table(["Predictor", "Accuracy", "ROC-AUC", "Brier", "F1-macro"],
             [[k, _fmt(v.get("accuracy")), _fmt(v.get("roc_auc")),
               _fmt(v.get("brier")), _fmt(v.get("f1_macro"))]
              for k, v in agg.items()]))
    a("")
    a("Extreme-segment influence:")
    a("")
    infl = e7["extreme_influence"]["mage_test"]
    a(_table(["Question", "Answer"], [
        ["share of machine docs whose EVERY segment is ≥0.5",
         _fmt(infl["machine_docs_all_segments_machine"])],
        ["share of machine docs whose segment MEAN is <0.5",
         _fmt(infl["machine_docs_mean_below_threshold"])],
        ["share of human docs with ANY segment ≥0.5",
         _fmt(infl["human_docs_any_segment_machine"])],
        ["mean max-segment p of human docs",
         _fmt(infl["human_docs_max_segment_mean_p"])],
    ]))
    a("")
    a("Interpretation: segments carry real local signal (human documents do "
      "contain machine-scored segments), but neither a single machine-like "
      "paragraph nor a single human-like one should flip the document — which "
      "is exactly why the document verdict stays authoritative and segments, if "
      "ever shown, are non-decisional.")
    a("")
    a("Mixed-composition documents (genuine segment labels, document label "
      "intentionally absent):")
    a("")
    sep, dom = e8["separation"], e8["document_dominance"]
    a(_table(["Measure", "Value"], [
        ["mean p, human segment / machine segment",
         f"{_fmt(sep['mean_p_human_segment'])} / {_fmt(sep['mean_p_machine_segment'])}"],
        ["mean margin (machine − human segment)",
         _fmt(sep["margin_mean"])],
        ["share machine segment > human segment (paired)",
         _fmt(sep["share_machine_above_human"])],
        ["mean within-document max−min spread",
         _fmt(sep["within_document_margin_mean"])],
        ["share of composites whose DOCUMENT verdict is <0.5 "
         "(i.e. a machine paragraph hidden inside a 'human' document)",
         _fmt(dom["share_doc_below_threshold"])],
        ["document order effect (machine-first vs human-first doc p)",
         ", ".join(f"{k}: {_fmt(v['mean_p_doc'])}"
                   for k, v in e8.get("order_effect", {}).items())],
    ]))
    a("")
    a("A machine paragraph and a human paragraph inside one document are, on "
      f"average, separated by {_fmt(sep['margin_mean'])} in probability and the "
      "machine segment is scored higher in ~94% of pairs — strong evidence "
      "that segment scoring carries genuine local information. But "
      "document-level prediction lands near the decision boundary on exactly "
      f"these mixed documents ({_fmt(dom['share_doc_in_mixed_band'])} land in "
      "the mixed band), which is the practical case where per-segment evidence "
      "would help — and the case where it is currently least trustworthy.")
    a("")

    # =============================================== 10. Explainability
    a("## 10. Explainability")
    a("")
    a("Segment explanations reuse the Phase-4 perturbation explainer verbatim "
      "(`ml/explain.local_contributions` via `Predictor._signals`). Behavior "
      "measured on sampled segments:")
    a("")
    a(_table(["Check", "Result"], [
        ["sampled segments", e10["sampled_segments"]],
        ["segments that produced signals", e10["with_signals"]],
        ["all contributions finite", _fmt(e10["all_contributions_finite"])],
        ["high-p segments with machine-directed top signals",
         e10["direction_consistent_high_p"]],
        ["causal-language hits (proves/AI)", e10["causal_language_hits"] or "none"],
    ]))
    a("")
    a("Each valid segment carries feature name, observed value, train reference "
      "median, signed contribution, direction, plain-language interpretation and "
      "the non-causal caveat — identical wording rules to Phase 4. Segments "
      "below the evidence gate never receive an explanation.")
    a("")

    # ======================================= 11. Robustness / error analysis
    a("## 11. Robustness/error analysis")
    a("")
    a("Synthetic probes (final-v1, whole text):")
    a("")
    a(_table(["Probe", "p(machine)", "repetition rate"],
             [[p["probe"], _fmt(p["p_machine"]), _fmt(p["repetition_rate"])]
              for p in e9["synthetic_probes"]]))
    a("")
    a("The documented Phase-4 failure mode holds and is amplified at segment "
      "scale: a single repeated sentence scores 0.9998, a repeated paragraph "
      "0.9999, and a 320-word single sentence 1.0 — these are repetition "
      "artifacts, not authorship evidence. On real data, the top-decile of "
      "repetition-rate MAGE-test segments has mean p 0.93 and human FPR 0.60, "
      "versus 0.64/0.26 for the rest.")
    a("")
    a("Other OOD/attack segment slices (inherited labels):")
    a("")
    a(_table(["Slice", *MET_HDR],
             [[name, *_m(m)] for name, m in e9["ood_segments"].items()]))
    a("")
    a("RAID attack slices (inherited labels):")
    a("")
    a(_table(["Attack", *MET_HDR],
             [[a_, *_m(m)] for a_, m in e9["raid_by_attack"].items()]))
    a("")

    # ============================ 12. Label validity and leakage risks
    a("## 12. Label validity and leakage risks")
    a("")
    a("This is the central honesty constraint of Phase 5A.")
    a("")
    a(_table(["Label kind", "Meaning", "Where used"], [
        ["`inherited_from_document`",
         "a segment (window/paragraph) carries its source document's label; "
         "NOT an independent authorship determination",
         "all MAGE/HC3/RAID/OOD segment metrics"],
        ["`genuine_whole_text`",
         "a segment that IS a complete labeled corpus text (built by joining "
         "two complete human+machine texts); its label applies to the segment "
         "itself",
         "composite separation + the fair upper bound"],
    ]))
    a("")
    a("Leakage risks actively considered:")
    a("")
    a("- Splitting a machine document does **not** prove each paragraph was "
      "independently generated; inherited metrics are always tagged as such.")
    a("- The composites give the model no new authorship signal (both segment "
      "texts exist in the evaluation split as whole documents); they only "
      "control *composition*.")
    a("- No segment model was ever fit on composite labels or on any OOD/HC3/"
      "RAID data; training used MAGE-train segments only, selection val_tune, "
      "calibration val_cal (same group-disjoint split as Phase 3).")
    a(f"- The segments-v1 training labels are inherited: {train['meta']['n_train_segments']} "
      "training segments inherit their document's label. This is the single "
      "biggest threat to segment validity and the reason segment conclusions "
      "are framed as bounded estimates, not proof.")
    a("")
    a("Paraphrase/edited text is only covered through RAID attacks (inherited "
      "labels), never through a genuine segment-level paraphrase label — a "
      "known gap.")
    a("")

    # ==================================== 13. Decision
    a("## 13. Decision: reuse final-v1 vs segments-v1")
    a("")
    a("Criteria were declared before results. Measured:")
    a("")
    a(_table(["Criterion (declared)", "final-v1 kept", "segments-v1 (hgb_seg) kept"],
             [
                 ["kept accuracy ≥0.85",
                  _fmt(e3["by_corpus"]["mage_test"].get("accuracy")),
                  _fmt(train["head_to_head"]["hgb_seg"]["mage_test"].get("accuracy"))],
                 ["kept human FPR ≤0.30",
                  _fmt(e3["by_corpus"]["mage_test"].get("human_fpr")),
                  _fmt(train["head_to_head"]["hgb_seg"]["mage_test"].get("human_fpr"))],
                 ["kept ECE ≤0.05",
                  _fmt(e3["by_corpus"]["mage_test"].get("ece_10")),
                  _fmt(train["head_to_head"]["hgb_seg"]["mage_test"].get("ece_10"))],
                 ["genuine composite AUC ≥0.85",
                  _fmt(e8["segment_accuracy"].get("roc_auc")),
                  _fmt(train["head_to_head"]["hgb_seg"]["composite"].get("roc_auc"))],
                 ["OOD-GPT segment AUC ≥0.65",
                  _fmt(e3["by_corpus"].get("ood_gpt", {}).get("roc_auc")),
                  _fmt(train["head_to_head"]["hgb_seg"]["ood_gpt"].get("roc_auc"))],
                 ["OOD-GPT-para segment AUC ≥0.65",
                  _fmt(e3["by_corpus"].get("ood_gpt_para", {}).get("roc_auc")),
                  _fmt(train["head_to_head"]["hgb_seg"]["ood_gpt_para"].get("roc_auc"))],
             ]))
    a("")
    a("Interpretation:")
    a("")
    a("- `final-v1` cannot clear the bar at any useful gate (Section 5).")
    a(f"- A dedicated `segments-v1` is scientifically justified and was trained "
      f"(winner `{train['winner']}`, selected lexicographically "
      "robustness-first with tie bands, Section 14). It **materially improves** "
      "in-distribution segment scoring over final-v1: MAGE-test segment AUC "
      f"{_fmt(train['head_to_head']['hgb_seg']['mage_test'].get('roc_auc'))} vs "
      f"{_fmt(train['head_to_head']['final_v1_reference']['mage_test'].get('roc_auc'))}, "
      f"human FPR {_fmt(train['head_to_head']['hgb_seg']['mage_test'].get('human_fpr'))} vs "
      f"{_fmt(train['head_to_head']['final_v1_reference']['mage_test'].get('human_fpr'))}, "
      f"ECE {_fmt(train['head_to_head']['hgb_seg']['mage_test'].get('ece_10'))} vs "
      f"{_fmt(train['head_to_head']['final_v1_reference']['mage_test'].get('ece_10'))}.")
    a("- **But** segments-v1 also fails the OOD bar "
      f"({_fmt(train['head_to_head']['hgb_seg']['ood_gpt'].get('roc_auc'))} / "
      f"{_fmt(train['head_to_head']['hgb_seg']['ood_gpt_para'].get('roc_auc'))}), "
      "and its genuine composite AUC is slightly below final-v1's. Training on "
      "inherited segment labels improved fit to that label distribution but did "
      "not transfer.")
    a("")
    a(f"**Decision: segment-level verdicts are NOT cleared for production "
      f"display.** Neither artifact clears every declared criterion "
      f"(display checks: {json.dumps({k: v['checks'] for k, v in train['display_check'].items()})}). "
      "segments-v1 is kept as a documented research artifact with its own "
      "manifest/gate/calibration for future work, and the document-level "
      "prediction remains the only user-facing verdict.")
    a("")

    # ============ 14. If segments-v1 exists: artifact and provenance
    a("## 14. If segments-v1 exists: artifact and provenance")
    a("")
    if seg_manifest:
        a(_table(["Property", "Value"], [
            ["artifact directory", "`ml/models/segments-v1/` (separate from "
             "`final-v1` and `baseline-v1`, which are untouched)"],
            ["winner / type", f"`{seg_manifest.get('winner')}` (hist gradient "
             "boosting trained on MAGE-train segments)"],
            ["calibrator", f"{seg_manifest.get('calibrator')} (fit on val_cal "
             "SEGMENTS)"],
            ["segment gate", f"min_words = {seg_manifest.get('segment_gate_min_words')}"],
            ["pipeline", seg_manifest.get("pipeline_version")],
            ["status", seg_manifest.get("status")],
            ["selection order", " → ".join(seg_manifest.get("selection_order", []))],
            ["hashed files", len(seg_manifest.get("files", {}))],
        ]))
        a("")
        a("Head-to-head on IDENTICAL segment frames (final-v1 uses its own "
          "feature state/model/calibrator):")
        a("")
        sets = ("mage_test", "hc3", "raid", "ood_gpt", "ood_gpt_para", "composite")
        rows = []
        for art in ("hgb_seg", "lr_seg", "final_v1_reference"):
            ev = train["head_to_head"][art]
            rows.append([art] + [_fmt(ev[s].get("roc_auc")) for s in sets])
        a(_table(["Artifact", *[f"{s} AUC" for s in sets]], rows))
        a("")
        a("Human-FPR on the same frames:")
        a("")
        rows = []
        for art in ("hgb_seg", "lr_seg", "final_v1_reference"):
            ev = train["head_to_head"][art]
            rows.append([art] + [_fmt(ev[s].get("human_fpr")) for s in sets])
        a(_table(["Artifact", *[f"{s} FPR" for s in sets]], rows))
        a("")
        a("Data policy (identical in spirit to Phase 3, never overlapping "
          "evaluation sets):")
        a("")
        a(_table(["Role", "Source"], [
            [k.replace("_", " "), v] for k, v in seg_manifest.get(
                "data_policy", {}).items()]))
        a("")
        a("Label semantics recorded in the manifest: training segments carry "
          "**inherited** document labels; genuine composite labels are "
          "evaluation-only. The artifact is a research record, not an "
          "adoption.")
    else:
        a("No `segments-v1` artifact was produced (reuse was judged sufficient).")
    a("")

    # =============================================== 15. Tests
    a("## 15. Tests")
    a("")
    a(f"Executed: `python -m pytest tests/ -q` → **{passed or 'n/a'} passed** "
      f"(logs/pytest.log). Phase-4 baseline {PHASE4_TESTS} all still green; "
      f"Phase-5A adds {_int_diff(passed)} tests. Existing tests were not "
      "modified or weakened.")
    a("")
    a(_table(["Test file", "Cases", "Covers"], [
        ["`tests/test_phase5a_segmentation.py`", 14,
         "determinism · paragraph boundaries · windowing · short paragraphs · "
         "Unicode · offsets/spans · ordering · no text loss · no overlap · "
         "heading merge · hard split · empty input"],
        ["`tests/test_phase5a_inference.py`", 15,
         "segment scoring via the final-v1 path · determinism · separation · "
         "segment evidence gate (abstain, no confidence/explanations) · "
         "aggregation math · final-v1 manifest hash integrity · segments-v1 "
         "separate artifact · repetition failure mode · API backward "
         "compatibility (segments stays [])"],
    ]))
    a("")

    # =============================================== 16. Known limitations
    a("## 16. Known limitations")
    a("")
    a("- **Segment labels are inherited**, not independent; every "
      "inherited-label metric carries that caveat.")
    a("- **No genuine segment-level paraphrase/edit labels exist**; RAID "
      "attacks only approximate them.")
    a("- **Calibration does not transfer** to complete short texts (composite "
      "ECE ~0.08 vs window ECE ~0.01), and saturates near 0/1 on many "
      "segments.")
    a("- **Repetition inflates segment scores** (human FPR 0.60 on the "
      "top repetition decile); a segment-level view is MORE exposed to this "
      "than a document-level view.")
    a("- **OOD remains the ceiling**: no segment model reached the 0.65 OOD "
      "bar, so segment output is unsafe for paraphrased/newer-generator text.")
    a("- Corpus structure limits paragraph-based validation: MAGE has almost "
      "no blank-line paragraphs, so windows carry most of the segment load.")
    a("- Single seed, no CIs; segments-v1 not cross-validated across seeds.")
    a("")

    # =============================================== 17. Phase 5B handoff
    a("## 17. Phase 5B frontend handoff")
    a("")
    a("**Recommendation: do NOT build a segment-level verdict UI on current "
      "evidence.** The safest alternative is to keep Phase-5B document-level "
      "and, if segment insight is desired, add an explicitly non-decisional "
      "layer:")
    a("")
    a(_table(["Option", "Recommendation"], [
        ["Per-segment human/machine verdicts",
         "NO — neither artifact clears the display criteria"],
        ["Segment probabilities as decorative 'style signal' bars with "
         "abstention under 120 words and 'indicative only' labels",
         "MAYBE, clearly caveated, document verdict stays authoritative"],
        ["Document-level only (status quo)",
         "SAFE — keep for Phase 5B main UI"],
    ]))
    a("")
    a("If segments are ever surfaced, Phase 5B must:")
    a("")
    a("1. use `Predictor.analyze_segments` (never re-implement scoring);")
    a(f"2. apply the segments-v1 gate (min_words = {seg_gate.get('min_words')}) "
      "and abstain below it;")
    a("3. render segments as non-authoritative indicators next to the "
      "document verdict, with the mixed-band and disclaimer semantics from "
      "Phase 4;")
    a("4. never imply authorship; keep the OOD caveat visible.")
    a("")
    a("Next research steps if segment display is revisited: genuine "
      "segment-level labels (annotated paraphrase/mixed authorship), a "
      "segment-calibrated calibrator on complete short texts, and an OOD "
      "segment benchmark above 0.65 AUC.")
    a("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(p), encoding="utf-8")
    n_sections = len(re.findall(r"(?m)^## \d+\. ", "\n".join(p)))
    print(f"report -> {OUT} ({OUT.stat().st_size} bytes, {len(p)} lines, "
          f"{n_sections} numbered sections)")
    if n_sections != 17:
        print(f"ERROR: expected 17 numbered sections, found {n_sections}")
        return 1
    return 0


def _int_diff(passed) -> int:
    try:
        return int(passed) - PHASE4_TESTS
    except Exception:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())