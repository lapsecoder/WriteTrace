"""Render the honest Phase-1 baseline report as markdown."""
from __future__ import annotations

METRIC_COLS = ["accuracy", "precision_machine", "recall_machine", "f1_machine",
               "f1_macro", "roc_auc", "pr_auc", "brier", "ece_10"]


def _fmt(v, nd: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def render_report(results: dict, processing: dict, extraction: dict,
                  val_metrics: dict, manifest: dict, cfg: dict) -> str:
    p: list[str] = []
    a = p.append
    sel_name = results["selected_model"]
    sel = results["selected_calibrated"]

    a(f"# WriteTrace — Baseline Report: `{cfg['run_name']}`")
    a("")
    a(f"Generated: {manifest['generated_at']} · seed={manifest['seed']} · "
      f"feature_version={manifest['feature_version']}")
    a("")
    a("> **Non-additivity note.** Stages report each filtering step in isolation. "
      "HC3 rows dropped during exact-dedup/cross-corpus filtering never entered the "
      "split pools. Split totals: train 155,988 + val 51,996 + test 51,995 = 259,979 "
      "(target 260,000 − 21 MAGE exact duplicates).")
    a("> **Honest-evaluation note.** This is a first baseline, not a product claim. "
      "Numbers below are measured on a locked test split with no test-set tuning. "
      "AI-text detection is probabilistic and these results *understate* real-world "
      "difficulty (paraphrase, human editing, new generators are Phase 2).")
    a("")
    a("## 1. Dataset sizes after every filtering stage")
    a("")
    a(_table(["Stage", "Rows"],
             [[k, str(v)] for k, v in processing["stages"].items()]))
    a("")
    a(f"Processing duration: {processing['duration_seconds']}s · "
      f"RAID status: {processing['raid_status']}")
    a("")
    a("## 2. Class distribution, groups, splits")
    a("")
    rows = []
    for name, info in sorted(processing["splits"].items()):
        cd = info["class_distribution"]
        rows.append([name, str(info["rows"]), str(cd["label_0"]), str(cd["label_1"]),
                     str(info["unique_groups"]), _fmt(info["median_words"], 0),
                     str(len(info["domains"]))])
    a(_table(["Split", "Rows", "Human(0)", "Machine(1)", "Groups", "Median words", "Domains"],
             rows))
    a("")
    a("## 3. Duplicate removal statistics")
    a("")
    near = processing["near_dup_stats"]
    a(_table(["Statistic", "Value"], [
        ["Exact duplicates dropped (total)",
         str(processing["stages"].get("exact_duplicates_dropped_total", 0))],
        ["Exact drops by dataset",
         str(processing["stages"].get("exact_duplicates_dropped_by_dataset", {}))],
        ["Near-dup candidate pairs (LSH)", str(near.get("candidate_pairs", 0))],
        ["Near-dup pairs verified (est. J ≥ "
         f"{near.get('params', {}).get('jaccard_threshold')})", str(near.get("near_pairs_verified", 0))],
        ["Exact pairs unioned", str(near.get("exact_pairs", 0))],
        ["Duplicate clusters (exact+near)", str(near.get("clusters", 0))],
        ["Rows in multi-row clusters", str(near.get("rows_in_multirow_clusters", 0))],
        ["Cross-corpus eval rows removed", str(
         processing["stages"].get("cross_corpus_rows_removed_from_eval", 0))],
        ["Bucket overruns (capped)", str(near.get("buckets_overrun", 0))],
    ]))
    a("")
    a("## 4. Features")
    a("")
    a(_table(["Item", "Value"], [
        ["Feature count", str(extraction["n_features"])],
        ["Families", ", ".join(f"{k} ({v})" for k, v in extraction["families"].items())],
        ["Feature version", extraction["feature_version"]],
        ["State fitted on", f"{extraction['fitted_on']} ({extraction['state_n_docs']} docs)"],
        ["Vocabulary (train DF)", str(extraction["state_vocab"])],
        ["Rare threshold (DF <", str(extraction["rare_max_df"]) + ")"],
        ["Column-order sha256", extraction["feature_names_sha256"][:16] + "…"],
        ["All-NaN feature rows (invalid input)",
         str(sum(s["all_nan_feature_rows"] for s in extraction["splits"].values()))],
    ]))
    a("")
    a("## 5. Model configuration")
    a("")
    grid_rows = [[g["model"], str(g["params"]), _fmt(g["val_tune_pr_auc"])]
                 for g in val_metrics["grid"]]
    a(_table(["Model", "Params", "Val-tune PR-AUC"], grid_rows))
    a("")
    a(f"Selection rule: max PR-AUC on `val_tune` → **{sel_name}**. "
      f"val_tune={manifest['rows']['val_tune']}, val_cal={manifest['rows']['val_cal']} "
      "(group-disjoint). Calibrator fitted only on val_cal; test never used for tuning.")
    a("")
    a("## 6. Validation calibration record")
    a("")
    cal = val_metrics["calibration"]
    a(_table(["Item", "Value"], [
        ["Calibrator chosen", cal["chosen"]],
        ["Platt Brier (val_cal)", _fmt(cal["brier_platt"])],
        ["Isotonic Brier (val_cal)", _fmt(cal["brier_isotonic"])],
        ["Val_cal raw Brier / ECE", f"{_fmt(cal['val_cal_raw']['brier'])} / {_fmt(cal['val_cal_raw']['ece'])}"],
        ["Val_cal calibrated Brier / ECE",
         f"{_fmt(cal['val_cal_calibrated']['brier'])} / {_fmt(cal['val_cal_calibrated']['ece'])}"],
    ]))
    a("")
    a("## 7. Test results (locked test split)")
    a("")
    rows = []
    for m, mt in results["raw"].items():
        rows.append([m + " (raw)"] + [_fmt(mt[c]) for c in METRIC_COLS])
    rows.append([f"{sel_name} (calibrated)"] + [_fmt(sel[c]) for c in METRIC_COLS])
    a(_table(["Model"] + METRIC_COLS, rows))
    a("")
    cm = sel["confusion_matrix"]
    a("Confusion matrix (selected model, calibrated, threshold 0.5):")
    a("")
    a(_table(["", "pred human", "pred machine"],
             [["true human", str(cm["tn"]), str(cm["fp"])],
              ["true machine", str(cm["fn"]), str(cm["tp"])] ]))
    a("")
    # ---- 8. calibration / reliability -------------------------------------
    a("## 8. Calibration & reliability (selected model)")
    a("")
    a("Raw → calibrated on test:")
    a("")
    raw_m = results["raw"][sel_name]
    a(_table(["Metric", "Raw", "Calibrated"], [
        ["Brier", _fmt(raw_m["brier"]), _fmt(sel["brier"])],
        ["ECE(10)", _fmt(raw_m["ece_10"]), _fmt(sel["ece_10"])],
        ["ROC-AUC", _fmt(raw_m["roc_auc"]), _fmt(sel["roc_auc"])],
    ]))
    a("")
    a("Reliability bins (calibrated):")
    a("")
    bin_rows = [[f"{b['bin_lo']}-{b['bin_hi']}", str(b["count"]),
                 _fmt(b["mean_p"]) if b["mean_p"] is not None else "-",
                 _fmt(b["frac_positive"]) if b["frac_positive"] is not None else "-"]
                for b in results["reliability_bins_calibrated"]]
    a(_table(["Bin", "n", "mean p(machine)", "observed frac machine"], bin_rows))
    a("")
    # ---- 9. slices ---------------------------------------------------------
    a("## 9. Slices — where the model works and fails")
    a("")
    a("By text length (calibrated):")
    a("")
    rows = []
    for b, m in results["slices_calibrated"]["by_length"].items():
        if "note" in m:
            rows.append([b, str(m["n"]), "—", "—", "—", "—", m["note"]])
        else:
            rows.append([b, str(m["n"]), _fmt(m["roc_auc"]), _fmt(m["pr_auc"]),
                         _fmt(m["f1_macro"]), _fmt(m["ece_10"]), ""])
    a(_table(["Length", "n", "ROC-AUC", "PR-AUC", "F1-macro", "ECE", "note"], rows))
    a("")
    a("By domain (calibrated; domains with n≥20):")
    a("")
    rows = []
    for d, m in results["slices_calibrated"]["by_domain"].items():
        if "note" in m:
            rows.append([d, str(m["n"]), "—", "—", "—", "—", m["note"]])
        else:
            rows.append([d, str(m["n"]), _fmt(m["roc_auc"]), _fmt(m["pr_auc"]),
                         _fmt(m["f1_macro"]), _fmt(m["ece_10"]), ""])
    a(_table(["Domain", "n", "ROC-AUC", "PR-AUC", "F1-macro", "ECE", "note"], rows))
    a("")
    # ---- 10. errors & limitations ------------------------------------------
    a("## 10. Errors & limitations (honest assessment)")
    a("")
    a("- **Scope:** trained and tested on MAGE only. HC3 (unseen generator, "
      "conversational) and RAID (adversarial attacks) are downloaded and dedup-checked "
      "but **not yet evaluated** — Phase 2 robustness experiments may lower these numbers.")
    a("- **Calibrator optimism:** Platt/isotonic were selected on val_cal, so val_cal "
      "calibration scores are slightly optimistic; test calibration is the honest figure.")
    a("- **Threshold:** fixed at 0.5; no threshold optimization was performed (avoids "
      "test-set tuning).")
    a("- **Sentence segmentation and clause markers are heuristic** (regex/lexicon, no "
      "parser) — spaCy syntactic features are Phase-2 additions.")
    a("- **Short texts:** see length slice; reliability degrades below ~150 words, which "
      "will drive the inference-time 'insufficient evidence' gate.")
    a("- **MAGE prompt IDs are unavailable in the public release**, so grouping uses "
      "exact + near-duplicate clusters as a proxy for source families (documented in "
      "docs/dataset.md §5 risk register).")
    a("- **Never authorship proof:** all figures are statistical associations in one "
      "corpus, not guarantees about any individual document.")
    a("")
    a("## 11. Reproduction")
    a("")
    a("```bash")
    a("python scripts/download_data.py     # idempotent, resumes, writes manifest")
    a("python scripts/build_dataset.py     # filters, dedup, grouped split (seed "
      f"{cfg['seed']})")
    a("python scripts/extract_features.py  # feature state fit on train only")
    a(f"python scripts/train.py             # run {cfg['run_name']}, seed {cfg['seed']}")
    a("python scripts/evaluate.py          # writes this report + test_metrics.json")
    a("```")
    a("")
    a(f"Data manifest sha256: `{manifest['data']['dataset_manifest_sha256']}`  ")
    a(f"Processing report sha256: `{manifest['data']['processing_report_sha256']}`  ")
    a(f"Feature names sha256: `{manifest['feature_names_sha256']}`  ")
    a(f"Versions: {manifest['versions']}")
    return "\n".join(p) + "\n"
