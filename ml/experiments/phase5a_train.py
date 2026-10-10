"""Train `segments-v1`: a SEPARATE artifact for segment-level inference.

WHY A SEPARATE MODEL
-------------------
The Phase-5A study (ml/experiments/phase5a.py) showed that final-v1 applied to
segments keeps its ranking quality only for long segments (150w+: AUC 0.962,
close to the document level) and degrades badly on shorter ones (30-49w: AUC
0.754, human FPR 0.672) while OOD segment AUC is ~0.55-0.59 (near chance).
No evidence gate could satisfy the declared reliability bars, so a dedicated
segment model is justified. It is written to `ml/models/segments-v1/` and
NEVER touches `final-v1` / `baseline-v1`.

DATA POLICY (mirrors Phase 3, no leakage)
----------------------------------------
* Training segments: windows/paragraphs of MAGE-train documents. Labels are
  INHERITED from the document (canonical y=1 machine). This is stated
  explicitly: a window of a machine-generated document is machine text by
  construction of the corpus, but it is NOT an independent authorship
  determination, so segment training data carries inherited labels.
* Selection: val_tune (group-disjoint 60% half of MAGE-val, same `_sha_groups`
  split as Phase 3).
* Calibration: val_cal (the group-disjoint 40% half), fit ONLY there.
* Evaluation-only (never trained/calibrated/tuned on): MAGE-test segments,
  genuine whole-text composite segments, HC3, RAID, OOD-GPT, OOD-GPT-para.
* Feature state: refit on the TRAINING segments (document statistics differ at
  segment scale), so segments-v1 owns its feature state.

SELECTION (lexicographic, robustness first - same philosophy as Phase 3)
-----------------------------------------------------------------------
    1. OOD segment robustness (mean AUC on OOD-GPT + OOD-GPT-para)
    2. calibration (Brier on val_cal segments)
    3. false-positive behaviour (human FPR on val_cal segments)
    4. PR-AUC (val_tune segments)
    5. in-distribution accuracy (val_tune segments)
In-distribution accuracy alone can never decide the winner.

Output: `ml/models/segments-v1/` (model, calibrator, feature_state, features,
config, calibration, gate, manifest) plus `segments-v1-results.json`.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.evaluation.calibration import IsotonicCalibrator, PlattCalibrator  # noqa: E402
from ml.evaluation.metrics import compute_metrics  # noqa: E402
from ml.experiments.phase2 import eval_hc3, meta_join, raid_frame  # noqa: E402
from ml.experiments.phase3 import load_final  # noqa: E402
from ml.experiments.phase5a import (CACHE_DIR, DECISION, LABEL_GENUINE,  # noqa: E402
                                    LABEL_INHERITED, bucket_of, build_composites,
                                    dump_json, human_fpr, load_json)
from ml.features.extract import FeatureState, extract_many  # noqa: E402
from ml.features.syntactic import extract_syntactic  # noqa: E402
from ml.segmentation import SegmentationPolicy, segment_text  # noqa: E402
from ml.training.train import _sha_groups  # noqa: E402

SEG_DIR = ROOT / "ml" / "models" / "segments-v1"
RESULTS_PATH = CACHE_DIR.parent / "segments-v1-results.json"
TRAIN_CACHE = CACHE_DIR / "train_segments.parquet"
TRAIN_META = CACHE_DIR / "train_segments.meta.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


DEFAULT_CONFIG = {
    "pipeline_version": "phase5a_segments_v1",
    "seed": 42,
    "train_docs": 60000,          # MAGE-train documents sampled (seeded)
    "val_tune_docs": 12000,
    "val_cal_docs": 12000,
    "policy": {"max_segment_words": 150, "min_window_words": 30,
               "heading_max_words": 12, "merge_headings": True},
    "candidates": {
        "hgb_seg": {"type": "hist_gradient_boosting", "learning_rate": 0.1,
                    "max_leaf_nodes": 63, "max_iter": 300, "random_state": 42},
        "lr_seg": {"type": "logistic_regression", "C_grid": [0.1, 1.0, 10.0]},
    },
    "calibrators": ["platt", "isotonic"],
    "selection_order": ["ood_auc_mean", "calibration_brier",
                        "false_positive_behavior", "pr_auc",
                        "in_distribution_performance"],
    "tie_bands": {"ood_auc_mean": 0.02, "calibration_brier": 0.005,
                  "false_positive_behavior": 0.01, "pr_auc": 0.005},
    "tie_band_note": ("Mirrors the Phase-3 lexicographic protocol: a criterion "
                      "decides only when the next criterion is outside its tie "
                      "band, so noise-level differences cannot hijack selection."),
    "gate_candidates": [30, 40, 50, 60, 80, 100, 120, 150],
    "composite": {"target_words": 110, "n_composites": 800,
                  "min_source_words": 60, "max_source_words": 150},
    "raid_attacks": ["none", "paraphrase", "insert_paragraphs", "synonym",
                     "article_deletion", "whitespace", "homoglyph",
                     "upper_lower", "number"],
    "workers": 4,
    "config_path": "ml/configs/phase5a_segments.json",
}


def load_config() -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    p = ROOT / DEFAULT_CONFIG["config_path"]
    if p.exists():
        cfg.update(load_json(p))
    return cfg


# --------------------------------------------------------------------------
# segment corpus construction (inherited labels, explicitly flagged)
# --------------------------------------------------------------------------

def _segment_frame(df: pd.DataFrame, corpus: str, policy: SegmentationPolicy,
                   y_col: str = "y", limit: int | None = None,
                   seed: int = 0) -> pd.DataFrame:
    """Segment each document into segments carrying the INHERITED label."""
    if limit is not None and limit < len(df):
        rng = np.random.default_rng(seed)
        df = df.iloc[np.sort(rng.choice(len(df), size=limit, replace=False))]
    recs = []
    for row_id, text, y, extra in zip(
            df["row_id"].astype(str), df["text"], df[y_col],
            df["extra"] if "extra" in df.columns else [""] * len(df)):
        for s in segment_text(str(text), policy):
            recs.append({"corpus": corpus, "doc_id": row_id,
                         "segment_id": s.segment_id, "words": s.word_count,
                         "kind": s.kind, "text": s.text, "y": int(y),
                         "extra": str(extra), "label_semantics": LABEL_INHERITED})
    return pd.DataFrame(recs)


def build_train_pool(policy: SegmentationPolicy, cfg: dict) -> pd.DataFrame:
    """Segments of MAGE-train / val_tune / val_cal, with inherited labels."""
    key = {"policy": policy.as_dict(), "cfg": {
        "train_docs": cfg["train_docs"], "val_tune_docs": cfg["val_tune_docs"],
        "val_cal_docs": cfg["val_cal_docs"], "seed": cfg["seed"]}}
    from ml.experiments.phase2 import sha_of
    key_sha = sha_of(key)
    if TRAIN_CACHE.exists() and TRAIN_META.exists() and not getattr(
            build_train_pool, "_force", False):
        try:
            meta = load_json(TRAIN_META)
            if meta.get("key_sha") == key_sha:
                blob = pd.read_parquet(TRAIN_CACHE)
                return blob.reset_index(drop=True)
        except Exception:
            pass
    joined = meta_join(("train", "val"))
    train = joined["train"].copy()
    train["text"] = train["text_norm"]
    train["extra"] = ""
    train["y"] = 1 - train["label"].astype(int)

    val = joined["val"].copy()
    val["text"] = val["text_norm"]
    val["extra"] = ""
    val["y"] = 1 - val["label"].astype(int)
    tune_mask = _sha_groups(val["group_id"], int(cfg["seed"]), 0.6)
    val_tune = val[tune_mask]
    val_cal = val[~tune_mask]

    parts = [
        _segment_frame(train, "train", policy, limit=int(cfg["train_docs"]),
                       seed=int(cfg["seed"])),
        _segment_frame(val_tune, "val_tune", policy,
                       limit=int(cfg["val_tune_docs"]), seed=int(cfg["seed"]) + 1),
        _segment_frame(val_cal, "val_cal", policy,
                       limit=int(cfg["val_cal_docs"]), seed=int(cfg["seed"]) + 2),
    ]
    out = pd.concat(parts, ignore_index=True)
    TRAIN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(TRAIN_CACHE, index=False)
    dump_json(TRAIN_META, {"key_sha": key_sha, "written_at": _now(),
                           "rows": int(len(out))})
    return out


def _extract(pool: pd.DataFrame, features: list[str], workers: int,
             state: FeatureState | None) -> pd.DataFrame:
    feats = extract_many(pool["text"].tolist(), state, workers=workers)
    syn = extract_syntactic(pool["text"].tolist(), n_process=workers)
    return pd.concat([feats.reset_index(drop=True), syn.reset_index(drop=True)],
                     axis=1)


def _candidate_fit(spec_key: str, spec: dict, X: np.ndarray, y: np.ndarray):
    if spec.get("type") == "logistic_regression":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        best, best_pr = None, -np.inf
        from sklearn.metrics import average_precision_score
        for C in spec.get("C_grid", [1.0]):
            pipe = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=C, max_iter=2000,
                                           class_weight="balanced",
                                           random_state=42)),
            ])
            pipe.fit(X, y)
            pr = average_precision_score(y, pipe.predict_proba(X)[:, 1])
            if pr > best_pr:
                best, best_pr = pipe, pr
        return best
    if spec.get("type") == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(
            learning_rate=spec.get("learning_rate", 0.1),
            max_leaf_nodes=spec.get("max_leaf_nodes", 63),
            max_iter=spec.get("max_iter", 300),
            random_state=spec.get("random_state", 42),
        )
        clf.fit(X, y)
        return clf
    raise ValueError(f"unknown candidate type {spec.get('type')}")


def _fit_calibrator(kind: str, raw: np.ndarray, y: np.ndarray):
    if kind == "platt":
        return PlattCalibrator().fit(raw, y)
    if kind == "isotonic":
        return IsotonicCalibrator().fit(raw, y)
    raise ValueError(kind)


# --------------------------------------------------------------------------
# held-out segment evaluation sets (evaluation-only corpora)
# --------------------------------------------------------------------------

def _eval_segment_sets(policy: SegmentationPolicy, cfg: dict, workers: int,
                       state: FeatureState, features: list[str]
                       ) -> dict[str, pd.DataFrame]:
    """Segments of the evaluation-only corpora (never trained/calibrated on)."""
    sets: dict[str, pd.DataFrame] = {}
    test = meta_join(("test",))["test"].copy()
    test["text"] = test["text_norm"]
    test["extra"] = ""
    test["y"] = 1 - test["label"].astype(int)
    sets["mage_test"] = _segment_frame(test, "mage_test", policy,
                                       limit=4000, seed=int(cfg["seed"]))
    hc3 = eval_hc3().copy()
    hc3["text"] = hc3["text_norm"]
    hc3["extra"] = ""
    hc3["y"] = hc3["label"].astype(int)
    sets["hc3"] = _segment_frame(hc3, "hc3", policy, limit=2000,
                                 seed=int(cfg["seed"]) + 1)
    raid = raid_frame().copy()
    raid["row_id"] = raid["id"].astype(str)
    raid["text"] = raid["text_norm"]
    raid = raid[raid["attack"].isin(cfg.get(
        "raid_attacks",
        ["none", "paraphrase", "insert_paragraphs", "synonym",
         "article_deletion", "whitespace", "homoglyph", "upper_lower",
         "number"]))]
    raid["extra"] = raid["attack"]
    sets["raid"] = _segment_frame(raid, "raid", policy, limit=1500,
                                  seed=int(cfg["seed"]) + 2)
    for name in ("ood_gpt", "ood_gpt_para"):
        import csv as _csv
        from ml.preprocessing.normalize import normalize_text
        fname = ("test_ood_set_gpt.csv" if name == "ood_gpt"
                 else "test_ood_set_gpt_para.csv")
        rows = []
        with (ROOT / "ml" / "data" / "raw" / "mage" / fname).open(
                "r", encoding="utf-8") as fh:
            r = _csv.reader(fh)
            next(r)
            for i, row in enumerate(r):
                nt = normalize_text(row[0])
                if nt:
                    rows.append({"row_id": f"{name}-{i}", "text": nt,
                                 "y": 1 - int(row[1]), "extra": ""})
        sets[name] = _segment_frame(pd.DataFrame(rows), name, policy,
                                    seed=int(cfg["seed"]) + 3)
    # genuine whole-text composites (segments that ARE complete labeled texts)
    from ml.preprocessing.normalize import word_count as _wc
    test_for_comp = test.assign(words=test["text"].map(_wc))
    comp = build_composites(cfg, test_for_comp)
    rows = []
    for _, rec in comp.iterrows():
        for s in segment_text(rec["text"], policy):
            nt = rec["human_text"] if s.text.strip() == rec["human_text"].strip() \
                else rec["machine_text"]
            y_true = 0 if nt == rec["human_text"] else 1
            rows.append({"corpus": "composite", "doc_id": rec["doc_id"],
                         "segment_id": s.segment_id, "words": s.word_count,
                         "kind": s.kind, "text": s.text, "y": y_true,
                         "extra": rec["order"],
                         "label_semantics": LABEL_GENUINE})
    sets["composite"] = pd.DataFrame(rows)
    return sets


def _eval_metrics(y, p, words=None) -> dict:
    m = compute_metrics(np.asarray(y, int), np.asarray(p, float))
    m["human_fpr"] = human_fpr(m["confusion_matrix"])
    if words is not None:
        buckets = {}
        words = np.asarray(words, int)
        for name in ("<30w", "30-49w", "50-79w", "80-149w", "150-299w"):
            mask = np.array([bucket_of(int(w)) == name for w in words])
            if mask.sum() >= 20 and len(np.unique(np.asarray(y, int)[mask])) > 1:
                buckets[name] = _eval_metrics(np.asarray(y, int)[mask],
                                              np.asarray(p, float)[mask])
        m["by_bucket"] = buckets
    return m


def _score_with_final(eval_sets: dict, final: dict, workers: int) -> dict:
    """Score the SAME segment frames with the locked final-v1 artifact.

    Uses final-v1's own feature state / model / calibrator, so the comparison
    is exactly "what the shipped document model does to these segments".
    """
    out = {}
    state = final["state"]
    features = list(final["features"])
    for name, df in eval_sets.items():
        feats = extract_many(df["text"].tolist(), state, workers=workers)
        syn = extract_syntactic(df["text"].tolist(), n_process=workers)
        X = pd.concat([feats.reset_index(drop=True), syn.reset_index(drop=True)],
                      axis=1)[features].to_numpy()
        raw = final["model"].predict_proba(X)[:, 1]
        p = final["calibrator"].predict(raw)
        out[name] = _eval_metrics(df["y"].to_numpy(), p, df["words"])
    return out


# --------------------------------------------------------------------------
# training driver
# --------------------------------------------------------------------------

def run(cfg: dict | None = None, workers: int | None = None,
        force: bool = False) -> dict:
    cfg = json.loads(json.dumps(cfg)) if cfg else load_config()
    if workers is not None:
        cfg["workers"] = int(workers)
    workers = int(cfg.get("workers", 0))
    policy = SegmentationPolicy(**cfg["policy"])
    t0 = time.perf_counter()

    # feature list: same 89 features as final-v1 (no new features in 5A)
    final = load_final()
    features = list(final["features"])

    pool = build_train_pool(policy, cfg)
    train = pool[pool["corpus"] == "train"]
    tune = pool[pool["corpus"] == "val_tune"]
    cal = pool[pool["corpus"] == "val_cal"]

    # refit feature state on TRAINING segments only
    state = FeatureState.fit(train["text"].tolist(),
                             rare_frac=0.001)
    Xtr = _extract(train, features, workers, state)[features].to_numpy()
    Xtu = _extract(tune, features, workers, state)[features].to_numpy()
    Xca = _extract(cal, features, workers, state)[features].to_numpy()
    ytr = train["y"].to_numpy()
    ytu = tune["y"].to_numpy()
    yca = cal["y"].to_numpy()

    # held-out evaluation sets (features extracted with the TRAIN-segment state)
    eval_sets = _eval_segment_sets(policy, cfg, workers, state, features)

    # OOD + HC3 + raid held-out raw p for selection (evaluation-only)
    ood_raw = {}
    for name in ("ood_gpt", "ood_gpt_para"):
        df = eval_sets[name]
        ood_raw[name] = (df["y"].to_numpy(),
                         _extract(df, features, workers, state)[features].to_numpy())

    results: dict = {
        "meta": {"generated_at": _now(),
                 "pipeline_version": cfg["pipeline_version"],
                 "seed": cfg["seed"], "workers": workers,
                 "policy": policy.as_dict(),
                 "features": len(features),
                 "n_train_segments": int(len(train)),
                 "n_val_tune_segments": int(len(tune)),
                 "n_val_cal_segments": int(len(cal)),
                 "python": sys.version.split()[0]},
        "data_policy": {
            "train": "MAGE-train windows/paragraphs (INHERITED document label)",
            "selection": "val_tune (group-disjoint 60% of MAGE-val)",
            "calibration": "val_cal (group-disjoint 40% of MAGE-val)",
            "eval_only": ["MAGE-test", "HC3", "RAID", "OOD-GPT",
                          "OOD-GPT-para", "composite (genuine)"],
            "feature_state": "refit on TRAINING segments only",
        },
        "candidates": {},
    }
    results["document_auc_reference"] = final["manifest"].get(
        "eval_summary", {}).get("test", {}).get("roc_auc")

    # ---- evaluation matrix: every candidate AND final-v1 on IDENTICAL
    # segment frames (apples-to-apples; final-v1 uses its OWN feature state,
    # which is the honest comparison because that is how it would be served).
    eval_feats: dict[str, np.ndarray] = {}
    for name, df in eval_sets.items():
        eval_feats[name] = _extract(df, features, workers, state)[features].to_numpy()
    final_scores = _score_with_final(eval_sets, final, workers)

    fitted: dict[str, tuple] = {}
    head_to_head: dict[str, dict] = {}
    best = None
    for spec_key, spec in cfg["candidates"].items():
        t_c = time.perf_counter()
        clf = _candidate_fit(spec_key, spec, Xtr, ytr)
        raw_tune = clf.predict_proba(Xtu)[:, 1]
        raw_cal = clf.predict_proba(Xca)[:, 1]
        pr_tune = float(compute_metrics(ytu, raw_tune)["pr_auc"])
        cal_choice = {}
        for kind in cfg["calibrators"]:
            c = _fit_calibrator(kind, raw_cal, yca)
            cal_choice[kind] = float(np.mean((c.predict(raw_cal) - yca) ** 2))
        cal_kind = min(cal_choice, key=cal_choice.get)
        calibrator = _fit_calibrator(cal_kind, raw_cal, yca)
        p_cal = calibrator.predict(raw_cal)
        fpr_cal = human_fpr(compute_metrics(yca, p_cal)["confusion_matrix"])
        ood_aucs = {}
        for name, (y, X) in ood_raw.items():
            p = calibrator.predict(clf.predict_proba(X)[:, 1])
            ood_aucs[name] = compute_metrics(y, p)["roc_auc"]
        ood_mean = float(np.mean([v for v in ood_aucs.values() if v is not None]))
        entry = {
            "spec": spec, "calibrator": cal_kind,
            "calibrator_brier_on_val_cal": {k: round(v, 4)
                                            for k, v in cal_choice.items()},
            "pr_auc_val_tune": round(pr_tune, 4),
            "brier_val_cal": round(float(np.mean((p_cal - yca) ** 2)), 4),
            "human_fpr_val_cal": fpr_cal,
            "ood_auc": {k: (round(v, 4) if v is not None else None)
                        for k, v in ood_aucs.items()},
            "ood_auc_mean": round(ood_mean, 4),
            "fit_seconds": round(time.perf_counter() - t_c, 1),
        }
        results["candidates"][spec_key] = entry
        crit = {"ood_auc_mean": ood_mean,
                "calibration_brier": entry["brier_val_cal"],
                "false_positive_behavior": fpr_cal if fpr_cal is not None else 1.0,
                "pr_auc": pr_tune,
                "in_distribution_performance": compute_metrics(
                    ytu, raw_tune)["accuracy"]}
        entry["criteria"] = {k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in crit.items()}
        fitted[spec_key] = (clf, calibrator, cal_kind)
        head_to_head[spec_key] = {
            name: _eval_metrics(df["y"].to_numpy(),
                                calibrator.predict(clf.predict_proba(
                                    eval_feats[name])[:, 1]),
                                df["words"])
            for name, df in eval_sets.items()}
        if best is None or _lexicographic_better(spec_key, crit, best[1],
                                                 cfg["selection_order"],
                                                 cfg.get("tie_bands")):
            best = (spec_key, crit, clf, calibrator, cal_kind, spec)

    head_to_head["final_v1_reference"] = final_scores
    results["head_to_head"] = head_to_head

    winner, _, clf, calibrator, cal_kind, _ = best
    results["winner"] = winner
    results["selection_order"] = cfg["selection_order"]
    results["tie_bands"] = cfg.get("tie_bands", {})

    # ---- is ANY artifact good enough to display? (declared criteria) ----
    def display_check(p_by_set: dict, label: str) -> dict:
        m = p_by_set.get("mage_test", {})
        oods = [p_by_set.get("ood_gpt", {}).get("roc_auc"),
                p_by_set.get("ood_gpt_para", {}).get("roc_auc")]
        comp_auc = p_by_set.get("composite", {}).get("roc_auc")
        doc_auc = results.get("document_auc_reference")
        checks = {
            "accuracy": (m.get("accuracy") is not None
                         and m["accuracy"] >= DECISION["min_kept_accuracy"]),
            "human_fpr": (m.get("human_fpr") is not None
                          and m["human_fpr"] <= DECISION["max_kept_human_fpr"]),
            "ece": (m.get("ece_10") is not None
                    and m["ece_10"] <= DECISION["max_kept_ece"]),
            "auc_vs_document": (doc_auc is None or m.get("roc_auc") is None
                                or m["roc_auc"] >= doc_auc
                                - DECISION["max_auc_drop_vs_document"]),
            "genuine_auc": (comp_auc is not None
                            and comp_auc >= DECISION["min_genuine_auc"]),
            "ood_auc": all(a is not None and a >= DECISION["min_ood_auc"]
                           for a in oods),
        }
        return {"artifact": label, "checks": {k: bool(v) for k, v in checks.items()},
                "passes_all": bool(all(checks.values())),
                "measured": {"mage_test_accuracy": m.get("accuracy"),
                             "mage_test_human_fpr": m.get("human_fpr"),
                             "mage_test_ece": m.get("ece_10"),
                             "mage_test_auc": m.get("roc_auc"),
                             "genuine_composite_auc": comp_auc,
                             "ood_auc": oods}}

    results["display_check"] = {
        spec_key: display_check(head_to_head[spec_key], f"segments-v1:{spec_key}")
        for spec_key in fitted}
    results["display_check"]["final_v1"] = display_check(
        final_scores, "final-v1 (reference, document-trained model)")

    # ---- final evaluation of the winner (kept for the report) ----
    winner_eval = head_to_head[winner]
    results["winner_eval_by_set"] = winner_eval
    # genuine composite accuracy specifically
    comp = eval_sets["composite"]
    Xc = _extract(comp, features, workers, state)[features].to_numpy()
    pc = calibrator.predict(clf.predict_proba(Xc)[:, 1])
    results["composite_genuine"] = _eval_metrics(comp["y"].to_numpy(), pc,
                                                comp["words"])

    # ---- gate sweep for the winner (segment evidence gate) ----
    gate_rows = []
    for gate in cfg["gate_candidates"]:
        m = eval_sets["mage_test"]
        keep = m["words"].to_numpy() >= gate
        Xm = _extract(m, features, workers, state)[features].to_numpy()
        pm = calibrator.predict(clf.predict_proba(Xm)[:, 1])
        kept = _eval_metrics(m["y"].to_numpy()[keep], pm[keep],
                             m["words"].to_numpy()[keep])
        ckeep = comp["words"].to_numpy() >= gate
        ck = _eval_metrics(comp["y"].to_numpy()[ckeep], pc[ckeep])
        gate_rows.append({"gate": gate,
                          "mage_test_kept_share": round(float(keep.mean()), 4),
                          "mage_test_kept": kept,
                          "composite_genuine_kept": ck})
    results["gate_sweep"] = gate_rows

    # Pick the segment evidence gate: the smallest gate that keeps most
    # segments (>= min_kept_share) AND whose kept MAGE slice has the lowest
    # human-FPR while genuine composite AUC stays healthy. Deterministic and
    # recorded, not tuned on the test labels.
    def gate_score(row):
        m = row["mage_test_kept"]
        c = row["composite_genuine_kept"]
        comp_auc = c.get("roc_auc")
        ok = (row["mage_test_kept_share"] >= DECISION["min_kept_share"]
              and (comp_auc is None or comp_auc >= DECISION["min_genuine_auc"]))
        return (1 if ok else 0,
                -(m.get("human_fpr") if m.get("human_fpr") is not None else 1.0))

    eligible = [r for r in gate_rows
                if r["mage_test_kept_share"] >= DECISION["min_kept_share"]]
    chosen_gate = (max(eligible, key=gate_score)["gate"] if eligible
                   else max(gate_rows, key=gate_score)["gate"])
    results["chosen_segment_gate"] = chosen_gate

    # ---- verdict computed BEFORE writing, so the manifest can record it ----
    passing = [k for k, v in results["display_check"].items()
               if v.get("passes_all")]
    results["segment_display_recommended"] = bool(passing)
    results["passing_artifacts"] = passing
    results["artifact_status"] = (
        "research_only_not_adopted_for_production"
        if not results["display_check"].get(winner, {}).get("passes_all")
        else "adopted_for_segment_scoring")
    results["recommendation"] = (
        "segment-level display is defensible with: " + ", ".join(passing)
        if passing else
        "NO artifact clears the declared segment-display criteria; segment-level "
        "verdicts must NOT be shown to users. Keep document-level prediction "
        "authoritative and, if segments are ever surfaced, expose them as "
        "non-decisional indicators with explicit abstention and warnings.")

    # ---- write the SEPARATE artifact ----
    import joblib
    SEG_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, SEG_DIR / "model.joblib")
    joblib.dump(calibrator, SEG_DIR / "calibrator.joblib")
    joblib.dump(state, SEG_DIR / "feature_state.joblib")
    dump_json(SEG_DIR / "features.json",
              {"version": "features_v1", "pipeline_version": cfg["pipeline_version"],
               "n_features": len(features), "features": features,
               "families": {}, "note": "same 89 features as final-v1; no new "
               "features in Phase 5A"})
    dump_json(SEG_DIR / "config.json", {
        "spec_key": winner, "spec": cfg["candidates"][winner],
        "calibrator": cal_kind, "threshold": 0.5,
        "label_policy": "canonical y=1 MACHINE; p = cal(model(x))",
        "pipeline_version": cfg["pipeline_version"],
        "selection_order": cfg["selection_order"],
        "trained_at": _now(), "seed": cfg["seed"],
        "note": "segment-level artifact; final-v1 remains the document model",
    })
    # segment evidence gate + mixed band (from the study, recorded not tuned)
    dump_json(SEG_DIR / "gate.json", {
        "status": "active",
        "kind": "segment_evidence_gate",
        "min_words": chosen_gate,
        "threshold": 0.5,
        "mixed_band": [0.4, 0.6],
        "mixed_band_rule": "inherited from the Phase-3 document gate; segments "
                           "inside it are flagged low-confidence",
        "decision_rule": ("a segment shorter than min_words yields "
                          "'insufficient evidence' (no label, no confidence); "
                          "chosen from the Phase-5A gate sweep as the largest "
                          "coverage with lowest kept human-FPR while genuine "
                          "whole-text composite AUC stays healthy"),
        "written_at": _now(),
        "document_gate_note": ("this artifact is for SEGMENTS; document-level "
                               "prediction continues to use final-v1 (min_words "
                               "50, same mixed band)"),
    })
    dump_json(SEG_DIR / "calibration.json", {
        "calibrator": cal_kind,
        "fit_on": "val_cal segments (group-disjoint 40% of MAGE-val)",
        "type_comparison_brier": results["candidates"][winner][
            "calibrator_brier_on_val_cal"],
        "note": "isotonic/platt fitted on SEGMENT val_cal; segment "
                "calibration is independent of the document calibrator",
    })

    # feature hashes for the manifest
    from ml.experiments.phase2 import file_sha
    files = {}
    for name in ("model.joblib", "calibrator.joblib", "feature_state.joblib",
                 "features.json", "config.json", "gate.json",
                 "calibration.json"):
        p = SEG_DIR / name
        files[name] = {"sha256": file_sha(p), "bytes": p.stat().st_size}

    results["meta"]["artifact_dir"] = str(SEG_DIR.relative_to(ROOT))
    results["meta"]["fit_total_seconds"] = round(time.perf_counter() - t0, 1)
    dump_json(SEG_DIR / "manifest.json", {
        "generated_at": _now(),
        "pipeline_version": cfg["pipeline_version"],
        "artifact_kind": "segments",
        "winner": winner,
        "calibrator": cal_kind,
        "threshold": 0.5,
        "segment_gate_min_words": chosen_gate,
        "selection_order": cfg["selection_order"],
        "tie_bands": cfg.get("tie_bands", {}),
        "status": results.get("artifact_status"),
        "eval_summary": {
            name: {k: (round(m[k], 4) if isinstance(m.get(k), float) else m.get(k))
                   for k in ("n", "roc_auc", "pr_auc", "accuracy", "f1_macro",
                             "human_fpr", "brier", "ece_10")}
            for name, m in (winner_eval or {}).items()},
        "data_policy": results["data_policy"],
        "label_semantics": {
            "training": LABEL_INHERITED,
            "note": ("training segments carry the document's label; this is NOT "
                     "an independent authorship determination per segment"),
            "genuine_evaluation": LABEL_GENUINE,
        },
        "files": files,
        "note": ("Segment-level research artifact. final-v1 and baseline-v1 are "
                 "untouched. Segment-level display is NOT cleared for "
                 "production (see segments-v1-results.json display_check): "
                 "OOD segment AUC remains below the declared bar."),
    })
    results["manifest_files"] = files
    dump_json(RESULTS_PATH, results)
    return results


def _lexicographic_better(name_a: str, crit_a: dict, crit_b: dict,
                          order: list[str],
                          bands: dict | None = None) -> bool:
    """True if candidate A beats B lexicographically.

    Mirrors the Phase-3 protocol INCLUDING tie bands: for each criterion, if
    the two values differ by less than that criterion's tie band, the
    comparison falls through to the next criterion instead of deciding on a
    noise-level difference. `calibration_brier` and `false_positive_behavior`
    are lower-is-better; the rest are higher-is-better.
    """
    lower_better = {"calibration_brier", "false_positive_behavior"}
    bands = bands or {}
    for key in order:
        a, b = crit_a.get(key), crit_b.get(key)
        if a is None and b is None:
            continue
        if a is None:
            return False
        if b is None:
            return True
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            continue
        diff = abs(float(a) - float(b))
        band = float(bands.get(key, 0.0))
        if diff <= band:                      # inside the tie band -> next one
            continue
        if key in lower_better:
            return float(a) < float(b)
        return float(a) > float(b)
    return False


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Train the segments-v1 artifact")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force-pool", action="store_true",
                    help="rebuild the training segment pool")
    args = ap.parse_args(argv)
    if args.force_pool:
        build_train_pool._force = True
    res = run(workers=args.workers)
    print(f"winner: {res['winner']}  ({res['meta']['fit_total_seconds']}s)")
    print("composite genuine:", {k: res["composite_genuine"].get(k)
                                 for k in ("n", "roc_auc", "accuracy",
                                           "human_fpr")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
