"""Phase 3: final model selection + explainability.

Canonical convention (ALL Phase-3 candidates): y=1 MACHINE, y=0 HUMAN.
MAGE stored labels are complemented at load time (`y = 1 - label`), so every
Phase-3 candidate is TRAINED on canonical labels; calibrators are fit in
canonical space and p(machine) = calibrator(model(x)) directly - NO complement
at inference. The Phase-1 locked baseline keeps its stored-orientation
semantics and is only ever scored through the Phase-2 machinery
(1 - calibrated stored score) as the reference to beat.

Protocol guarantees (enforced by code structure + tests):
  * training rows: MAGE-train ONLY; model/feature/hyperparameter selection:
    MAGE-val_tune ONLY; calibrator parameter fitting: MAGE-val_cal ONLY.
  * HC3, RAID, OOD, MAGE-test: evaluation ONLY. Never trained, tuned,
    calibrated or threshold-selected on. Threshold stays 0.5 (Phase-1 fixed).
  * calibration type comparison (E4) uses a group-disjoint cal_fit/cal_eval
    split INSIDE val_cal; the test split is reported descriptively only.
  * Phase-1 artifacts (ml/models/baseline-v1) are read-only; new candidates
    live in ml/experiments/phase3/models/, the deliverable in
    ml/models/final-v1/.
  * Selection is lexicographic on robustness/generalization first (never
    headline MAGE accuracy); every criterion value is recorded for audit.

Experiment order: E3 (models) -> E1 (ablations) -> E2 (syntactic) ->
E4 (calibration) -> E5 (generalization + LODO) -> E6 (leakage/importance) ->
SELECTION (ranking + final artifact) -> E7 (error analysis) ->
E8 (evidence gate; writes gate.json into the final artifact).
"""
from __future__ import annotations

import functools
import hashlib
import json
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..evaluation.calibration import (
    IsotonicCalibrator, PlattCalibrator, brier_score,
    expected_calibration_error, reliability_bins,
)
from ..evaluation.metrics import (
    BUCKET_NAMES, LENGTH_BUCKETS, compute_metrics, slice_metrics,
)
from ..features.extract import FEATURE_FAMILIES, FeatureState, feature_names
from ..preprocessing.build_dataset import PROCESSED, ROOT
from ..preprocessing.normalize import normalize_text, word_count
from ..training.train import _sha_groups
from .phase2 import (
    RESULTS_DIR as P2_RESULTS_DIR,
    _drift_block, _qwen_rewrite, _raid_paraphrase_analysis,
    build_bundle, eval_hc3, extract_texts,
    gen_family, hc3_features, meta_join, raid_frame, raid_features,
    replace_state_features, sha_of, file_sha,
)

RESULTS_DIR = ROOT / "ml" / "experiments" / "phase3"
CACHE_DIR = RESULTS_DIR / "cache"
MODEL_DIR = RESULTS_DIR / "models"
FINAL_DIR = ROOT / "ml" / "models" / "final-v1"
SYN_DIR = RESULTS_DIR / "cache"
P2_RESULTS_FILE = P2_RESULTS_DIR / "phase2-results.json"
V1_FEATURES = feature_names()

# metric keys used for higher-is-better / lower-is-better ranking
HIGHER_BETTER = {
    "robustness_generalization": True,
    "calibration": False,
    "false_positive_behavior": False,
    "pr_auc": True,
    "interpretability": True,
    "computational_practicality": False,
    "in_distribution_performance": True,
}


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------

def _np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_np_default,
                               allow_nan=False), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _set_metrics(y, p) -> dict:
    m = compute_metrics(np.asarray(y), np.asarray(p))
    cm = m.pop("confusion_matrix")
    out = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}
    out["confusion_matrix"] = cm
    tn, fp = cm["tn"], cm["fp"]
    out["fpr"] = round(fp / (fp + tn), 4) if (fp + tn) else None
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    hum = y == 0
    out["human_fpr"] = round(float((p[hum] >= 0.5).mean()), 4) if hum.any() else None
    mach = y == 1
    out["machine_fnr"] = round(float((p[mach] < 0.5).mean()), 4) if mach.any() else None
    return out


def _mean(vals) -> float | None:
    v = [float(x) for x in vals if x is not None and x == x]
    return round(float(np.mean(v)), 4) if v else None


# --------------------------------------------------------------------------
# context
# --------------------------------------------------------------------------

def _ood_frame(name: str) -> pd.DataFrame:
    fname = "test_ood_set_gpt.csv" if name == "ood_gpt" else "test_ood_set_gpt_para.csv"
    import csv as _csv
    path = ROOT / "ml" / "data" / "raw" / "mage" / fname
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        r = _csv.reader(fh)
        next(r)
        for row in r:
            nt = normalize_text(row[0])
            rows.append({"text_norm": nt if nt else "", "raw_label": int(row[1]),
                         "src": row[2] if len(row) > 2 else ""})
    df = pd.DataFrame(rows)
    df["source_dataset"] = "mage"
    df["y"] = 1 - df["raw_label"].to_numpy()
    df["words"] = df["text_norm"].map(word_count)
    return df


def _ood_with_features(name: str, state: FeatureState, workers: int) -> pd.DataFrame:
    df = _ood_frame(name)
    fname = "test_ood_set_gpt.csv" if name == "ood_gpt" else "test_ood_set_gpt_para.csv"
    src = ROOT / "ml" / "data" / "raw" / "mage" / fname
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"ood_{name}_features.parquet"
    meta_path = Path(str(path) + ".meta.json")
    key = {"src_sha": file_sha(src), "features_sha": sha_of(V1_FEATURES)}
    if meta_path.exists() and path.exists():
        try:
            got = load_json(meta_path)
        except Exception:
            got = {}
        if all(got.get(k) == v for k, v in key.items()):
            feats = pd.read_parquet(path)
            if len(feats) == len(df):
                return pd.concat([df.reset_index(drop=True), feats], axis=1)
    feats = extract_texts(df["text_norm"].tolist(), state, workers=workers)
    assert len(feats) == len(df), "ood feature rows misaligned"
    feats.to_parquet(path, index=False)
    dump_json(meta_path, key)
    return pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)],
                     axis=1)


def _attach_syn(ctx: dict) -> bool:
    """Positionally attach cached syntactic columns to every base frame.

    Alignment contract (guaranteed by scripts/extract_syntactic.py reading the
    same source parquets/CSVs in the same row order): syn_X.parquet row i
    belongs to frame X row i. All 7 caches must be present and row-matched,
    otherwise the syntactic family is unavailable (E2 then fails loudly).
    """
    from ..features.syntactic import SYNTACTIC_FEATURES
    names = list(SYNTACTIC_FEATURES)
    base = {"train": ctx["train"], "val": ctx["val_full"], "test": ctx["test"],
            "ood_gpt": ctx["sets"]["ood_gpt"], "ood_gpt_para": ctx["sets"]["ood_gpt_para"],
            "hc3": ctx["sets"]["hc3"], "raid": ctx["sets"]["raid"]}
    for key, frame in base.items():
        path = SYN_DIR / f"syn_{key}.parquet"
        if not path.exists():
            return False
        syn = pd.read_parquet(path)
        if len(syn) != len(frame):
            raise AssertionError(
                f"syn_{key} row mismatch: {len(syn)} vs frame {len(frame)}")
        frame[names] = syn[names].to_numpy()
    ctx["syn_names"] = names
    ctx["syn_ok"] = True
    return True


def build_ctx(cfg: dict) -> dict:
    t0 = time.time()
    p2_cfg = load_json(ROOT / "ml" / "configs" / "phase2.json")
    p2_cfg.setdefault("device", "cpu")
    bundle = build_bundle(p2_cfg)          # locked baseline scores (cached)
    joined = meta_join(("train", "val"))
    train, val_full = joined["train"], joined["val"]
    train["y"] = 1 - train["label"].astype(int)
    val_full["y"] = 1 - val_full["label"].astype(int)
    test = bundle["test"].copy()
    test["y"] = 1 - test["label"].astype(int)

    sets: dict[str, pd.DataFrame] = {
        "test": test,
        "ood_gpt": _ood_with_features("ood_gpt", bundle["state"],
                                      int(cfg.get("feature_workers", 0))),
        "ood_gpt_para": _ood_with_features("ood_gpt_para", bundle["state"],
                                           int(cfg.get("feature_workers", 0))),
    }
    hc3 = eval_hc3().reset_index(drop=True)
    hc3_feats = hc3_features(p2_cfg, bundle["state"], workers=0)
    hc3 = pd.concat([hc3.reset_index(drop=True),
                     hc3_feats.drop(columns=["row_id"], errors="ignore")
                     .reset_index(drop=True)], axis=1)
    hc3["y"] = hc3["label"].astype(int)
    if "words" not in hc3.columns:
        hc3["words"] = hc3["text_norm"].map(word_count)
    sets["hc3"] = hc3

    raid = raid_frame()
    r_feats = raid_features(p2_cfg, raid, bundle["state"], workers=0)
    n0 = len(raid)
    r_feats = r_feats.reset_index(drop=True)
    assert len(r_feats) == n0, "raid feature rows misaligned"
    raid = pd.concat([raid.reset_index(drop=True), r_feats], axis=1)
    sets["raid"] = raid

    ctx: dict = {
        "p2_cfg": p2_cfg, "bundle": bundle, "state": bundle["state"],
        "train": train, "val_full": val_full,
        "test": test, "sets": sets, "syn_ok": False,
        "data_key": sha_of({
            "train_sha": file_sha(PROCESSED / "train.parquet"),
            "ftrain_sha": file_sha(PROCESSED / "features_train.parquet"),
            "val_sha": file_sha(PROCESSED / "val.parquet"),
            "fval_sha": file_sha(PROCESSED / "features_val.parquet"),
            "state_sha": file_sha(PROCESSED / "feature_state.joblib"),
            "splits": cfg["splits"], "seed": cfg["seed"],
            "pipeline_version": cfg["pipeline_version"],
        }),
        "built_at_s": None,
    }
    try:
        _attach_syn(ctx)                    # BEFORE the val split so slices inherit
    except AssertionError:
        ctx["syn_ok"] = False

    seed = int(cfg["seed"])
    tune_mask = _sha_groups(val_full["group_id"], seed,
                            float(cfg["splits"]["val_tune_fraction"]))
    cal_seed = int(hashlib.sha1(str(cfg["splits"]["cal_fit_salt"]).encode())
                   .hexdigest()[:8], 16)
    ctx["val_tune"] = val_full[tune_mask].reset_index(drop=True)
    ctx["val_cal"] = val_full[~tune_mask].reset_index(drop=True)
    cal_fit_mask = _sha_groups(ctx["val_cal"]["group_id"], cal_seed,
                               float(cfg["splits"]["cal_fit_fraction"]))
    ctx["cal_fit_mask"] = cal_fit_mask.to_numpy(dtype=bool)
    ctx["built_at_s"] = round(time.time() - t0, 1)
    return ctx


# --------------------------------------------------------------------------
# candidates: fit / load / score / evaluate
# --------------------------------------------------------------------------

def _build_estimator(spec_key: str, spec: dict, C: float | None):
    if spec_key == "logistic_regression" or spec.get("type") == "logistic_regression":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=float(C), solver="lbfgs",
                                       class_weight="balanced", max_iter=1000,
                                       random_state=42)),
        ])
    if spec.get("type") == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            learning_rate=float(spec["learning_rate"]),
            max_leaf_nodes=int(spec["max_leaf_nodes"]),
            max_iter=int(spec["max_iter"]),
            random_state=int(spec.get("random_state", 42)),
        )
    if spec.get("type") == "extratrees":
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", ExtraTreesClassifier(
                n_estimators=int(spec["n_estimators"]),
                min_samples_leaf=int(spec["min_samples_leaf"]),
                class_weight=spec.get("class_weight", "balanced_subsample"),
                random_state=int(spec.get("random_state", 42)),
                n_jobs=int(spec.get("n_jobs", -1))),
            ),
        ])
    raise ValueError(f"unknown model spec: {spec}")


def fit_candidate(cfg: dict, ctx: dict, name: str, features: list[str],
                  spec_key: str, C: float | None = None,
                  select_C: bool = False) -> dict:
    """Fit (or load from cache) a canonical candidate with both calibrators."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name}.joblib"
    spec = dict(cfg["candidates"][spec_key])
    spec["type"] = {"lr": "logistic_regression",
                    "hgb": "hist_gradient_boosting",
                    "extratrees": "extratrees"}[spec_key]
    meta_expect = {"name": name, "features": features, "spec_key": spec_key,
                   "C": C, "data_key": ctx["data_key"]}
    if path.exists():
        cand = joblib.load(path)
        if all(cand.get(k) == v for k, v in meta_expect.items()):
            return cand

    t0 = time.time()
    tr, va_cal = ctx["train"], ctx["val_cal"]
    y_tr = tr["y"].to_numpy()
    y_cal = va_cal["y"].to_numpy()
    grid_records = []
    if select_C:
        best_C, best_pr = None, -1.0
        for C_try in cfg["candidates"][spec_key]["C_grid"]:
            est = _build_estimator(spec_key, spec, C=float(C_try))
            est.fit(tr[features].to_numpy(), y_tr)
            raw = est.predict_proba(ctx["val_tune"][features].to_numpy())[:, 1]
            pr = float(compute_metrics(ctx["val_tune"]["y"].to_numpy(), raw)["pr_auc"])
            grid_records.append({"C": float(C_try), "val_tune_pr_auc": round(pr, 4)})
            if pr > best_pr:
                best_pr, best_C = pr, float(C_try)
        C = best_C
    model = _build_estimator(spec_key, spec, C=C)
    model.fit(tr[features].to_numpy(), y_tr)
    raw_cal = model.predict_proba(va_cal[features].to_numpy())[:, 1]
    calibrators = {
        "isotonic": IsotonicCalibrator().fit(raw_cal, y_cal),
        "platt": PlattCalibrator().fit(raw_cal, y_cal),
    }
    cand = {
        "name": name, "model": model, "calibrators": calibrators,
        "default_cal": "isotonic", "features": list(features),
        "spec_key": spec_key, "spec": spec, "C": C,
        "grid_records": grid_records, **meta_expect,
        "fit_seconds": round(time.time() - t0, 1),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "label_policy": "canonical y=1 MACHINE (MAGE label complemented); "
                        "calibrators fit in canonical space; p = cal(model(x))",
        "train_rows": int(len(tr)), "cal_rows": int(len(va_cal)),
    }
    joblib.dump(cand, path)
    return cand


def load_candidate(name: str) -> dict | None:
    path = MODEL_DIR / f"{name}.joblib"
    return joblib.load(path) if path.exists() else None


def raw_p(cand: dict, df: pd.DataFrame) -> np.ndarray:
    feats = df[cand["features"]].to_numpy()
    return cand["model"].predict_proba(feats)[:, 1]


def p_of(cand: dict, df: pd.DataFrame, cal: str | None = None) -> np.ndarray:
    cal_type = cal or cand.get("default_cal", "isotonic")
    return cand["calibrators"][cal_type].predict(raw_p(cand, df))


def eval_suite(cfg: dict, ctx: dict, cand: dict) -> dict:
    """Metrics for the candidate on every evaluation set (canonical p)."""
    sets = ctx["sets"]
    out: dict = {
        "val_tune": _set_metrics(ctx["val_tune"]["y"], p_of(cand, ctx["val_tune"])),
        "val_cal": _set_metrics(ctx["val_cal"]["y"], p_of(cand, ctx["val_cal"])),
        "test": _set_metrics(sets["test"]["y"], p_of(cand, sets["test"])),
        "ood_gpt": _set_metrics(sets["ood_gpt"]["y"], p_of(cand, sets["ood_gpt"])),
        "ood_gpt_para": _set_metrics(sets["ood_gpt_para"]["y"],
                                     p_of(cand, sets["ood_gpt_para"])),
        "hc3": _set_metrics(sets["hc3"]["y"], p_of(cand, sets["hc3"])),
    }
    clean = sets["raid"][(sets["raid"]["attack"] == "none") & sets["raid"]["in_policy"]]
    out["raid_clean"] = _set_metrics(clean["y"], p_of(cand, clean))
    return out


def _need_syn(ctx: dict, features: list[str]) -> None:
    if any(f.startswith("syn_") for f in features) and not ctx.get("syn_ok"):
        raise RuntimeError("candidate needs syntactic features but syn caches "
                           "are incomplete - run scripts/extract_syntactic.py")


# --------------------------------------------------------------------------
# locked baseline reference (from Phase-2 results; never re-scored here)
# --------------------------------------------------------------------------

def _norm_locked(m: dict) -> dict:
    out = {k: v for k, v in m.items()
           if k in ("n", "accuracy", "f1_macro", "precision_macro", "recall_macro",
                    "recall_machine", "roc_auc", "pr_auc", "brier", "ece_10",
                    "mean_abs_p_minus_y")}
    cm = m.get("confusion_matrix") or {}
    if cm:
        tn, fp = cm.get("tn"), cm.get("fp")
        out["fpr"] = round(fp / (fp + tn), 4) if (tn is not None and (tn + fp)) else None
        hum = tn + fp
        if hum:
            out["human_fpr"] = out["fpr"]
    for k in ("n", "human_fpr_at_0_5"):
        if k in m and k not in out:
            if k == "human_fpr_at_0_5":
                out["human_fpr"] = m[k]
            else:
                out["n"] = m[k]
    if "confusion_matrix" in m:
        out["confusion_matrix"] = m["confusion_matrix"]
    return out


def _locked_reference() -> dict:
    if not P2_RESULTS_FILE.exists():
        return {"available": False,
                "note": "phase2-results.json not found - locked comparisons skipped"}
    p2 = load_json(P2_RESULTS_FILE)
    e1, e2, e5 = p2.get("E1", {}), p2.get("E2", {}), p2.get("E5", {})
    ref = p2.get("reference", {})
    test_entry = _norm_locked({**ref.get("metrics", {}),
                               "confusion_matrix": ref.get("confusion_matrix", {})})
    sets = {
        "test": test_entry,
        "ood_gpt": _norm_locked(e1.get("ood_test_ood_set_gpt", {})),
        "ood_gpt_para": _norm_locked(e1.get("ood_test_ood_set_gpt_para", {})),
        "hc3": _norm_locked(e1.get("hc3", {})),
        "raid_clean": _norm_locked(e1.get("raid_labelled", {}).get("overall_clean", {})),
    }
    raid_hf = e1.get("raid_labelled", {}).get("human_fpr_at_0_5")
    if raid_hf is not None:
        sets["raid_clean"]["human_fpr"] = raid_hf
    pol = e2.get("raid", {}).get("policy_40_5000_words", {})
    return {
        "available": True, "sets": sets,
        "paraphrase_policy": {k: pol.get(k) for k in
                              ("machine_clean", "machine_paraphrase",
                               "delta_paraphrase_minus_clean", "human_control")},
        "lodo_summary": e5.get("lodo", {}).get("summary", {}),
        "phase2_report": "docs/experiments/phase2-report.md",
        "score_map": "p(machine) = 1 - calibrated stored score (Phase-2 machinery)",
    }


# --------------------------------------------------------------------------
# selection pool + ranking
# --------------------------------------------------------------------------

def _read_cache(key: str) -> dict | None:
    path = CACHE_DIR / f"{key}.json"
    return load_json(path) if path.exists() else None


def selection_pool(cfg: dict) -> list[str]:
    pool = list(cfg["selection_pool"]["primary"])
    for ch in cfg["selection_pool"]["challengers"]:
        if ch.startswith("abl_"):
            if (MODEL_DIR / f"{ch}.joblib").exists():
                pool.append(ch)
        elif ch == "hgb_syn":
            e2 = _read_cache("E2")
            if e2 and e2.get("verdict", {}).get("include"):
                pool.append(ch)
    return pool


def lexicographic_rank(criteria: dict[str, dict], order: list[str],
                       bands: dict[str, float], higher: dict[str, bool]
                       ) -> tuple[list[str], list[dict]]:
    """Lexicographic comparison with per-criterion tie bands.

    Returns (sorted names, trace) where trace records the winner's value per
    criterion for auditability. Deterministic for ties (stable input order).
    """
    names = list(criteria.keys())

    def cmp(a: str, b: str) -> int:
        for crit in order:
            va = criteria[a][crit]
            vb = criteria[b][crit]
            band = float(bands.get(crit, 0.0))
            if va is None or vb is None:
                continue
            if abs(va - vb) <= band:
                continue
            better = (va > vb) if higher[crit] else (vb > va)
            return -1 if better else 1
        return 0

    ranked = sorted(names, key=functools.cmp_to_key(cmp))
    trace = [{"criterion": c, "band": bands.get(c, 0.0),
              "higher_is_better": bool(higher[c]),
              "values": {n: criteria[n][c] for n in names}} for c in order]
    return ranked, trace


# --------------------------------------------------------------------------
# E3 - model comparison (LR / HGB / ExtraTrees)
# --------------------------------------------------------------------------

def run_e3(cfg: dict, ctx: dict) -> dict:
    t0 = time.time()
    res: dict = {"name": "E3-model-comparison", "status": "ok",
                 "orientation": "canonical training (y=1 machine); "
                                "p = calibrator(model(x)) with no complement",
                 "candidates": {}}
    names = V1_FEATURES
    _need_syn(ctx, names)
    for spec_key in ("lr", "hgb", "extratrees"):
        cand = fit_candidate(cfg, ctx, name=spec_key, features=names,
                             spec_key=spec_key,
                             select_C=(spec_key == "lr"))
        suite = eval_suite(cfg, ctx, cand)
        res["candidates"][spec_key] = {
            "n_features": len(names), "C": cand.get("C"),
            "grid": cand.get("grid_records"),
            "fit_seconds": cand.get("fit_seconds"),
            "default_cal": cand.get("default_cal"),
            "sets": suite,
        }
        print(f"[E3] {spec_key}: test roc={suite['test']['roc_auc']} "
              f"pr={suite['test']['pr_auc']} hc3_pr={suite['hc3']['pr_auc']}",
              flush=True)
    res["locked_baseline"] = _locked_reference()
    res["duration_s"] = round(time.time() - t0, 1)
    return res


# --------------------------------------------------------------------------
# E1 - feature-family ablation (HGB, canonical)
# --------------------------------------------------------------------------

def _ablation_features(cfg: dict, ab: dict) -> list[str]:
    removed: list[str] = []
    if "remove_family" in ab:
        fam = ab["remove_family"]
        removed = list(FEATURE_FAMILIES[fam])
    removed += list(ab.get("remove", []))
    unknown = [f for f in removed if f not in V1_FEATURES]
    if unknown:
        raise ValueError(f"ablation {ab['key']} removes unknown features: {unknown}")
    return [f for f in V1_FEATURES if f not in set(removed)], removed


def run_e1(cfg: dict, ctx: dict) -> dict:
    t0 = time.time()
    res: dict = {"name": "E1-feature-ablation", "status": "ok",
                 "model": {k: cfg["ablation_model"][k]
                           for k in ("type", "learning_rate", "max_leaf_nodes",
                                     "max_iter", "random_state")},
                 "calibration": "isotonic on val_cal (canonical); threshold 0.5",
                 "rows": {}, "family_analysis": {}}
    spec = dict(cfg["ablation_model"])
    reference: dict | None = None
    for ab in cfg["ablations"]:
        feats, removed = _ablation_features(cfg, ab)
        key = ab["key"]
        name = "hgb" if key == "all" else f"abl_{key}"
        cand = fit_candidate(cfg, ctx, name=name, features=feats,
                             spec_key="hgb")
        suite = eval_suite(cfg, ctx, cand)
        row = {"removed": removed, "n_features": len(feats),
               "model_file": f"ml/experiments/phase3/models/{name}.joblib",
               "fit_seconds": cand.get("fit_seconds"), "sets": suite}
        if key == "all":
            reference = suite
            row["role"] = "reference (full 73 features)"
        elif reference is not None:
            row["deltas_vs_all"] = {
                s: {k: (round(suite[s][k] - reference[s][k], 4)
                        if isinstance(suite[s].get(k), float)
                        and isinstance(reference[s].get(k), float) else None)
                    for k in ("roc_auc", "pr_auc", "f1_macro", "brier", "ece_10",
                              "human_fpr")}
                for s in suite
            }
        res["rows"][key] = row
        print(f"[E1] {key}: n={len(feats)} test_pr="
              f"{suite['test']['pr_auc']} ood_roc={suite['ood_gpt']['roc_auc']}",
              flush=True)

    ref = res["rows"].get("all", {}).get("sets", {})
    for ab in cfg["ablations"]:
        key = ab["key"]
        if key == "all" or "remove_family" not in ab:
            continue
        row = res["rows"].get(key, {})
        d = row.get("deltas_vs_all", {})
        in_domain_pr = d.get("test", {}).get("pr_auc")
        ood_roc = _mean([d.get(s, {}).get("roc_auc")
                         for s in ("ood_gpt", "ood_gpt_para", "hc3", "raid_clean")])
        hc3_fpr = d.get("hc3", {}).get("human_fpr")
        if in_domain_pr is None or ood_roc is None:
            verdict = "insufficient data"
        elif in_domain_pr > 0.002 and ood_roc < -0.002:
            verdict = "in-domain helpful but HURTS generalization"
        elif in_domain_pr < -0.002 and ood_roc > 0.002:
            verdict = "in-domain harmful but HELPS generalization"
        elif ood_roc < -0.002:
            verdict = "hurts generalization"
        elif ood_roc > 0.002 and (in_domain_pr or 0) >= -0.002:
            verdict = "helps generalization without in-domain cost"
        else:
            verdict = "negligible on both axes (|delta| <= 0.002)"
        res["family_analysis"][ab["remove_family"]] = {
            "in_domain_delta_pr_auc": in_domain_pr,
            "generalization_delta_mean_roc": ood_roc,
            "hc3_delta_human_fpr": hc3_fpr,
            "verdict": verdict,
        }
    leak_row = res["rows"].get("minus_leakage_preregistered", {})
    if leak_row:
        d = leak_row.get("deltas_vs_all", {})
        res["preregistered_leakage_ablation"] = {
            "justification": cfg["ablations"][-1].get("justification", ""),
            "n_removed": len(leak_row.get("removed", [])),
            "in_domain_delta_pr_auc": d.get("test", {}).get("pr_auc"),
            "generalization_delta_mean_roc": _mean(
                [d.get(s, {}).get("roc_auc")
                 for s in ("ood_gpt", "ood_gpt_para", "hc3", "raid_clean")]),
            "test_delta_human_fpr": d.get("test", {}).get("human_fpr"),
            "hc3_delta_human_fpr": d.get("hc3", {}).get("human_fpr"),
        }
    res["reference_sets"] = ref
    res["duration_s"] = round(time.time() - t0, 1)
    return res


# --------------------------------------------------------------------------
# E2 - syntactic (spaCy) features
# --------------------------------------------------------------------------

def _parse_syn_logs() -> dict:
    out: dict = {}
    for logp in (ROOT / "logs" / "phase3_syn2.log",
                 ROOT / "logs" / "phase3_syn.log"):
        if not logp.exists():
            continue
        rows = re.findall(r"^(\w+): (\d+) rows in ([\d.]+)s", logp.read_text(
            encoding="utf-8", errors="replace"), flags=re.M)
        for name, n, s in rows:
            out[name] = {"rows": int(n), "seconds": float(s),
                         "rows_per_s": round(int(n) / max(float(s), 1e-9), 1)}
    return out


def run_e2(cfg: dict, ctx: dict) -> dict:
    t0 = time.time()
    scfg = cfg["syntactic"]
    if not ctx.get("syn_ok"):
        raise RuntimeError("syntactic caches incomplete/absent - run "
                           "scripts/extract_syntactic.py before E2")
    res: dict = {"name": "E2-syntactic-features", "status": "ok",
                 "version": scfg["version"], "model": scfg["model"]}
    features = V1_FEATURES + list(ctx["syn_names"])
    cand = fit_candidate(cfg, ctx, name="hgb_syn", features=features,
                         spec_key="hgb")
    suite = eval_suite(cfg, ctx, cand)
    base = load_candidate("hgb")
    if base is None:
        base = fit_candidate(cfg, ctx, name="hgb", features=V1_FEATURES,
                             spec_key="hgb")
    base_suite = eval_suite(cfg, ctx, base)
    delta = {
        s: {k: (round(suite[s][k] - base_suite[s][k], 4)
                if isinstance(suite[s].get(k), float)
                and isinstance(base_suite[s].get(k), float) else None)
            for k in ("roc_auc", "pr_auc", "f1_macro", "brier", "ece_10",
                      "human_fpr")}
        for s in suite
    }
    gen_mean_new = _mean([suite[s]["roc_auc"] for s in
                          ("ood_gpt", "ood_gpt_para", "hc3", "raid_clean")])
    gen_mean_base = _mean([base_suite[s]["roc_auc"] for s in
                           ("ood_gpt", "ood_gpt_para", "hc3", "raid_clean")])
    rule = scfg["verdict_rule"]["include_if_all"]
    observed = {
        "generalization_mean_roc_delta": round(gen_mean_new - gen_mean_base, 4),
        "test_ece": suite["test"]["ece_10"],
        "hc3_fpr_delta": delta["hc3"]["human_fpr"],
    }
    include = bool(
        observed["generalization_mean_roc_delta"] >=
        float(rule["generalization_mean_roc_delta"])
        and observed["test_ece"] <= float(rule["test_ece_max"])
        and (observed["hc3_fpr_delta"] is None or
             observed["hc3_fpr_delta"] <= float(rule["hc3_fpr_delta_max"]))
    )
    bench_src = ROOT / "ml" / "experiments" / "phase3" / "cache" / "syn_val.parquet"
    meta = {}
    if bench_src.exists():
        meta = load_json(Path(str(bench_src) + ".meta.json"))
    res.update({
        "n_features": len(features),
        "n_syntactic": len(ctx["syn_names"]),
        "feature_list": features,
        "extraction": {
            "cached_splits": _parse_syn_logs(),
            "cache_meta": meta,
            "note": "extraction ran in background via scripts/extract_syntactic.py "
                    "(n_process=4, batch_size=48 after an 8-process OOM on the "
                    "long train split); deterministic en_core_web_sm inference",
        },
        "candidate": {"fit_seconds": cand.get("fit_seconds"), "sets": suite},
        "hgb_v1_reference": {"sets": base_suite},
        "deltas_syn_minus_v1": delta,
        "verdict_rule": scfg["verdict_rule"],
        "observed": observed,
        "verdict": {"include": include,
                    "reason": "all pre-declared conditions met" if include else
                              "at least one pre-declared condition failed "
                              f"(observed: {observed})"},
        "duration_s": round(time.time() - t0, 1),
    })
    print(f"[E2] hgb_syn vs hgb: gen_delta="
          f"{observed['generalization_mean_roc_delta']} include={include}",
          flush=True)
    return res


# --------------------------------------------------------------------------
# E4 - calibration comparison (platt vs isotonic)
# --------------------------------------------------------------------------

def run_e4(cfg: dict, ctx: dict) -> dict:
    t0 = time.time()
    ecfg = cfg["e4"]
    res: dict = {"name": "E4-calibration", "status": "ok", "protocol": {
        "split": "val_cal further split group-disjointly into cal_fit/cal_eval",
        "note": "winner chosen on cal_eval Brier only; test metrics are "
                "descriptive and NEVER used for the choice; winner refit on "
                "full val_cal and attached as the candidate's default_cal",
        "tie_rule": f"prefer platt when |brier_platt - brier_isotonic| <= "
                    f"{ecfg['tie_brier_eps']}",
    }, "candidates": {}}
    mask = ctx["cal_fit_mask"]
    va = ctx["val_cal"]
    y = va["y"].to_numpy()
    names = sorted(p.stem for p in MODEL_DIR.glob("*.joblib"))
    for nm in names:
        cand = joblib.load(MODEL_DIR / f"{nm}.joblib")
        _need_syn(ctx, cand["features"])
        raw = raw_p(cand, va)
        cands = {
            "isotonic": IsotonicCalibrator().fit(raw[mask], y[mask]),
            "platt": PlattCalibrator().fit(raw[mask], y[mask]),
        }
        ev = {}
        for ctype, cal in cands.items():
            p_ev = cal.predict(raw[~mask])
            ev[ctype] = {
                "cal_eval_brier": round(float(brier_score(p_ev, y[~mask])), 4),
                "cal_eval_ece": round(float(expected_calibration_error(
                    p_ev, y[~mask])), 4),
                "cal_eval_mean_abs_p_minus_y": round(float(np.mean(
                    np.abs(p_ev - y[~mask]))), 4),
            }
        diff = ev["isotonic"]["cal_eval_brier"] - ev["platt"]["cal_eval_brier"]
        if abs(diff) <= float(ecfg["tie_brier_eps"]):
            chosen = ecfg["prefer_on_tie"]
        else:
            chosen = "platt" if ev["platt"]["cal_eval_brier"] < \
                ev["isotonic"]["cal_eval_brier"] else "isotonic"
        full = {}
        for ctype in ("isotonic", "platt"):
            cal_full = cands[ctype].fit(raw, y)   # refit on full val_cal
            p_test = cal_full.predict(raw_p(cand, ctx["sets"]["test"]))
            full[ctype] = {"test_brier": round(float(brier_score(
                p_test, ctx["sets"]["test"]["y"])), 4)}
        cand["default_cal"] = chosen
        joblib.dump(cand, MODEL_DIR / f"{nm}.joblib")
        res["candidates"][nm] = {
            "cal_fit_rows": int(mask.sum()), "cal_eval_rows": int((~mask).sum()),
            "eval": ev, "brier_diff_isotonic_minus_platt": round(diff, 5),
            "chosen": chosen, "test_descriptive": full,
        }
        print(f"[E4] {nm}: chosen={chosen} (iso={ev['isotonic']['cal_eval_brier']} "
              f"platt={ev['platt']['cal_eval_brier']})", flush=True)
    res["duration_s"] = round(time.time() - t0, 1)
    return res


# --------------------------------------------------------------------------
# E5 - generalization table + LODO
# --------------------------------------------------------------------------

def _robustness(suite: dict) -> float | None:
    return _mean([suite[s]["roc_auc"] for s in
                  ("ood_gpt", "ood_gpt_para", "hc3", "raid_clean")])


def _raid_scored_for(cfg: dict, ctx: dict, cand: dict) -> pd.DataFrame:
    raid = ctx["sets"]["raid"].copy()
    raid["p_machine_cal"] = p_of(cand, raid)
    return raid


def _lodo_for_candidate(cfg: dict, ctx: dict, cand: dict,
                        cal_type: str) -> dict:
    """Leave-one-domain-out for a Phase-3 candidate (canonical training).

    Mirrors Phase-2 E5 LODO protocol: train excluding domain d, refit the
    document-frequency feature state excluding d, calibrate on val rows of
    domains != d (chosen calibrator type), evaluate on test rows of domain d.
    Never touches HC3/RAID/OOD.
    """
    names = cand["features"]
    tr_all, va_all = ctx["train"], ctx["val_full"]
    te_all = ctx["test"]
    domains = sorted(te_all[te_all["source_dataset"] == "mage"]
                     ["domain"].astype(str).unique())
    rare = float(cfg.get("rare_frac", 0.001))
    out: dict = {"protocol": {
        "train": "MAGE-train rows of all domains except d (canonical labels)",
        "feature_state": "document-frequency stats refit on train-minus-d text",
        "calibration": f"{cal_type} fit on MAGE-val rows of domains != d",
        "eval": "MAGE-test rows of domain d only",
        "model": cand["spec"], "C": cand.get("C"), "features": len(names),
    }, "folds": {}}
    has_state_cols = ("rare_token_ratio" in names) and ("mean_log_df" in names)
    for d in domains:
        ts = time.time()
        tr = tr_all[tr_all["domain"].astype(str) != d]
        va = va_all[va_all["domain"].astype(str) != d]
        te = te_all[(te_all["domain"].astype(str) == d)
                    & (te_all["source_dataset"] == "mage")]
        if len(te) < 20:
            out["folds"][d] = {"n_test": int(len(te)), "note": "too few rows"}
            continue
        if has_state_cols:
            state_d = FeatureState.fit(tr["text_norm"].tolist(), rare_frac=rare)
            Xtr = replace_state_features(tr[names], tr["text_norm"].tolist(), state_d)
            Xva = replace_state_features(va[names], va["text_norm"].tolist(), state_d)
            Xte = replace_state_features(te[names], te["text_norm"].tolist(), state_d)
        else:
            Xtr, Xva, Xte = tr[names], va[names], te[names]
        clf = _build_estimator(cand["spec_key"], cand["spec"], C=cand.get("C"))
        clf.fit(Xtr.to_numpy(), tr["y"].to_numpy())
        raw_va = clf.predict_proba(Xva.to_numpy())[:, 1]
        cal = (IsotonicCalibrator() if cal_type == "isotonic"
               else PlattCalibrator()).fit(raw_va, va["y"].to_numpy())
        p = cal.predict(clf.predict_proba(Xte.to_numpy())[:, 1])
        y = te["y"].to_numpy()
        m = _set_metrics(y, p)
        p_locked = te["p_machine_cal"].to_numpy()
        ml = _set_metrics(y, p_locked)
        out["folds"][d] = {
            "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
            "candidate": {k: m[k] for k in ("roc_auc", "pr_auc", "f1_macro",
                                            "brier", "ece_10", "human_fpr")},
            "locked_baseline_same_rows": {k: ml[k] for k in
                                          ("roc_auc", "pr_auc", "f1_macro",
                                           "brier", "ece_10", "human_fpr")},
            "delta_pr_auc_candidate_minus_locked": round(
                m["pr_auc"] - ml["pr_auc"], 4),
            "duration_s": round(time.time() - ts, 1),
        }
        print(f"[E5] lodo {cand['name']}/{d}: pr={m['pr_auc']:.4f} "
              f"(locked {ml['pr_auc']:.4f})", flush=True)
    prs = [f["candidate"]["pr_auc"] for f in out["folds"].values()
           if "candidate" in f]
    prl = [f["locked_baseline_same_rows"]["pr_auc"] for f in out["folds"].values()
           if "candidate" in f]
    if prs:
        out["summary"] = {
            "n_folds": len(prs),
            "mean_pr_auc_candidate": round(float(np.mean(prs)), 4),
            "min_pr_auc_candidate": round(float(np.min(prs)), 4),
            "mean_pr_auc_locked": round(float(np.mean(prl)), 4),
            "worst_domains": sorted(
                [d for d, f in out["folds"].items() if "candidate" in f],
                key=lambda d: out["folds"][d]["candidate"]["pr_auc"])[:3],
        }
    return out


def run_e5(cfg: dict, ctx: dict) -> dict:
    t0 = time.time()
    pool = selection_pool(cfg)
    res: dict = {"name": "E5-generalization", "status": "ok",
                 "pool": pool, "locked_baseline": _locked_reference(),
                 "table": {}, "paraphrase": {}, "lodo": {}}
    for nm in pool:
        cand = load_candidate(nm)
        if cand is None:
            res["table"][nm] = {"status": "missing model file"}
            continue
        _need_syn(ctx, cand["features"])
        suite = eval_suite(cfg, ctx, cand)
        res["table"][nm] = {"sets": suite,
                            "robustness_mean_roc": _robustness(suite),
                            "default_cal": cand.get("default_cal"),
                            "fit_seconds": cand.get("fit_seconds")}
        raid_sc = _raid_scored_for(cfg, ctx, cand)
        res["paraphrase"][nm] = _raid_paraphrase_analysis(raid_sc)
        print(f"[E5] {nm}: robustness={res['table'][nm]['robustness_mean_roc']}",
              flush=True)

    ranked = sorted(
        [n for n in pool if "robustness_mean_roc" in res["table"].get(n, {})],
        key=lambda n: -res["table"][n]["robustness_mean_roc"])
    res["ranking_pre_lodo"] = ranked

    lodo_cfg = cfg["e5"]["lodo"]
    if lodo_cfg.get("enabled") and ranked:
        top = ranked[:int(lodo_cfg.get("top_k", 2))]
        for nm in top:
            cand = load_candidate(nm)
            _need_syn(ctx, cand["features"])
            res["lodo"][nm] = _lodo_for_candidate(
                cfg, ctx, cand, cand.get("default_cal", "isotonic"))
        s1 = res["lodo"].get(top[0], {}).get("summary", {}).get(
            "mean_pr_auc_candidate")
        if len(top) > 1:
            s2 = res["lodo"].get(top[1], {}).get("summary", {}).get(
                "mean_pr_auc_candidate")
            res["lodo"]["override_check"] = {
                "rule": lodo_cfg.get("use", ""),
                "leader": top[0], "runner_up": top[1],
                "gap": (round(s1 - s2, 4)
                        if s1 is not None and s2 is not None else None),
                "triggered": bool(s1 is not None and s2 is not None
                                  and (s1 - s2) < -0.02),
            }
    res["locked_lodo_reference"] = res["locked_baseline"].get("lodo_summary", {})
    res["duration_s"] = round(time.time() - t0, 1)
    return res


# --------------------------------------------------------------------------
# E6 - leakage / importance
# --------------------------------------------------------------------------

def _univariate_block(frames: dict[str, tuple], names: list[str]) -> dict:
    from sklearn.metrics import roc_auc_score
    out: dict = {}
    for f in names:
        row: dict = {}
        direction = None
        for sname, (df, y) in frames.items():
            x = pd.to_numeric(df[f], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x)
            yy = np.asarray(y)[mask]
            xx = x[mask]
            if len(xx) < 50 or len(np.unique(yy)) < 2:
                row[f"{sname}_auc"] = None
                continue
            auc = float(roc_auc_score(yy, xx))
            row[f"{sname}_auc"] = round(auc, 4)
            row[f"{sname}_sep"] = round(max(auc, 1 - auc), 4)
            if sname == "val_tune":
                direction = 1 if auc >= 0.5 else -1
            elif direction is not None:
                d = 1 if auc >= 0.5 else -1
                if d != direction:
                    row["direction_flip"] = True
        out[f] = row
    return out


def _perm_importance(model, features: list[str], df: pd.DataFrame,
                     n_rows: int, n_repeats: int, seed: int) -> dict:
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    sub = df.sample(n=min(n_rows, len(df)), random_state=seed)
    y = sub["y"].to_numpy()
    X = sub[features].to_numpy(dtype=float)
    base = float(average_precision_score(y, model.predict_proba(X)[:, 1]))
    out: dict = {}
    for j, f in enumerate(features):
        drops = []
        for _ in range(n_repeats):
            Xs = X.copy()
            Xs[:, j] = rng.permutation(Xs[:, j])
            pr = float(average_precision_score(
                y, model.predict_proba(Xs)[:, 1]))
            drops.append(base - pr)
        out[f] = round(float(np.mean(drops)), 5)
    return {"baseline_pr_auc": round(base, 4), "n_rows": int(len(sub)),
            "n_repeats": n_repeats, "importance_delta_pr_auc": out}


def _probe_model(cfg: dict, ctx: dict, features: list[str]) -> dict:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    tr, va = ctx["train"], ctx["val_tune"]
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, solver="lbfgs",
                                                class_weight="balanced",
                                                max_iter=1000, random_state=42))])
    pipe.fit(tr[features].to_numpy(), tr["y"].to_numpy())
    sets = ctx["sets"]
    return {"n_features": len(features), "features": features,
            "val_tune": _set_metrics(va["y"], pipe.predict_proba(
                va[features].to_numpy())[:, 1]),
            "test": _set_metrics(sets["test"]["y"], pipe.predict_proba(
                sets["test"][features].to_numpy())[:, 1]),
            "ood_gpt": _set_metrics(sets["ood_gpt"]["y"], pipe.predict_proba(
                sets["ood_gpt"][features].to_numpy())[:, 1]),
            "hc3": _set_metrics(sets["hc3"]["y"], pipe.predict_proba(
                sets["hc3"][features].to_numpy())[:, 1]),
            "raid_clean": _set_metrics(
                sets["raid"].query("attack == 'none' and in_policy")["y"],
                pipe.predict_proba(sets["raid"].query(
                    "attack == 'none' and in_policy")[features].to_numpy())[:, 1]),
            }


def run_e6(cfg: dict, ctx: dict) -> dict:
    t0 = time.time()
    ecfg = cfg["e6"]
    res: dict = {"name": "E6-leakage-importance", "status": "ok"}
    mage_tr = ctx["train"]
    hc3 = ctx["sets"]["hc3"]
    raid = ctx["sets"]["raid"]
    raid_clean = raid[(raid["attack"] == "none") & raid["in_policy"]]
    vt = ctx["val_tune"]
    frames = {
        "val_tune": (vt, vt["y"].to_numpy()),
        "ood_gpt": (ctx["sets"]["ood_gpt"],
                    ctx["sets"]["ood_gpt"]["y"].to_numpy()),
        "hc3": (hc3, hc3["y"].to_numpy()),
        "raid_clean": (raid_clean, raid_clean["y"].to_numpy()),
    }
    uni = _univariate_block(frames, V1_FEATURES)
    res["univariate_auc_by_corpus"] = uni

    y_tr = mage_tr["y"]
    mage_mach, mage_hum = mage_tr[y_tr == 1], mage_tr[y_tr == 0]
    y_h, y_r = hc3["y"], raid_clean["y"]
    res["drift_vs_mage_train"] = {
        "hc3_machine": _drift_block(hc3[y_h == 1], mage_mach, V1_FEATURES,
                                    "hc3_machine", "mage_train_machine"),
        "hc3_human": _drift_block(hc3[y_h == 0], mage_hum, V1_FEATURES,
                                  "hc3_human", "mage_train_human"),
        "raid_machine": _drift_block(raid_clean[y_r == 1], mage_mach,
                                     V1_FEATURES, "raid_machine",
                                     "mage_train_machine"),
        "raid_human": _drift_block(raid_clean[y_r == 0], mage_hum,
                                   V1_FEATURES, "raid_human",
                                   "mage_train_human"),
    }

    thresh = float(ecfg["artifact_flag"]["min_abs_delta_auc"])
    flags = {}
    for f, row in uni.items():
        val_sep = row.get("val_tune_sep")
        if val_sep is None or val_sep < 0.60:
            continue
        reasons = []
        for other in ("hc3", "raid_clean"):
            osep = row.get(f"{other}_sep")
            if osep is not None and abs(val_sep - osep) >= thresh:
                reasons.append(f"|val_sep - {other}_sep| >= {thresh} "
                               f"({val_sep} vs {osep})")
        if row.get("direction_flip"):
            reasons.append("separability direction flips across corpora")
        if reasons:
            flags[f] = {"val_tune_sep": val_sep, "reasons": reasons}
    res["artifact_flags"] = {
        "rule": ecfg["artifact_flag"], "features": flags,
        "n_flagged": len(flags),
        "note": "flags = high in-domain separability + large cross-corpus drop "
                "or direction flip; evidence of corpus-dependent (possibly "
                "leaky) features, not proof of causal leakage",
    }

    imp: dict = {}
    pool = selection_pool(cfg)
    for nm in pool:
        cand = load_candidate(nm)
        if cand is None:
            continue
        _need_syn(ctx, cand["features"])
        imp[nm] = _perm_importance(
            cand["model"], cand["features"], vt,
            int(ecfg["permutation"]["n_rows"]),
            int(ecfg["permutation"]["n_repeats"]),
            int(ecfg["permutation"]["seed"]))
        print(f"[E6] permutation importance: {nm}", flush=True)
    res["permutation_importance"] = imp

    fam_of = {f: fam for fam, fs in FEATURE_FAMILIES.items() for f in fs}
    fam_agg: dict = {}
    for nm, block in imp.items():
        agg: dict = {}
        for f, v in block["importance_delta_pr_auc"].items():
            fam = fam_of.get(f, "syntactic_spacy")
            agg.setdefault(fam, []).append(v)
        fam_agg[nm] = {k: {"mean_delta_pr_auc": round(float(np.mean(v)), 5),
                           "n": len(v)} for k, v in agg.items()}
    res["family_aggregation"] = fam_agg

    lexical = list(FEATURE_FAMILIES["lexical"])
    formatting = [f for f in ("hash_rate", "bullet_rate", "header_rate",
                              "uppercase_word_ratio") if f in V1_FEATURES]
    res["probes"] = {
        "lexical_only": _probe_model(cfg, ctx, lexical),
        "formatting_only": _probe_model(cfg, ctx, formatting),
        "note": "probe models isolate how far lexical/formatting features "
                "alone generalize (eval-only corpora never trained on)",
    }
    res["duration_s"] = round(time.time() - t0, 1)
    return res


# --------------------------------------------------------------------------
# SELECTION - ranking + final artifact
# --------------------------------------------------------------------------

def _criteria_values(pool: list[str], e5tab: dict, e4: dict) -> dict:
    crit: dict[str, dict] = {}
    for nm in pool:
        tab = e5tab.get(nm, {})
        suite = tab.get("sets", {})
        if not suite:
            continue
        e4r = e4.get("candidates", {}).get(nm, {})
        cand = load_candidate(nm)
        spec = cand.get("spec", {}) if cand else {}
        crit[nm] = {
            "robustness_generalization": tab.get("robustness_mean_roc"),
            "calibration": _mean([suite[s]["brier"]
                                  for s in ("test", "hc3", "raid_clean")]),
            "false_positive_behavior": _mean([suite[s]["human_fpr"]
                                              for s in ("test", "hc3",
                                                        "raid_clean", "ood_gpt")]),
            "pr_auc": suite.get("test", {}).get("pr_auc"),
            "interpretability": 1 if spec.get("type") == "logistic_regression"
            else 0,
            "computational_practicality": cand.get("fit_seconds") if cand else None,
            "in_distribution_performance": suite.get("test", {}).get("roc_auc"),
            "_chosen_cal": e4r.get("chosen"),
        }
    return crit


def _ref_stats_block(df: pd.DataFrame, names: list[str]) -> dict:
    out = {}
    for n in names:
        x = pd.to_numeric(df[n], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        sd = float(x.std())
        out[n] = {"median": round(float(np.median(x)), 6),
                  "mean": round(float(x.mean()), 6),
                  "std": round(sd if sd > 1e-12 else 1.0, 6)}
    return out


def run_selection(cfg: dict, ctx: dict) -> dict:
    t0 = time.time()
    need = ("E3", "E1", "E4", "E5")
    missing = [k for k in need if _read_cache(k) is None]
    if missing:
        raise RuntimeError(f"selection needs cached results for {missing} - "
                           f"run them first")
    e3, e1, e2 = _read_cache("E3"), _read_cache("E1"), _read_cache("E2")
    e4, e5, e6 = _read_cache("E4"), _read_cache("E5"), _read_cache("E6")
    pool = selection_pool(cfg)
    e5tab = e5.get("table", {})
    crit = _criteria_values(pool, e5tab, e4.get("candidates", {}))
    order = list(cfg["selection"]["criteria_order"])
    bands = dict(cfg["selection"]["tie_bands"])
    ranked, trace = lexicographic_rank(crit, order, bands, HIGHER_BETTER)

    override = e5.get("lodo", {}).get("override_check", {})
    override_applied = False
    if override.get("triggered") and len(ranked) >= 2 \
            and ranked[0] == override.get("leader"):
        ranked[0], ranked[1] = ranked[1], ranked[0]
        override_applied = True
    winner = ranked[0]

    cand = load_candidate(winner)
    if cand is None:
        raise RuntimeError(f"winner {winner} has no model file")
    _need_syn(ctx, cand["features"])
    cal_type = (e4.get("candidates", {}).get(winner, {}) or {}).get("chosen") \
        or cand.get("default_cal", "isotonic")
    raw_cal = raw_p(cand, ctx["val_cal"])
    calibrator = (IsotonicCalibrator() if cal_type == "isotonic"
                  else PlattCalibrator()).fit(
        raw_cal, ctx["val_cal"]["y"].to_numpy())
    cand["default_cal"] = cal_type
    joblib.dump(cand, MODEL_DIR / f"{winner}.joblib")

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(cand["model"], FINAL_DIR / "model.joblib")
    joblib.dump(calibrator, FINAL_DIR / "calibrator.joblib")
    joblib.dump(ctx["state"], FINAL_DIR / "feature_state.joblib")
    features_doc = {
        "version": sha_of(cand["features"]),
        "pipeline_version": cfg["pipeline_version"],
        "n_features": len(cand["features"]),
        "features": cand["features"],
        "families": {f: fam for fam, fs in FEATURE_FAMILIES.items()
                     for f in fs},
        "note": "syn_* = spacy en_core_web_sm syntactic family "
                "(syntactic_spacy_v1), excluded from features_v1",
    }
    dump_json(FINAL_DIR / "features.json", features_doc)
    dump_json(FINAL_DIR / "reference.json", {
        "train_reference_stats": _ref_stats_block(
            ctx["train"], cand["features"]),
        "note": "MAGE-train distribution of the final feature set; used by "
                "ml/explain.py for local perturbation explanations",
        "n_train_rows": int(len(ctx["train"])),
    })
    imp_block = (e6 or {}).get("permutation_importance", {}).get(winner)
    dump_json(FINAL_DIR / "importance.json", {
        "method": "permutation importance (mean PR-AUC drop over n_repeats "
                  "shuffles on a seeded val_tune sample); SHAP/xgboost/lightgbm "
                  "not installed - documented deviation, no fabricated values",
        "selection_pool_coverage": sorted((e6 or {})
                                          .get("permutation_importance", {})),
        "winner": imp_block,
    })
    dump_json(FINAL_DIR / "config.json", {
        "spec_key": cand["spec_key"], "spec": cand["spec"], "C": cand.get("C"),
        "calibrator": cal_type, "threshold": 0.5,
        "label_policy": cand.get("label_policy"),
        "train_rows": int(len(ctx["train"])),
        "val_tune_rows": int(len(ctx["val_tune"])),
        "val_cal_rows": int(len(ctx["val_cal"])),
        "seed": cfg["seed"], "trained_at": cand.get("trained_at"),
        "hc3_raid_policy": "evaluation-only: never trained/calibrated/tuned on",
    })
    suite = e5tab.get(winner, {}).get("sets", {})
    dump_json(FINAL_DIR / "calibration.json", {
        "calibrator": cal_type,
        "e4_study": e4.get("candidates", {}).get(winner),
        "reliability_bins_val_cal": reliability_bins(
            calibrator.predict(raw_cal),
            ctx["val_cal"]["y"].to_numpy(), n_bins=10),
        "reliability_bins_test": reliability_bins(
            calibrator.predict(raw_p(cand, ctx["sets"]["test"])),
            ctx["sets"]["test"]["y"].to_numpy(), n_bins=10),
    })
    dump_json(FINAL_DIR / "gate.json", {
        "status": "pending", "note": "written by experiment E8"},
    )

    eval_summary = {
        s: {k: suite[s].get(k) for k in
            ("n", "roc_auc", "pr_auc", "f1_macro", "accuracy", "brier",
             "ece_10", "human_fpr")}
        for s in ("test", "ood_gpt", "ood_gpt_para", "hc3", "raid_clean")
        if s in suite
    }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": cfg["pipeline_version"],
        "winner": winner, "ranking": ranked,
        "calibrator": cal_type, "threshold": 0.5,
        "selection_criteria_order": order,
        "criteria_values": {n: {k: v for k, v in c.items()
                                if not k.startswith("_")}
                            for n, c in crit.items()},
        "tie_bands": bands,
        "lodo_override_applied": override_applied,
        "lodo_override_record": override,
        "files": {}, "eval_summary": eval_summary,
        "data_policy": {
            "trained_on": "MAGE-train only",
            "selected_on": "MAGE-val_tune (+ E4 cal_fit/cal_eval inside val_cal)",
            "calibrated_on": "MAGE-val_cal",
            "eval_only": ["MAGE-test", "OOD-GPT", "OOD-GPT-para", "HC3", "RAID"],
            "threshold": "0.5 fixed from Phase 1, never tuned",
        },
    }
    for fn in ("model.joblib", "calibrator.joblib", "feature_state.joblib",
               "features.json", "reference.json", "importance.json",
               "config.json", "calibration.json"):
        p = FINAL_DIR / fn
        if p.exists():
            manifest["files"][fn] = {"sha256": file_sha(p),
                                     "bytes": p.stat().st_size}
    dump_json(FINAL_DIR / "manifest.json", manifest)

    res = {
        "name": "SELECTION", "status": "ok",
        "pool": pool, "criteria_values": crit, "ranking": ranked,
        "trace": trace, "winner": winner, "calibrator": cal_type,
        "lodo_override_applied": override_applied,
        "lodo_override_record": override,
        "artifact_dir": str(FINAL_DIR.relative_to(ROOT)).replace("\\", "/"),
        "manifest": {k: manifest[k] for k in ("winner", "ranking",
                                              "eval_summary", "files")},
        "decision_note": (
            "lexicographic criteria (robustness first, in-distribution last); "
            "every criterion value recorded above for audit; NOT selected by "
            "headline MAGE accuracy"),
        "e1_reference_test": (e1 or {}).get("reference_sets", {}).get("test"),
        "e3_test": {k: v.get("sets", {}).get("test")
                    for k, v in (e3 or {}).get("candidates", {}).items()},
        "duration_s": round(time.time() - t0, 1),
    }
    print(f"[SELECTION] winner={winner} ranking={ranked}", flush=True)
    return res


def load_final(final_dir: Path = FINAL_DIR) -> dict:
    if not (final_dir / "manifest.json").exists():
        raise RuntimeError(f"final artifact not found at {final_dir} - run "
                           f"SELECTION first")
    manifest = load_json(final_dir / "manifest.json")
    model = joblib.load(final_dir / "model.joblib")
    calibrator = joblib.load(final_dir / "calibrator.joblib")
    state = joblib.load(final_dir / "feature_state.joblib")
    features_doc = load_json(final_dir / "features.json")
    gate = load_json(final_dir / "gate.json")
    cfg_doc = load_json(final_dir / "config.json")
    return {"manifest": manifest, "model": model, "calibrator": calibrator,
            "state": state, "features": features_doc["features"],
            "features_doc": features_doc, "gate": gate, "config": cfg_doc,
            "dir": final_dir}


def final_predict(final: dict, df: pd.DataFrame) -> np.ndarray:
    feats = df[final["features"]].to_numpy()
    return final["calibrator"].predict(final["model"].predict_proba(feats)[:, 1])


# --------------------------------------------------------------------------
# E7 - structured error analysis on the final model
# --------------------------------------------------------------------------

def _pick_errors(df: pd.DataFrame, y, p, mask, descending: bool, stats: dict,
                 n: int, extra_cols: tuple = ()) -> list[dict]:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    idx = idx[np.argsort(-p[idx]) if descending else np.argsort(p[idx])]
    out = []
    for i in idx[:n]:
        row = df.iloc[i]
        d = {"text": str(row.get("text_norm", ""))[:240],
             "p_machine": round(float(p[i]), 4),
             "true": "machine" if int(y[i]) == 1 else "human"}
        if "words" in df.columns:
            try:
                d["words"] = int(row["words"])
            except (TypeError, ValueError):
                pass
        devs = []
        for f, st in stats.items():
            try:
                v = float(row[f])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(v):
                devs.append({"feature": f, "value": round(v, 4),
                             "z": round((v - st["mean"]) / st["std"], 3)})
        devs.sort(key=lambda x: -abs(x["z"]))
        d["top_feature_deviations"] = devs[:4]
        for c in extra_cols:
            if c in df.columns:
                d[c] = str(row[c])
        out.append(d)
    return out


def _slice_fpr_fnr(df: pd.DataFrame, y, p, group_col: str,
                   min_n: int = 20) -> dict:
    out = {}
    for g, idx in pd.Series(range(len(df))).groupby(
            df[group_col].astype(str).to_numpy()):
        ii = np.asarray(idx)
        if len(ii) < min_n:
            out[str(g)] = {"n": int(len(ii)), "note": "too few rows"}
            continue
        yy, pp = y[ii], p[ii]
        hum, mach = yy == 0, yy == 1
        out[str(g)] = {
            "n": int(len(ii)),
            "machine_share": round(float(yy.mean()), 4),
            "human_fpr": round(float((pp[hum] >= 0.5).mean()), 4) if hum.any() else None,
            "machine_fnr": round(float((pp[mach] < 0.5).mean()), 4) if mach.any() else None,
            "roc_auc": round(float(compute_metrics(yy, pp)["roc_auc"]), 4)
            if len(np.unique(yy)) > 1 else None,
        }
    return out


def _edited_text_block(cfg: dict, ctx: dict, final: dict) -> dict:
    import torch as _torch
    c = cfg["e7"]["edited_text"]
    cache_path = CACHE_DIR / "edited_texts.json"
    test = ctx["sets"]["test"]
    machine = test[(test["source_dataset"] == "mage") & (test["y"] == 1)]
    rng = np.random.default_rng(int(cfg["seed"]) + int(c["seed_offset"]))
    take = min(int(c["sample_n"]), len(machine))
    samp = machine.iloc[rng.choice(len(machine), size=take,
                                   replace=False)].copy()
    key = sha_of({"row_ids": samp["row_id"].tolist(),
                  "cfg": {k: c[k] for k in ("model_id", "sample_n",
                                            "temperature", "top_p",
                                            "max_new_tokens", "seed_offset")},
                  "seed": cfg["seed"]})
    if cache_path.exists():
        cached = load_json(cache_path)
        if cached.get("key") == key:
            items = cached["items"]
        else:
            items = None
    else:
        items = None
    if items is None:
        device = _torch.device("cpu")
        outs = _qwen_rewrite(samp["text_norm"].tolist(), c["prompt_template"],
                             c, device, greedy=bool(c.get("greedy", False)))
        items = [{"row_id": int(r), "original": o, "edited": e}
                 for r, o, e in zip(samp["row_id"].tolist(),
                                    samp["text_norm"].tolist(), outs)]
        dump_json(cache_path, {"key": key, "items": items,
                               "generated_at": datetime.now(timezone.utc)
                               .isoformat()})
    orig = [i["original"] for i in items]
    edited = [i["edited"] or i["original"] for i in items]

    def _score_batch(texts: list[str], with_syn: bool) -> np.ndarray:
        feats = extract_texts(texts, ctx["state"], workers=4)
        if with_syn:
            from ..features.syntactic import extract_syntactic
            feats = pd.concat([feats.reset_index(drop=True),
                               extract_syntactic(texts)], axis=1)
        return final_predict(final, feats)

    def _score_locked(texts: list[str]) -> np.ndarray:
        feats = extract_texts(texts, ctx["state"], workers=4)
        names = V1_FEATURES
        raw = ctx["bundle"]["model"].predict_proba(
            feats[names].to_numpy())[:, 1]
        return 1.0 - ctx["bundle"]["calibrator"].predict(raw)

    with_syn = any(f.startswith("syn_") for f in final["features"])
    p_pre_final = final_predict(final, samp)
    p_post_final = _score_batch(edited, with_syn)
    p_pre_locked = samp["p_machine_cal"].to_numpy()
    p_post_locked = _score_locked(edited)
    y = np.ones(len(items), dtype=int)

    def _rec(p):
        return round(float((p >= 0.5).mean()), 4)

    return {
        "model": c["model_id"], "n": len(items), "seed": int(cfg["seed"]) + 3,
        "protocol": "identical sampling (seed+3, 32 MAGE-test machine rows), "
                    "prompt and Qwen params as Phase-2 E3",
        "final_model": {
            "pre_recall_machine": _rec(p_pre_final),
            "post_recall_machine": _rec(p_post_final),
            "mean_p_pre": round(float(p_pre_final.mean()), 4),
            "mean_p_post": round(float(p_post_final.mean()), 4),
            "delta_recall": round(_rec(p_post_final) - _rec(p_pre_final), 4),
        },
        "locked_baseline": {
            "pre_recall_machine": _rec(p_pre_locked),
            "post_recall_machine": _rec(p_post_locked),
            "mean_p_pre": round(float(p_pre_locked.mean()), 4),
            "mean_p_post": round(float(p_post_locked.mean()), 4),
            "delta_recall": round(_rec(p_post_locked) - _rec(p_pre_locked), 4),
        },
        "label_view": "ground truth stays MACHINE (authorship lineage); both "
                      "views coincide because the editor is a model",
        "examples": [
            {"original": i["original"][:260], "edited": i["edited"][:260],
             "p_final_pre": round(float(p_pre_final[k]), 4),
             "p_final_post": round(float(p_post_final[k]), 4),
             "p_locked_pre": round(float(p_pre_locked[k]), 4),
             "p_locked_post": round(float(p_post_locked[k]), 4)}
            for k, i in enumerate(items[:6])
        ],
        "mean_abs_delta_final": round(float(np.mean(
            np.abs(p_post_final - p_pre_final))), 4),
        "mean_abs_delta_locked": round(float(np.mean(
            np.abs(p_post_locked - p_pre_locked))), 4),
    }


def run_e7(cfg: dict, ctx: dict) -> dict:
    t0 = time.time()
    final = load_final()
    res: dict = {"name": "E7-error-analysis-final", "status": "ok",
                 "model": final["manifest"]["winner"],
                 "threshold": 0.5, "slices": {}}
    names = final["features"]
    _need_syn(ctx, names)
    stats = _ref_stats_block(ctx["train"], names)
    sets = ctx["sets"]

    test = sets["test"]
    y = test["y"].to_numpy()
    p = final_predict(final, test)
    mage = test.reset_index(drop=True)
    ym, pm = y, p
    res["slices"]["mage_test_by_domain"] = _slice_fpr_fnr(mage, ym, pm, "domain")
    mage_mach = mage[ym == 1].reset_index(drop=True).copy()
    mage_mach["_family"] = mage_mach["src"].map(gen_family).astype(str)
    res["slices"]["mage_test_machine_by_generator"] = _slice_fpr_fnr(
        mage_mach, np.ones(len(mage_mach), int),
        pm[ym == 1], "_family", min_n=30)
    res["slices"]["mage_test_by_length"] = {
        k: {kk: vv for kk, vv in v.items() if kk != "confusion_matrix"}
        for k, v in slice_metrics(ym, pm, mage["words"].to_numpy(),
                                  mage["domain"].astype(str).to_numpy())
        ["by_length"].items()}
    fp = (ym == 0) & (pm >= 0.5)
    fn = (ym == 1) & (pm < 0.5)
    res["slices"]["mage_test_human_fp"] = _pick_errors(
        mage, ym, pm, fp, True, stats, 6, ("domain", "src"))
    res["slices"]["mage_test_machine_fn"] = _pick_errors(
        mage, ym, pm, fn, False, stats, 6, ("domain", "src"))

    hc3 = sets["hc3"]
    yh = hc3["y"].to_numpy()
    ph = final_predict(final, hc3)
    res["slices"]["hc3_by_domain"] = _slice_fpr_fnr(hc3, yh, ph, "domain",
                                                    min_n=50)
    res["slices"]["hc3_human_fp"] = _pick_errors(
        hc3, yh, ph, (yh == 0) & (ph >= 0.5), True, stats, 6, ("domain",))
    res["slices"]["hc3_machine_fn"] = _pick_errors(
        hc3, yh, ph, (yh == 1) & (ph < 0.5), False, stats, 6, ("domain",))

    raid = sets["raid"].copy()
    raid["p_machine_cal"] = final_predict(final, raid)
    pol = raid[raid["in_policy"]]
    res["paraphrase"] = _raid_paraphrase_analysis(raid)
    yr, pr_ = pol["y"].to_numpy(), pol["p_machine_cal"].to_numpy()
    res["slices"]["raid_policy_by_domain"] = _slice_fpr_fnr(
        pol.reset_index(drop=True), yr, pr_, "domain")
    res["slices"]["raid_human_fp"] = _pick_errors(
        pol, yr, pr_, (yr == 0) & (pr_ >= 0.5), True, stats, 6,
        ("model", "domain"))
    res["slices"]["raid_machine_fn"] = _pick_errors(
        pol, yr, pr_, (yr == 1) & (pr_ < 0.5), False, stats, 6,
        ("model", "domain"))

    for key in ("ood_gpt", "ood_gpt_para"):
        od = sets[key]
        yo = od["y"].to_numpy()
        po = final_predict(final, od)
        res["slices"][f"{key}_by_src"] = _slice_fpr_fnr(od, yo, po, "src",
                                                        min_n=30)
        res["slices"][f"{key}_machine_fn"] = _pick_errors(
            od, yo, po, (yo == 1) & (po < 0.5), False, stats, 6, ("src",))
        res["slices"][f"{key}_human_fp"] = _pick_errors(
            od, yo, po, (yo == 0) & (po >= 0.5), True, stats, 6, ("src",))

    dev_counts: dict = {}
    for k, v in res["slices"].items():
        if isinstance(v, list):
            for ex in v:
                for d in ex.get("top_feature_deviations", []):
                    if abs(d["z"]) >= 3.0:
                        dev_counts.setdefault(d["feature"], 0)
                        dev_counts[d["feature"]] += 1
    res["recurring_patterns"] = {
        "method": "count of |z| >= 3 feature deviations among the top error "
                  "examples above (z vs MAGE-train distribution)",
        "counts_by_feature": dict(sorted(dev_counts.items(),
                                         key=lambda kv: -kv[1])[:12]),
        "interpretation_rule": "recurring deviations describe WHERE errors "
                               "cluster in feature space; they are not causal "
                               "attributions",
    }

    try:
        res["edited_text"] = _edited_text_block(cfg, ctx, final)
    except Exception as exc:
        res["edited_text"] = {"status": "failed",
                              "error": f"{type(exc).__name__}: {exc}",
                              "traceback": traceback.format_exc()[-1500:]}
    res["slice_counts"] = {k: (len(v) if isinstance(v, list) else None)
                           for k, v in res["slices"].items()}
    res["duration_s"] = round(time.time() - t0, 1)
    return res


# --------------------------------------------------------------------------
# E8 - evidence gate (writes gate.json into the final artifact)
# --------------------------------------------------------------------------

def decide_gate(gate_table: list[dict], abstain_frac_max: float) -> dict:
    """Pre-declared rule: smallest min_words whose EXCLUDED bucket has
    F1-macro strictly below the KEPT subset AND abstained fraction <= max."""
    eligible = []
    for row in gate_table:
        if row.get("note"):
            continue
        if (row["excluded_f1_macro"] is not None
                and row["kept_f1_macro"] is not None
                and row["excluded_f1_macro"] < row["kept_f1_macro"]
                and row["frac_abstained"] <= abstain_frac_max):
            eligible.append(row)
    if not eligible:
        return {"min_words": None, "rule_applied": False,
                "reason": "no gate candidate satisfied the pre-declared rule; "
                          "no length gate applied"}
    pick = min(eligible, key=lambda r: r["min_words"])
    return {"min_words": int(pick["min_words"]), "rule_applied": True,
            "abstain_frac_test": pick["frac_abstained"],
            "kept_f1_macro": pick["kept_f1_macro"],
            "excluded_f1_macro": pick["excluded_f1_macro"],
            "eligible": [r["min_words"] for r in eligible]}


def mixed_band(p: np.ndarray, y: np.ndarray, edges: list[float],
               acc_threshold: float = 0.75) -> dict:
    """Smallest reliability interval strictly straddling 0.5 (lo < 0.5 < hi)
    with observed accuracy < acc_threshold on the given split."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    i0 = next(i for i in range(len(edges) - 1)
              if edges[i] <= 0.5 <= edges[i + 1])
    best = None
    for i in range(i0 + 1):                 # lo = edges[i] <= 0.5
        for j in range(i0 + 1, len(edges) - 1):   # hi = edges[j+1] > 0.5
            lo, hi = edges[i], edges[j + 1]
            if not (lo < 0.5 < hi):
                continue
            m = (p >= lo) & (p <= hi)
            if not m.any():
                continue
            acc = float(((p[m] >= 0.5) == y[m]).mean())
            if acc < acc_threshold:
                width = hi - lo
                if best is None or width < best[2]:
                    best = (lo, hi, width, acc, int(m.sum()))
    if best is None:
        return {"band": None,
                "note": "no interval around 0.5 met accuracy < "
                        f"{acc_threshold} on val_cal; no mixed band declared"}
    lo, hi, _, acc, n = best
    return {"band": [round(lo, 2), round(hi, 2)],
            "accuracy_inside_band": round(acc, 4), "n_inside": n,
            "rule": f"smallest interval strictly containing 0.5 with "
                    f"val_cal accuracy < {acc_threshold}"}


def run_e8(cfg: dict, ctx: dict) -> dict:
    t0 = time.time()
    final = load_final()
    ecfg = cfg["e8"]
    res: dict = {"name": "E8-evidence-gate", "status": "ok",
                 "model": final["manifest"]["winner"], "buckets": {},
                 "fine_buckets": {}, "gate_table": []}
    sets = ctx["sets"]
    test = sets["test"]
    y = test["y"].to_numpy()
    p = final_predict(final, test)
    words = test["words"].to_numpy()
    sl = slice_metrics(y, p, words, test["domain"].astype(str).to_numpy())
    for k, v in sl["by_length"].items():
        res["buckets"][k] = {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                             for kk, vv in v.items() if kk != "confusion_matrix"}
    fine = [0, 25, 50, 75, 100, 150, 200, 300, 400, 10 ** 9]
    for lo, hi in zip(fine[:-1], fine[1:]):
        m = (words >= lo) & (words < hi)
        name = f"{lo}-{hi if hi < 10 ** 9 else 'inf'}w"
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            res["fine_buckets"][name] = {"n": int(m.sum()),
                                         "note": "too few / single-class"}
            continue
        mm = _set_metrics(y[m], p[m])
        res["fine_buckets"][name] = {k: mm[k] for k in
                                     ("n", "accuracy", "f1_macro", "roc_auc",
                                      "pr_auc", "brier", "ece_10",
                                      "mean_abs_p_minus_y", "human_fpr")}

    for g in ecfg["gate_candidates"]:
        keep = words >= g
        excl = ~keep
        if keep.sum() < 20 or len(np.unique(y[keep])) < 2:
            res["gate_table"].append({"min_words": int(g), "note":
                                      "too few / single-class kept"})
            continue
        mk = compute_metrics(y[keep], p[keep])
        row = {
            "min_words": int(g),
            "n_kept": int(keep.sum()), "n_abstained": int(excl.sum()),
            "frac_abstained": round(float(excl.mean()), 4),
            "kept_f1_macro": round(float(mk["f1_macro"]), 4),
            "kept_ece": round(float(mk["ece_10"]), 4),
            "kept_accuracy": round(float(mk["accuracy"]), 4),
        }
        if excl.sum() >= 20 and len(np.unique(y[excl])) == 2:
            me = compute_metrics(y[excl], p[excl])
            row["excluded_f1_macro"] = round(float(me["f1_macro"]), 4)
            row["excluded_mean_abs_p_minus_y"] = round(float(
                me["mean_abs_p_minus_y"]), 4)
        else:
            row["excluded_f1_macro"] = None
            row["note"] = "excluded bucket too small/single-class"
        res["gate_table"].append(row)

    decision = decide_gate(res["gate_table"],
                           float(ecfg["abstain_frac_max"]))
    decision["rule"] = ecfg["decision_rule"]
    validation: dict = {}
    if decision.get("min_words") is not None:
        g = decision["min_words"]
        for sname in ("hc3", "ood_gpt", "ood_gpt_para"):
            sdf = sets[sname]
            sy, sp = sdf["y"].to_numpy(), final_predict(final, sdf)
            sw = sdf["words"].to_numpy()
            keep = sw >= g
            full_h = sy == 0
            keep_h = keep & full_h
            validation[sname] = {
                "n_kept": int(keep.sum()),
                "frac_abstained": round(float((~keep).mean()), 4),
                "kept_human_fpr": round(float((sp[keep_h] >= 0.5).mean()), 4)
                if keep_h.any() else None,
                "full_human_fpr": round(float((sp[full_h] >= 0.5).mean()), 4)
                if full_h.any() else None,
            }
        validation["note"] = "kept-subset human FPR must not exceed full-set"
    decision["validation"] = validation
    res["gate_decision"] = decision

    cal_type = final["config"]["calibrator"]
    p_val = final["calibrator"].predict(
        raw_p({"model": final["model"], "features": final["features"]},
              ctx["val_cal"]))
    y_val = ctx["val_cal"]["y"].to_numpy()
    edges = [i / 10 for i in range(11)]
    mb = mixed_band(p_val, y_val, edges)
    mb["rule"] = ecfg["mixed_band_rule"]
    mb["threshold"] = 0.5
    res["mixed_band"] = mb

    gate_doc = {
        "status": "active",
        "min_words": decision.get("min_words"),
        "decision_rule": ecfg["decision_rule"],
        "abstain_frac_max": float(ecfg["abstain_frac_max"]),
        "abstain_frac_test": decision.get("abstain_frac_test"),
        "threshold": 0.5,
        "mixed_band": mb.get("band"),
        "mixed_band_rule": ecfg["mixed_band_rule"],
        "mixed_band_accuracy": mb.get("accuracy_inside_band"),
        "validation": validation,
        "threshold_policy": cfg["threshold_policy"],
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    dump_json(FINAL_DIR / "gate.json", gate_doc)
    manifest_path = FINAL_DIR / "manifest.json"
    manifest = load_json(manifest_path)
    manifest["files"]["gate.json"] = {"sha256": file_sha(FINAL_DIR / "gate.json"),
                                      "bytes": (FINAL_DIR / "gate.json").stat().st_size}
    manifest["gate"] = {"min_words": gate_doc["min_words"],
                        "mixed_band": gate_doc["mixed_band"]}
    manifest["artifact_complete"] = True
    dump_json(manifest_path, manifest)
    res["gate_written"] = "ml/models/final-v1/gate.json"
    res["duration_s"] = round(time.time() - t0, 1)
    print(f"[E8] gate={decision.get('min_words')} mixed_band={mb.get('band')}",
          flush=True)
    return res


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

EXPERIMENTS: dict = {
    "E3": run_e3, "E1": run_e1, "E2": run_e2, "E4": run_e4,
    "E5": run_e5, "E6": run_e6, "SELECTION": run_selection,
    "E7": run_e7, "E8": run_e8,
}


def run_all(cfg: dict, only: list[str] | None = None,
            force: bool = False) -> dict:
    cfg.setdefault("device", "cpu")
    for d in (RESULTS_DIR, CACHE_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    ctx = build_ctx(cfg)
    out: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha_of(cfg),
        "pipeline_version": cfg["pipeline_version"],
        "convention": "canonical y=1 MACHINE; Phase-3 candidates trained on "
                      "canonical labels, p = cal(model(x)) with no complement",
        "note": "HC3/RAID/OOD are evaluation-only; threshold 0.5 fixed; "
                "Phase-1 baseline untouched; selection lexicographic "
                "(robustness first)",
        "context": {
            "train_rows": int(len(ctx["train"])),
            "val_tune_rows": int(len(ctx["val_tune"])),
            "val_cal_rows": int(len(ctx["val_cal"])),
            "test_rows": int(len(ctx["test"])),
            "hc3_rows": int(len(ctx["sets"]["hc3"])),
            "raid_rows": int(len(ctx["sets"]["raid"])),
            "syn_ok": bool(ctx.get("syn_ok")),
            "build_seconds": ctx["built_at_s"],
        },
    }
    keys = only or list(EXPERIMENTS)
    for key in keys:
        cache_file = CACHE_DIR / f"{key}.json"
        if not force and key != "SELECTION" and cache_file.exists():
            # invalidate downstream caches when SELECTION changed the winner
            if key in ("E7", "E8"):
                cached = load_json(cache_file)
                winner = out.get("SELECTION", {}).get("winner")
                if winner and cached.get("model") not in (None, winner):
                    cache_file.unlink()
                    cached = None
                else:
                    out[key] = cached
                    out[key]["cached"] = True
                    continue
            else:
                out[key] = load_json(cache_file)
                out[key]["cached"] = True
                continue
        t0 = time.time()
        try:
            result = EXPERIMENTS[key](cfg, ctx)
        except Exception as exc:
            result = {"name": key, "status": "failed",
                      "error": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc()[-3000:]}
        out[key] = result
        if result.get("status") != "failed":
            dump_json(cache_file, result)
            print(f"[{key}] ok in {round(time.time() - t0, 1)}s", flush=True)
        else:
            if cache_file.exists():
                cache_file.unlink()
            print(f"[{key}] FAILED: {result.get('error')}", flush=True)
    return out
