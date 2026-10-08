"""Phase 2: robustness & generalization evaluation. Locked baseline ONLY.

Canonical convention (phase-2 target everywhere): y=1 means MACHINE, y=0 HUMAN.
Raw corpora encode labels differently:
  MAGE raw CSVs:  label 0 = machine, 1 = human   (verified from src tokens)
  MAGE stored parquet: same raw values (passthrough, no mapping in Phase 1)
  HC3 stored parquet:  label 0 = human answers, 1 = chatgpt answers (canonical)
  HC3 raw jsonl: human_answers / chatgpt_answers lists
  RAID eval subset: model != 'human' -> machine (canonical), model == 'human' -> human
  RAID test.csv: id+generation only - UNLABELED, distribution diagnostics only.

ORIENTATION (verified 2026-10-07 from raw src tokens):
  Phase 1 trained directly on stored labels, where stored-1 == HUMAN for MAGE.
  The locked model therefore estimates p(stored-1) = p(human-like) for ANY input
  text, corpus-agnostic. Canonical p(machine) = 1 - p_raw EVERYWHERE (MAGE,
  HC3, RAID, OOD) - including HC3, where the labels themselves are already
  canonical (labels and scores are mapped independently). The isotonic map was
  fit in stored space, so it is ALWAYS applied to stored-space scores first;
  the canonical complement happens after calibration.

  Phase-1 stored-space metrics label the positive class "machine" but the
  positive class is actually HUMAN (stored-1). The Phase-2 report reproduces
  them with corrected headers.

No training, no tuning, no threshold changes on evaluation data. The only
model training allowed here is documented *experimental* models (E1
single-generator gap, E5 leave-one-domain-out) that never touch HC3/RAID/
held-out evaluation rows and are reported separately from the locked baseline.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..features.extract import FeatureState, extract_many, feature_names
from ..preprocessing.build_dataset import PROCESSED, ROOT
from ..preprocessing.normalize import normalize_text, word_count, WORD_RE
from ..evaluation.metrics import compute_metrics, slice_metrics, LENGTH_BUCKETS, BUCKET_NAMES
from ..evaluation.calibration import (
    IsotonicCalibrator, reliability_bins, brier_score, expected_calibration_error,
)

CANON_MACHINE = 1
CANON_HUMAN = 0
RESULTS_DIR = ROOT / "ml" / "experiments" / "phase2"
CACHE_DIR = RESULTS_DIR / "cache"
MODELS_DIR = RESULTS_DIR / "models"
RAID_SUBSET = ROOT / "ml" / "data" / "raw" / "raid" / "raid_eval_subset.csv"
MIN_WORDS = 40   # Phase-1 inclusion policy (docs/dataset.md)
MAX_WORDS = 5000


def sha_of(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def _round_metrics(m: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def _met(y, p, threshold: float = 0.5) -> dict:
    m = compute_metrics(np.asarray(y), np.asarray(p), threshold=threshold)
    return _round_metrics(m)


def load_locked_model(run_dir: Path):
    out = {}
    for name in ("hist_gradient_boosting", "logistic_regression", "dummy_prior"):
        out[name] = joblib.load(run_dir / f"model_{name}.joblib")
    out["calibrator"] = joblib.load(run_dir / "calibrator.joblib")
    out["feature_state"] = joblib.load(run_dir / "feature_state.joblib")
    return out


_META_COLS = ["row_id", "label", "source_dataset", "src", "domain", "words",
              "text_norm", "group_id"]


def meta_join(splits: tuple[str, ...] = ("test",)) -> dict[str, pd.DataFrame]:
    """Join feature rows with their metadata (row_id key).

    features_*.parquet carry features only; test/val/train.parquet carry the
    text_norm/src/domain/label metadata. Joined deterministically on row_id
    (asserted 1:1). Only requested splits are loaded (memory).
    """
    out = {}
    for s in splits:
        meta = pd.read_parquet(PROCESSED / f"{s}.parquet", columns=_META_COLS)
        feats = pd.read_parquet(PROCESSED / f"features_{s}.parquet")
        m = meta.merge(feats, on="row_id", how="inner", suffixes=("", "_feat"))
        assert len(m) == len(meta) == len(feats), \
            f"meta/feature row mismatch on {s}: {len(meta)} vs {len(feats)}"
        if "label_feat" in m.columns:
            m = m.drop(columns=["label_feat"])
        m = m.reset_index(drop=True)
        out[s] = m
    return out


def eval_hc3() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "eval_hc3.parquet")


def canonical_labels(df: pd.DataFrame) -> np.ndarray:
    """Map stored labels to canonical (1=machine, 0=human) using source_dataset.

    MAGE stored labels pass the raw CSV convention through (1=human) -> complement.
    HC3 stored labels were assigned human_answers=0 / chatgpt_answers=1 -> as-is.
    Unknown sources raise (never silently guess).
    """
    y = df["label"].to_numpy().astype(int)
    src = df["source_dataset"].astype(str).to_numpy()
    canon = np.full_like(y, -1)
    canon[src == "mage"] = 1 - y[src == "mage"]
    canon[src == "hc3"] = y[src == "hc3"]
    if (canon == -1).any():
        bad = sorted(set(src[canon == -1]))
        raise ValueError(f"unmapped label convention for sources: {bad}")
    return canon


def machine_probability(p_raw: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Map model predict_proba[:,1] to canonical p(machine).

    The locked model was trained on MAGE stored labels where stored-1 == HUMAN,
    so p_raw estimates p(human-like) REGARDLESS of which corpus the text comes
    from. Canonical p(machine) is therefore 1 - p_raw for EVERY known source
    (MAGE, HC3, RAID). This is the cross-corpus transfer assumption behind all
    out-of-corpus evaluation and is stated explicitly in the report. Unknown
    sources -> NaN (never silently guess).
    """
    p = np.asarray(p_raw, dtype=float)
    src = df["source_dataset"].astype(str).to_numpy()
    out = np.full_like(p, np.nan)
    known = (src == "mage") | (src == "hc3") | (src == "raid")
    out[known] = 1.0 - p[known]
    return out


def gen_family(src: str) -> str:
    """Extract the generator family token from a MAGE src string."""
    s = str(src)
    if s.endswith("_human") or s.endswith("_human_para"):
        return "human"
    for marker in ("machine_continuation_", "machine_specified_", "machine_topical_"):
        if marker in s:
            tok = s.split(marker, 1)[1]
            return tok.strip("_") or "continuation_unknown"
    if "machine_" in s:
        return s.split("machine_", 1)[1].strip("_") or "machine_unknown"
    return "other"


def extract_texts(texts: list[str], state: FeatureState, workers: int = 0) -> pd.DataFrame:
    return extract_many(texts, state, workers=workers)


def score_texts(model, calibrator, feats: pd.DataFrame, df_meta: pd.DataFrame) -> dict:
    """Return stored + canonical probabilities.

    Calibration coherence: the isotonic map was fit on stored-space
    (p(human), stored labels). Reuse it in stored space, THEN complement to
    canonical p(machine). Never complement-then-calibrate.
    """
    names = feature_names()
    p_stored = model.predict_proba(feats[names].to_numpy())[:, 1]  # p(human-like)
    p_stored_cal = calibrator.predict(p_stored)
    return {"p_stored": p_stored, "p_stored_cal": p_stored_cal,
            "p_machine": 1.0 - p_stored, "p_machine_cal": 1.0 - p_stored_cal}


def ood_label_map(df: pd.DataFrame) -> pd.DataFrame:
    """Attach canonical labels to raw OOD csv rows (MAGE convention: 1 - raw)."""
    df = df.copy()
    df["source_dataset"] = "mage"
    df["ystored"] = df["label"].astype(int)
    df["y"] = 1 - df["ystored"]
    return df


# --------------------------------------------------------------------------
# caches (deterministic artifacts; keyed by content hashes)
# --------------------------------------------------------------------------

def _cache_meta_write(path: Path, meta: dict) -> None:
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _cache_meta_ok(path: Path, expect: dict) -> bool:
    if not path.exists():
        return False
    try:
        got = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return all(got.get(k) == v for k, v in expect.items())


def build_bundle(cfg: dict) -> dict:
    """Load locked artifacts and score the locked test split once (canonical).

    Scores are cached to parquet keyed by (model, calibrator, feature-names,
    test-meta) hashes; scoring is deterministic so a cache hit is bit-identical.
    """
    run_dir = ROOT / cfg["artifacts"]["run_dir"]
    locked = load_locked_model(run_dir)
    model = locked[cfg.get("primary_model", "hist_gradient_boosting")]
    calibrator = locked["calibrator"]
    state = locked["feature_state"]
    key = {
        "model_sha": file_sha(run_dir / f"model_{cfg.get('primary_model', 'hist_gradient_boosting')}.joblib"),
        "cal_sha": file_sha(run_dir / "calibrator.joblib"),
        "features_sha": sha_of(feature_names()),
        "test_meta_sha": file_sha(PROCESSED / "test.parquet"),
        "features_meta_sha": file_sha(PROCESSED / "features_test.parquet"),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    score_path = CACHE_DIR / "test_scores.parquet"
    meta_path = CACHE_DIR / "test_scores.meta.json"
    joined = meta_join(("test",))
    test = joined["test"]
    if _cache_meta_ok(meta_path, key) and score_path.exists():
        sc = pd.read_parquet(score_path)
        assert (sc["row_id"].to_numpy() == test["row_id"].to_numpy()).all(), \
            "cached test scores misaligned with test split"
        p_stored = sc["p_stored"].to_numpy()
        p_stored_cal = sc["p_stored_cal"].to_numpy()
    else:
        names = feature_names()
        sc = score_texts(model, calibrator, test[names], test)
        p_stored, p_stored_cal = sc["p_stored"], sc["p_stored_cal"]
        pd.DataFrame({"row_id": test["row_id"].to_numpy(),
                      "p_stored": p_stored, "p_stored_cal": p_stored_cal,
                      "p_machine": sc["p_machine"], "p_machine_cal": sc["p_machine_cal"],
                      }).to_parquet(score_path, index=False)
        _cache_meta_write(meta_path, key)
    test = test.copy()
    test["p_stored"] = p_stored
    test["p_stored_cal"] = p_stored_cal
    test["p_machine"] = 1.0 - p_stored
    test["p_machine_cal"] = 1.0 - p_stored_cal
    return {
        "model": model, "calibrator": calibrator, "state": state,
        "test": test, "feat_test": test,
        "p_raw": p_stored, "p_machine": 1.0 - p_stored,
        "p_machine_cal": 1.0 - p_stored_cal,
        "run_dir": run_dir,
    }


# --------------------------------------------------------------------------
# HC3 + RAID shared resources (feature caches; never trained on)
# --------------------------------------------------------------------------

def hc3_features(cfg: dict, state: FeatureState, workers: int = 0) -> pd.DataFrame:
    """HC3 eval rows + v1 features, cached by content hash."""
    ev = eval_hc3()
    key = {"hc3_sha": file_sha(PROCESSED / "eval_hc3.parquet"),
           "features_sha": sha_of(feature_names())}
    path = CACHE_DIR / "hc3_features.parquet"
    meta_path = CACHE_DIR / "hc3_features.meta.json"
    if _cache_meta_ok(meta_path, key) and path.exists():
        feats = pd.read_parquet(path)
        assert len(feats) == len(ev)
        return feats
    feats = extract_texts(ev["text_norm"].tolist(), state, workers=workers)
    out = pd.concat([ev[["row_id"]].reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    out.to_parquet(path, index=False)
    _cache_meta_write(meta_path, key)
    return out


def hc3_frame(cfg: dict, ctx: dict) -> pd.DataFrame:
    """HC3 rows with features + scores + canonical labels (cached scoring)."""
    ev = eval_hc3().reset_index(drop=True)
    feats = hc3_features(cfg, ctx["bundle"]["state"], workers=int(cfg.get("score_workers", 0)))
    names = feature_names()
    p_stored = ctx["bundle"]["model"].predict_proba(feats[names].to_numpy())[:, 1]
    p_stored_cal = ctx["bundle"]["calibrator"].predict(p_stored)
    ev["p_machine_cal"] = 1.0 - p_stored_cal
    ev["p_stored_cal"] = p_stored_cal
    return ev


def raid_frame() -> pd.DataFrame:
    """RAID labelled eval subset with normalized text + canonical labels."""
    df = pd.read_csv(RAID_SUBSET)
    df["source_dataset"] = "raid"
    norm = df["generation"].map(normalize_text)
    keep = norm.notna()
    dropped_invalid = int((~keep).sum())
    df = df[keep].copy()
    df["text_norm"] = norm[keep].to_numpy()
    df["words"] = df["text_norm"].map(word_count)
    df["y"] = (df["model"].astype(str) != "human").astype(int)
    df["in_policy"] = (df["words"] >= MIN_WORDS) & (df["words"] <= MAX_WORDS)
    df.attrs["dropped_invalid_text"] = dropped_invalid
    return df.reset_index(drop=True)


def raid_features(cfg: dict, df: pd.DataFrame, state: FeatureState,
                  workers: int = 0) -> pd.DataFrame:
    """RAID subset features, cached by (subset sha, feature names) hash."""
    key = {"raid_sha": file_sha(RAID_SUBSET), "features_sha": sha_of(feature_names())}
    path = CACHE_DIR / "raid_features.parquet"
    meta_path = CACHE_DIR / "raid_features.meta.json"
    if _cache_meta_ok(meta_path, key) and path.exists():
        feats = pd.read_parquet(path)
        assert len(feats) == len(df)
        return feats
    feats = extract_texts(df["text_norm"].tolist(), state, workers=workers)
    feats.to_parquet(path, index=False)
    _cache_meta_write(meta_path, key)
    return feats


def raid_scored(cfg: dict, ctx: dict) -> pd.DataFrame:
    """RAID subset rows + features + canonical p(machine) (cached features)."""
    df = raid_frame()
    attrs = dict(df.attrs)
    feats = raid_features(cfg, df, ctx["bundle"]["state"],
                          workers=int(cfg.get("score_workers", 0)))
    names = feature_names()
    p_stored = ctx["bundle"]["model"].predict_proba(feats[names].to_numpy())[:, 1]
    p_stored_cal = ctx["bundle"]["calibrator"].predict(p_stored)
    df = df.copy()
    df.attrs.update(attrs)
    df["p_machine_cal"] = 1.0 - p_stored_cal
    df["p_stored_cal"] = p_stored_cal
    return df


# --------------------------------------------------------------------------
# experimental (NON-baseline) training helpers - recorded separately
# --------------------------------------------------------------------------

def hgb_from_cfg(tcfg: dict):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        learning_rate=float(tcfg["learning_rate"]),
        max_leaf_nodes=int(tcfg["max_leaf_nodes"]),
        max_iter=int(tcfg["max_iter"]),
        random_state=int(tcfg.get("random_state", 42)),
    )


def fit_isotonic(p_val, y_val) -> IsotonicCalibrator:
    """Calibrator type is FIXED to isotonic (the Phase-1 selection); its
    parameters are fit on validation rows only. No selection/tuning happens
    on any evaluation set."""
    return IsotonicCalibrator().fit(p_val, y_val)


def state_dependent_features(texts, state: FeatureState) -> pd.DataFrame:
    """Recompute the two state-dependent columns (rare_token_ratio,
    mean_log_df) under an arbitrary FeatureState - exact mirror of
    ml.features.extract.extract_one for those columns. Used by E5 LODO so the
    document-frequency statistics never see the held-out domain's text.
    """
    rows = []
    for t in texts:
        if not isinstance(t, str) or not t.strip():
            rows.append((float("nan"), float("nan")))
            continue
        words = WORD_RE.findall(t)
        n = len(words)
        if n < 2:
            rows.append((float("nan"), float("nan")))
            continue
        lower = [w.lower() for w in words]
        dfs = [state.df_of(w) for w in lower]
        rare = sum(1 for d in dfs if d < state.rare_max_df) / n
        mlog = float(np.mean([math.log10(1 + d) for d in dfs]))
        rows.append((rare, mlog))
    return pd.DataFrame(rows, columns=["rare_token_ratio", "mean_log_df"])


def replace_state_features(feats: pd.DataFrame, texts, state: FeatureState) -> pd.DataFrame:
    out = feats.copy()
    sd = state_dependent_features(texts, state)
    out["rare_token_ratio"] = sd["rare_token_ratio"].to_numpy()
    out["mean_log_df"] = sd["mean_log_df"].to_numpy()
    return out


def cohen_d(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = math.sqrt((float(a.var()) + float(b.var())) / 2.0)
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


# --------------------------------------------------------------------------
# E0 - label-orientation audit (pre-evaluation gate)
# --------------------------------------------------------------------------

def run_e0(cfg: dict, ctx: dict) -> dict:
    """E0: label-orientation audit + score-orientation sanity + determinism.

    Proves stored-1 semantics per corpus from src tokens, asserts MAGE audit
    rates, verifies canonical-machine rows score higher on average, and checks
    that scoring is bit-identical across repeated calls. Fails loudly.
    """
    t0 = time.time()
    out: dict = {"name": "E0-label-orientation-audit", "findings": {}, "status": "ok"}
    joined = meta_join(("test",))
    mt = joined["test"]
    hum = mt[mt["src"].str.endswith("_human", na=False)]
    mac = mt[mt["src"].str.contains("machine_", na=False)]
    out["findings"]["mage_stored"] = {
        "n_human_src": int(len(hum)),
        "stored1_rate_human_src": float((hum["label"] == 1).mean()) if len(hum) else None,
        "n_machine_src": int(len(mac)),
        "stored1_rate_machine_src": float((mac["label"] == 1).mean()) if len(mac) else None,
        "conclusion": "stored-1 == HUMAN for MAGE (complement required for canonical)",
    }
    ev_p = PROCESSED / "eval_hc3.parquet"
    if ev_p.exists():
        ev = pd.read_parquet(ev_p, columns=["label", "source_dataset"])
        out["findings"]["hc3_stored"] = {
            "n": int(len(ev)),
            "stored1_rate": float((ev["label"] == 1).mean()),
            "note": "HC3 build assigned human_answers=0, chatgpt_answers=1; labels canonical as-is; SCORES still complemented (model semantics are corpus-agnostic)",
        }
    raid_p = ROOT / "ml" / "data" / "raw" / "raid" / "test.csv"
    if raid_p.exists():
        import csv as _csv
        with raid_p.open("r", encoding="utf-8") as fh:
            r = _csv.reader(fh)
            hdr = next(r)
            first = [next(r) for _ in range(3)]
        out["findings"]["raid_schema"] = {
            "columns": hdr, "n_columns_first_rows": [len(x) for x in first],
            "conclusion": "UNLABELED (id+generation only): distribution diagnostics only",
        }
    if RAID_SUBSET.exists():
        sub = raid_frame()
        out["findings"]["raid_labelled_subset"] = {
            "rows": int(len(sub)),
            "canonical_balance": {str(k): int(v) for k, v in sub["y"].value_counts().items()},
            "policy_rows": int(sub["in_policy"].sum()),
            "note": "RAID-train derived subset; EVALUATION-ONLY, never trained on",
        }
    assert out["findings"]["mage_stored"]["stored1_rate_human_src"] == 1.0, "MAGE human-src audit failed"
    assert out["findings"]["mage_stored"]["stored1_rate_machine_src"] == 0.0, "MAGE machine-src audit failed"
    # Score-orientation sanity: canonical-machine rows must score higher on average.
    y = canonical_labels(mt)
    p = ctx["bundle"]["p_machine_cal"]
    m1 = float(np.mean(p[y == 1])) if (y == 1).any() else None
    m0 = float(np.mean(p[y == 0])) if (y == 0).any() else None
    out["findings"]["score_orientation"] = {
        "mean_p_machine_given_canonical_machine": round(m1, 4) if m1 is not None else None,
        "mean_p_machine_given_canonical_human": round(m0, 4) if m0 is not None else None,
        "sanity": "mean p(machine|y=1) must exceed mean p(machine|y=0)",
    }
    assert m1 is not None and m0 is not None and m1 > m0, (
        f"score orientation flipped: E[p|y=1]={m1} vs E[p|y=0]={m0}")
    # Determinism: repeated scoring must be bit-identical.
    names = feature_names()
    X = mt[names].to_numpy()
    p1 = ctx["bundle"]["model"].predict_proba(X)[:, 1]
    p2 = ctx["bundle"]["model"].predict_proba(X)[:, 1]
    out["findings"]["determinism"] = {
        "scores_bit_identical": bool(np.array_equal(p1, p2, equal_nan=True)),
        "config_sha256": sha_of(cfg),
        "feature_names_sha256": sha_of(names),
    }
    assert out["findings"]["determinism"]["scores_bit_identical"], "scoring not deterministic"
    out["duration_s"] = round(time.time() - t0, 1)
    return out


# --------------------------------------------------------------------------
# E1 - unseen generators
# --------------------------------------------------------------------------

def _family_of_src(df: pd.DataFrame) -> pd.Series:
    return df["src"].map(gen_family)


def _train_machine_tokens() -> set[str]:
    """Model-family tokens in MAGE TRAIN machine rows (for seen/unseen split)."""
    import csv as _csv
    toks: set[str] = set()
    with (ROOT / "ml" / "data" / "raw" / "mage" / "train.csv").open("r", encoding="utf-8") as fh:
        r = _csv.reader(fh)
        next(r)
        for row in r:
            if "machine_" in row[2]:
                toks.add(gen_family(row[2]))
    return toks


def _raid_unseen_generator_metrics(raid: pd.DataFrame) -> dict:
    """Locked model on RAID: machine models are ALL unseen generators."""
    clean = raid[raid["attack"] == "none"]
    pol = clean[clean["in_policy"]]
    human = pol[pol["y"] == 0]
    machine = pol[pol["y"] == 1]
    out: dict = {"n_policy_clean": int(len(pol)),
                 "n_human_clean": int(len(human)), "n_machine_clean": int(len(machine)),
                 "per_model": {}}
    if len(human) and len(machine):
        y = np.concatenate([machine["y"].to_numpy(), human["y"].to_numpy()])
        p = np.concatenate([machine["p_machine_cal"].to_numpy(), human["p_machine_cal"].to_numpy()])
        out["overall_clean"] = _met(y, p)
    for mname, g in machine.groupby("model"):
        rec = {"n": int(len(g)),
               "recall_at_0.5": round(float((g["p_machine_cal"] >= 0.5).mean()), 4),
               "mean_p_machine": round(float(g["p_machine_cal"].mean()), 4)}
        out["per_model"][str(mname)] = rec
    if len(human):
        out["human_fpr_at_0.5"] = round(float((human["p_machine_cal"] >= 0.5).mean()), 4)
        out["human_mean_p_machine"] = round(float(human["p_machine_cal"].mean()), 4)
    return out


def _single_vs_multi_generator(cfg: dict, ctx: dict, test: pd.DataFrame,
                               test_only_families: list[str]) -> dict:
    """Plan E1: train on single-generator subsets vs the multi-generator
    (locked) model to quantify the generalization gap.

    Experimental models ONLY - never presented as the baseline. Training rows:
    MAGE-train human + machine rows of exactly one generator family. Eval rows:
    MAGE-test machine rows of *other* families + all MAGE-test human rows.
    """
    tcfg = cfg["experimental_training"]
    res: dict = {"models": {}, "eval_protocol": {
        "train": "MAGE-train: all human rows + machine rows of the named family",
        "eval": "MAGE-test: machine rows of OTHER families + all human rows",
        "calibration": "isotonic fit on MAGE-validation (all rows); no eval rows touched",
        "params": tcfg,
    }}
    joined = meta_join(("train", "val"))
    train, val = joined["train"], joined["val"]
    names = feature_names()
    human_tr = train[~train["src"].str.contains("machine_", na=False)]
    fam_series = _family_of_src(train)
    test_fam = _family_of_src(test)
    y_test = canonical_labels(test)
    human_test_mask = (y_test == 0)
    for fam in cfg["e1"]["single_gen_families"]:
        mach = train[fam_series == fam]
        if len(mach) < 500:
            res["models"][fam] = {"status": "skipped", "reason": f"only {len(mach)} train rows"}
            continue
        sub = pd.concat([human_tr, mach], ignore_index=True)
        clf = hgb_from_cfg(tcfg)
        t_start = time.time()
        clf.fit(sub[names].to_numpy(), sub["label"].to_numpy())
        p_val = clf.predict_proba(val[names].to_numpy())[:, 1]
        cal = fit_isotonic(p_val, val["label"].to_numpy())
        train_s = time.time() - t_start
        eval_mask = human_test_mask | ((y_test == 1) & (test_fam != fam))
        sub_test = test[eval_mask]
        y_eval = y_test[eval_mask]
        p_single = 1.0 - cal.predict(clf.predict_proba(sub_test[names].to_numpy())[:, 1])
        p_multi = sub_test["p_machine_cal"].to_numpy()
        gen = {
            "n_train_machine": int(len(mach)), "n_train_human": int(len(human_tr)),
            "n_eval": int(len(sub_test)),
            "n_eval_machine_other_families": int(((y_eval == 1)).sum()),
            "train_families_excluded_from_eval": fam,
            "train_seconds": round(train_s, 1),
            "single_generator": _met(y_eval, p_single),
            "multi_generator_locked": _met(y_eval, p_multi),
        }
        gen["gap_pr_auc_multi_minus_single"] = round(
            gen["multi_generator_locked"]["pr_auc"] - gen["single_generator"]["pr_auc"], 4)
        gen["gap_roc_multi_minus_single"] = round(
            gen["multi_generator_locked"]["roc_auc"] - gen["single_generator"]["roc_auc"], 4)
        gen["gap_recall_machine_multi_minus_single"] = round(
            gen["multi_generator_locked"]["recall_machine"]
            - gen["single_generator"]["recall_machine"], 4)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": clf, "calibrator": cal, "train_family": fam,
                     "params": tcfg, "trained_at": datetime.now(timezone.utc).isoformat()},
                    MODELS_DIR / f"single_gen_{fam.replace('/', '_')}.joblib")
        res["models"][fam] = gen
    res["note"] = ("experimental models: fitted on MAGE-train only, calibrated on "
                   "MAGE-validation; evaluated on the same held-out rows as the "
                   "locked model. NOT the baseline; baseline untouched.")
    res["test_only_family_tokens"] = test_only_families
    return res


def run_e1(cfg: dict, ctx: dict) -> dict:
    """E1: unseen generators - locked model + single-vs-multi generator gap."""
    t0 = time.time()
    c = cfg["e1"]
    bundle = ctx["bundle"]
    res: dict = {"name": "E1-unseen-generators", "status": "ok", "parts": {}}
    test = ctx["test"]
    train_tokens = _train_machine_tokens()
    res["train_machine_family_tokens"] = sorted(train_tokens)
    mage_test = test[test["source_dataset"] == "mage"]
    tfam = _family_of_src(mage_test)
    heldout = sorted(set(tfam) - train_tokens - {"human"})
    res["test_only_family_tokens"] = heldout

    def _met_sub(sub: pd.DataFrame, tag: str) -> dict:
        y = canonical_labels(sub)
        p = sub["p_machine_cal"].to_numpy()
        m = compute_metrics(y, p)
        cm = m.pop("confusion_matrix")
        return {"n": int(len(sub)), "tag": tag, **_round_metrics(m),
                "confusion_matrix": cm}

    for fam in c["heldout_generators"]:
        sub = mage_test[tfam == fam]
        res["parts"][fam] = _met_sub(sub, "heldout-generator-family") if len(sub) >= 5 else {
            "n": int(len(sub)), "note": "too few rows"}

    sub_all = mage_test[tfam.isin(set(heldout))]
    if len(sub_all) >= 5:
        res["test_only_rollup"] = _met_sub(sub_all, "all-test-only-families")
    else:
        res["test_only_rollup"] = {"n": int(len(sub_all)),
                                   "note": "no test-only generator families in MAGE test "
                                           "(train covers all test families); unseen-generator "
                                           "evidence comes from OOD/HC3/RAID parts below"}

    # per-family recall across ALL families (seen generators, reference)
    per_fam = {}
    for fam, g in mage_test[tfam != "human"].groupby(tfam[tfam != "human"]):
        y = canonical_labels(g)
        p = g["p_machine_cal"].to_numpy()
        per_fam[str(fam)] = {
            "n": int(len(g)),
            "recall_at_0.5": round(float((p >= 0.5).mean()), 4),
            "mean_p_machine": round(float(p.mean()), 4),
            "machine_share": round(float(y.mean()), 4),
        }
    res["seen_family_behavior"] = per_fam

    # --- MAGE OOD sets (unseen GPT-family generators, supplied by MAGE) ---
    import csv as _csv
    for f in c["ood_files"]:
        path = ROOT / f
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as fh:
            r = _csv.reader(fh)
            next(r)
            for row in r:
                rows.append({"text": row[0], "raw_label": int(row[1]), "src": row[2]})
        odf = pd.DataFrame(rows)
        odf["source_dataset"] = "mage"
        odf["y"] = 1 - odf["raw_label"].to_numpy()
        keep = odf["text"].map(normalize_text).notna()
        odf = odf[keep].copy()
        odf["text_norm"] = odf["text"].map(normalize_text)
        feats = extract_texts(odf["text_norm"].tolist(), bundle["state"],
                              workers=int(cfg.get("feature_workers", 0)))
        p_mc = 1.0 - bundle["calibrator"].predict(
            bundle["model"].predict_proba(feats[feature_names()].to_numpy())[:, 1])
        odf["p_machine_cal"] = p_mc
        m = compute_metrics(odf["y"].to_numpy(), p_mc)
        res[f"ood_{Path(f).stem}"] = {
            "n": int(len(odf)),
            "class_balance": {str(k): int(v) for k, v in odf["y"].value_counts().items()},
            **{k: v for k, v in _round_metrics(m).items() if k != "confusion_matrix"},
            "confusion_matrix": m["confusion_matrix"],
            "src_breakdown": {str(k): int(v) for k, v in odf["src"].value_counts().items()},
            "generator_note": "unseen GPT-family generators not present in MAGE training",
        }

    # --- HC3 (unseen ChatGPT generator + conversational domain) ---
    ev = hc3_frame(cfg, ctx)
    yh = ev["label"].to_numpy().astype(int)  # canonical already
    ph = ev["p_machine_cal"].to_numpy()
    mh = compute_metrics(yh, ph)
    res["hc3"] = {
        "n": int(len(ev)),
        "class_balance_canonical": {str(k): int(v) for k, v in pd.Series(yh).value_counts().items()},
        **{k: v for k, v in _round_metrics(mh).items() if k != "confusion_matrix"},
        "confusion_matrix": mh["confusion_matrix"],
        "score_map": "p(machine) = 1 - calibrated p_stored (corpus-agnostic model semantics)",
    }

    # --- RAID labelled subset: ALL 11 machine models are unseen generators ---
    if RAID_SUBSET.exists():
        raid = raid_scored(cfg, ctx)
        res["raid_labelled"] = _raid_unseen_generator_metrics(raid)
        res["raid_labelled"]["source"] = "raid_eval_subset.csv (RAID-train derived, eval-only)"

    # --- unlabeled RAID test.csv: raw-score distribution diagnostics ---
    raid_test = ROOT / "ml" / "data" / "raw" / "raid" / "test.csv"
    if raid_test.exists():
        rng = np.random.default_rng(int(cfg["seed"]))
        reservoir: list[tuple[str, str]] = []
        with raid_test.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            r = _csv.reader(fh)
            next(r)
            n = 0
            for row in r:
                n += 1
                if len(row) < 2:
                    continue
                if len(reservoir) < int(c["raid_sample_n"]):
                    reservoir.append((row[0], row[1]))
                else:
                    j = int(rng.integers(0, n))
                    if j < int(c["raid_sample_n"]):
                        reservoir[j] = (row[0], row[1])
        rtexts = [t for t in (normalize_text(s[1]) or "" for s in reservoir) if t]
        rfeats = extract_texts(rtexts, bundle["state"],
                               workers=int(cfg.get("feature_workers", 0)))
        rp_stored = bundle["model"].predict_proba(rfeats[feature_names()].to_numpy())[:, 1]
        rp_stored_cal = bundle["calibrator"].predict(rp_stored)
        p_machine = 1.0 - rp_stored_cal
        res["raid_distribution"] = {
            "n_rows_in_file": int(n), "n_sampled": int(len(rtexts)),
            "p_machine_mean": round(float(np.mean(p_machine)), 4),
            "p_machine_std": round(float(np.std(p_machine)), 4),
            "frac_p_machine_ge_0_5": round(float(np.mean(p_machine >= 0.5)), 4),
            "frac_p_machine_ge_0_9": round(float(np.mean(p_machine >= 0.9)), 4),
            "frac_p_machine_le_0_1": round(float(np.mean(p_machine <= 0.1)), 4),
            "note": "UNLABELED test.csv: distribution diagnostics only, no supervised claims; "
                    "p(machine) = 1 - calibrated stored score",
        }

    # --- single- vs multi-generator generalization gap (experimental models) ---
    res["single_vs_multi"] = _single_vs_multi_generator(cfg, ctx, test, heldout)

    res["duration_s"] = round(time.time() - t0, 1)
    return res


# --------------------------------------------------------------------------
# E2 - paraphrased machine text
# --------------------------------------------------------------------------

def _raid_paraphrase_analysis(raid: pd.DataFrame) -> dict:
    """E2 core: clean vs paraphrased machine text against the SAME human control."""
    pol = raid[raid["in_policy"]].copy()
    allr = raid.copy()
    out: dict = {"dropped_invalid_text": int(raid.attrs.get("dropped_invalid_text", 0))}

    def _task(frame: pd.DataFrame, pos_stratum: str) -> dict:
        pos = frame[(frame["y"] == 1) & (frame["stratum_key"] == pos_stratum)]
        # negatives = RAID clean human rows only (attack held unchanged)
        neg = frame[(frame["y"] == 0) & (frame["attack"] == "none")]
        if len(pos) < 30 or len(neg) < 30:
            return {"n_pos": int(len(pos)), "n_neg": int(len(neg)), "note": "too few rows"}
        y = np.concatenate([pos["y"].to_numpy(), neg["y"].to_numpy()])
        p = np.concatenate([pos["p_machine_cal"].to_numpy(), neg["p_machine_cal"].to_numpy()])
        m = compute_metrics(y, p)
        cm = m.pop("confusion_matrix")
        return {"n_pos": int(len(pos)), "n_neg": int(len(neg)),
                **_round_metrics(m), "confusion_matrix": cm}

    for frame, tag in ((pol, "policy_40_5000_words"), (allr, "all_lengths")):
        f = frame.copy()
        f["stratum_key"] = np.where(
            f["model"] == "human", "human_nonattack",
            np.where(f["attack"] == "none", "machine_clean",
                     np.where(f["attack"] == "paraphrase", "machine_paraphrase",
                              "machine_other_attack")))
        clean = _task(f, "machine_clean")
        para = _task(f, "machine_paraphrase")
        other = _task(f, "machine_other_attack")
        entry = {"machine_clean": clean, "machine_paraphrase": para,
                 "machine_other_attacks": other}
        if isinstance(clean, dict) and isinstance(para, dict) and "roc_auc" in clean and "roc_auc" in para:
            entry["delta_paraphrase_minus_clean"] = {
                k: round(para[k] - clean[k], 4)
                for k in ("roc_auc", "pr_auc", "recall_machine", "f1_macro",
                          "accuracy", "brier", "ece_10") if k in clean and k in para}
        # human control: clean vs paraphrased human text (same corpus, same scorer)
        hc = f[(f["model"] == "human") & (f["attack"] == "none")]
        hp = f[(f["model"] == "human") & (f["attack"] == "paraphrase")]
        entry["human_control"] = {
            "n_clean": int(len(hc)), "n_paraphrased": int(len(hp)),
            "clean_fpr_at_0_5": round(float((hc["p_machine_cal"] >= 0.5).mean()), 4) if len(hc) else None,
            "clean_mean_p_machine": round(float(hc["p_machine_cal"].mean()), 4) if len(hc) else None,
            "paraphrased_fpr_at_0_5": round(float((hp["p_machine_cal"] >= 0.5).mean()), 4) if len(hp) else None,
            "paraphrased_mean_p_machine": round(float(hp["p_machine_cal"].mean()), 4) if len(hp) else None,
        }
        out[tag] = entry

    # per-model recall: clean vs paraphrase (paired)
    f = pol.copy()
    per = {}
    for mname, g in f[f["y"] == 1].groupby("model"):
        cl = g[g["attack"] == "none"]
        pa = g[g["attack"] == "paraphrase"]
        if len(cl) >= 20 and len(pa) >= 20:
            per[str(mname)] = {
                "n_clean": int(len(cl)), "n_paraphrase": int(len(pa)),
                "recall_clean": round(float((cl["p_machine_cal"] >= 0.5).mean()), 4),
                "recall_paraphrase": round(float((pa["p_machine_cal"] >= 0.5).mean()), 4),
                "delta": round(float((pa["p_machine_cal"] >= 0.5).mean())
                               - float((cl["p_machine_cal"] >= 0.5).mean()), 4),
            }
    out["per_model_clean_vs_paraphrase"] = per
    return out


def _clip_sentence(text: str, max_chars: int) -> str:
    """Cut at max_chars, backing off to the last sentence boundary so the
    model is never asked to rewrite a text that ends mid-sentence."""
    t = str(text)[:max_chars]
    if len(str(text)) <= max_chars:
        return t
    m = max(t.rfind(". "), t.rfind("? "), t.rfind("! "), t.rfind("\n"))
    if m > max_chars // 2:
        return t[:m + 1]
    return t


def _qwen_rewrite(texts: list[str], user_tpl: str, c: dict, device,
                  greedy: bool) -> list[str]:
    """Local instruct-model rewrite via the model's native chat template.

    The chat template matters: feeding a raw completion prompt made Qwen2.5-
    0.5B ignore the instruction and continue the story instead of editing it
    (observed and corrected during Phase 2; the correction is logged).
    """
    import torch as _torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed as _set_seed
    _set_seed(int(c.get("seed", 42)))
    tok = AutoTokenizer.from_pretrained(c["model_id"])
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    mdl = AutoModelForCausalLM.from_pretrained(c["model_id"])
    mdl.to(device)
    mdl.eval()
    system = c.get("system", "")
    max_chars = int(c.get("max_input_chars", 1500))
    outs: list[str] = []
    with _torch.no_grad():
        for t in texts:
            user = user_tpl.format(text=_clip_sentence(t, max_chars))
            msgs = ([{"role": "system", "content": system}] if system else [])
            msgs.append({"role": "user", "content": user})
            full = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
            enc = tok(full, return_tensors="pt").to(device)
            in_len = int(enc["input_ids"].shape[1])
            budget = min(int(c.get("max_new_tokens_cap", 800)),
                         max(int(c["max_new_tokens"]), round(1.4 * in_len)))
            kw: dict = dict(max_new_tokens=budget, pad_token_id=tok.eos_token_id)
            if greedy:
                kw["do_sample"] = False
            else:
                kw.update(do_sample=True,
                          temperature=float(c.get("temperature", 0.3)),
                          top_p=float(c.get("top_p", 0.9)))
            gen = mdl.generate(**enc, **kw)
            outs.append(tok.decode(gen[0][in_len:],
                                   skip_special_tokens=True).strip())
    return outs


def run_e2(cfg: dict, ctx: dict) -> dict:
    """E2: paraphrased machine text - RAID attack subsets + local T5 rewrite."""
    import torch as _torch
    t0 = time.time()
    c = cfg["e2"]
    bundle = ctx["bundle"]
    res: dict = {"name": "E2-paraphrase", "status": "ok"}

    # --- Part A: RAID adversarial paraphrase subsets (documented protocol) ---
    if RAID_SUBSET.exists():
        raid = raid_scored(cfg, ctx)
        res["raid"] = _raid_paraphrase_analysis(raid)
        res["raid"]["method"] = {
            "data": "RAID-train derived stratified subset (eval-only)",
            "human_control": "RAID clean human rows held unchanged for both tasks",
        }

    # --- Part B: local instruct-model paraphrase of MAGE test samples
    #     (+ human control through the same paraphraser) ---
    device = _torch_device(cfg)
    test = ctx["test"]
    machine = test[(test["source_dataset"] == "mage") & (canonical_labels(test) == 1)]
    human = test[(test["source_dataset"] == "mage") & (canonical_labels(test) == 0)]
    rng = np.random.default_rng(int(cfg["seed"]) + 2)
    take_m = min(int(c["sample_n"]), len(machine))
    take_h = min(int(c.get("human_control_n", c["sample_n"])), len(human))
    samp_m = machine.iloc[rng.choice(len(machine), size=take_m, replace=False)].copy()
    samp_h = human.iloc[rng.choice(len(human), size=take_h, replace=False)].copy()
    para_m = _qwen_rewrite(samp_m["text_norm"].tolist(), c["prompt_template"],
                           c, device, greedy=bool(c.get("greedy", True)))
    para_h = _qwen_rewrite(samp_h["text_norm"].tolist(), c["prompt_template"],
                           c, device, greedy=bool(c.get("greedy", True)))
    fm = extract_texts(para_m, bundle["state"], workers=int(cfg.get("feature_workers", 0)))
    fh_ = extract_texts(para_h, bundle["state"], workers=int(cfg.get("feature_workers", 0)))
    names = feature_names()
    p_m = 1.0 - bundle["calibrator"].predict(
        bundle["model"].predict_proba(fm[names].to_numpy())[:, 1])
    p_h = 1.0 - bundle["calibrator"].predict(
        bundle["model"].predict_proba(fh_[names].to_numpy())[:, 1])
    pre_m = samp_m["p_machine_cal"].to_numpy()
    pre_h = samp_h["p_machine_cal"].to_numpy()

    def _exps(orig, para, pre, post, want, nmax=6):
        idx = [i for i in range(len(pre)) if want(float(pre[i]), float(post[i]))]
        idx.sort(key=lambda i: abs(float(post[i]) - float(pre[i])), reverse=True)
        return [{"original": str(orig[i])[:220], "paraphrase": str(para[i])[:220],
                 "p_pre": round(float(pre[i]), 4),
                 "p_post": round(float(post[i]), 4)} for i in idx[:nmax]]

    def _strength(origs, paras):
        import difflib
        sims = [difflib.SequenceMatcher(None, o[:600], q[:600]).ratio()
                for o, q in zip(origs, paras)]
        wp = [max(1, len(o.split())) for o in origs]
        wq = [len(q.split()) for q in paras]
        return {
            "mean_similarity_first_600_chars": round(float(np.mean(sims)), 4),
            "mean_words_pre": round(float(np.mean(wp)), 1),
            "mean_words_post": round(float(np.mean(wq)), 1),
            "mean_words_ratio_post_over_pre": round(float(np.mean(
                [b / a for a, b in zip(wp, wq)])), 4),
        }

    res["local_rewrite"] = {
        "model": c["model_id"],
        "prompting": "native chat template (system + user), instruction at the end "
                     "of a bounded input (<=1500 chars, sentence-boundary cut)",
        "model_config": {k: c.get(k) for k in
                         ("system", "prompt_template", "greedy", "max_new_tokens",
                          "max_new_tokens_cap", "max_input_chars", "sample_n",
                          "human_control_n")},
        "seed": int(c.get("seed", cfg["seed"])),
        "generation": ("greedy (do_sample=False)" if c.get("greedy", True) else
                       f"sampled (temperature={c.get('temperature')}, top_p={c.get('top_p')})")
                      + f"; new-token budget = max({c.get('max_new_tokens')}, "
                        f"1.4 x input tokens) capped at {c.get('max_new_tokens_cap')}",
        "device": str(device),
        "machine_control": {
            "n": int(len(samp_m)),
            "recall_pre": round(float((pre_m >= 0.5).mean()), 4),
            "recall_post": round(float((p_m >= 0.5).mean()), 4),
            "mean_p_pre": round(float(pre_m.mean()), 4),
            "mean_p_post": round(float(p_m.mean()), 4),
            "mean_delta_p": round(float((p_m - pre_m).mean()), 4),
            "frac_post_below_0_5": round(float((p_m < 0.5).mean()), 4),
        },
        "human_control": {
            "n": int(len(samp_h)),
            "fpr_pre": round(float((pre_h >= 0.5).mean()), 4),
            "fpr_post": round(float((p_h >= 0.5).mean()), 4),
            "mean_p_pre": round(float(pre_h.mean()), 4),
            "mean_p_post": round(float(p_h.mean()), 4),
            "mean_delta_p": round(float((p_h - pre_h).mean()), 4),
        },
        "edit_strength_machine": _strength(samp_m["text_norm"].tolist(), para_m),
        "edit_strength_human_control": _strength(samp_h["text_norm"].tolist(), para_h),
        "examples_machine_evasion": _exps(
            samp_m["text_norm"].tolist(), para_m, pre_m, p_m,
            lambda a, b: a >= 0.5 and b < 0.5),
        "examples_human_control_flips": _exps(
            samp_h["text_norm"].tolist(), para_h, pre_h, p_h,
            lambda a, b: a < 0.5 and b >= 0.5),
        "note": "all machine rows are TRUE machine origin; recall drop = evasion rate. "
                "Human control passes through the same paraphraser to check that "
                "paraphrasing alone does not flip human text.",
    }
    res["duration_s"] = round(time.time() - t0, 1)
    return res


# --------------------------------------------------------------------------
# E3 - human-edited machine text
# --------------------------------------------------------------------------

def _torch_device(cfg: dict):
    import os as _os
    import torch
    forced = (cfg.get("device") or "").lower()
    torch.set_num_threads(max(1, (_os.cpu_count() or 4) - 2))
    if forced == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def run_e3(cfg: dict, ctx: dict) -> dict:
    """E3: human-edited machine text - local instruct LLM rewrite.

    Both documented label views are reported:
      * machine-origin (ground truth = authorship lineage) -> y=1 always;
      * predominant authorship -> under LLM editing the predominant author is
        the editing model, so this view is ALSO y=1; the two views coincide
        here by construction (the editor is a machine, not a human) and this
        limitation is stated in the report rather than hidden.
    """
    t0 = time.time()
    c = cfg["e3"]
    bundle = ctx["bundle"]
    res: dict = {"name": "E3-human-edits", "status": "ok"}
    test = ctx["test"]
    machine = test[(test["source_dataset"] == "mage") & (canonical_labels(test) == 1)]
    rng = np.random.default_rng(int(cfg["seed"]) + 3)
    take = min(int(c["sample_n"]), len(machine))
    samp = machine.iloc[rng.choice(len(machine), size=take, replace=False)].copy()
    device = _torch_device(cfg)
    outs = _qwen_rewrite(samp["text_norm"].tolist(), c["prompt_template"],
                         c, device, greedy=False)
    samp = samp.copy()
    samp["text_humanized"] = [o if o else t for o, t in zip(outs, samp["text_norm"])]
    usable = [t for t in samp["text_humanized"] if t and t.strip()]
    feats = extract_texts(samp["text_humanized"].tolist(), bundle["state"],
                          workers=int(cfg.get("feature_workers", 0)))
    names = feature_names()
    p_post = 1.0 - bundle["calibrator"].predict(
        bundle["model"].predict_proba(feats[names].to_numpy())[:, 1])
    pre = samp["p_machine_cal"].to_numpy()
    y = np.ones(len(samp), dtype=int)
    m_pre = compute_metrics(y, pre)
    m_post = compute_metrics(y, p_post)
    res.update({
        "model": c["model_id"],
        "model_source": "local, free, CPU/GPU via transformers; no paid API",
        "prompting": "native chat template (system + user); inputs cut at a "
                     "sentence boundary <=1500 chars; decode sliced by token count",
        "device": str(device),
        "n": int(len(samp)),
        "n_nonempty_edits": int(len(usable)),
        "prompt_template": c["prompt_template"],
        "temperature": float(c["temperature"]), "top_p": float(c["top_p"]),
        "max_new_tokens": int(c["max_new_tokens"]),
        "seed": int(cfg["seed"]),
        "pre_recall_machine": round(float(m_pre["recall_machine"]), 4),
        "post_recall_machine": round(float(m_post["recall_machine"]), 4),
        "mean_p_pre": round(float(pre.mean()), 4),
        "mean_p_post": round(float(p_post.mean()), 4),
        "mean_delta_p": round(float((p_post - pre).mean()), 4),
        "frac_post_below_0_5": round(float((p_post < 0.5).mean()), 4),
        "post": {k: v for k, v in _round_metrics(m_post).items() if k != "confusion_matrix"},
        "confusion_post": m_post["confusion_matrix"],
        "label_views": {
            "machine_origin": {
                "y": 1, "recall": round(float(m_post["recall_machine"]), 4),
                "meaning": "ground truth = authorship lineage (originally machine-written)",
            },
            "predominant_authorship": {
                "y": 1, "recall": round(float(m_post["recall_machine"]), 4),
                "meaning": "under LLM editing the predominant author of the final "
                           "text is the editing model, so this view is ALSO machine; "
                           "the views coincide by construction here (no human editors "
                           "were available for this study) - stated, not hidden",
            },
        },
        "edit_examples": [
            {"original": str(t)[:260], "edited": str(e)[:260],
             "p_pre": round(float(pre[i]), 4),
             "p_post": round(float(p_post[i]), 4)}
            for i, (t, e) in list(enumerate(
                zip(samp["text_norm"].tolist(), samp["text_humanized"].tolist())))
            if i < 6
        ],
        "largest_drops": [
            {"original": str(samp["text_norm"].iloc[i])[:260],
             "edited": str(samp["text_humanized"].iloc[i])[:260],
             "p_pre": round(float(pre[i]), 4),
             "p_post": round(float(p_post[i]), 4)}
            for i in np.argsort(-(pre - p_post))[:4]
        ],
        "note": "ground truth stays MACHINE (authorship, not final form); "
                "recall drop = humanization gain",
    })
    res["duration_s"] = round(time.time() - t0, 1)
    return res


# --------------------------------------------------------------------------
# E4 - mixed authorship
# --------------------------------------------------------------------------

def run_e4(cfg: dict, ctx: dict) -> dict:
    """E4: mixed authorship - synthetic docs with 25/50/75% machine paragraphs.

    NOT a measurement of exact human/AI percentages; a behavioral probe of the
    document-level model's response to mixed evidence. Documents are built
    WITHIN a single MAGE domain so domain cannot confound the mixture signal.
    Also reports the first segment-level benchmark: paragraph-level predictions
    against ground truth known by construction.
    """
    t0 = time.time()
    c = cfg["e4"]
    bundle = ctx["bundle"]
    res: dict = {"name": "E4-mixed-authorship", "status": "ok", "mixtures": {}}
    test = ctx["test"]
    mage = test[test["source_dataset"] == "mage"].copy()
    mage["y_canon"] = canonical_labels(mage)
    rng = np.random.default_rng(int(cfg["seed"]) + 4)
    npara = int(c["paragraphs_per_doc"])
    names = feature_names()
    all_paras: list[dict] = []
    doc_rows: list[dict] = []

    for ratio in c["ratios"]:
        n_machine = max(1, min(npara - 1, int(round(ratio * npara))))
        n_human = npara - n_machine
        domains = sorted(mage["domain"].astype(str).unique())
        docs: list[str] = []
        ratio_rows: list[dict] = []
        for i in range(int(c["docs_per_ratio"])):
            # choose a domain where both pools suffice (same-domain documents)
            eligible = [d for d in domains
                        if ((mage["domain"] == d) & (mage["y_canon"] == 1)).sum() >= n_machine
                        and ((mage["domain"] == d) & (mage["y_canon"] == 0)).sum() >= n_human]
            if not eligible:
                raise RuntimeError(f"no domain supports mixture ratio {ratio}")
            d = str(rng.choice(eligible))
            mpool = mage[(mage["domain"] == d) & (mage["y_canon"] == 1)]
            hpool = mage[(mage["domain"] == d) & (mage["y_canon"] == 0)]
            mi = rng.choice(len(mpool), size=n_machine, replace=False)
            hi = rng.choice(len(hpool), size=n_human, replace=False)
            chosen = [(mpool.iloc[i], 1) for i in mi] + [(hpool.iloc[i], 0) for i in hi]
            rng.shuffle(chosen)
            docs.append("\n\n".join(r["text_norm"] for r, _ in chosen))
            ratio_rows.append({"ratio": ratio, "domain": d, "doc_idx": i})
            for j, (r, y_p) in enumerate(chosen):
                all_paras.append({"ratio": ratio, "doc_idx": i, "para_idx": j,
                                  "y": int(y_p), "text": r["text_norm"],
                                  "row_id": int(r["row_id"])})
        feats = extract_texts(docs, bundle["state"],
                              workers=int(cfg.get("feature_workers", 0)))
        p_mc = 1.0 - bundle["calibrator"].predict(
            bundle["model"].predict_proba(feats[names].to_numpy())[:, 1])
        for r, pval in zip(ratio_rows, p_mc):
            r["p_machine"] = float(pval)
        doc_rows.extend(ratio_rows)
        res["mixtures"][str(ratio)] = {
            "n_docs": len(docs), "machine_paragraphs": n_machine,
            "human_paragraphs": n_human,
            "p_machine_mean": round(float(np.mean(p_mc)), 4),
            "p_machine_std": round(float(np.std(p_mc)), 4),
            "p_machine_min": round(float(np.min(p_mc)), 4),
            "p_machine_max": round(float(np.max(p_mc)), 4),
            "frac_ge_0_5": round(float(np.mean(p_mc >= 0.5)), 4),
            "domains_used": sorted({r["domain"] for r in ratio_rows}),
        }

    ratios = list(c["ratios"])
    means = [res["mixtures"][str(r)]["p_machine_mean"] for r in ratios]
    res["monotonic"] = bool(all(b >= a for a, b in zip(means, means[1:])))
    # Spearman rank correlation over all documents
    from scipy.stats import spearmanr
    rs = np.array([r["ratio"] for r in doc_rows], dtype=float)
    ps = np.array([r["p_machine"] for r in doc_rows], dtype=float)
    rho, pval = spearmanr(rs, ps)
    res["spearman_doc_level"] = {
        "rho": round(float(rho), 4), "p_value": float(f"{pval:.3e}"),
        "n_docs": int(len(doc_rows)),
        "interpretation": "rank correlation between machine share and document p(machine); "
                          "failure to be monotonic is itself a finding",
    }
    # --- segment-level benchmark (paragraph ground truth known by construction) ---
    pfeats = extract_texts([p["text"] for p in all_paras], bundle["state"],
                           workers=int(cfg.get("feature_workers", 0)))
    p_scores = 1.0 - bundle["calibrator"].predict(
        bundle["model"].predict_proba(pfeats[names].to_numpy())[:, 1])
    y_para = np.array([p["y"] for p in all_paras])
    seg = compute_metrics(y_para, p_scores)
    seg_cm = seg.pop("confusion_matrix")
    res["segment_benchmark"] = {
        "n_paragraphs": int(len(all_paras)),
        "class_balance": {str(k): int(v) for k, v in pd.Series(y_para).value_counts().items()},
        **_round_metrics(seg), "confusion_matrix": seg_cm,
        "mean_p_machine_given_machine_para": round(float(p_scores[y_para == 1].mean()), 4),
        "mean_p_machine_given_human_para": round(float(p_scores[y_para == 0].mean()), 4),
        "note": "paragraph-level predictions on E4 documents; ground truth known by "
                "construction - an honest first segment-analysis benchmark (no segment "
                "model trained; locked document-level model applied to paragraphs)",
    }
    res["disclaimer"] = ("Controlled synthetic mixtures. The detector is NOT claimed to "
                         "measure exact AI authorship percentages; this probes whether "
                         "document-level p(machine) responds monotonically to evidence.")
    res["duration_s"] = round(time.time() - t0, 1)
    return res


# --------------------------------------------------------------------------
# E5 - domain transfer + hswag forensics + cross-corpus
# --------------------------------------------------------------------------

def _hswag_forensics(ctx: dict, test: pd.DataFrame) -> dict:
    hs = test[test["domain"].astype(str) == "hswag"]
    yh = canonical_labels(hs)
    ph = hs["p_machine_cal"].to_numpy()
    out: dict = {"n": int(len(hs)),
                 "class_balance_canonical": {str(k): int(v) for k, v in
                                             pd.Series(yh).value_counts().items()},
                 "positive_rate_machine": round(float(yh.mean()), 4)}
    if len(np.unique(yh)) > 1:
        mh = compute_metrics(yh, ph)
        cm = mh.pop("confusion_matrix")
        out["metrics_canonical"] = _round_metrics(mh)
        out["confusion_matrix"] = cm
    # reproduce the Phase-1 stored-space numbers (positive class = stored-1 = HUMAN)
    from sklearn.metrics import average_precision_score, f1_score
    y_stored = hs["label"].to_numpy()
    p_stored = hs["p_stored_cal"].to_numpy()
    out["stored_space_reproduction"] = {
        "positive_class": "stored-1 == HUMAN (Phase-1 header said 'machine')",
        "human_share": round(float(y_stored.mean()), 4),
        "pr_auc_stored_calibrated": round(float(average_precision_score(y_stored, p_stored)), 4),
        "pr_auc_stored_raw": round(float(average_precision_score(y_stored, hs["p_stored"].to_numpy())), 4),
        "f1_stored_macro": round(float(f1_score(y_stored, (p_stored >= 0.5).astype(int),
                                                average="macro", zero_division=0)), 4),
        "note": "Phase-1 reported hswag PR-AUC=0.21 / F1=0.60 in this stored space; "
                "reproduced here with corrected orientation labels",
    }
    out["src_breakdown"] = {str(k): int(v) for k, v in hs["src"].value_counts().items()}
    # length by class inside hswag
    for name, mask in (("machine", yh == 1), ("human", yh == 0)):
        if mask.sum():
            w = hs.loc[mask, "words"].to_numpy()
            out[f"words_{name}"] = {"mean": round(float(w.mean()), 1),
                                    "median": round(float(np.median(w)), 1),
                                    "min": int(w.min()), "max": int(w.max()),
                                    "n": int(mask.sum())}
    # feature-level evidence within hswag
    names = feature_names()
    fh = hs[names]
    g1 = fh[yh == 1].mean(numeric_only=True)
    g0 = fh[yh == 0].mean(numeric_only=True)
    diffs = (g1 - g0).dropna().sort_values(key=abs, ascending=False)
    out["top_feature_diffs_machine_minus_human"] = {str(k): round(float(v), 4)
                                                    for k, v in diffs.head(12).items()}
    # effect-size comparison: same features on the REST of the test split
    rest = test[(test["domain"].astype(str) != "hswag") & (test["source_dataset"] == "mage")]
    yr = canonical_labels(rest)
    fr = rest[names]
    r1 = fr[yr == 1].mean(numeric_only=True)
    r0 = fr[yr == 0].mean(numeric_only=True)
    d_rest = (r1 - r0).dropna()
    rows = []
    for feat in diffs.head(12).index:
        rows.append({"feature": feat,
                     "hswag_diff": round(float(diffs[feat]), 4),
                     "rest_of_test_diff": round(float(d_rest.get(feat, float("nan"))), 4)})
    out["top_diffs_vs_rest_of_test"] = rows
    # per-family machine behavior inside hswag
    fam_beh = {}
    mhs = hs[yh == 1]
    for fam, g in mhs.groupby(_family_of_src(mhs)):
        fam_beh[str(fam)] = {"n": int(len(g)),
                             "recall_at_0_5": round(float((g["p_machine_cal"] >= 0.5).mean()), 4),
                             "mean_p_machine": round(float(g["p_machine_cal"].mean()), 4)}
    out["machine_family_behavior"] = fam_beh
    hum = hs[yh == 0]
    if len(hum):
        out["human_behavior"] = {
            "n": int(len(hum)),
            "mean_p_machine": round(float(hum["p_machine_cal"].mean()), 4),
            "frac_ge_0_5": round(float((hum["p_machine_cal"] >= 0.5).mean()), 4),
            "frac_le_0_3": round(float((hum["p_machine_cal"] <= 0.3).mean()), 4),
        }
    # domain contrast: how hswag differs from other domains in feature space
    contra = []
    for feat in diffs.head(12).index:
        contra.append({"feature": feat,
                       "hswag_machine_minus_human": round(float(diffs[feat]), 4),
                       "other_domains_machine_minus_human": round(float(d_rest.get(feat, float("nan"))), 4)})
    out["domain_contrast"] = contra
    return out


def _lodo(cfg: dict, ctx: dict) -> dict:
    """Leave-one-domain-out: train an experimental model excluding domain d,
    evaluate on domain d. Never touches HC3/RAID. Feature document-frequency
    statistics are REFIT excluding domain d (no held-out-domain leakage)."""
    c = cfg["e5"].get("lodo", {})
    if not c.get("enabled", False):
        return {"enabled": False, "note": "disabled in config"}
    tcfg = cfg["experimental_training"]
    rare_frac = float(cfg.get("rare_frac", 0.001))
    joined = meta_join(("train", "val"))
    train, val = joined["train"], joined["val"]
    test = ctx["test"]
    names = feature_names()
    domains = sorted(test[test["source_dataset"] == "mage"]["domain"].astype(str).unique())
    out: dict = {"enabled": True, "protocol": {
        "train": "MAGE-train rows of all domains except d (stored labels)",
        "feature_state": ("document-frequency stats refit on train-minus-d text "
                          "(rare_token_ratio/mean_log_df recomputed; other 71 "
                          "features are state-independent)"),
        "calibration": "isotonic fit on MAGE-val rows of domains != d",
        "eval": "MAGE-test rows of domain d only",
        "params": tcfg,
        "note": "experimental models - reported separately from the locked baseline",
    }, "folds": {}}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for d in domains:
        t_start = time.time()
        tr = train[train["domain"].astype(str) != d]
        va = val[val["domain"].astype(str) != d]
        te = test[(test["domain"].astype(str) == d) & (test["source_dataset"] == "mage")]
        if len(te) < 20:
            out["folds"][d] = {"n_test": int(len(te)), "note": "too few test rows"}
            continue
        state_d = FeatureState.fit(tr["text_norm"].tolist(), rare_frac=rare_frac)
        Xtr = replace_state_features(tr[names], tr["text_norm"].tolist(), state_d)
        Xva = replace_state_features(va[names], va["text_norm"].tolist(), state_d)
        Xte = replace_state_features(te[names], te["text_norm"].tolist(), state_d)
        clf = hgb_from_cfg(tcfg)
        clf.fit(Xtr.to_numpy(), tr["label"].to_numpy())
        p_val = clf.predict_proba(Xva.to_numpy())[:, 1]
        cal = fit_isotonic(p_val, va["label"].to_numpy())
        p_stored_cal = cal.predict(clf.predict_proba(Xte.to_numpy())[:, 1])
        p_machine = 1.0 - p_stored_cal
        y = canonical_labels(te)
        m = compute_metrics(y, p_machine)
        cm = m.pop("confusion_matrix")
        # locked baseline on the same rows (reference; frozen full-train state)
        p_locked = te["p_machine_cal"].to_numpy()
        ml = compute_metrics(y, p_locked)
        cml = ml.pop("confusion_matrix")
        fold = {
            "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
            "machine_share": round(float(y.mean()), 4),
            "lodo_model": {**_round_metrics(m), "confusion_matrix": cm},
            "locked_baseline_same_rows": {**_round_metrics(ml), "confusion_matrix": cml},
            "delta_pr_auc_lodo_minus_locked": round(float(m["pr_auc"] - ml["pr_auc"]), 4),
            "duration_s": round(time.time() - t_start, 1),
        }
        if c.get("save_models", False):
            joblib.dump({"model": clf, "calibrator": cal, "held_out_domain": d,
                         "params": tcfg, "feature_state": state_d,
                         "trained_at": datetime.now(timezone.utc).isoformat()},
                        MODELS_DIR / f"lodo_{d}.joblib")
            fold["model_saved"] = f"ml/experiments/phase2/models/lodo_{d}.joblib"
        out["folds"][d] = fold
    # summary
    prs = [f["lodo_model"]["pr_auc"] for f in out["folds"].values()
           if isinstance(f, dict) and "lodo_model" in f]
    prl = [f["locked_baseline_same_rows"]["pr_auc"] for f in out["folds"].values()
           if isinstance(f, dict) and "locked_baseline_same_rows" in f]
    if prs:
        out["summary"] = {
            "n_folds": len(prs),
            "mean_pr_auc_lodo": round(float(np.mean(prs)), 4),
            "min_pr_auc_lodo": round(float(np.min(prs)), 4),
            "mean_pr_auc_locked": round(float(np.mean(prl)), 4),
            "worst_domains": sorted(
                [d for d, f in out["folds"].items() if "lodo_model" in f],
                key=lambda d: out["folds"][d]["lodo_model"]["pr_auc"])[:3],
        }
    return out


def _raid_cross_corpus(raid: pd.DataFrame) -> dict:
    """Cross-corpus view: MAGE-trained locked model on RAID domains
    (different generators AND domains simultaneously)."""
    pol = raid[raid["in_policy"]]
    out: dict = {"note": "cross-corpus supervised view (locked MAGE-trained model)",
                 "policy": "40-5000 words (Phase-1 inclusion range)", "per_domain": {}}
    human = pol[pol["y"] == 0]
    machine = pol[pol["y"] == 1]
    if len(human) and len(machine):
        y = np.concatenate([machine["y"].to_numpy(), human["y"].to_numpy()])
        p = np.concatenate([machine["p_machine_cal"].to_numpy(), human["p_machine_cal"].to_numpy()])
        m = compute_metrics(y, p)
        cm = m.pop("confusion_matrix")
        out["overall"] = {**_round_metrics(m), "confusion_matrix": cm}
    for d, g in pol.groupby("domain"):
        gm, gh = g[g["y"] == 1], g[g["y"] == 0]
        if len(gm) < 20 or len(gh) < 20:
            out["per_domain"][str(d)] = {"n_machine": int(len(gm)), "n_human": int(len(gh)),
                                         "note": "too few rows"}
            continue
        y = np.concatenate([gm["y"].to_numpy(), gh["y"].to_numpy()])
        p = np.concatenate([gm["p_machine_cal"].to_numpy(), gh["p_machine_cal"].to_numpy()])
        m = compute_metrics(y, p)
        cm = m.pop("confusion_matrix")
        out["per_domain"][str(d)] = {**_round_metrics(m), "confusion_matrix": cm,
                                     "n_machine": int(len(gm)), "n_human": int(len(gh))}
    return out


def run_e5(cfg: dict, ctx: dict) -> dict:
    """E5: domain transfer (LODO) + per-domain matrix + hswag forensics + cross-corpus."""
    t0 = time.time()
    bundle = ctx["bundle"]
    res: dict = {"name": "E5-domains-hswag", "status": "ok", "domains": {}, "hswag": {}}
    test = ctx["test"]
    mage = test[test["source_dataset"] == "mage"]
    domains = sorted(mage["domain"].astype(str).unique().tolist())
    for d in domains:
        sub = mage[mage["domain"].astype(str) == d]
        if len(sub) < 20:
            res["domains"][d] = {"n": int(len(sub)), "note": "too few"}
            continue
        y = canonical_labels(sub)
        p = sub["p_machine_cal"].to_numpy()
        m = compute_metrics(y, p)
        cm = m.pop("confusion_matrix")
        res["domains"][d] = {"n": int(len(sub)),
                             "machine_share": round(float(y.mean()), 4),
                             "words_mean": round(float(sub["words"].mean()), 1),
                             **_round_metrics(m), "confusion_matrix": cm}
    res["hswag"] = _hswag_forensics(ctx, test)
    res["lodo"] = _lodo(cfg, ctx)
    if RAID_SUBSET.exists():
        raid = raid_scored(cfg, ctx)
        res["raid_cross_corpus"] = _raid_cross_corpus(raid)
    res["duration_s"] = round(time.time() - t0, 1)
    return res


# --------------------------------------------------------------------------
# E6 - length sensitivity + evidence gate
# --------------------------------------------------------------------------

def run_e6(cfg: dict, ctx: dict) -> dict:
    """E6: confidence/evidence gate - empirical min-length investigation.

    Reports the plan's canonical buckets, finer buckets, and a gate table so
    the min-evidence threshold can be READ from the data instead of being
    hand-tuned. No threshold is changed anywhere in the pipeline.
    """
    t0 = time.time()
    res: dict = {"name": "E6-length-sensitivity", "status": "ok",
                 "buckets": {}, "plan_buckets": {}, "gate_table": []}
    test = ctx["test"]
    y = canonical_labels(test)
    p = test["p_machine_cal"].to_numpy()
    words = test["words"].to_numpy()

    def _bucket(mask, name, store=None):
        store = res["buckets"] if store is None else store
        if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
            store[name] = {"n": int(mask.sum()), "note": "too few / single-class"}
            return
        m = compute_metrics(y[mask], p[mask])
        conf = np.abs(p[mask] - 0.5)
        store[name] = {
            "n": int(mask.sum()),
            "machine_share": round(float(y[mask].mean()), 4),
            **{k: v for k, v in _round_metrics(m).items() if k != "confusion_matrix"},
            "mean_abs_confidence": round(float(conf.mean()), 4),
            "frac_uncertain_0_4_0_6": round(float(((p[mask] >= 0.4) & (p[mask] <= 0.6)).mean()), 4),
        }

    for (lo, hi), name in zip(LENGTH_BUCKETS, BUCKET_NAMES):
        _bucket((words >= lo) & (words < hi), name, store=res["plan_buckets"])
    edges = [0, 25, 50, 75, 100, 150, 200, 300, 400, 10 ** 9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        name = f"{lo}-{hi if hi < 10 ** 9 else 'inf'}w"
        _bucket((words >= lo) & (words < hi), name)

    # confidence/evidence gate candidates - descriptive, never applied here
    for g in cfg["e6"]["gate_candidates"]:
        mask = words >= g
        if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
            res["gate_table"].append({"min_words": int(g), "n_kept": int(mask.sum()),
                                      "note": "too few / single-class"})
            continue
        m = compute_metrics(y[mask], p[mask])
        res["gate_table"].append({
            "min_words": int(g),
            "n_kept": int(mask.sum()),
            "n_abstained": int((~mask).sum()),
            "frac_abstained": round(float((~mask).mean()), 4),
            "roc_auc": round(float(m["roc_auc"]), 4),
            "pr_auc": round(float(m["pr_auc"]), 4),
            "ece_10": round(float(m["ece_10"]), 4),
            "brier": round(float(m["brier"]), 4),
            "f1_macro": round(float(m["f1_macro"]), 4),
            "accuracy": round(float(m["accuracy"]), 4),
        })
    # behavior of the shortest bucket: should it be "insufficient evidence"?
    short = words < 50
    res["short_text_behavior"] = {
        "n_under_50w": int(short.sum()),
        "machine_share": round(float(y[short].mean()), 4) if short.sum() else None,
        "mean_abs_confidence": round(float(np.abs(p[short] - 0.5).mean()), 4) if short.sum() else None,
        "frac_uncertain_0_4_0_6": round(float(((p[short] >= 0.4) & (p[short] <= 0.6)).mean()), 4) if short.sum() else None,
        "frac_extreme_ge_0_9_or_le_0_1": round(float(((p[short] >= 0.9) | (p[short] <= 0.1)).mean()), 4) if short.sum() else None,
        "note": "the gate RECOMMENDATION is derived in the report from these tables; "
                "no threshold was tuned against them",
    }
    res["duration_s"] = round(time.time() - t0, 1)
    return res


# --------------------------------------------------------------------------
# E7 - temporal / generator drift + reproducibility
# --------------------------------------------------------------------------

_DRIFT_FEATURES = [
    "hedge_rate", "boilerplate_rate", "transition_rate", "first_person_rate",
    "second_person_rate", "contraction_rate", "pron_i", "pron_we", "ttr",
    "hapax_ratio", "avg_sent_len", "cv_sent_len", "flesch_reading_ease",
    "clause_density", "comma_rate", "exclaim_rate_per_sent", "qmark_rate",
    "rare_token_ratio", "mean_log_df", "stopword_ratio",
]


def _drift_block(a: pd.DataFrame, b: pd.DataFrame, names: list[str],
                 label_a: str, label_b: str) -> dict:
    """Feature-space drift between two corpora (matched by class)."""
    rows = []
    for f in names:
        va = pd.to_numeric(a[f], errors="coerce").to_numpy()
        vb = pd.to_numeric(b[f], errors="coerce").to_numpy()
        d = cohen_d(va, vb)
        ma, mb = float(np.nanmean(va)), float(np.nanmean(vb))
        rows.append({"feature": f, f"mean_{label_a}": round(ma, 4),
                     f"mean_{label_b}": round(mb, 4),
                     "cohen_d": round(d, 4) if d == d else None})
    rows_sorted = sorted([r for r in rows if r["cohen_d"] is not None],
                         key=lambda r: abs(r["cohen_d"]), reverse=True)
    lex = [r for r in rows if r["feature"] in _DRIFT_FEATURES]
    return {"n_a": int(len(a)), "n_b": int(len(b)),
            "top_shifted_features": rows_sorted[:15],
            "lexicon_and_stylistic_rates": lex,
            "positive_d_means": f"positive d = {label_a} higher than {label_b}"}


def run_e7(cfg: dict, ctx: dict) -> dict:
    """E7: temporal/generator drift (HC3 = Jan-2023 ChatGPT time shift).

    Compares the time-shifted, unseen-generator HC3 corpus against MAGE in
    performance AND in feature space (hedge/boilerplate/lexicon rates), and
    documents what drift looks like as part of the limitations narrative.
    Also re-verifies pipeline reproducibility (bit-identical scoring).
    """
    t0 = time.time()
    bundle = ctx["bundle"]
    res: dict = {"name": "E7-temporal-drift", "status": "ok"}
    names = feature_names()
    test = ctx["test"]
    mage = test[test["source_dataset"] == "mage"]
    y_mage = canonical_labels(mage)
    perf: dict = {}

    # reference: in-domain MAGE test
    perf["mage_test_in_domain"] = _met(y_mage, mage["p_machine_cal"].to_numpy())

    # HC3: time-shifted ChatGPT (Jan-2023), unseen generator + conversational domain
    ev = hc3_frame(cfg, ctx)
    hc3_feats = hc3_features(cfg, bundle["state"],
                             workers=int(cfg.get("score_workers", 0)))
    ev = pd.concat([ev.reset_index(drop=True),
                    hc3_feats.drop(columns=["row_id"], errors="ignore").reset_index(drop=True)],
                   axis=1)
    y_hc3 = ev["label"].to_numpy().astype(int)
    perf["hc3_overall"] = _met(y_hc3, ev["p_machine_cal"].to_numpy())
    hc3_by_domain = {}
    for d, g in ev.groupby("domain"):
        if len(g) < 50:
            hc3_by_domain[str(d)] = {"n": int(len(g)), "note": "too few rows"}
            continue
        m = _met(g["label"].to_numpy(), g["p_machine_cal"].to_numpy())
        hc3_by_domain[str(d)] = m
    perf["hc3_by_domain"] = hc3_by_domain

    # newer-generator slices: MAGE OOD GPT set vs MAGE gpt-3.5 test rows
    import csv as _csv
    for f in cfg["e1"]["ood_files"]:
        path = ROOT / f
        rows = []
        with path.open("r", encoding="utf-8") as fh:
            r = _csv.reader(fh)
            next(r)
            for row in r:
                rows.append({"text": row[0], "raw_label": int(row[1])})
        odf = pd.DataFrame(rows)
        odf["y"] = 1 - odf["raw_label"].to_numpy()
        odf["text_norm"] = odf["text"].map(normalize_text)
        odf = odf[odf["text_norm"].notna()]
        feats = extract_texts(odf["text_norm"].tolist(), bundle["state"],
                              workers=int(cfg.get("feature_workers", 0)))
        p_mc = 1.0 - bundle["calibrator"].predict(
            bundle["model"].predict_proba(feats[names].to_numpy())[:, 1])
        perf[Path(f).stem] = _met(odf["y"].to_numpy(), p_mc)

    res["performance_by_corpus"] = perf

    # --- feature-space drift: class-matched corpus contrasts ---
    mage_mach = mage[y_mage == 1]
    mage_hum = mage[y_mage == 0]
    hc3_mach = ev[y_hc3 == 1]
    hc3_hum = ev[y_hc3 == 0]
    res["feature_drift_machine_hc3_vs_mage"] = _drift_block(
        hc3_mach, mage_mach, names, "hc3_machine", "mage_machine")
    res["feature_drift_human_hc3_vs_mage"] = _drift_block(
        hc3_hum, mage_hum, names, "hc3_human", "mage_human")

    # explicit age-sensitive lexicon narrative numbers
    def _rate(df, feat):
        return round(float(pd.to_numeric(df[feat], errors="coerce").mean()), 4) if len(df) else None
    res["lexicon_rates"] = {
        f: {"mage_human": _rate(mage_hum, f), "mage_machine": _rate(mage_mach, f),
            "hc3_human": _rate(hc3_hum, f), "hc3_machine": _rate(hc3_mach, f)}
        for f in ("hedge_rate", "boilerplate_rate", "transition_rate",
                  "first_person_rate", "contraction_rate")
    }

    # reproducibility re-verification (moved here from the original E7 stub)
    X = mage[names].to_numpy()
    p1 = bundle["model"].predict_proba(X)[:, 1]
    p2 = bundle["model"].predict_proba(X)[:, 1]
    res["reproducibility"] = {
        "score_bit_identical": bool(np.array_equal(p1, p2, equal_nan=True)),
        "config_sha256": sha_of(cfg),
        "feature_names_sha256": sha_of(names),
    }
    res["duration_s"] = round(time.time() - t0, 1)
    return res


EXPERIMENTS = {
    "E0": run_e0, "E1": run_e1, "E2": run_e2, "E3": run_e3,
    "E4": run_e4, "E5": run_e5, "E6": run_e6, "E7": run_e7,
}


# --------------------------------------------------------------------------
# qualitative error examples with standardized-feature attributions
# --------------------------------------------------------------------------

_REF_STATS: dict[str, tuple[float, float]] | None = None


def _ref_stats(names: list[str]) -> dict[str, tuple[float, float]]:
    """MAGE-train feature mean/std, used for z-score attributions (cached)."""
    global _REF_STATS
    if _REF_STATS is None:
        feats = pd.read_parquet(PROCESSED / "features_train.parquet", columns=names)
        stats: dict[str, tuple[float, float]] = {}
        for n in names:
            x = pd.to_numeric(feats[n], errors="coerce").to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            sd = float(x.std())
            stats[n] = (float(x.mean()), sd if sd > 1e-12 else 1.0)
        _REF_STATS = stats
    return _REF_STATS


def _top_deviations(feat_row, stats, k: int = 4) -> list[dict]:
    items = []
    for n, (mu, sd) in stats.items():
        try:
            v = float(feat_row[n])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        items.append({"feature": n, "value": round(v, 4),
                      "z": round((v - mu) / sd, 3)})
    items.sort(key=lambda d: -abs(d["z"]))
    return items[:k]


def _pick_examples(df: pd.DataFrame, feats: pd.DataFrame, y: np.ndarray,
                   p: np.ndarray, err_mask: np.ndarray, descending: bool,
                   stats, n: int = 6, extra_cols: tuple = ()) -> list[dict]:
    idx = np.where(err_mask)[0]
    if len(idx) == 0:
        return []
    idx = idx[np.argsort(-p[idx]) if descending else np.argsort(p[idx])]
    out = []
    for i in idx[:n]:
        d: dict = {
            "text": str(df["text_norm"].iloc[i])[:240],
            "p_machine_cal": round(float(p[i]), 4),
            "true": "machine" if int(y[i]) == 1 else "human",
            "top_feature_deviations": _top_deviations(feats.iloc[i], stats),
        }
        if "words" in df.columns:
            try:
                d["words"] = int(df["words"].iloc[i])
            except (TypeError, ValueError):
                pass
        for c in extra_cols:
            if c in df.columns:
                d[c] = str(df[c].iloc[i])
        out.append(d)
    return out


def error_examples_block(cfg: dict, ctx: dict, n: int = 6) -> dict:
    """Misclassified examples (5-10 per slice, plan requirement) with top
    standardized-feature deviations vs the MAGE-train distribution."""
    t0 = time.time()
    names = feature_names()
    stats = _ref_stats(names)
    bundle = ctx["bundle"]
    workers = int(cfg.get("score_workers", 0))
    out: dict = {
        "attribution_method": "per-row z-scores of the 73 v1 features against "
                              "MAGE-train mean/std; the top-|z| features are "
                              "shown (standardized deviation, NOT SHAP)",
        "threshold": 0.5,
        "slice_policy": "false positives of humans and false negatives of "
                        "machine text, sorted by confidence in the wrong direction",
        "slices": {},
    }
    sl = out["slices"]

    # --- MAGE locked test ---
    test = bundle["test"]
    y = canonical_labels(test)
    p = test["p_machine_cal"].to_numpy()
    sl["mage_test_human_fp"] = _pick_examples(
        test, test, y, p, (y == 0) & (p >= 0.5), True, stats, n, ("domain", "src"))
    sl["mage_test_machine_fn"] = _pick_examples(
        test, test, y, p, (y == 1) & (p < 0.5), False, stats, n, ("domain", "src"))

    # --- HC3 (time-shifted ChatGPT) ---
    ev = hc3_frame(cfg, ctx)
    hf = hc3_features(cfg, bundle["state"], workers=workers)
    assert (ev["row_id"].to_numpy() == hf["row_id"].to_numpy()).all(), \
        "hc3 scores/features misaligned"
    yh = ev["label"].to_numpy().astype(int)
    ph = ev["p_machine_cal"].to_numpy()
    sl["hc3_human_fp"] = _pick_examples(
        ev, hf, yh, ph, (yh == 0) & (ph >= 0.5), True, stats, n, ("domain", "src"))
    sl["hc3_machine_fn"] = _pick_examples(
        ev, hf, yh, ph, (yh == 1) & (ph < 0.5), False, stats, n, ("domain", "src"))

    # --- RAID labelled subset (in-policy rows only) ---
    if RAID_SUBSET.exists():
        raid = raid_scored(cfg, ctx)
        rf = raid_features(cfg, raid, bundle["state"], workers=workers)
        yr = raid["y"].to_numpy().astype(int)
        pr_ = raid["p_machine_cal"].to_numpy()
        pol = raid["in_policy"].to_numpy().astype(bool)
        sl["raid_human_fp"] = _pick_examples(
            raid, rf, yr, pr_, (yr == 0) & (pr_ >= 0.5) & pol, True, stats, n,
            ("model", "attack", "domain"))
        sl["raid_machine_fn"] = _pick_examples(
            raid, rf, yr, pr_, (yr == 1) & (pr_ < 0.5) & pol, False, stats, n,
            ("model", "attack", "domain"))

    # --- MAGE OOD GPT-4 sets (unseen generator + unseen domains) ---
    import csv as _csv
    for f in cfg["e1"]["ood_files"]:
        path = ROOT / f
        if not path.exists():
            continue
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as fh:
            r = _csv.reader(fh)
            next(r)
            for row in r:
                rows.append({"text": row[0], "raw_label": int(row[1]), "src": row[2]})
        odf = pd.DataFrame(rows)
        odf["source_dataset"] = "mage"
        odf["y"] = 1 - odf["raw_label"].to_numpy()
        keep = odf["text"].map(normalize_text).notna()
        odf = odf[keep].copy()
        odf["text_norm"] = odf["text"].map(normalize_text)
        of = extract_texts(odf["text_norm"].tolist(), bundle["state"],
                           workers=int(cfg.get("feature_workers", 0)))
        po = 1.0 - bundle["calibrator"].predict(
            bundle["model"].predict_proba(of[names].to_numpy())[:, 1])
        odf["p_machine_cal"] = po
        yo = odf["y"].to_numpy().astype(int)
        stem = Path(f).stem
        sl[f"ood_{stem}_machine_fn"] = _pick_examples(
            odf, of, yo, po, (yo == 1) & (po < 0.5), False, stats, n, ("src",))
        sl[f"ood_{stem}_human_fp"] = _pick_examples(
            odf, of, yo, po, (yo == 0) & (po >= 0.5), True, stats, n, ("src",))

    out["slice_counts"] = {k: len(v) for k, v in sl.items()}
    out["duration_s"] = round(time.time() - t0, 1)
    return out


def reference_block(cfg: dict, ctx: dict) -> dict:
    """Locked-baseline reference on the canonical test split (for the report)."""
    test = ctx["test"]
    y = canonical_labels(test)
    p = test["p_machine_cal"].to_numpy()
    m = compute_metrics(y, p)
    cm = m.pop("confusion_matrix")
    sl = slice_metrics(y, p, test["words"].to_numpy(),
                       test["domain"].astype(str).to_numpy())
    for bucket in sl["by_length"].values():
        if "confusion_matrix" in bucket:
            bucket.pop("confusion_matrix")
    for dom in sl["by_domain"].values():
        if "confusion_matrix" in dom:
            dom.pop("confusion_matrix")
    return {
        "convention": "y=1 machine, y=0 human (canonical); Phase-1 positive class was HUMAN",
        "n": int(len(test)),
        "machine_share": round(float(y.mean()), 4),
        "metrics": {k: v for k, v in _round_metrics(m).items() if k != "confusion_matrix"},
        "confusion_matrix": cm,
        "by_length": sl["by_length"],
        "by_domain": sl["by_domain"],
        "reliability_bins": reliability_bins(p, y, n_bins=10),
        "brier": round(float(brier_score(p, y)), 4),
        "ece_10": round(float(expected_calibration_error(p, y, n_bins=10)), 4),
        "threshold": 0.5,
        "threshold_policy": "0.5 fixed from Phase 1; never tuned on evaluation data",
    }


def run_all(cfg: dict, only: list[str] | None = None, force: bool = False) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bundle = build_bundle(cfg)
    ctx = {"bundle": bundle, "test": bundle["test"]}
    out: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha_of(cfg),
        "canonical_convention": "y=1 MACHINE, y=0 HUMAN; scores always 1 - calibrated stored score",
        "model": cfg.get("primary_model", "hist_gradient_boosting"),
        "note": "Locked baseline. No training, no tuning, no threshold changes on eval data. "
                "Experimental models (E1 single-generator, E5 LODO) are trained only on "
                "MAGE-train/val and reported separately.",
    }
    out["reference"] = reference_block(cfg, ctx)
    try:
        out["error_examples"] = error_examples_block(cfg, ctx)
    except Exception as exc:
        import traceback
        out["error_examples"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                 "traceback": traceback.format_exc()[-3000:]}
    keys = only or list(EXPERIMENTS)
    for key in keys:
        cache_file = CACHE_DIR / f"{key}.json"
        if not force and cache_file.exists():
            out[key] = json.loads(cache_file.read_text(encoding="utf-8"))
            out[key]["cached"] = True
            continue
        fn = EXPERIMENTS[key]
        try:
            result = fn(cfg, ctx)
        except Exception as exc:  # never silently skip: record the failure
            import traceback
            result = {"name": key, "status": "failed",
                      "error": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc()[-3000:]}
        out[key] = result
        if result.get("status") != "failed":
            cache_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        elif cache_file.exists():
            cache_file.unlink()  # never cache a failed run
    return out
