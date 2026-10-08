"""Phase-4 verification helpers: reference parity, latency bench, samples.

Everything here MEASURES actual execution against the locked Phase-3
artifacts - no number is hard-coded. `scripts/verify_phase4.py` and
`scripts/bench_phase4.py` write the JSON files that the Phase-4 report
generator reads.
"""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..inference import Predictor
from ..preprocessing.build_dataset import PROCESSED, ROOT
from ..preprocessing.normalize import word_count
from .phase2 import meta_join
from .phase3 import CACHE_DIR as P3_CACHE, FINAL_DIR, load_final

LOGS = ROOT / "logs"
SYN_TEST = P3_CACHE / "syn_test.parquet"
E7_CACHE = P3_CACHE / "E7.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_text_meta() -> pd.DataFrame:
    """MAGE-test metadata only (no features) - light path for sampling."""
    cols = ["row_id", "text_norm", "label", "domain", "words"]
    df = pd.read_parquet(PROCESSED / "test.parquet", columns=cols)
    return df.sort_values("row_id").reset_index(drop=True)


def sample_texts() -> dict[str, str]:
    """Deterministic representative texts (MAGE-test, first row_id match)."""
    df = load_text_meta()
    y = 1 - df["label"].astype(int)          # canonical: 1 = machine
    w = df["words"].astype(int)
    picks: dict[str, str] = {}

    def first(mask) -> str | None:
        idx = df.index[mask]
        return str(df.loc[idx[0], "text_norm"]) if len(idx) else None

    picks["short_under_gate"] = first((w >= 5) & (w < 50))
    picks["typical_human"] = first((y == 0) & (w >= 50) & (w <= 150))
    picks["typical_machine"] = first((y == 1) & (w >= 50) & (w <= 150))
    picks["long"] = first(w >= 400)
    missing = [k for k, v in picks.items() if not v]
    if missing:
        raise RuntimeError(f"sample_texts could not find rows: {missing}")
    return picks


# --------------------------------------------------------------------------
# reference parity: Phase-4 inference vs the Phase-3 cached-feature path
# --------------------------------------------------------------------------

def reference_check(n_per_class: int = 5, seed: int = 0,
                    artifact_dir: Path | None = None) -> dict:
    """Re-score MAGE-test rows through ml.inference.Predictor and compare
    with p computed from the Phase-3 cached features (the numbers the
    locked artifact, reports and E7 error slices were built from).

    Proves Phase 4 did not change preprocessing, feature order,
    calibration, orientation or rounding behavior.
    """
    artifact_dir = Path(artifact_dir) if artifact_dir else FINAL_DIR
    final = load_final(artifact_dir)
    feats = final["features"]

    test = meta_join(("test",))["test"].reset_index(drop=True)
    syn = pd.read_parquet(SYN_TEST)
    assert len(syn) == len(test), "syntactic cache row mismatch"
    for c in [f for f in feats if f.startswith("syn_")]:
        test[c] = syn[c].to_numpy()

    y = (1 - test["label"].astype(int)).to_numpy()
    X = test[feats].to_numpy()
    p_cache = final["calibrator"].predict(
        final["model"].predict_proba(X)[:, 1])

    # orientation sanity over the WHOLE locked test split (free: p_cache is
    # already computed). Canonical y=1 MACHINE means mean p(machine) for
    # machine rows must sit well above the human-row mean.
    human_p = p_cache[y == 0]
    mach_p = p_cache[y == 1]
    orientation = {
        "n_human": int(human_p.size),
        "n_machine": int(mach_p.size),
        "mean_p_human": round(float(human_p.mean()), 4),
        "mean_p_machine": round(float(mach_p.mean()), 4),
        "median_p_human": round(float(np.median(human_p)), 4),
        "median_p_machine": round(float(np.median(mach_p)), 4),
    }
    orientation["separated"] = bool(
        orientation["mean_p_machine"] > orientation["mean_p_human"] + 0.3)

    pred = Predictor(artifact_dir)
    rng = np.random.default_rng(seed)
    human_idx = np.where(y == 0)[0]
    mach_idx = np.where(y == 1)[0]
    picks = (list(rng.choice(human_idx, n_per_class, replace=False)) +
             list(rng.choice(mach_idx, n_per_class, replace=False)))

    rows = []
    for i in picks:
        text = str(test.iloc[int(i)]["text_norm"])
        p_api = pred.score(text)
        p_ref = float(p_cache[int(i)])
        rows.append({
            "row_id": int(test.iloc[int(i)]["row_id"]),
            "domain": str(test.iloc[int(i)]["domain"]),
            "true": "machine" if int(y[int(i)]) == 1 else "human",
            "words": int(word_count(text)),
            "p_phase3_cached": p_ref,
            "p_phase4_inference": p_api,
            "abs_delta": abs(p_api - p_ref),
        })
    max_delta = max(r["abs_delta"] for r in rows)

    # -- cross-check against the Phase-3 E7 record for the top human FP ----
    e7_block = {}
    top = int(human_idx[int(np.argmax(p_cache[human_idx]))])
    if E7_CACHE.exists():
        e7 = json.loads(E7_CACHE.read_text(encoding="utf-8"))
        rec = ((e7.get("slices") or {}).get("mage_test_human_fp") or [None])[0]
        if rec:
            p_phase4_top = pred.score(str(test.iloc[top]["text_norm"]))
            e7_block = {
                "e7_recorded_p_machine": rec.get("p_machine"),
                "recomputed_p_machine": round(float(p_cache[top]), 4),
                "phase4_p_machine": round(p_phase4_top, 4),
                "e7_recorded_words": rec.get("words"),
                "recomputed_words": int(test.iloc[top]["words"]),
                "e7_recorded_domain": rec.get("domain"),
                "recomputed_domain": str(test.iloc[top]["domain"]),
                "match": (round(float(p_cache[top]), 4) == rec.get("p_machine")
                          and round(p_phase4_top, 4) == rec.get("p_machine")
                          and int(test.iloc[top]["words"]) == rec.get("words")
                          and str(test.iloc[top]["domain"]) ==
                          rec.get("domain")),
            }

    return {
        "generated_at": _now(),
        "artifact": artifact_dir.name,
        "n_rows": len(rows),
        "seed": seed,
        "tolerance": 1e-9,
        "max_abs_delta": max_delta,
        "all_within_tolerance": bool(max_delta <= 1e-9),
        "rows": rows,
        "orientation": orientation,
        "e7_cross_check": e7_block,
    }


# --------------------------------------------------------------------------
# latency bench
# --------------------------------------------------------------------------

def _timed(fn, n: int) -> dict:
    xs = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(xs)
    return {
        "n": n,
        "median_ms": round(float(np.median(arr)), 2),
        "mean_ms": round(float(arr.mean()), 2),
        "min_ms": round(float(arr.min()), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "max_ms": round(float(arr.max()), 2),
    }


def bench(n_runs: int = 20, artifact_dir: Path | None = None) -> dict:
    """Measure real local latency: cold load, first request (lazy spaCy
    load), warm per-request analyze with/without signals."""
    artifact_dir = Path(artifact_dir) if artifact_dir else FINAL_DIR
    texts = sample_texts()

    t0 = time.perf_counter()
    pred = Predictor(artifact_dir)
    cold_load_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    # first extraction pays the lazy spaCy pipeline load (one-time; the API
    # does this during startup via Predictor.warmup())
    warmup_ms = pred.warmup()

    warm: dict[str, dict] = {}
    for name, text in texts.items():
        warm[name] = {
            "words": int(word_count(text)),
            "chars": len(text),
            "analyze_with_signals_ms": _timed(
                lambda t=text: pred.analyze(t, include_signals=True),
                n_runs),
            "analyze_no_signals_ms": _timed(
                lambda t=text: pred.analyze(t, include_signals=False),
                n_runs),
        }

    import sklearn
    return {
        "generated_at": _now(),
        "artifact": artifact_dir.name,
        "n_runs": n_runs,
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "platform": platform.platform(),
            "cpu_count": __import__("os").cpu_count(),
            "device": "cpu",
        },
        "cold_load_ms": cold_load_ms,
        "warmup_including_spacy_load_ms": warmup_ms,
        "warm": warm,
    }


def write_json(path: Path, obj: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    return path
