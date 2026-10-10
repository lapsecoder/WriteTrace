"""Phase-5A experiments: is segment-level inference scientifically defensible?

This module is the measurement layer of Phase 5A. It answers the question
BEFORE any new model or any UI: *may WriteTrace say anything about a part of a
document, and on what evidence?*

Ground-truth semantics (the central honesty constraint)
------------------------------------------------------
MAGE / HC3 / RAID / OOD corpora carry ONE label per document. Splitting such a
document into paragraphs does NOT create segment-level authorship labels, so
every segment derived that way is labelled

    inherited_from_document

and all reported metrics for it are explicitly flagged as inherited-label
metrics. The only GENUINE segment-level labels available without new data are
segments that are, by construction, complete corpus texts:

  * composite documents built from two complete corpus texts (one human, one
    machine) joined by a blank line -> each segment IS a whole labeled text
    ("genuine_whole_text"), while the composite itself has NO document label.

Both kinds are reported separately and never pooled into one headline number.

Reuse policy
------------
final-v1 is consumed read-only through the same preprocessing, feature state,
feature order, model and calibrator as Phase 3/4 (ml.inference.Predictor). The
batched scoring helper below calls the same functions in parallel and is checked
for exact equality against the serial Predictor path (recorded as
`batch_parity` in the results).
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.evaluation.metrics import compute_metrics
from ml.experiments.phase2 import (PROCESSED, eval_hc3, meta_join,  # noqa: E402
                                   raid_frame)
from ml.experiments.phase3 import FINAL_DIR, load_final, sha_of  # noqa: E402
from ml.features.extract import extract_many  # noqa: E402
from ml.features.syntactic import extract_syntactic  # noqa: E402
from ml.inference import Predictor  # noqa: E402
from ml.preprocessing.normalize import normalize_text, word_count  # noqa: E402
from ml.segmentation import (SegmentationPolicy, coverage_report,  # noqa: E402
                             segment_text)

RESULTS_DIR = ROOT / "ml" / "experiments" / "phase5a"
CACHE_DIR = RESULTS_DIR / "cache"
RESULTS_PATH = RESULTS_DIR / "phase5a-results.json"
CONFIG_PATH = ROOT / "ml" / "configs" / "phase5a.json"

# length buckets required by the Phase-5A spec
BUCKETS = [(0, 30), (30, 50), (50, 80), (80, 150), (150, 300), (300, 10 ** 9)]
BUCKET_NAMES = ["<30w", "30-49w", "50-79w", "80-149w", "150-299w", "300w+"]

LABEL_INHERITED = "inherited_from_document"
LABEL_GENUINE = "genuine_whole_text"

# reuse-vs-train decision thresholds. These encode the scientific bar for
# exposing segment-level output to a user, declared BEFORE looking at results.
#
#   * A segment verdict shown to a user is a claim about ONE passage, so the
#     relevant bar is the GATE that keeps segments where the model is actually
#     reliable: kept accuracy/AUC on inherited labels, kept human-FPR (a false
#     "machine" accusation on one paragraph is the costly error), and kept ECE
#     (probabilities must be trustworthy, not just correctly ranked).
#   * OOD segments (GPT-4 / paraphrased) must not be far worse than chance at the
#     kept gate; if they are, segment output is not safe to surface generically.
#   * "Genuine" (whole-text) segments are the fair ceiling; a reused final-v1 is
#     acceptable ONLY if it clears these bars at some gate that keeps a useful
#     share of segments. If no gate clears them, final-v1 is NOT defensible at
#     segment level and a dedicated segments-v1 model is required.
DECISION = {
    "min_kept_accuracy": 0.85,        # kept inherited-label accuracy
    "max_kept_human_fpr": 0.30,       # kept false-'machine' rate on human segs
    "max_kept_ece": 0.05,             # kept expected calibration error (10 bins)
    "min_genuine_auc": 0.85,          # segment AUC on genuine whole-text labels
    "min_kept_share": 0.50,           # kept gate must retain >=50% of segments
    "max_auc_drop_vs_document": 0.10,  # allowed kept-AUC drop vs document level
    "min_ood_auc": 0.65,              # kept OOD segment AUC floor (vs ~0.55-0.59)
    "ood_for_gate": "mage_test",      # corpus whose kept share gates usability
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _np(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and math.isnan(o):
        return None
    return str(o)


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_np) + "\n",
                    encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def default_config() -> dict:
    return {
        "pipeline_version": "phase5a_v1",
        "seed": 0,
        "artifact": "final-v1",
        "policy": {"max_segment_words": 150, "min_window_words": 30,
                   "heading_max_words": 12, "merge_headings": True},
        "composite": {"target_words": 110, "n_composites": 800,
                      "min_source_words": 60, "max_source_words": 150},
        "samples": {"mage_test": 4000, "hc3": 2000, "raid": 1500,
                    "ood_gpt": None, "ood_gpt_para": None},
        "raid_attacks": ["none", "paraphrase", "insert_paragraphs",
                         "synonym", "article_deletion", "whitespace",
                         "homoglyph", "upper_lower", "number"],
        "workers": 4,
        "gate_candidates": [30, 40, 50, 60, 80, 100, 120, 150],
    }


# --------------------------------------------------------------------------
# corpora
# --------------------------------------------------------------------------

def _sample(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(df), size=n, replace=False))
    return df.iloc[idx].reset_index(drop=True)


def load_corpus(name: str, cfg: dict) -> pd.DataFrame:
    """Return a frame with columns [row_id, text, y, domain, src, extra].

    y is canonical (1 = machine-generated), matching Phase 3.
    """
    seed = int(cfg["seed"])
    if name == "mage_test":
        df = meta_join(("test",))["test"][["row_id", "text_norm", "label",
                                          "domain", "src"]]
        df = df.rename(columns={"text_norm": "text"})
        df["y"] = 1 - df["label"].astype(int)
        out = _sample(df, int(cfg["samples"]["mage_test"]), seed)
        out["extra"] = ""
        return out
    if name == "hc3":
        df = eval_hc3()[["row_id", "text_norm", "label", "domain", "src"]]
        df = df.rename(columns={"text_norm": "text"})
        df["y"] = df["label"].astype(int)          # HC3: 0 human, 1 chatgpt
        out = _sample(df, int(cfg["samples"]["hc3"]), seed + 1)
        out["extra"] = ""
        return out
    if name == "raid":
        df = raid_frame()[["id", "text_norm", "y", "domain", "attack"]]
        df = df.rename(columns={"id": "row_id", "text_norm": "text",
                                "attack": "extra"})
        df = df[df["extra"].isin(cfg["raid_attacks"])]
        per = max(int(cfg["samples"]["raid"]) // max(len(cfg["raid_attacks"]), 1), 1)
        parts = [_sample(g, per, seed + 2 + i)
                 for i, (_, g) in enumerate(df.groupby("extra"))]
        out = pd.concat(parts, ignore_index=True)
        return out
    if name in ("ood_gpt", "ood_gpt_para"):
        fname = ("test_ood_set_gpt.csv" if name == "ood_gpt"
                 else "test_ood_set_gpt_para.csv")
        import csv as _csv
        rows = []
        with (ROOT / "ml" / "data" / "raw" / "mage" / fname).open(
                "r", encoding="utf-8") as fh:
            r = _csv.reader(fh)
            next(r)
            for i, row in enumerate(r):
                nt = normalize_text(row[0])
                if not nt:
                    continue
                rows.append({"row_id": f"{name}-{i}", "text": nt,
                             "y": 1 - int(row[1]),          # MAGE convention
                             "domain": "ood", "src": name,
                             "extra": row[2] if len(row) > 2 else ""})
        out = _sample(pd.DataFrame(rows), cfg["samples"].get(name), seed + 3)
        return out
    raise ValueError(f"unknown corpus: {name}")


def build_composites(cfg: dict, mage: pd.DataFrame) -> pd.DataFrame:
    """Genuine-label segments: complete corpus texts joined by a blank line.

    Only sources whose WHOLE text fits inside the segment limit are used, so a
    composite segment is a complete labeled document - its label applies to the
    segment itself, not by inheritance. The composite has no document label.
    """
    ccfg = cfg["composite"]
    pool = mage[(mage["words"] >= ccfg["min_source_words"])
                & (mage["words"] <= ccfg["max_source_words"])].copy() \
        if "words" in mage.columns else None
    if pool is None:
        pool = mage.assign(words=mage["text"].map(word_count))
        pool = pool[(pool["words"] >= ccfg["min_source_words"])
                    & (pool["words"] <= ccfg["max_source_words"])]
    human = pool[pool["y"] == 0].reset_index(drop=True)
    machine = pool[pool["y"] == 1].reset_index(drop=True)
    n = min(int(ccfg["n_composites"]), len(human), len(machine))
    if n == 0:
        return pd.DataFrame(columns=["doc_id", "text", "seg_true_y", "order"])
    rng = np.random.default_rng(int(cfg["seed"]) + 11)
    hi = rng.choice(len(human), size=n, replace=False)
    mi = rng.choice(len(machine), size=n, replace=False)
    rows = []
    for i in range(n):
        h_text = human.at[int(hi[i]), "text"].strip()
        m_text = machine.at[int(mi[i]), "text"].strip()
        machine_first = bool(rng.integers(0, 2))
        if machine_first:
            text = f"{m_text}\n\n{h_text}"
            order = "machine_first"
        else:
            text = f"{h_text}\n\n{m_text}"
            order = "human_first"
        rows.append({"doc_id": f"comp-{i:05d}", "text": text, "order": order,
                     "human_text": h_text, "machine_text": m_text})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# segmentation + scoring
# --------------------------------------------------------------------------

@dataclass
class SegmentRow:
    corpus: str
    doc_id: str
    segment_id: int
    kind: str
    paragraph_index: int
    window_index: int | None
    words: int
    hard_split: bool
    text: str


def segment_corpus(df: pd.DataFrame, corpus: str, policy: SegmentationPolicy,
                   doc_col: str = "doc_id", text_col: str = "text",
                   label_semantics: str = LABEL_INHERITED
                   ) -> tuple[list[SegmentRow], list[dict]]:
    rows: list[SegmentRow] = []
    covs: list[dict] = []
    for _, rec in df.iterrows():
        text = str(rec[text_col])
        segs = segment_text(text, policy)
        cov = coverage_report(text, segs)
        cov["doc_id"] = str(rec[doc_col])
        covs.append(cov)
        for s in segs:
            rows.append(SegmentRow(
                corpus=corpus, doc_id=str(rec[doc_col]), segment_id=s.segment_id,
                kind=s.kind, paragraph_index=s.paragraph_index,
                window_index=s.window_index, words=s.word_count,
                hard_split=s.hard_split, text=s.text))
    if rows:
        missing = sum(c["non_whitespace_missing"] for c in covs)
        if missing:
            raise AssertionError(f"{corpus}: {missing} characters lost")
    return rows, covs


def score_segments(pred: Predictor, rows, workers: int = 0,
                   batch: int = 2000) -> np.ndarray:
    """p(machine) for every segment text through the final-v1 path.

    `rows` may be SegmentRow objects or plain strings. Uses the same extraction
    functions as Predictor._extract_rows (optionally in parallel) and one
    batched model+calibrator call.
    """
    final = pred.final
    raw_texts = [r.text if hasattr(r, "text") else str(r) for r in rows]
    texts = [normalize_text(t) or "" for t in raw_texts]
    out = np.full(len(texts), np.nan, dtype=float)
    state = final["state"]
    for start in range(0, len(texts), batch):
        chunk = texts[start:start + batch]
        feats = extract_many(chunk, state, workers=workers)
        if pred.needs_syn:
            feats = pd.concat([feats.reset_index(drop=True),
                               extract_syntactic(chunk, n_process=workers)],
                              axis=1)
        raw = final["model"].predict_proba(feats[pred.features].to_numpy())[:, 1]
        out[start:start + len(chunk)] = np.asarray(
            final["calibrator"].predict(raw)).ravel()
    return out


# --------------------------------------------------------------------------
# metrics helpers
# --------------------------------------------------------------------------

def bucket_of(words: int) -> str:
    for (lo, hi), name in zip(BUCKETS, BUCKET_NAMES):
        if lo <= words < hi:
            return name
    return BUCKET_NAMES[-1]


def human_fpr(cm: dict) -> float | None:
    tn, fp = cm.get("tn"), cm.get("fp")
    return round(fp / (fp + tn), 4) if tn is not None and (fp + tn) else None


def eval_frame(y, p, words, extra_label: str | None = None) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(y) == 0:
        return {"n": 0, "label_semantics": extra_label}
    if len(np.unique(y)) < 2:
        return {"n": int(len(y)), "label_semantics": extra_label,
                "note": "single class in this slice"}
    m = compute_metrics(y, p)
    m["human_fpr"] = human_fpr(m["confusion_matrix"])
    m["label_semantics"] = extra_label
    return m


def by_bucket_counts(p, words) -> dict:
    """Distribution-only bucket stats (no labels -> no metrics)."""
    p = np.asarray(p, dtype=float)
    words = np.asarray(words, dtype=int)
    out = {}
    for name in BUCKET_NAMES:
        mask = np.array([bucket_of(int(w)) == name for w in words])
        if mask.sum() == 0:
            out[name] = {"n": 0}
            continue
        out[name] = {
            "n": int(mask.sum()),
            "mean_p": round(float(p[mask].mean()), 4),
            "share_pred_machine": round(float((p[mask] >= 0.5).mean()), 4),
        }
    return out


def by_bucket(y, p, words, label_semantics: str | None = None) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    words = np.asarray(words, dtype=int)
    out = {}
    for name in BUCKET_NAMES:
        mask = np.array([bucket_of(int(w)) == name for w in words])
        if mask.sum() == 0:
            out[name] = {"n": 0}
            continue
        out[name] = eval_frame(y[mask], p[mask], words[mask], label_semantics)
        out[name]["mean_p"] = round(float(p[mask].mean()), 4)
        out[name]["share_pred_machine"] = round(
            float((p[mask] >= 0.5).mean()), 4)
    return out


def reliability(y, p, n_bins: int = 10) -> list[dict]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    bins = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if mask.sum() == 0:
            continue
        bins.append({"bin_lo": round(float(lo), 2), "bin_hi": round(float(hi), 2),
                     "count": int(mask.sum()),
                     "mean_p": round(float(p[mask].mean()), 4),
                     "frac_positive": round(float(y[mask].mean()), 4)})
    return bins


def confident_subset(y, p) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    conf = (p <= 0.2) | (p >= 0.8)
    if conf.sum() == 0:
        return {"n": 0}
    yc, pc = y[conf], p[conf]
    pred = (pc >= 0.5).astype(int)
    return {"n": int(conf.sum()),
            "share_of_segments": round(float(conf.mean()), 4),
            "accuracy": round(float((pred == yc).mean()), 4),
            "human_fpr": (round(float((pred[yc == 0] == 1).mean()), 4)
                          if (yc == 0).any() else None),
            "machine_recall": (round(float((pred[yc == 1] == 1).mean()), 4)
                               if (yc == 1).any() else None)}

# --------------------------------------------------------------------------
# synthetic robustness probes (Phase 4 limitation: repetition false positives)
# --------------------------------------------------------------------------

PROBES: dict[str, str] = {
    "normal_human": (
        "so we ended up going to that new cafe near the station, the one everyone "
        "keeps talking about. honestly the coffee was fine, nothing special, but "
        "the cake was unreal - i had this lemon thing with poppy seeds and now i "
        "want to go back every weekend. anyway we sat there for two hours just "
        "catching up, you know how it is. sarah brought her dog, which was a whole "
        "situation on its own, but whatever, it was really nice to just sit and not "
        "think about work for a while. we should go together sometime, i think youd "
        "like it a lot. it is the kind of place where you end up staying for hours."),
    "repeated_sentence": "The committee announced yesterday that the new library "
                         "will open in March. " * 12,
    "repeated_paragraph": " ".join(
        ["The quick brown fox jumps over the lazy dog again and again. " * 2] * 4),
    "punctuation_heavy": ("Wait—really?! Yes; truly... (the \"final\" count) "
                          "wasn't 42; it was 43!!! #refresh 100% -- see below... "
                          "Okay? Right. ") * 6,
    "list_like": ("- first item on the list of things we discussed at length\n"
                  "- second item, which everyone agreed was important\n"
                  "- third item that turned out to be less relevant in hindsight\n"
                  "- fourth item, mentioned briefly and then forgotten entirely\n"
                  "- fifth item that nobody could explain clearly at the meeting"),
    "very_short": "Too short to judge.",
    "heading_only": "Introduction",
    "unicode_multilingual": (
        "Le comité a annoncé hier que la nouvelle bibliothèque ouvrira en mars. "
        "Elle a coûté environ 4,2 millions d'euros, ont dit les responsables. "
        "這個項目是本地運行的，不會把文字發送到任何外部服務。 "
        "I have to admit the delays frustrated quite a few residents, but the "
        "finished building looks wonderful. Bring your library cards! Nous sommes "
        "ravis, a déclaré la maire, et elle restera ouverte sept jours sur sept."),
    "one_long_sentence": "word " * 320,
    "formally_machine": (
        "The implementation follows a modular architecture that separates data "
        "ingestion, feature computation, and model inference into discrete "
        "components. Each module exposes a well-defined interface, enabling "
        "independent testing and straightforward replacement. Furthermore, the "
        "design ensures reproducibility by pinning all random seeds and recording "
        "configuration metadata for every experiment. In conclusion, this approach "
        "reduces coupling between subsystems and simplifies long-term maintenance "
        "of the pipeline across environments and contributors. Moreover, each "
        "component is validated against a fixed set of reference measurements."),
}


def repetition_rate(text: str, n: int = 5) -> float:
    """Fraction of repeated n-grams (text-level proxy, no features needed)."""
    words = normalize_text(text).lower().split() if text else []
    if len(words) < n:
        return 0.0
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    return round(1.0 - len(set(grams)) / len(grams), 4)


# --------------------------------------------------------------------------
# aggregation / consistency statistics
# --------------------------------------------------------------------------

def _corr(a: np.ndarray, b: np.ndarray, method: str = "pearson") -> float | None:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3:
        return None
    if method == "pearson":
        if a.std() == 0 or b.std() == 0:
            return None
        return round(float(np.corrcoef(a, b)[0, 1]), 4)
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return None
    return round(float(np.corrcoef(ra, rb)[0, 1]), 4)


def aggregation_stats(seg_p: np.ndarray, weights: np.ndarray,
                      doc_p: float) -> dict:
    mean = float(np.mean(seg_p))
    median = float(np.median(seg_p))
    wsum = float(weights.sum())
    weighted = float((seg_p * weights).sum() / wsum)
    return {
        "mean": round(mean, 6), "median": round(median, 6),
        "weighted_mean": round(weighted, 6),
        "max": round(float(np.max(seg_p)), 6),
        "min": round(float(np.min(seg_p)), 6),
        "std": round(float(np.std(seg_p)), 6),
        "doc_p": round(float(doc_p), 6),
        "doc_minus_mean": round(float(doc_p) - mean, 6),
        "doc_minus_weighted": round(float(doc_p) - weighted, 6),
    }


AGG_KEYS = ("mean", "median", "weighted_mean", "max", "min", "std")


def decision_agreement(agg_value: float, doc_p: float, threshold: float = 0.5
                       ) -> int:
    return int((agg_value >= threshold) == (doc_p >= threshold))


# --------------------------------------------------------------------------
# experiment driver
# --------------------------------------------------------------------------

def _cache_key(cfg: dict, policy: SegmentationPolicy) -> str:
    final = load_final(FINAL_DIR)
    manifest = final["manifest"]
    return sha_of({"cfg": cfg, "policy": policy.as_dict(),
                   "artifact_files": manifest.get("files"),
                   "pipeline": manifest.get("pipeline_version")})


def _build_tables(cfg: dict, policy: SegmentationPolicy, pred: Predictor,
                  workers: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Segment every corpus + composites, then score everything once."""
    seg_records: list[dict] = []
    doc_records: list[dict] = []
    cov_summary: dict = {}

    frames = {name: load_corpus(name, cfg)
              for name in ("mage_test", "hc3", "raid", "ood_gpt", "ood_gpt_para")}

    for name, df in frames.items():
        rows, covs = segment_corpus(df, name, policy, doc_col="row_id",
                                    text_col="text")
        extra_map = (dict(zip(df["row_id"].astype(str), df["extra"]))
                     if "extra" in df.columns else {})
        for r in rows:
            seg_records.append({
                "corpus": name, "doc_id": r.doc_id, "segment_id": r.segment_id,
                "kind": r.kind, "paragraph_index": r.paragraph_index,
                "window_index": r.window_index, "words": r.words,
                "hard_split": r.hard_split, "text": r.text,
                "label_semantics": LABEL_INHERITED,
                "extra": extra_map.get(r.doc_id, ""),
            })
        doc_records.extend({"corpus": name, "doc_id": str(r), "words": w,
                            "y": int(y), "text": t, "label_semantics": LABEL_INHERITED,
                            "extra": e}
                           for r, w, y, t, e in zip(
                               df["row_id"].astype(str), df["text"].map(word_count),
                               df["y"], df["text"], df["extra"]
                               if "extra" in df.columns else [""] * len(df)))
        cov_summary[name] = {
            "n_docs": int(len(df)),
            "n_segments": int(len(rows)),
            "segments_per_doc": round(len(rows) / max(len(df), 1), 3),
            "paragraph_segments": sum(1 for r in rows if r.kind == "paragraph"),
            "window_segments": sum(1 for r in rows if r.kind == "window"),
            "hard_splits": sum(1 for r in rows if r.hard_split),
            "non_whitespace_missing": int(sum(c["non_whitespace_missing"]
                                             for c in covs)),
        }

    # --- composites: genuine segment labels -------------------------------
    comp = build_composites(cfg, frames["mage_test"])
    label_map: dict[str, int] = {}
    for _, rec in comp.iterrows():
        label_map[normalize_text(rec["human_text"])] = 0
        label_map[normalize_text(rec["machine_text"])] = 1
    crows, ccovs = segment_corpus(comp, "composite", policy, doc_col="doc_id")
    unmatched = 0
    for r in crows:
        nt = normalize_text(r.text)
        if nt in label_map:
            y_true = label_map[nt]
        else:
            y_true = -1
            unmatched += 1
        seg_records.append({
            "corpus": "composite", "doc_id": r.doc_id, "segment_id": r.segment_id,
            "kind": r.kind, "paragraph_index": r.paragraph_index,
            "window_index": r.window_index, "words": r.words,
            "hard_split": r.hard_split, "text": r.text,
            "label_semantics": LABEL_GENUINE, "extra": "",
            "y_true": y_true,
        })
    order_map = dict(zip(comp["doc_id"], comp["order"]))
    doc_records.extend({"corpus": "composite", "doc_id": r, "words": w, "y": -1,
                        "text": t, "label_semantics": "no_document_label",
                        "extra": order_map.get(r, "")}
                       for r, w, t in zip(comp["doc_id"],
                                          comp["text"].map(word_count),
                                          comp["text"]))
    cov_summary["composite"] = {
        "n_docs": int(len(comp)),
        "n_segments": int(len(crows)),
        "segments_per_doc": round(len(crows) / max(len(comp), 1), 3),
        "paragraph_segments": sum(1 for r in crows if r.kind == "paragraph"),
        "window_segments": sum(1 for r in crows if r.kind == "window"),
        "hard_splits": sum(1 for r in crows if r.hard_split),
        "non_whitespace_missing": int(sum(c["non_whitespace_missing"]
                                         for c in ccovs)),
        "unmatched_genuine_labels": int(unmatched),
    }

    segs = pd.DataFrame(seg_records)
    docs = pd.DataFrame(doc_records)

    # document-level labels (canonical) for inherited corpora
    t0 = time.perf_counter()
    segs["p"] = score_segments(pred, segs["text"].tolist(), workers=workers)
    docs["p_doc"] = score_segments(pred, docs["text"].tolist(), workers=workers)
    scoring_s = round(time.perf_counter() - t0, 1)

    # genuine segment truth for composites
    if "y_true" not in segs.columns:
        segs["y_true"] = -1
    segs.loc[segs["corpus"] != "composite", "y_true"] = -1
    return segs, docs, {"coverage": cov_summary, "scoring_s": scoring_s}


def _path_parity(pred: Predictor, segs: pd.DataFrame, docs: pd.DataFrame,
                 n: int = 200, seed: int = 0) -> dict:
    """Serial Predictor path vs batched study path, plus Phase-3 cached parity."""
    rng = np.random.default_rng(seed)
    out: dict = {}
    if len(segs):
        idx = rng.choice(len(segs), size=min(n, len(segs)), replace=False)
        texts = segs.iloc[idx]["text"].tolist()
        serial = pred.score_texts(texts)
        batched = segs.iloc[idx]["p"].to_numpy()
        out["segments"] = {
            "n": int(len(texts)),
            "max_abs_delta": round(float(np.max(np.abs(
                np.asarray(serial) - batched))), 12),
        }
    if len(docs):
        idx = rng.choice(len(docs), size=min(n, len(docs)), replace=False)
        texts = docs.iloc[idx]["text"].tolist()
        serial = pred.score_texts(texts)
        batched = docs.iloc[idx]["p_doc"].to_numpy()
        out["documents"] = {
            "n": int(len(texts)),
            "max_abs_delta": round(float(np.max(np.abs(
                np.asarray(serial) - batched))), 12),
        }
    # Phase-3 cached-feature oracle for the same MAGE-test documents
    try:
        from ml.experiments.phase3 import V1_FEATURES, final_predict
        test = meta_join(("test",))["test"]
        final = pred.final
        from ml.features.syntactic import SYNTACTIC_FEATURES
        syn = pd.read_parquet(ROOT / "ml" / "experiments" / "phase3" / "cache"
                              / "syn_test.parquet")
        frame = test.copy()
        for c in SYNTACTIC_FEATURES:
            frame[c] = syn[c].to_numpy()
        p_cached = final_predict(final, frame)
        id_to_p = dict(zip(frame["row_id"].astype(str), p_cached))
        sample = docs[docs["corpus"] == "mage_test"].head(400)
        deltas = [abs(float(id_to_p[r]) - float(p))
                  for r, p in zip(sample["doc_id"], sample["p_doc"])
                  if r in id_to_p]
        out["mage_test_vs_phase3_cached"] = {
            "n": len(deltas),
            "max_abs_delta": round(float(max(deltas)), 9) if deltas else None,
        }
    except Exception as exc:                      # recorded, never hidden
        out["mage_test_vs_phase3_cached"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out

# --------------------------------------------------------------------------
# analysis blocks
# --------------------------------------------------------------------------

def _mask(segs: pd.DataFrame, **kw) -> pd.Series:
    m = pd.Series(True, index=segs.index)
    for k, v in kw.items():
        m &= (segs[k] == v) if not isinstance(v, (list, set)) else segs[k].isin(v)
    return m


def e1_inventory(segs: pd.DataFrame, docs: pd.DataFrame, cov: dict) -> dict:
    wc = segs["words"].to_numpy()
    return {
        "n_documents_scored": int(len(docs)),
        "n_segments_scored": int(len(segs)),
        "coverage": cov,
        "segment_words": {
            "mean": round(float(wc.mean()), 2),
            "median": int(np.median(wc)),
            "p10": int(np.percentile(wc, 10)),
            "p90": int(np.percentile(wc, 90)),
            "min": int(wc.min()), "max": int(wc.max()),
        },
        "label_semantics_counts": {k: int(v) for k, v in
                                   segs["label_semantics"].value_counts().items()},
        "kind_counts": {k: int(v) for k, v in
                        segs["kind"].value_counts().items()},
        "hard_split_segments": int(segs["hard_split"].sum()),
    }


def e2_length_analysis(segs: pd.DataFrame) -> dict:
    out: dict = {"buckets": dict(zip(BUCKET_NAMES, []))}
    overall = by_bucket_counts(segs["p"], segs["words"])
    share = {b: round(v.get("n", 0) / max(len(segs), 1), 4) for b, v in overall.items()}
    out["share_of_segments"] = share
    out["mean_p_by_bucket"] = {b: v.get("mean_p") for b, v in overall.items()}
    out["share_pred_machine_by_bucket"] = {
        b: v.get("share_pred_machine") for b, v in overall.items()}
    out["by_corpus"] = {}
    for name in sorted(segs["corpus"].unique()):
        sub = segs[segs["corpus"] == name]
        b = by_bucket_counts(sub["p"], sub["words"])
        out["by_corpus"][name] = {
            "n_segments": int(len(sub)),
            "mean_words": round(float(sub["words"].mean()), 1),
            "share_by_bucket": {k: round(v.get("n", 0) / max(len(sub), 1), 4)
                                for k, v in b.items()},
            "mean_p_by_bucket": {k: v.get("mean_p") for k, v in b.items()},
        }
    # genuine-label composites: expect a clean split by construction
    comp = segs[segs["corpus"] == "composite"]
    if len(comp):
        h = comp[comp["y_true"] == 0]["p"].to_numpy()
        m = comp[comp["y_true"] == 1]["p"].to_numpy()
        out["composite_word_profile"] = {
            "mean_words_human_segments": round(float(
                comp[comp["y_true"] == 0]["words"].mean()), 1),
            "mean_words_machine_segments": round(float(
                comp[comp["y_true"] == 1]["words"].mean()), 1),
            "min_words": int(comp["words"].min()),
            "max_words": int(comp["words"].max()),
        }
    return out


def e3_segment_eval(segs: pd.DataFrame, docs: pd.DataFrame) -> dict:
    """final-v1 applied to segments, per corpus and per bucket.

    inherited-label corpora are reported with their semantics visible; the
    genuine-label composite numbers are reported separately and never pooled.
    """
    out: dict = {"by_corpus": {}, "by_corpus_by_bucket": {}}
    for name in sorted(segs["corpus"].unique()):
        sub = segs[segs["corpus"] == name]
        if name == "composite":
            keep = sub[sub["y_true"] >= 0]
            out["by_corpus"][name] = eval_frame(
                keep["y_true"], keep["p"], keep["words"], LABEL_GENUINE)
            out["by_corpus_by_bucket"][name] = by_bucket(
                keep["y_true"], keep["p"], keep["words"], LABEL_GENUINE)
            continue
        dm = docs[docs["corpus"] == name]
        truth = dm.set_index("doc_id")["y"]
        y = sub["doc_id"].map(truth).to_numpy()
        out["by_corpus"][name] = eval_frame(y, sub["p"], sub["words"],
                                            LABEL_INHERITED)
        out["by_corpus_by_bucket"][name] = by_bucket(y, sub["p"], sub["words"],
                                                     LABEL_INHERITED)
    # document-level reference on the same samples (for contrast)
    out["document_level_reference"] = {}
    for name in sorted(docs["corpus"].unique()):
        if name == "composite":
            continue
        dm = docs[docs["corpus"] == name]
        out["document_level_reference"][name] = eval_frame(
            dm["y"], dm["p_doc"], dm["words"], LABEL_INHERITED)
    return out


def e4_calibration(segs: pd.DataFrame, docs: pd.DataFrame,
                   manifest: dict) -> dict:
    out: dict = {"document_level_locked": {
        "test_brier": manifest.get("eval_summary", {}).get("test", {}).get("brier"),
        "test_ece_10": manifest.get("eval_summary", {}).get("test", {}).get("ece_10"),
        "source": "final-v1 manifest eval_summary.test (whole documents)",
    }}
    for name in ("mage_test", "hc3", "ood_gpt", "raid", "composite"):
        sub = segs[segs["corpus"] == name]
        if not len(sub):
            continue
        if name == "composite":
            keep = sub[sub["y_true"] >= 0]
            y, p = keep["y_true"].to_numpy(), keep["p"].to_numpy()
            sem = LABEL_GENUINE
        else:
            dm = docs[docs["corpus"] == name].set_index("doc_id")["y"]
            y = sub["doc_id"].map(dm).to_numpy()
            p = sub["p"].to_numpy()
            sem = LABEL_INHERITED
        out[name] = {
            "label_semantics": sem,
            "n": int(len(y)),
            "brier": round(float(np.mean((p - y) ** 2)), 4),
            "mean_abs_p_minus_y": round(float(np.mean(np.abs(p - y))), 4),
            "reliability_bins": reliability(y, p),
            "confident_subset": confident_subset(y, p),
        }
    # per-bucket calibration for the main corpus
    sub = segs[segs["corpus"] == "mage_test"]
    dm = docs[docs["corpus"] == "mage_test"].set_index("doc_id")["y"]
    y = sub["doc_id"].map(dm).to_numpy()
    p = sub["p"].to_numpy()
    out["mage_test_by_bucket"] = {}
    for name in BUCKET_NAMES:
        mask = np.array([bucket_of(int(w)) == name for w in sub["words"]])
        if mask.sum() < 20:
            out["mage_test_by_bucket"][name] = {"n": int(mask.sum()),
                                                "note": "too few segments"}
            continue
        out["mage_test_by_bucket"][name] = {
            "n": int(mask.sum()),
            "brier": round(float(np.mean((p[mask] - y[mask]) ** 2)), 4),
            "mean_abs_p_minus_y": round(float(np.mean(
                np.abs(p[mask] - y[mask]))), 4),
            "reliability_bins": reliability(y[mask], p[mask]),
            "confident_subset": confident_subset(y[mask], p[mask]),
            "label_semantics": LABEL_INHERITED,
        }
    return out


def _corpus_kept(segs: pd.DataFrame, docs: pd.DataFrame, corpus: str,
                 gate: int) -> dict:
    sub = segs[segs["corpus"] == corpus]
    if not len(sub):
        return {"kept_share": 0.0, "kept": {"n": 0}}
    dm = docs[docs["corpus"] == corpus].set_index("doc_id")["y"]
    y = sub["doc_id"].map(dm).to_numpy()
    p = sub["p"].to_numpy()
    w = sub["words"].to_numpy()
    kept = w >= gate
    return {
        "kept_share": round(float(kept.mean()), 4),
        "kept": eval_frame(y[kept], p[kept], w[kept], LABEL_INHERITED),
        "excluded": eval_frame(y[~kept], p[~kept], w[~kept], LABEL_INHERITED),
    }


def e5_gate_policy(segs: pd.DataFrame, docs: pd.DataFrame, cfg: dict,
                   doc_auc: float | None) -> dict:
    """Candidate segment evidence gates, scored the way Phase 3 scored its gate:
    the excluded slice must be worse than the kept slice, coverage must stay
    useful, and the kept slice must clear the user-facing reliability bars."""
    rows = []
    for gate in cfg["gate_candidates"]:
        entry: dict = {"segment_gate_min_words": gate}
        for corpus in ("mage_test", "hc3", "ood_gpt", "ood_gpt_para"):
            entry[corpus] = _corpus_kept(segs, docs, corpus, gate)
        # genuine-label check at the same gate
        comp = segs[(segs["corpus"] == "composite") & (segs["y_true"] >= 0)]
        kept = comp["words"].to_numpy() >= gate
        entry["composite_genuine"] = {
            "kept_share": round(float(kept.mean()), 4) if len(comp) else 0.0,
            "kept": eval_frame(comp["y_true"].to_numpy()[kept],
                               comp["p"].to_numpy()[kept],
                               comp["words"].to_numpy()[kept], LABEL_GENUINE),
        }
        rows.append(entry)

    def clears(entry: dict) -> tuple[bool, dict]:
        m = entry["mage_test"]["kept"]
        comp_auc = entry["composite_genuine"]["kept"].get("roc_auc")
        ood_aucs = [entry[c]["kept"].get("roc_auc") for c in ("ood_gpt", "ood_gpt_para")]
        checks = {
            "kept_share": (entry["mage_test"]["kept_share"]
                           >= DECISION["min_kept_share"]),
            "kept_accuracy": (m.get("accuracy") is not None
                              and m["accuracy"] >= DECISION["min_kept_accuracy"]),
            "kept_human_fpr": (m.get("human_fpr") is not None
                               and m["human_fpr"] <= DECISION["max_kept_human_fpr"]),
            "kept_ece": (m.get("ece_10") is not None
                         and m["ece_10"] <= DECISION["max_kept_ece"]),
            "kept_auc_vs_document": (
                doc_auc is None or m.get("roc_auc") is None
                or m["roc_auc"] >= doc_auc - DECISION["max_auc_drop_vs_document"]),
            "genuine_auc": (comp_auc is not None
                            and comp_auc >= DECISION["min_genuine_auc"]),
            "ood_auc": (all(a is not None and a >= DECISION["min_ood_auc"]
                            for a in ood_aucs) if any(
                a is not None for a in ood_aucs) else False),
        }
        return all(checks.values()), checks

    chosen = None
    for entry in rows:
        ok, checks = clears(entry)
        entry["criteria_met"] = bool(ok)
        entry["criteria_checks"] = {k: bool(v) for k, v in checks.items()}
        if ok and chosen is None:
            chosen = entry["segment_gate_min_words"]
    return {
        "criteria": DECISION,
        "document_auc_reference": doc_auc,
        "candidates": rows,
        "chosen_segment_gate_min_words": chosen,
        "rule": ("smallest candidate gate whose kept slice keeps >= "
                 f"{DECISION['min_kept_share']:.0%} of MAGE-test segments with "
                 f"accuracy >= {DECISION['min_kept_accuracy']}, human FPR <= "
                 f"{DECISION['max_kept_human_fpr']}, ECE-10 <= "
                 f"{DECISION['max_kept_ece']}, kept AUC within "
                 f"{DECISION['max_auc_drop_vs_document']} of the document AUC, "
                 f"genuine-label composite AUC >= {DECISION['min_genuine_auc']}, "
                 f"and BOTH OOD corpora kept AUC >= {DECISION['min_ood_auc']}"),
    }


def _doc_seg_frame(segs: pd.DataFrame, docs: pd.DataFrame, corpus: str
                   ) -> pd.DataFrame:
    sub = segs[segs["corpus"] == corpus]
    dm = docs[docs["corpus"] == corpus].set_index("doc_id")
    recs = []
    for doc_id, g in sub.groupby("doc_id"):
        if doc_id not in dm.index:
            continue
        recs.append({
            "doc_id": doc_id, "n_segments": len(g),
            "words": int(dm.at[doc_id, "words"]),
            "y": int(dm.at[doc_id, "y"]),
            "p_doc": float(dm.at[doc_id, "p_doc"]),
            "order": str(dm.at[doc_id, "extra"]),
            **{k: v for k, v in aggregation_stats(
                g["p"].to_numpy(), g["words"].to_numpy().astype(float),
                float(dm.at[doc_id, "p_doc"])).items()},
        })
    return pd.DataFrame(recs)


def e6_consistency(segs: pd.DataFrame, docs: pd.DataFrame) -> dict:
    out: dict = {"by_corpus": {}}
    for corpus in sorted(segs["corpus"].unique()):
        f = _doc_seg_frame(segs, docs, corpus)
        multi = f[f["n_segments"] >= 2] if len(f) else f
        block: dict = {"n_docs": int(len(f)), "n_docs_multi_segment": int(len(multi))}
        if len(multi):
            for key in AGG_KEYS:
                block[key] = {
                    "pearson_vs_doc": _corr(multi[key], multi["p_doc"], "pearson"),
                    "spearman_vs_doc": _corr(multi[key], multi["p_doc"], "spearman"),
                    "mean_abs_diff_vs_doc": round(float(
                        (multi[key] - multi["p_doc"]).abs().mean()), 4),
                    "decision_agreement_with_doc": round(float(
                        np.mean([decision_agreement(a, d)
                                 for a, d in zip(multi[key], multi["p_doc"])])),
                        4),
                }
            block["mean"]["mean_minus_doc"] = round(float(
                (multi["mean"] - multi["p_doc"]).mean()), 4)
            block["std"]["mean_std"] = round(float(multi["std"].mean()), 4)
        out["by_corpus"][corpus] = block
    # within-document spread on genuine composites
    comp = _doc_seg_frame(segs, docs, "composite")
    if len(comp):
        out["composite"] = {
            "n_docs": int(len(comp)),
            "mean_std": round(float(comp["std"].mean()), 4),
            "mean_max_minus_min": round(float((comp["max"] - comp["min"]).mean()), 4),
            "pearson_max_vs_doc": _corr(comp["max"], comp["p_doc"], "pearson"),
            "pearson_mean_vs_doc": _corr(comp["mean"], comp["p_doc"], "pearson"),
        }
    return out


def e7_aggregation_experiments(segs: pd.DataFrame, docs: pd.DataFrame) -> dict:
    """Each aggregation as a document-level predictor (inherited labels only)
    plus the extreme-segment influence test the spec asks for."""
    out: dict = {"by_corpus": {}, "extreme_influence": {}}
    for corpus in ("mage_test", "hc3"):
        f = _doc_seg_frame(segs, docs, corpus)
        if not len(f):
            continue
        block = {"n_docs": int(len(f)), "predictors": {}}
        for key in list(AGG_KEYS) + ["p_doc"]:
            block["predictors"][key] = eval_frame(f["y"], f[key], f["words"],
                                                  LABEL_INHERITED)
        out["by_corpus"][corpus] = block
    for corpus in ("mage_test", "hc3"):
        f = _doc_seg_frame(segs, docs, corpus)
        if not len(f):
            continue
        machine = f[f["y"] == 1]
        human = f[f["y"] == 0]
        out["extreme_influence"][corpus] = {
            "n_machine_docs": int(len(machine)),
            "n_human_docs": int(len(human)),
            "machine_docs_all_segments_machine": round(float(
                (machine["min"] >= 0.5).mean()), 4) if len(machine) else None,
            "machine_docs_mean_below_threshold": round(float(
                (machine["mean"] < 0.5).mean()), 4) if len(machine) else None,
            "human_docs_any_segment_machine": round(float(
                (human["max"] >= 0.5).mean()), 4) if len(human) else None,
            "human_docs_all_segments_below_threshold": round(float(
                (human["max"] < 0.5).mean()), 4) if len(human) else None,
            "human_docs_max_segment_mean_p": round(float(human["max"].mean()), 4)
            if len(human) else None,
        }
    return out


def e8_composites(segs: pd.DataFrame, docs: pd.DataFrame, cfg: dict) -> dict:
    """Mixed-composition documents with genuine per-segment labels.

    The composite has NO document label - it is used only to measure (a) how
    well segment p separates two known-provenance paragraphs inside one
    document, and (b) whether one segment dominates the document verdict.
    """
    comp = segs[(segs["corpus"] == "composite") & (segs["y_true"] >= 0)]
    dm = docs[docs["corpus"] == "composite"].set_index("doc_id")
    order = {d: str(dm.at[d, "extra"]) if d in dm.index else "" for d in comp["doc_id"].unique()}
    out: dict = {"label_semantics": LABEL_GENUINE,
                 "n_documents": int(comp["doc_id"].nunique()),
                 "n_segments": int(len(comp))}
    h = comp[comp["y_true"] == 0]["p"].to_numpy()
    m = comp[comp["y_true"] == 1]["p"].to_numpy()
    out["separation"] = {
        "mean_p_human_segment": round(float(h.mean()), 4),
        "mean_p_machine_segment": round(float(m.mean()), 4),
        "median_p_human_segment": round(float(np.median(h)), 4),
        "median_p_machine_segment": round(float(np.median(m)), 4),
        "margin_mean": round(float(m.mean() - h.mean()), 4),
        "share_machine_above_human": round(float((m > h).mean()), 4),
        "within_document_margin_mean": round(float(np.mean([
            comp[comp["doc_id"] == d]["p"].max()
            - comp[comp["doc_id"] == d]["p"].min()
            for d in comp["doc_id"].unique()])), 4),
    }
    # segment-level accuracy on the genuine labels
    y = comp["y_true"].to_numpy()
    p = comp["p"].to_numpy()
    out["segment_accuracy"] = eval_frame(y, p, comp["words"], LABEL_GENUINE)
    out["segment_accuracy"]["by_gate"] = {}
    for gate in cfg["gate_candidates"]:
        keep = comp["words"].to_numpy() >= gate
        if keep.sum() < 20:
            out["segment_accuracy"]["by_gate"][str(gate)] = {"n": int(keep.sum())}
            continue
        out["segment_accuracy"]["by_gate"][str(gate)] = eval_frame(
            y[keep], p[keep], comp["words"].to_numpy()[keep], LABEL_GENUINE)
    # order effect + document dominance
    rows = []
    for d, g in comp.groupby("doc_id"):
        hp = g[g["y_true"] == 0]["p"].to_numpy()
        mp = g[g["y_true"] == 1]["p"].to_numpy()
        if not len(hp) or not len(mp):
            continue
        rows.append({
            "doc_id": d, "order": order.get(d, ""),
            "p_human": float(hp.mean()), "p_machine": float(mp.mean()),
            "p_doc": float(dm.at[d, "p_doc"]) if d in dm.index else float("nan"),
        })
    df = pd.DataFrame(rows)
    if len(df):
        out["order_effect"] = {}
        for o in sorted(df["order"].unique()):
            sub = df[df["order"] == o]
            out["order_effect"][o] = {
                "n": int(len(sub)),
                "mean_p_doc": round(float(sub["p_doc"].mean()), 4),
                "doc_machine_rate": round(float((sub["p_doc"] >= 0.5).mean()), 4),
                "mean_margin": round(float((sub["p_machine"]
                                             - sub["p_human"]).mean()), 4),
            }
        out["document_dominance"] = {
            "share_doc_below_threshold": round(float((df["p_doc"] < 0.5).mean()), 4),
            "share_doc_in_mixed_band": round(float(
                ((df["p_doc"] >= 0.4) & (df["p_doc"] <= 0.6)).mean()), 4),
            "share_doc_above_threshold": round(float(
                (df["p_doc"] >= 0.5).mean()), 4),
            "share_machine_segment_above_threshold": round(float(
                (df["p_machine"] >= 0.5).mean()), 4),
            "share_human_segment_below_threshold": round(float(
                (df["p_human"] < 0.5).mean()), 4),
            "mean_p_human": round(float(df["p_human"].mean()), 4),
            "mean_p_machine": round(float(df["p_machine"].mean()), 4),
        }
    return out


def e9_robustness(segs: pd.DataFrame, docs: pd.DataFrame, pred: Predictor,
                  cfg: dict) -> dict:
    out: dict = {}
    # synthetic probes
    probe_rows = []
    probe_p = pred.score_texts(list(PROBES.values()))
    for (name, text), p in zip(PROBES.items(), probe_p):
        segs_probe = segment_text(text, SegmentationPolicy(**cfg["policy"]))
        probe_rows.append({
            "probe": name, "p_machine": round(float(p), 4),
            "words": word_count(text),
            "n_segments": len(segs_probe),
            "min_segment_words": min((s.word_count for s in segs_probe),
                                     default=0),
            "repetition_rate": repetition_rate(text),
        })
    out["synthetic_probes"] = probe_rows
    # per-attack (RAID) segment behaviour, inherited labels
    raid = segs[segs["corpus"] == "raid"].copy()
    if len(raid):
        dm = docs[docs["corpus"] == "raid"].set_index("doc_id")["y"]
        raid["y"] = raid["doc_id"].map(dm)
        out["raid_by_attack"] = {}
        for attack, g in raid.groupby("extra"):
            if len(g) < 20:
                continue
            out["raid_by_attack"][str(attack)] = eval_frame(
                g["y"], g["p"], g["words"], LABEL_INHERITED)
    # OOD corpora
    out["ood_segments"] = {}
    for name in ("ood_gpt", "ood_gpt_para", "hc3"):
        sub = segs[segs["corpus"] == name]
        if not len(sub):
            continue
        dm = docs[docs["corpus"] == name].set_index("doc_id")["y"]
        out["ood_segments"][name] = eval_frame(
            sub["doc_id"].map(dm), sub["p"], sub["words"], LABEL_INHERITED)
    # repetition slice (text-level proxy)
    sub = segs[segs["corpus"] == "mage_test"].copy()
    sub["rep"] = sub["text"].map(repetition_rate)
    thr = float(np.percentile(sub["rep"], 90))
    dm = docs[docs["corpus"] == "mage_test"].set_index("doc_id")["y"]
    sub["y"] = sub["doc_id"].map(dm)
    hi = sub[sub["rep"] >= thr]
    lo = sub[sub["rep"] < thr]
    out["repetition_slice"] = {
        "rep_p90": round(thr, 4),
        "high_repetition": {
            "n": int(len(hi)), "mean_p": round(float(hi["p"].mean()), 4),
            "share_pred_machine": round(float((hi["p"] >= 0.5).mean()), 4),
            "human_fpr": (round(float((hi.loc[hi["y"] == 0, "p"] >= 0.5).mean()), 4)
                          if (hi["y"] == 0).any() else None),
            "label_semantics": LABEL_INHERITED,
        },
        "low_repetition": {
            "n": int(len(lo)), "mean_p": round(float(lo["p"].mean()), 4),
            "share_pred_machine": round(float((lo["p"] >= 0.5).mean()), 4),
            "human_fpr": (round(float((lo.loc[lo["y"] == 0, "p"] >= 0.5).mean()), 4)
                          if (lo["y"] == 0).any() else None),
            "label_semantics": LABEL_INHERITED,
        },
    }
    # length sensitivity on the segment level (inherited labels, main corpus)
    out["length_sensitivity"] = {
        b: {"n": int(mask.sum()),
            "mean_p": round(float(sub.loc[mask, "p"].mean()), 4),
            "share_pred_machine": round(float(
                (sub.loc[mask, "p"] >= 0.5).mean()), 4)}
        for b, mask in ((b, np.array([bucket_of(int(w)) == b
                                      for w in sub["words"]]))
                        for b in BUCKET_NAMES)
        if mask.sum() > 0}
    return out


def e10_explainability(segs: pd.DataFrame, pred: Predictor, n: int = 30,
                       seed: int = 0) -> dict:
    """Per-segment signals: populated, finite, direction-consistent, non-causal."""
    forbidden = ["proves", "prove that", "this is ai", "written by ai",
                 "definitely ai", "guarantees"]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(segs), size=min(n, len(segs)), replace=False)
    sample = segs.iloc[idx]
    ok_signals = 0
    finite = 0
    direction_ok = 0
    causal_hits: list[str] = []
    empty = 0
    rows = []
    for _, r in sample.iterrows():
        res = pred.analyze(r["text"], include_signals=True, max_signals=4)
        if not res.signals:
            empty += 1
            continue
        ok_signals += 1
        if all(np.isfinite(s.contribution) for s in res.signals):
            finite += 1
        machine_dir = [s for s in res.signals if s.direction == "machine"]
        human_dir = [s for s in res.signals if s.direction == "human"]
        if (res.machine_probability >= 0.8 and machine_dir
                and len(machine_dir) >= len(human_dir)):
            direction_ok += 1
        for s in res.signals:
            blob = f"{s.phrase} {s.interpretation}".lower()
            for bad in forbidden:
                if bad in blob:
                    causal_hits.append(bad)
        rows.append({"words": int(r["words"]), "p_machine": round(
            res.machine_probability, 4), "n_signals": len(res.signals),
            "top": [s.feature for s in res.signals[:3]]})
    return {
        "sampled_segments": int(len(sample)),
        "with_signals": ok_signals,
        "without_signals": empty,
        "all_contributions_finite": finite == ok_signals,
        "direction_consistent_high_p": direction_ok,
        "causal_language_hits": sorted(set(causal_hits)),
        "sample_rows": rows[:10],
    }


def e11_decision(e3: dict, e4: dict, e5: dict, e8: dict, cfg: dict) -> dict:
    """Verdict: reuse final-v1 behind a gate, or train a separate segments-v1.

    The verdict is computed from the declared criteria (DECISION) applied to the
    measured kept-slice / genuine / OOD numbers. It records WHY, including the
    specific criteria each candidate gate failed, so the decision is auditable
    and not a vibe.
    """
    genuine = e8.get("segment_accuracy", {})
    chosen_gate = e5.get("chosen_segment_gate_min_words")
    mage_segments = e3.get("by_corpus", {}).get("mage_test", {})
    mage_document = e3.get("document_level_reference", {}).get("mage_test", {})
    cand = next((c for c in e5.get("candidates", [])
                 if c["segment_gate_min_words"] == chosen_gate), None)
    kept = (cand or {}).get("mage_test", {}).get("kept", {}) or {}
    seg_auc = kept.get("roc_auc")
    doc_auc = mage_document.get("roc_auc")
    auc_drop = (round(doc_auc - seg_auc, 4)
                if isinstance(doc_auc, float) and isinstance(seg_auc, float)
                else None)
    reuse_ok = chosen_gate is not None
    train_ok = not reuse_ok

    # Which criteria did the best candidate fail, and why (audit trail)?
    per_gate_failures = {}
    for c in e5.get("candidates", []):
        failed = [k for k, v in (c.get("criteria_checks") or {}).items() if not v]
        per_gate_failures[str(c["segment_gate_min_words"])] = failed

    measured = {
        "kept_gate": chosen_gate,
        "kept_segment_auc_mage_test": seg_auc,
        "kept_segment_accuracy_mage_test": kept.get("accuracy"),
        "kept_segment_human_fpr_mage_test": kept.get("human_fpr"),
        "kept_segment_ece_mage_test": kept.get("ece_10"),
        "document_auc_mage_test": doc_auc,
        "auc_drop": auc_drop,
        "genuine_label_segment_auc": genuine.get("roc_auc"),
        "genuine_label_segment_accuracy": genuine.get("accuracy"),
        "genuine_label_segment_human_fpr": genuine.get("human_fpr"),
        "composite_document_below_threshold_rate": e8.get(
            "document_dominance", {}).get("share_doc_below_threshold"),
        "best_gate_kept_share": (cand or {}).get("mage_test", {}).get("kept_share"),
    }
    if reuse_ok:
        decision = ("reuse final-v1 for segments behind an evidence gate of "
                    f"{chosen_gate} words; no new model")
    else:
        decision = ("final-v1 not defensible at segment level; a dedicated "
                    "segments-v1 model is required")
    return {
        "declared_criteria": DECISION,
        "measured": measured,
        "reuse_final_v1_supported": reuse_ok,
        "train_segments_v1_justified": train_ok,
        "decision": decision,
        "criteria_failed_per_gate": per_gate_failures,
        "caveat": ("If segments-v1 is trained, it is a SEPARATE artifact with its "
                   "own config/calibrator/manifest/eval and is selected on "
                   "generalization + calibration, never on in-distribution "
                   "accuracy alone. final-v1 and baseline-v1 stay untouched."),
    }

# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run(cfg: dict | None = None, workers: int | None = None,
        force: bool = False, only: list[str] | None = None,
        verbose: bool = True) -> dict:
    provided = json.loads(json.dumps(cfg)) if cfg else {}
    base = load_json(CONFIG_PATH) if CONFIG_PATH.exists() else default_config()
    base.update(provided)                     # explicit args win over file
    cfg = base
    workers = int(workers if workers is not None else cfg.get("workers", 0))
    policy = SegmentationPolicy(**cfg["policy"])
    t_start = time.perf_counter()

    pred = Predictor(FINAL_DIR)
    warmup_ms = pred.warmup()

    cache_path = CACHE_DIR / "scored_segments.parquet"
    meta_path = CACHE_DIR / "scored_segments.meta.json"
    key = _cache_key(cfg, policy)
    segs = docs = None
    cov: dict = {}
    scoring_s = None
    if cache_path.exists() and meta_path.exists() and not force:
        try:
            meta = load_json(meta_path)
            if meta.get("key") == key:
                blob = pd.read_parquet(cache_path)
                segs = blob["segments"].reset_index(drop=True)
                docs = blob["documents"].reset_index(drop=True)
                cov = meta.get("coverage", {})
                scoring_s = meta.get("scoring_s")
        except Exception:
            segs = None
    if segs is None:
        segs, docs, build = _build_tables(cfg, policy, pred, workers)
        cov = build["coverage"]
        scoring_s = build["scoring_s"]
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        pd.concat([segs.assign(_table="segments"),
                   docs.assign(_table="documents")], axis=0,
                  ignore_index=True, sort=False).to_parquet(cache_path,
                                                            index=False)
        dump_json(meta_path, {"key": key, "coverage": cov, "scoring_s": scoring_s,
                              "written_at": _now()})

    manifest = pred.manifest
    results: dict = {
        "meta": {
            "generated_at": _now(),
            "pipeline_version": cfg["pipeline_version"],
            "artifact": pred.model_version,
            "winner": manifest.get("winner"),
            "calibrator": manifest.get("calibrator"),
            "threshold": manifest.get("threshold"),
            "seed": cfg["seed"],
            "config": cfg,
            "policy": policy.as_dict(),
            "workers": workers,
            "warmup_ms": warmup_ms,
            "scoring_seconds": scoring_s,
            "total_seconds": None,
            "python": sys.version.split()[0],
        },
        "path_parity": _path_parity(pred, segs, docs, seed=int(cfg["seed"])),
    }

    def maybe(key: str, fn):
        if only and key not in only:
            return
        t0 = time.perf_counter()
        results[key] = fn()
        if verbose:
            print(f"  {key} done in {time.perf_counter() - t0:.1f}s")

    maybe("E1_inventory", lambda: e1_inventory(segs, docs, cov))
    maybe("E2_length_analysis", lambda: e2_length_analysis(segs))
    maybe("E3_segment_eval", lambda: e3_segment_eval(segs, docs))
    maybe("E4_calibration", lambda: e4_calibration(segs, docs, manifest))
    e3 = results.get("E3_segment_eval") or e3_segment_eval(segs, docs)
    doc_auc = ((e3.get("document_level_reference", {})
                .get("mage_test", {}) or {}).get("roc_auc"))
    maybe("E5_gate_policy",
          lambda: e5_gate_policy(segs, docs, cfg, doc_auc))
    maybe("E6_consistency", lambda: e6_consistency(segs, docs))
    maybe("E7_aggregation_experiments",
          lambda: e7_aggregation_experiments(segs, docs))
    maybe("E8_composites", lambda: e8_composites(segs, docs, cfg))
    maybe("E9_robustness", lambda: e9_robustness(segs, docs, pred, cfg))
    maybe("E10_explainability", lambda: e10_explainability(segs, pred,
                                                          seed=int(cfg["seed"])))
    e5 = results.get("E5_gate_policy") or e5_gate_policy(segs, docs, cfg, doc_auc)
    e8 = results.get("E8_composites") or e8_composites(segs, docs, cfg)
    e4 = results.get("E4_calibration") or e4_calibration(segs, docs, manifest)
    maybe("E11_decision", lambda: e11_decision(e3, e4, e5, e8, cfg))

    results["meta"]["total_seconds"] = round(time.perf_counter() - t_start, 1)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dump_json(RESULTS_PATH, results)
    return results


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Run the Phase-5A segment study")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="ignore the scored-segment cache")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these blocks (E1..E11)")
    args = ap.parse_args(argv)
    res = run(workers=args.workers, force=args.force, only=args.only)
    print(f"results -> {RESULTS_PATH}")
    inv = res.get("E1_inventory", {})
    print(f"  documents={inv.get('n_documents_scored')} "
          f"segments={inv.get('n_segments_scored')}")
    dec = res.get("E11_decision", {})
    print(f"  decision: {dec.get('decision')}")
    print(f"  segment gate: {dec.get('measured', {}).get('chosen_segment_gate_min_words')}")
    print(f"  parity: {res.get('path_parity')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
