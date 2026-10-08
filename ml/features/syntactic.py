"""Syntactic feature family (spacy en_core_web_sm) - Phase 3 Experiment 2 ONLY.

Phase 1 deliberately kept spaCy POS/parse features OFF (features_v1 has a
regex-based syntactic PROXY family instead). This module adds a separate,
explicitly-flagged family (`syntactic_spacy`, 16 features, `syn_` prefix) that
is never part of `features_v1`:

- 9 universal-POS ratios (noun/pron/verb/adj/adv/det/adp/sconj/cconj per 1k words)
- POS Shannon entropy
- clause rate (advcl/ccomp/xcomp/relcl/acl per sentence)
- mark rate (dependency `mark` per 100 words)
- parse-depth mean/std (token -> root hops)
- passive rate (deps containing "pass" per sentence)
- mean head-dependent distance

Deterministic given (text, model): inference is CPU-only, no sampling.
Invalid/short input yields all-NaN, mirroring extract_one's contract.
Extraction cost is recorded by the caller (Experiment 2 must report it).
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

SYNTACTIC_VERSION = "syntactic_spacy_v1"

SYNTACTIC_FEATURES: tuple[str, ...] = (
    "syn_pos_noun_ratio", "syn_pos_pron_ratio", "syn_pos_verb_ratio",
    "syn_pos_adj_ratio", "syn_pos_adv_ratio", "syn_pos_det_ratio",
    "syn_pos_adp_ratio", "syn_pos_sconj_ratio", "syn_pos_cconj_ratio",
    "syn_pos_entropy",
    "syn_clause_rate", "syn_mark_rate",
    "syn_depth_mean", "syn_depth_std",
    "syn_passive_rate", "syn_dep_len_mean",
)

_CLAUSE_DEPS = frozenset({"advcl", "ccomp", "xcomp", "relcl", "acl"})
_MIN_WORDS = 2
_MAX_DEPTH = 512


def _nan_row() -> dict[str, float]:
    return {n: float("nan") for n in SYNTACTIC_FEATURES}


_NLP_CACHE: dict[tuple, object] = {}


def _load_nlp(disable=("ner", "lemmatizer")):
    """Load en_core_web_sm once per process (Phase-4: inference must not
    reload the pipeline per request). Values are identical either way - the
    model on disk is fixed and `nlp.pipe` is deterministic."""
    key = tuple(disable)
    nlp = _NLP_CACHE.get(key)
    if nlp is None:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=list(disable))
        _NLP_CACHE[key] = nlp
    return nlp


def _token_depth(tok) -> int:
    d = 0
    t = tok
    while t.head.i != t.i and d < _MAX_DEPTH:
        t = t.head
        d += 1
    return d


def features_from_doc(doc) -> dict[str, float]:
    """Vector for one spaCy Doc (all-NaN for <2 words, like extract_one)."""
    words = [t for t in doc if not t.is_space]
    n_words = len(words)
    if n_words < _MIN_WORDS:
        return _nan_row()
    pos_counts = Counter(t.pos_ for t in words)
    n = n_words
    inv = 1000.0 / n
    f: dict[str, float] = {
        "syn_pos_noun_ratio": pos_counts.get("NOUN", 0) * inv,
        "syn_pos_pron_ratio": pos_counts.get("PRON", 0) * inv,
        "syn_pos_verb_ratio": pos_counts.get("VERB", 0) * inv,
        "syn_pos_adj_ratio": pos_counts.get("ADJ", 0) * inv,
        "syn_pos_adv_ratio": pos_counts.get("ADV", 0) * inv,
        "syn_pos_det_ratio": pos_counts.get("DET", 0) * inv,
        "syn_pos_adp_ratio": pos_counts.get("ADP", 0) * inv,
        "syn_pos_sconj_ratio": pos_counts.get("SCONJ", 0) * inv,
        "syn_pos_cconj_ratio": pos_counts.get("CCONJ", 0) * inv,
    }
    ent = 0.0
    for c in pos_counts.values():
        p = c / n
        ent -= p * math.log2(p)
    f["syn_pos_entropy"] = ent
    n_sents = max(sum(1 for _ in doc.sents), 1)
    dep_counts = Counter(t.dep_ for t in words)
    n_clause = sum(dep_counts.get(d, 0) for d in _CLAUSE_DEPS)
    f["syn_clause_rate"] = n_clause / n_sents
    f["syn_mark_rate"] = dep_counts.get("mark", 0) * inv
    depths = np.fromiter((_token_depth(t) for t in words), dtype=float, count=n)
    f["syn_depth_mean"] = float(depths.mean())
    f["syn_depth_std"] = float(depths.std())
    passive = sum(1 for t in words if "pass" in t.dep_)
    f["syn_passive_rate"] = passive / n_sents
    dists = [abs(t.i - t.head.i) for t in doc if t.head.i != t.i]
    f["syn_dep_len_mean"] = float(np.mean(dists)) if dists else float("nan")
    if set(f) != set(SYNTACTIC_FEATURES):
        raise RuntimeError(f"syntactic set mismatch: {set(f) ^ set(SYNTACTIC_FEATURES)}")
    return f


def extract_syntactic(texts: list[str], n_process: int = 0,
                      batch_size: int = 256, disable=("ner", "lemmatizer")) -> pd.DataFrame:
    """Deterministic DataFrame with SYNTACTIC_FEATURES columns (input order)."""
    nlp = _load_nlp(disable=disable)
    rows: list[dict[str, float]] = []
    kwargs = {"batch_size": batch_size}
    if n_process and n_process > 1:
        kwargs["n_process"] = n_process
    for doc in nlp.pipe(texts, **kwargs):
        rows.append(features_from_doc(doc))
    return pd.DataFrame(rows, columns=list(SYNTACTIC_FEATURES))
