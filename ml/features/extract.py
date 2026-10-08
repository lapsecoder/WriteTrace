"""Deterministic, named feature extraction (v1) for WriteTrace.

~70 scalar features across families from docs/technical-specification.md §4:
structure, lexical, punctuation, stylometric lexicons, repetition,
clause-density proxies (syntactic family without a parser), readability,
positional. spaCy POS/parse features are a Phase-2 optional add-on (feature-
flagged OFF in v1) — documented in the baseline report.

FeatureState holds corpus statistics fitted on the TRAINING split only
(document frequency for rare-word features). Output columns are named and
stable; invalid input yields NaNs, never a crash.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from . import lexicons as LX
from ..preprocessing.normalize import WORD_RE

FEATURE_VERSION = "features_v1"
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])(?:[\"'”’)]*)\s+(?=[A-Z0-9\"“‘(])")
_MARKDOWN_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.M)
_HEADER_RE = re.compile(r"^\s*#{1,6}\s", re.M)
_NUMERIC_TOKEN_RE = re.compile(r"^\d+(?:[.,]\d+)?$")
_URL_TOKEN = "[URL]"
_EMAIL_TOKEN = "[EMAIL]"

# --- sentence / token helpers ----------------------------------------------

def split_sentences(text: str) -> list[str]:
    """Heuristic sentence segmentation (no parser in v1; documented limitation):
    split after . ! ? (with optional closing quotes) when followed by a capital,
    digit or opening quote; newlines always break sentences."""
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    sents: list[str] = []
    for part in parts:
        for line in part.split("\n"):
            line = line.strip()
            if line:
                sents.append(line)
    return sents


def _words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def _syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 1
    count, prev_vowel = 0, False
    for ch in w:
        is_v = ch in "aeiouy"
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    if w.endswith("e") and count > 1 and not w.endswith(("le", "ee")):
        count -= 1
    return max(count, 1)


@lru_cache(maxsize=8)
def _phrase_regex(phrases: tuple[str, ...]) -> re.Pattern:
    body = "|".join(re.escape(p) for p in phrases)
    return re.compile(rf"\b(?:{body})\b")


_TRANS_PHRASE_RE = _phrase_regex(tuple(LX.TRANSITIONS_PHRASES))
_BOILER_PHRASE_RE = _phrase_regex(tuple(LX.BOILERPLATE_PHRASES))


def _mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def _std(xs: Sequence[float]) -> float:
    return float(np.std(xs)) if xs else float("nan")


def _median(xs: Sequence[float]) -> float:
    return float(np.median(xs)) if xs else float("nan")


def _cv(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    m = float(np.mean(xs))
    if m == 0:
        return float("nan")
    return float(np.std(xs)) / m


def _pearson_r(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    x, y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rate(count: float, denom: float, scale: float = 1000.0) -> float:
    return float(count) / denom * scale if denom > 0 else float("nan")


def _nan_row() -> dict[str, float]:
    return {name: float("nan") for name in feature_names()}


# --- feature families (stable, named columns) -------------------------------

_PRON_SETS: dict[str, frozenset] = {
    "pron_i": frozenset("i me my mine myself".split()),
    "pron_you": frozenset("you your yours yourself yourselves".split()),
    "pron_we": frozenset("we us our ours ourselves".split()),
    "pron_they": LX.THIRD_PLURAL,
    "pron_heshe": LX.HE_SHE,
    "pron_it": frozenset(["it", "its", "itself"]),
}

_ALL_SETS: dict[str, frozenset] = {
    **_PRON_SETS,
    "first_person": LX.FIRST_PERSON,
    "second_person": LX.SECOND_PERSON,
    "hedge": LX.HEDGES,
    "certainty": LX.CERTAINTY,
    "transition": LX.TRANSITIONS_SINGLE,
    "boiler_word": LX.BOILERPLATE_WORDS,
    "enum": LX.ENUM_MARKERS,
    "subordinator": LX.SUBORDINATORS,
    "coordinator": LX.COORDINATORS,
    "relative": LX.RELATIVE_PRONOUNS,
    "aux": LX.AUXILIARIES,
}

FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "structure": (
        "n_chars", "n_chars_no_space", "n_words", "n_sentences", "n_paragraphs",
        "avg_word_len", "std_word_len", "median_word_len",
        "avg_sent_len", "median_sent_len", "std_sent_len", "cv_sent_len",
        "min_sent_len", "max_sent_len", "first_sent_len", "last_sent_len",
        "avg_para_len", "cv_para_len",
    ),
    "lexical": (
        "ttr", "root_ttr", "log_ttr", "hapax_ratio", "rare_token_ratio",
        "mean_log_df", "stopword_ratio", "contraction_rate",
        "uppercase_word_ratio", "digit_word_ratio",
    ),
    "punctuation": (
        "comma_rate", "semicolon_rate", "colon_rate", "em_dash_rate",
        "en_dash_rate", "ellipsis_rate", "quote_rate", "paren_rate",
        "qmark_rate", "exclaim_rate", "pct_rate", "hash_rate", "hyphen_rate",
        "bullet_rate", "digit_char_ratio",
    ),
    "stylometric": (
        "pron_i", "pron_you", "pron_we", "pron_they", "pron_heshe", "pron_it",
        "first_person_rate", "second_person_rate", "hedge_rate", "certainty_rate",
        "transition_rate", "connective_phrase_rate", "boilerplate_rate",
        "enum_rate", "question_rate_per_sent", "exclaim_rate_per_sent",
    ),
    "repetition": (
        "trigram_repeat_rate", "unique_bigram_ratio", "max_token_repeat_ratio",
    ),
    "syntactic": (
        "clause_density", "subordinator_rate", "coordinator_rate",
        "relative_clause_rate", "aux_rate",
    ),
    "readability": (
        "flesch_reading_ease", "flesch_kincaid_grade", "avg_syllables_per_word",
        "polysyllable_ratio",
    ),
    "positional": (
        "sent_len_trend", "header_rate",
    ),
}


def feature_names() -> list[str]:
    """Stable, ordered feature column names."""
    return [n for names in FEATURE_FAMILIES.values() for n in names]


_NAME_SET = frozenset(feature_names())

class FeatureState:
    """Corpus statistics fitted on the TRAINING split only (docs/methodology.md).

    v1 uses document frequencies derived from the training corpus instead of an
    external frequency list (documented deviation: fully reproducible, no extra
    data assets)."""

    def __init__(self, df: dict[str, int], n_docs: int, rare_max_df: int):
        self.df = df
        self.n_docs = n_docs
        self.rare_max_df = rare_max_df

    @classmethod
    def fit(cls, texts: Iterable[str], rare_frac: float = 0.001) -> "FeatureState":
        df_counts: Counter = Counter()
        n_docs = 0
        for t in texts:
            if not isinstance(t, str) or not t.strip():
                continue
            n_docs += 1
            df_counts.update(set(w.lower() for w in WORD_RE.findall(t)))
        rare_max = max(1, int(round(rare_frac * max(n_docs, 1))))
        return cls(dict(df_counts), n_docs, rare_max)

    def df_of(self, token: str) -> int:
        return self.df.get(token, 0)


def extract_one(text, state: FeatureState | None = None) -> dict[str, float]:
    """Extract all v1 features for one document. Invalid/empty input -> all-NaN."""
    if not isinstance(text, str) or not text.strip():
        return _nan_row()
    n_chars = len(text)
    words = _words(text)
    n_words = len(words)
    if n_words < 2:
        return _nan_row()
    lower = [w.lower() for w in words]
    sents = split_sentences(text) or [text]
    sent_lens = [len(_words(s)) for s in sents]
    paras = [p for p in text.split("\n\n") if p.strip()] or [text]
    para_lens = [len(_words(p)) for p in paras]
    word_lens = [len(w) for w in words]
    n_s = len(sents)
    low_join = " ".join(lower)

    f: dict[str, float] = {}
    # structure
    f.update(
        n_chars=float(n_chars),
        n_chars_no_space=float(n_chars - text.count(" ")),
        n_words=float(n_words),
        n_sentences=float(n_s),
        n_paragraphs=float(len(paras)),
        avg_word_len=_mean(word_lens), std_word_len=_std(word_lens),
        median_word_len=_median(word_lens),
        avg_sent_len=_mean(sent_lens), median_sent_len=_median(sent_lens),
        std_sent_len=_std(sent_lens), cv_sent_len=_cv(sent_lens),
        min_sent_len=float(min(sent_lens)), max_sent_len=float(max(sent_lens)),
        first_sent_len=float(sent_lens[0]), last_sent_len=float(sent_lens[-1]),
        avg_para_len=_mean(para_lens), cv_para_len=_cv(para_lens),
    )
    # lexical
    tok_counts = Counter(lower)
    unique = len(tok_counts)
    f["ttr"] = unique / n_words
    f["root_ttr"] = unique / math.sqrt(n_words)
    f["log_ttr"] = math.log(unique) / math.log(n_words) if unique > 1 else float("nan")
    f["hapax_ratio"] = sum(1 for c in tok_counts.values() if c == 1) / n_words
    if state is not None and state.n_docs > 0:
        dfs = [state.df_of(t) for t in lower]
        f["rare_token_ratio"] = sum(1 for d in dfs if d < state.rare_max_df) / n_words
        f["mean_log_df"] = float(np.mean([math.log10(1 + d) for d in dfs]))
    else:
        f["rare_token_ratio"] = float("nan")
        f["mean_log_df"] = float("nan")
    f["stopword_ratio"] = sum(1 for t in lower if t in LX.STOPWORDS) / n_words
    f["contraction_rate"] = sum(1 for t in lower if t in LX.CONTRACTIONS) / n_words
    f["uppercase_word_ratio"] = sum(
        1 for w in words if len(w) > 1 and w.isupper() and w not in ("URL", "EMAIL")
    ) / n_words
    f["digit_word_ratio"] = sum(1 for t in lower if _NUMERIC_TOKEN_RE.match(t)) / n_words

    # punctuation (per 1,000 chars)
    f.update(
        comma_rate=_rate(text.count(","), n_chars),
        semicolon_rate=_rate(text.count(";"), n_chars),
        colon_rate=_rate(text.count(":"), n_chars),
        em_dash_rate=_rate(text.count("\u2014"), n_chars),
        en_dash_rate=_rate(text.count("\u2013"), n_chars),
        ellipsis_rate=_rate(text.count("...") + text.count("\u2026"), n_chars),
        quote_rate=_rate(text.count('"') + text.count("\u201c") + text.count("\u201d"), n_chars),
        paren_rate=_rate(text.count("(") + text.count(")"), n_chars),
        qmark_rate=_rate(text.count("?"), n_chars),
        exclaim_rate=_rate(text.count("!"), n_chars),
        pct_rate=_rate(text.count("%"), n_chars),
        hash_rate=_rate(text.count("#"), n_chars),
        hyphen_rate=_rate(text.count("-"), n_chars),
        bullet_rate=_rate(len(_MARKDOWN_LIST_RE.findall(text)), n_chars),
        digit_char_ratio=sum(1 for ch in text if ch.isdigit()) / n_chars,
    )
    # stylometric lexicons (per 1,000 words)
    hits: Counter = Counter()
    for t in lower:
        for name, sset in _ALL_SETS.items():
            if t in sset:
                hits[name] += 1
    trans_phrases = len(_TRANS_PHRASE_RE.findall(low_join))
    boiler_phrases = len(_BOILER_PHRASE_RE.findall(low_join))
    for name in _PRON_SETS:
        f[name] = _rate(hits[name], n_words)
    f.update(
        first_person_rate=_rate(hits["first_person"], n_words),
        second_person_rate=_rate(hits["second_person"], n_words),
        hedge_rate=_rate(hits["hedge"], n_words),
        certainty_rate=_rate(hits["certainty"], n_words),
        transition_rate=_rate(hits["transition"], n_words),
        connective_phrase_rate=_rate(trans_phrases, n_words),
        boilerplate_rate=_rate(boiler_phrases + hits["boiler_word"], n_words),
        enum_rate=_rate(hits["enum"], n_words),
        question_rate_per_sent=_rate(text.count("?"), n_s),
        exclaim_rate_per_sent=_rate(text.count("!"), n_s),
    )

    # repetition
    bigrams = list(zip(lower, lower[1:]))
    trigrams = list(zip(lower, lower[1:], lower[2:]))
    f["trigram_repeat_rate"] = (1 - len(set(trigrams)) / len(trigrams)) if trigrams else float("nan")
    f["unique_bigram_ratio"] = (len(set(bigrams)) / len(bigrams)) if bigrams else float("nan")
    f["max_token_repeat_ratio"] = max(tok_counts.values()) / n_words

    # syntactic proxies (clause density & function words; no parser in v1)
    f["clause_density"] = (hits["subordinator"] + hits["coordinator"]) / n_s
    f["subordinator_rate"] = _rate(hits["subordinator"], n_words)
    f["coordinator_rate"] = _rate(hits["coordinator"], n_words)
    f["relative_clause_rate"] = hits["relative"] / n_s
    f["aux_rate"] = _rate(hits["aux"], n_words)

    # readability
    syls = [_syllables(w) for w in words]
    syl_total = sum(syls)
    f["avg_syllables_per_word"] = syl_total / n_words
    f["flesch_reading_ease"] = 206.835 - 1.015 * (n_words / n_s) - 84.6 * (syl_total / n_words)
    f["flesch_kincaid_grade"] = 0.39 * (n_words / n_s) + 11.8 * (syl_total / n_words) - 15.59
    f["polysyllable_ratio"] = sum(1 for s in syls if s >= 3) / n_words

    # positional
    f["sent_len_trend"] = _pearson_r(list(range(len(sent_lens))), sent_lens)
    f["header_rate"] = _rate(len(_HEADER_RE.findall(text)), n_words)

    if set(f) != _NAME_SET:  # guard against name/computation drift
        raise RuntimeError(f"feature set mismatch: {set(f) ^ _NAME_SET}")
    return f


# --- parallel extraction (spawn-safe) --------------------------------------
_WORKER_STATE: FeatureState | None = None


def _init_worker(state: FeatureState | None) -> None:
    global _WORKER_STATE
    _WORKER_STATE = state


def _work(text) -> dict[str, float]:
    return extract_one(text, _WORKER_STATE)


def extract_many(texts: Sequence[str], state: FeatureState | None = None,
                 workers: int = 0) -> pd.DataFrame:
    """Deterministic DataFrame with stable named columns (order = feature_names())."""
    if workers and workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker, initargs=(state,)) as pool:
            rows = list(pool.map(_work, list(texts), chunksize=64))
    else:
        rows = [extract_one(t, state) for t in texts]
    return pd.DataFrame(rows, columns=feature_names())



