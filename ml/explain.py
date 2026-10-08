"""Explainability for the WriteTrace final model.

Global: permutation importance is produced by Phase-3 E6 (stored in the
artifact's importance.json); SHAP/xgboost/lightgbm are not installed in this
environment - documented deviation, nothing is fabricated.

Local: one-at-a-time reference perturbation. For each feature, its value is
replaced by the MAGE-train median and the calibrated p(machine) delta is
recorded:

    contribution_j = p(x) - p(x with feature_j set to train median)

Positive contribution = the feature's observed value pushed the prediction
toward MACHINE relative to a typical training document; negative = toward
HUMAN. This is a model-behavior description, NOT a causal claim about
authorship, and contributions do not sum exactly to p (interactions).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .experiments.phase3 import FINAL_DIR, load_json
from .preprocessing.normalize import normalize_text, word_count

FEATURE_PHRASES: dict[str, str] = {
    # structure
    "n_chars": "document length in characters",
    "n_chars_no_space": "document length excluding spaces",
    "n_words": "document length in words",
    "n_sentences": "number of sentences",
    "n_paragraphs": "number of paragraphs",
    "avg_word_len": "average word length",
    "std_word_len": "word-length variation",
    "median_word_len": "median word length",
    "avg_sent_len": "average sentence length",
    "median_sent_len": "median sentence length",
    "std_sent_len": "sentence-length variation",
    "cv_sent_len": "sentence-length uniformity (coefficient of variation)",
    "min_sent_len": "shortest sentence length",
    "max_sent_len": "longest sentence length",
    "first_sent_len": "first-sentence length",
    "last_sent_len": "last-sentence length",
    "avg_para_len": "average paragraph length",
    "cv_para_len": "paragraph-length uniformity",
    # lexical
    "ttr": "type-token ratio (vocabulary richness)",
    "root_ttr": "root type-token ratio",
    "log_ttr": "log type-token ratio",
    "hapax_ratio": "share of words used only once",
    "rare_token_ratio": "share of rare words (train document frequency)",
    "mean_log_df": "mean word document frequency in training corpus",
    "stopword_ratio": "stopword share",
    "contraction_rate": "contraction usage",
    "uppercase_word_ratio": "uppercase-word share",
    "digit_word_ratio": "numeric-word share",
    # punctuation
    "comma_rate": "comma usage",
    "semicolon_rate": "semicolon usage",
    "colon_rate": "colon usage",
    "em_dash_rate": "em-dash usage",
    "en_dash_rate": "en-dash usage",
    "ellipsis_rate": "ellipsis usage",
    "quote_rate": "quotation-mark usage",
    "paren_rate": "parenthesis usage",
    "qmark_rate": "question-mark usage",
    "exclaim_rate": "exclamation-mark usage",
    "pct_rate": "percent-sign usage",
    "hash_rate": "hashtag/heading-mark usage",
    "hyphen_rate": "hyphen usage",
    "bullet_rate": "bullet/list-marker usage",
    "digit_char_ratio": "digit share of characters",
    # stylometric
    "pron_i": "first-person singular pronouns",
    "pron_you": "second-person pronouns",
    "pron_we": "first-person plural pronouns",
    "pron_they": "third-person plural pronouns",
    "pron_heshe": "gendered pronouns",
    "pron_it": "impersonal 'it' pronouns",
    "first_person_rate": "first-person narration",
    "second_person_rate": "second-person address",
    "hedge_rate": "hedging language",
    "certainty_rate": "certainty language",
    "transition_rate": "transition phrases",
    "connective_phrase_rate": "connective phrases",
    "boilerplate_rate": "boilerplate/formulaic phrases",
    "enum_rate": "enumeration patterns",
    "question_rate_per_sent": "questions per sentence",
    "exclaim_rate_per_sent": "exclamations per sentence",
    # repetition
    "trigram_repeat_rate": "trigram repetition",
    "unique_bigram_ratio": "bigram uniqueness",
    "max_token_repeat_ratio": "maximum token repetition",
    # syntactic proxy (regex, part of features_v1)
    "clause_density": "clause density",
    "subordinator_rate": "subordinating conjunctions",
    "coordinator_rate": "coordinating conjunctions",
    "relative_clause_rate": "relative clauses",
    "aux_rate": "auxiliary verbs",
    # readability
    "flesch_reading_ease": "Flesch reading ease",
    "flesch_kincaid_grade": "Flesch-Kincaid grade level",
    "avg_syllables_per_word": "syllables per word",
    "polysyllable_ratio": "polysyllable share",
    # positional
    "sent_len_trend": "sentence-length trend across the document",
    "header_rate": "markdown heading usage",
    # syntactic (spacy, Phase-3 optional family)
    "syn_pos_noun_ratio": "noun density (spaCy)",
    "syn_pos_pron_ratio": "pronoun density (spaCy)",
    "syn_pos_verb_ratio": "verb density (spaCy)",
    "syn_pos_adj_ratio": "adjective density (spaCy)",
    "syn_pos_adv_ratio": "adverb density (spaCy)",
    "syn_pos_det_ratio": "determiner density (spaCy)",
    "syn_pos_adp_ratio": "adposition density (spaCy)",
    "syn_pos_sconj_ratio": "subordinating-conjunction density (spaCy)",
    "syn_pos_cconj_ratio": "coordinating-conjunction density (spaCy)",
    "syn_pos_entropy": "part-of-speech diversity",
    "syn_clause_rate": "parsed clause rate",
    "syn_mark_rate": "clause-marker rate",
    "syn_depth_mean": "parse depth (mean)",
    "syn_depth_std": "parse depth variation",
    "syn_passive_rate": "passive constructions",
    "syn_dep_len_mean": "mean head-dependent distance",
}


def phrase(feature: str) -> str:
    return FEATURE_PHRASES.get(feature, feature.replace("_", " "))


def load_reference(final_dir: Path = FINAL_DIR) -> dict:
    return load_json(Path(final_dir) / "reference.json")


def local_contributions(model, calibrator, features: list[str],
                        x_row: np.ndarray, ref_stats: dict,
                        k: int | None = None) -> list[dict]:
    """One-at-a-time perturbation of a single feature row (canonical space)."""
    x_row = np.asarray(x_row, dtype=float)
    p_base = float(calibrator.predict(
        model.predict_proba(x_row.reshape(1, -1))[:, 1])[0])
    out = []
    for j, f in enumerate(features):
        st = ref_stats.get(f)
        if st is None:
            continue
        ref = float(st["median"])
        x2 = x_row.copy()
        x2[j] = ref
        p2 = float(calibrator.predict(
            model.predict_proba(x2.reshape(1, -1))[:, 1])[0])
        contrib = p_base - p2
        val = float(x_row[j])
        out.append({
            "feature": f, "phrase": phrase(f),
            "value": round(val, 4) if np.isfinite(val) else None,
            "reference_median": ref,
            "contribution_to_p_machine": round(contrib, 4),
            "direction": "toward machine" if contrib > 1e-6
            else ("toward human" if contrib < -1e-6 else "no effect"),
        })
    out.sort(key=lambda d: -abs(d["contribution_to_p_machine"]))
    return out[:k] if k else out


def format_explanation(p_machine: float, label: str, contribs: list[dict],
                       n_words: int | None = None) -> str:
    head = f"p(machine) = {p_machine:.3f} -> {label}"
    if n_words is not None:
        head += f"  ({n_words} words)"
    lines = [head]
    for c in contribs:
        v = "n/a" if c["value"] is None else f"{c['value']:g}"
        lines.append(
            f"  - {c['phrase']}: {c['contribution_to_p_machine']:+.3f} "
            f"({c['direction']}; value {v}, train median "
            f"{c['reference_median']:g})")
    lines.append("  method: one-at-a-time perturbation to the train median; "
                 "model behavior, not causal attribution")
    return "\n".join(lines)


def importance_summary(final_dir: Path = FINAL_DIR) -> dict:
    path = Path(final_dir) / "importance.json"
    if not path.exists():
        return {"available": False}
    doc = load_json(path)
    block = doc.get("winner") or {}
    items = sorted(
        ({"feature": f, "phrase": phrase(f), "delta_pr_auc": v}
         for f, v in block.get("importance_delta_pr_auc", {}).items()),
        key=lambda d: -d["delta_pr_auc"])
    return {"method": doc.get("method"),
            "baseline_pr_auc": block.get("baseline_pr_auc"),
            "top": items[:15], "available": bool(items)}
