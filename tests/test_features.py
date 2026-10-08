"""Tests for feature extraction: sanity, edge cases, determinism, fitted state."""
import math

import numpy as np
import pandas as pd
import pytest

from ml.features.extract import (FEATURE_FAMILIES, FeatureState, extract_many,
                                 extract_one, feature_names, split_sentences)

TEXT = (
    "I've been thinking about this — really, quite a lot — and I'm not convinced. "
    "The experiment ran for three weeks; results were mixed, though promising. "
    "However, one must ask: does the evidence hold up under scrutiny? "
    "It is important to note that our sample was small; furthermore, the controls "
    "were imperfect. The researchers carefully measured every variable; they "
    "reported their findings clearly, and the paper was published eventually. "
    "Because the effect size was large, we repeated the study with new participants "
    "who had not taken part before, and the second run produced similar results overall."
)
OTHER = ("Totally different prose about cooking pasta with garlic and olive oil "
         "until the sauce thickens nicely on low heat for a while.")


def test_named_stable_columns_and_families():
    names = feature_names()
    assert len(names) == sum(len(v) for v in FEATURE_FAMILIES.values())
    assert len(set(names)) == len(names)  # no duplicates
    assert names == feature_names()       # stable across calls


def test_sentence_splitting_heuristic():
    assert len(split_sentences("One two three. Four five! Six?")) == 3
    assert split_sentences("") == []
    assert len(split_sentences("A line\nanother line.")) >= 1


def test_extract_sanity_ranges():
    r = extract_one(TEXT)
    assert set(r) == set(feature_names())
    assert r["n_words"] > 50
    assert r["n_sentences"] >= 5
    assert r["comma_rate"] > 0 and r["semicolon_rate"] > 0
    assert r["em_dash_rate"] > 0 and r["qmark_rate"] > 0
    assert r["contraction_rate"] > 0        # I've / I'm
    assert r["first_person_rate"] > 0       # I / I'm / our
    assert r["hedge_rate"] > 0              # quite / must? (quite)
    assert -200 < r["flesch_reading_ease"] < 250
    assert 0 <= r["hapax_ratio"] <= 1
    assert 0 <= r["stopword_ratio"] <= 1
    assert r["n_paragraphs"] >= 1


def test_invalid_and_tiny_inputs_are_all_nan():
    for bad in (None, 42, "", "   ", "hi"):
        r = extract_one(bad)
        assert all(math.isnan(v) for v in r.values()), f"expected NaN row for {bad!r}"


def test_deterministic_serial_output():
    a = extract_many([TEXT, OTHER], workers=0)
    b = extract_many([TEXT, OTHER], workers=0)
    pd.testing.assert_frame_equal(a, b)
    r1, r2 = extract_one(TEXT), extract_one(TEXT)
    pd.testing.assert_frame_equal(pd.DataFrame([r1]), pd.DataFrame([r2]))


def test_fitted_state_changes_rare_features():
    common_state = FeatureState.fit([TEXT] * 20)
    r_common = extract_one(TEXT, common_state)
    assert r_common["rare_token_ratio"] == 0.0
    assert r_common["mean_log_df"] > 0
    alien_state = FeatureState.fit([OTHER])
    r_alien = extract_one(TEXT, alien_state)
    assert r_alien["rare_token_ratio"] > 0.7  # alien corpus shares only function words
    # without state -> NaN, not crash
    assert math.isnan(extract_one(TEXT)["rare_token_ratio"])


def test_parallel_matches_serial():
    import os
    if (os.cpu_count() or 1) < 2:
        import pytest
        pytest.skip("needs >=2 cpus")
    state = FeatureState.fit([TEXT, OTHER])
    serial = extract_many([TEXT, OTHER, TEXT], state, workers=0)
    parallel = extract_many([TEXT, OTHER, TEXT], state, workers=2)
    pd.testing.assert_frame_equal(serial, parallel)


def test_all_nan_rows_are_countable():
    df = extract_many([None, TEXT], workers=0)
    assert df.isna().all(axis=1).sum() == 1
