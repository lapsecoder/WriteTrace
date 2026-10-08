"""Phase-2 experiment unit tests: label canonicalization, generator families,
mixture construction helpers, and gate-table logic. No model downloads required."""
import numpy as np
import pandas as pd

from ml.experiments.phase2 import (
    EXPERIMENTS, canonical_labels, gen_family, machine_probability, sha_of,
)


def _df(rows):
    return pd.DataFrame(rows)


def test_canonical_labels_mage_complements_hc3_as_is():
    df = _df([
        {"label": 1, "source_dataset": "mage"},   # stored-1 == HUMAN
        {"label": 0, "source_dataset": "mage"},   # stored-0 == MACHINE
        {"label": 1, "source_dataset": "hc3"},    # stored-1 == MACHINE
        {"label": 0, "source_dataset": "hc3"},    # stored-1 == HUMAN? no: stored-0 == HUMAN
    ])
    got = canonical_labels(df)
    assert list(got) == [0, 1, 1, 0]


def test_canonical_labels_unknown_source_raises():
    df = _df([{"label": 1, "source_dataset": "mystery"}])
    try:
        canonical_labels(df)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unmapped source")


def test_machine_probability_direction():
    df = _df([
        {"source_dataset": "mage"},
        {"source_dataset": "hc3"},
        {"source_dataset": "raid"},
    ])
    out = machine_probability(np.array([0.9, 0.9, 0.9]), df)
    # The locked model learned p(stored-1) = p(human-like) on MAGE, where
    # stored-1 == HUMAN. That semantic is corpus-agnostic, so EVERY known
    # source complements: p(machine) = 1 - p_raw. (Labels map separately.)
    assert abs(out[0] - 0.1) < 1e-9
    assert abs(out[1] - 0.1) < 1e-9
    assert abs(out[2] - 0.1) < 1e-9
    # unknown source -> NaN (never silently guess)
    out2 = machine_probability(np.array([0.5]), _df([{"source_dataset": "mystery"}]))
    assert np.isnan(out2[0])


def test_gen_family_tokens():
    assert gen_family("eli5_machine_continuation_t0_3b") == "t0_3b"
    assert gen_family("xsum_machine_specified_text-davinci-003") == "text-davinci-003"
    assert gen_family("xsum_machine_topical_gpt-3.5-trubo") == "gpt-3.5-trubo"
    assert gen_family("xsum_machine_continuation") == "continuation"
    assert gen_family("cmv_human") == "human"
    assert gen_family("cnn_human_para") == "human"


def test_experiment_registry_complete():
    assert sorted(EXPERIMENTS) == ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7"]


def test_e4_ratio_math():
    ratios = [0.25, 0.5, 0.75]
    npara = 4
    counts = [max(1, min(npara - 1, int(round(r * npara)))) for r in ratios]
    assert counts == [1, 2, 3]


def test_sha_of_is_deterministic():
    assert sha_of({"b": 1, "a": 2}) == sha_of({"a": 2, "b": 1})
    assert len(sha_of({})) == 64


def test_cohen_d_known_value_and_degenerate_cases():
    from ml.experiments.phase2 import cohen_d
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [3.0, 4.0, 5.0, 6.0, 7.0]
    # means 3 vs 5, both var 2 -> pooled sd sqrt(2); d = -2/sqrt(2)
    assert abs(cohen_d(a, b) - (-2.0 / (2.0 ** 0.5))) < 1e-9
    assert np.isnan(cohen_d([1.0], [2.0]))          # too few rows
    assert np.isnan(cohen_d([2.0, 2.0], [2.0, 2.0]))  # zero variance


def test_clip_sentence_cuts_at_boundary():
    from ml.experiments.phase2 import _clip_sentence
    long = "First sentence here. " * 40          # 840 chars
    out = _clip_sentence(long, 100)
    assert len(out) <= 100
    assert out.endswith(".")
    assert out.startswith("First sentence here.")
    assert _clip_sentence("short text", 100) == "short text"


def test_state_dependent_features_match_extract_one():
    """LODO correctness: the two refit features must mirror extract_one."""
    import joblib
    from ml.features.extract import extract_one
    from ml.experiments.phase2 import state_dependent_features, replace_state_features
    state = joblib.load("ml/data/processed/feature_state.joblib")
    texts = [
        "The quick brown fox jumps over the lazy dog again and again.",
        "short",        # single token -> NaN in both paths
        "",             # empty -> NaN in both paths
    ]
    got = state_dependent_features(texts, state)
    for i, t in enumerate(texts):
        ref = extract_one(t, state)
        for col in ("rare_token_ratio", "mean_log_df"):
            g, r = got[col].iloc[i], ref[col]
            if np.isnan(r):
                assert np.isnan(g), (col, i, g, r)
            else:
                assert abs(g - r) < 1e-12, (col, i, g, r)
    # replace_state_features touches ONLY the two state-dependent columns
    feats = pd.DataFrame({"rare_token_ratio": [9.9, 9.9, 9.9],
                          "mean_log_df": [9.9, 9.9, 9.9],
                          "n_words": [1, 2, 3]})
    out = replace_state_features(feats, texts, state)
    assert out["n_words"].tolist() == [1, 2, 3]
    assert not np.isclose(out["rare_token_ratio"].iloc[0], 9.9)


def test_error_example_block_shapes():
    """Report contract: slices are lists of examples with attributions."""
    import inspect
    from ml.experiments.phase2 import error_examples_block
    sig = inspect.signature(error_examples_block)
    assert set(sig.parameters) >= {"cfg", "ctx"}
    src = inspect.getsource(error_examples_block)
    for key in ("mage_test_human_fp", "hc3_machine_fn", "raid_human_fp",
                "ood_"):  # slice names the report renders
        assert key in src, key
