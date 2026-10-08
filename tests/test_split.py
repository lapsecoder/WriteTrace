"""Tests for grouped, stratified splitting (leakage guarantees)."""
import numpy as np
import pandas as pd
import pytest

from ml.preprocessing.split import grouped_split, make_stratum


def make_df(n_groups: int = 60, per_group: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        label = int(rng.integers(0, 2))
        src = str(rng.choice(["a", "b", "c"]))
        for _ in range(per_group):
            rows.append({"group_id": f"g{g}", "label": label, "src": src})
    df = pd.DataFrame(rows)
    df["stratum"] = make_stratum(df["label"], df["src"])
    return df


def test_every_row_assigned():
    df = make_df()
    split = grouped_split(df, seed=42)
    assert len(split) == len(df)
    assert set(split.unique()) <= {"train", "val", "test"}


def test_no_group_leakage_across_splits():
    df = make_df()
    out = df.assign(split=grouped_split(df, seed=42))
    group_sets = {s: set(g["group_id"]) for s, g in out.groupby("split")}
    keys = sorted(group_sets)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            assert not (group_sets[a] & group_sets[b]), f"leakage {a}/{b}"


def test_split_is_deterministic_for_fixed_seed():
    df = make_df()
    s1 = grouped_split(df, seed=42)
    s2 = grouped_split(df, seed=42)
    assert (s1 == s2).all()
    s3 = grouped_split(df, seed=7)
    assert (s1 != s3).any(), "different seeds should (almost surely) differ"


def test_fractions_roughly_respected():
    df = make_df(n_groups=200, per_group=5)
    out = grouped_split(df, seed=42, fractions=(0.6, 0.2, 0.2))
    frac = out.value_counts(normalize=True)
    assert 0.45 <= frac.get("train", 0) <= 0.75
    assert 0.10 <= frac.get("val", 0) <= 0.30
    assert 0.10 <= frac.get("test", 0) <= 0.30


def test_both_classes_present_in_each_split():
    df = make_df(n_groups=100, per_group=6)
    out = df.assign(split=grouped_split(df, seed=42))
    for _, g in out.groupby("split"):
        assert set(g["label"].unique()) == {0, 1}


def test_rare_strata_are_coarsened():
    df = pd.DataFrame({"label": [0, 0] + [1] * 12,
                       "src": ["rare1", "rare2"] + ["common"] * 12})
    st = make_stratum(df["label"], df["src"])
    assert st[0] == "0|rare" and st[1] == "0|rare"          # rare coarsened
    assert st[2] == "1|common" and (st[2:] == "1|common").all()


def test_bad_fractions_rejected():
    df = make_df(n_groups=10)
    with pytest.raises(AssertionError):
        grouped_split(df, fractions=(0.5, 0.5, 0.5))


def test_empty_frame_yields_empty_split():
    df = pd.DataFrame({"group_id": [], "stratum": []})
    split = grouped_split(df, seed=1)
    assert len(split) == 0
