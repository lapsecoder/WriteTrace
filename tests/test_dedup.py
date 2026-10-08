"""Tests for exact/near duplicate clustering (deduplication before splitting)."""
import numpy as np

from ml.preprocessing.dedup import cluster_duplicates, exact_duplicate_pairs

_WORDS = """alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima
mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray
yankee zulu red blue green cloud river stone tree wind fire""".split()
# ~78-word documents so a single token change keeps shingle Jaccard well above
# the 0.75 near-dup threshold (shorter texts sit right at the boundary).
BASE = " ".join(_WORDS + _WORDS)
NEAR = BASE.replace(_WORDS[10], "storm", 1)                 # one token differs
OTHER = " ".join(list(reversed(_WORDS)) * 2)  # same words, other order


def test_exact_duplicates_cluster_and_are_counted():
    labels, stats = cluster_duplicates([BASE, BASE, BASE, OTHER], workers=1)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]
    assert stats["exact_pairs"] == 2


def test_near_duplicates_cluster_when_shingle_overlap_is_high():
    labels, stats = cluster_duplicates([BASE, NEAR], workers=1)
    assert labels[0] == labels[1], f"near-dup not clustered; stats={stats}"
    assert stats["near_pairs_verified"] >= 1
    assert stats["candidate_pairs"] >= 1


def test_unrelated_texts_stay_separate():
    labels, _ = cluster_duplicates([BASE, OTHER, OTHER + " extra tail words here"], workers=1)
    assert labels[0] != labels[1]


def test_cluster_labels_are_deterministic():
    texts = [BASE, NEAR, OTHER, BASE]
    l1, s1 = cluster_duplicates(texts, workers=1, seed=13)
    l2, s2 = cluster_duplicates(texts, workers=1, seed=13)
    assert (l1 == l2).all()
    assert s1["candidate_pairs"] == s2["candidate_pairs"]
    # relabeling by first occurrence
    assert l1[0] == min(l1)


def test_empty_and_singleton_inputs():
    labels, stats = cluster_duplicates([], workers=1)
    assert labels.size == 0 and stats["n"] == 0
    labels, stats = cluster_duplicates([BASE], workers=1)
    assert labels.tolist() == [0]


def test_exact_duplicate_pairs_helper():
    pairs = exact_duplicate_pairs(["x", "x", "y", "x"])
    assert pairs == [(0, 1), (0, 3)]


def test_outputs_are_int_labels():
    labels, _ = cluster_duplicates([BASE, OTHER], workers=1)
    assert labels.dtype == np.int64
    assert labels.min() == 0
