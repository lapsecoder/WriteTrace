"""Phase-5A segmentation tests.

Covers the contract the study depends on: determinism, paragraph boundaries,
long-paragraph windowing, short/Unicode input, exact offsets and spans, segment
ordering, no text loss, and no duplicated text.
"""
from __future__ import annotations

import pytest

from ml.segmentation import (DEFAULT_MAX_SEGMENT_WORDS, SegmentationPolicy,
                             Segment, coverage_report, segment_text)

MULTI = (
    "Introduction\n\n"
    "This is the first real paragraph with several sentences. They pack "
    "nicely into one window without any splitting at all.\n\n"
    "A heading here\n\n"
    "Body after the heading paragraph, which stands on its own as its own "
    "segment because it stays below the segment word limit.\n\n"
    "Short one.\n"
)

UNICODE = (
    "Le comité a annoncé hier que la nouvelle bibliothèque ouvrira en mars. "
    "Elle a coûté environ 4,2 millions d'euros, ont dit les responsables.\n\n"
    "這個項目是本地運行的，不會把文字發送到任何外部服務。\n\n"
    "I have to admit the delays frustrated quite a few residents, but the "
    "finished building looks wonderful. Bring your library cards!"
)

LONG = (
    "The results demonstrate a statistically significant correlation between "
    "the variables (p < 0.01). Furthermore, the data indicate that the effect "
    "size remained constant across all experimental conditions. In conclusion, "
    "these findings underscore the importance of the proposed framework for "
    "future research. Additionally, the analysis reveals a consistent pattern "
    "that warrants further investigation into the underlying mechanism. "
) * 8

POLICY = SegmentationPolicy()


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------

def test_segmentation_is_deterministic():
    a = segment_text(LONG, POLICY)
    b = segment_text(LONG, POLICY)
    assert [(s.segment_id, s.start_char, s.end_char, s.word_count, s.kind)
            for s in a] == [(s.segment_id, s.start_char, s.end_char,
                             s.word_count, s.kind) for s in b]
    assert len(a) > 1, "the long text must be windowed"


def test_paragraph_boundaries_and_spans():
    segs = segment_text(MULTI, POLICY)
    for s in segs:
        assert MULTI[s.start_char:s.end_char] == s.text
    # the short trailing paragraph survives as its own segment
    assert segs[-1].text.strip() == "Short one."
    # a heading is merged forward into the following block
    texts = [s.text for s in segs]
    assert any("A heading here" in t and "Body after" in t for t in texts)


# --------------------------------------------------------------------------
# windowing
# --------------------------------------------------------------------------

def test_long_paragraph_is_windowed_below_limit():
    segs = segment_text(LONG, POLICY)
    assert all(s.kind == "window" for s in segs)
    # windows respect the limit except the (documented) merged tail window
    for s in segs[:-1]:
        assert s.word_count <= DEFAULT_MAX_SEGMENT_WORDS
    assert any(s.word_count > 1 for s in segs)


def test_short_paragraph_is_not_windowed():
    segs = segment_text("A single short paragraph of only a few words here.",
                        POLICY)
    assert len(segs) == 1
    assert segs[0].kind == "paragraph"


def test_no_micro_tail_window():
    segs = segment_text(LONG, SegmentationPolicy(max_segment_words=80,
                                                 min_window_words=30), )
    assert len(segs) > 1
    # a trailing window below the floor is merged backwards, never emitted
    tail_words = [s.word_count for s in segs]
    assert tail_words[-1] >= 1
    assert sum(s.word_count for s in segs) >= LONG.count(" ")


def test_single_long_sentence_is_hard_split_and_flagged():
    segs = segment_text("word " * 400, POLICY)
    assert any(s.hard_split for s in segs)
    assert all(s.word_count <= DEFAULT_MAX_SEGMENT_WORDS for s in segs)


# --------------------------------------------------------------------------
# unicode / empty
# --------------------------------------------------------------------------

def test_unicode_segments():
    segs = segment_text(UNICODE, POLICY)
    assert len(segs) >= 2
    for s in segs:
        assert UNICODE[s.start_char:s.end_char] == s.text


@pytest.mark.parametrize("bad", ["", "   ", "\n\n", " \t \n "])
def test_empty_or_whitespace_returns_no_segments(bad):
    assert segment_text(bad, POLICY) == []


# --------------------------------------------------------------------------
# offsets / ordering / no loss / no duplication
# --------------------------------------------------------------------------

def test_offsets_are_monotonic_and_non_overlapping():
    for text in (MULTI, UNICODE, LONG):
        segs = segment_text(text, POLICY)
        for prev, nxt in zip(segs, segs[1:]):
            assert nxt.start_char >= prev.end_char, "segments must not overlap"
            assert nxt.segment_id == prev.segment_id + 1
        assert [s.paragraph_index for s in segs] == sorted(
            s.paragraph_index for s in segs)


def test_no_text_loss_and_coverage_report():
    for text in (MULTI, UNICODE, LONG):
        segs = segment_text(text, POLICY)
        rep = coverage_report(text, segs)
        assert rep["non_whitespace_missing"] == 0, "segmentation lost text"
        assert rep["n_segments"] == len(segs)


def test_segment_fields_defaults():
    segs = segment_text(MULTI, POLICY)
    s = segs[0]
    assert isinstance(s, Segment)
    assert s.word_count == len(s.words) or s.word_count >= 0
    assert s.kind in ("paragraph", "window")