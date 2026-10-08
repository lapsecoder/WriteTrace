"""Tests for conservative normalization, artifact stripping, and language heuristics."""
from ml.preprocessing.normalize import (ARTIFACT_PATTERNS, TRANSFORMATIONS,
                                        looks_english, normalize_text,
                                        strip_artifacts, word_count)


def test_normalize_preserves_stylistic_signals():
    t = "I think — really — that's fine… “quoted” and IMPORTANT case."
    out = normalize_text(t)
    assert "—" in out          # em dash preserved
    assert "…" in out          # ellipsis preserved
    assert "“quoted”" in out   # curly quotes preserved
    assert "that's" in out     # contraction preserved
    assert "IMPORTANT" in out  # casing preserved
    assert "I think" in out


def test_normalize_replaces_urls_and_emails():
    out = normalize_text("See https://example.com/x?y=1 or write a.b@c.de about it.")
    assert "[URL]" in out and "[EMAIL]" in out
    assert "https" not in out and "@" not in out


def test_normalize_strips_control_and_collapses_whitespace():
    out = normalize_text("a\x00b   c\t\td\n\n\n\n\ne")
    assert "\x00" not in out
    assert "   " not in out and "\t" not in out
    assert "\n\n\n" not in out and "\n\n" in out  # paragraph structure kept
    assert out.endswith("e")


def test_normalize_invalid_inputs_return_none():
    assert normalize_text(None) is None
    assert normalize_text(123) is None
    assert normalize_text(b"bytes") is None
    assert normalize_text("   \n  ") is None


def test_artifact_strip_removes_llm_leadins_only():
    assert not strip_artifacts("Sure, here is an answer about cats.").lower().startswith("sure")
    assert not strip_artifacts("As an AI language model, I cannot.").startswith("As an AI")
    plain = "Surely the results are inconclusive though, no?"
    assert strip_artifacts(plain) == plain          # "Surely" is a human word
    body = "The team conducted an evaluation of the methods. However, limits remain."
    assert strip_artifacts(body) == body             # ordinary prose untouched


def test_artifact_patterns_and_transformations_are_documented():
    assert ARTIFACT_PATTERNS and all({"name", "pattern", "replacement"} <= set(p)
                                      for p in ARTIFACT_PATTERNS)
    assert "PRESERVED" in TRANSFORMATIONS and "url_placeholder" in TRANSFORMATIONS


def test_looks_english():
    assert looks_english("The quick brown fox jumps over the lazy dog and keeps running.")
    assert not looks_english("你好 世界 这是一个测试 这是一个很长的句子用来测试语言检测 这是另一个句子")
    assert looks_english("ok")  # too short to judge -> defer to length filter


def test_word_count_matches_tokenization():
    assert word_count("one two three") == 3
    assert word_count("don't stop believing") == 3
    assert word_count(None) == 0 and word_count("") == 0
