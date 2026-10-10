"""Deterministic text segmentation for Phase-5A segment-level analysis.

WHY THIS EXISTS
---------------
Phase 4 proved document-level inference (ml/inference.py). Phase 5A asks the
prior question first: *is segment-level inference scientifically defensible?*
To ask it we need a segmentation unit that is

  * deterministic (same text -> same segments, always),
  * faithful (no silent rewriting; every non-whitespace character of the input
    appears in exactly one segment, in order),
  * offset-carrying (start_char/end_char index into the ORIGINAL text, so a
    future UI can map a segment back to its source),
  * conservative (never overlaps segments; never invents text).

ALGORITHM (v1)
--------------
1. Paragraph blocks: split on blank lines (``\\n`` + optional spaces + ``\\n``).
   Each block is trimmed to its first/last non-whitespace character and keeps
   exact offsets.
2. Heading merge: blocks that look like headings (<= ``heading_max_words``
   words, no sentence-final punctuation) are merged forward into the next
   non-heading block (span union). A heading carries almost no stylistic
   evidence on its own; merging avoids orphan micro-segments. A trailing
   heading with nothing to merge into is kept as its own segment.
3. Windowing: a block longer than ``max_segment_words`` is packed into
   non-overlapping, sentence-aligned windows. A single sentence longer than the
   limit is hard-split at word boundaries (counted and reported - a real cost
   of segmentation, not hidden). A trailing window shorter than
   ``min_window_words`` is merged backwards so no micro-segment is emitted.
4. Sentences come from the project's own deterministic splitter
   (``ml.features.extract.split_sentences`` - the same heuristic the v1
   features use) and are mapped back to the original text with a sequential
   cursor, so offsets stay exact even though sentences are stripped internally.

Word counts use ``ml.preprocessing.normalize.word_count`` - the same counter
as the Phase-4 gate, so segment counts are comparable with document counts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .features.extract import split_sentences
from .preprocessing.normalize import WORD_RE, word_count

__all__ = [
    "Segment", "SegmentationPolicy", "DEFAULT_POLICY", "DEFAULT_MAX_SEGMENT_WORDS",
    "DEFAULT_MIN_WINDOW_WORDS", "HEADING_MAX_WORDS", "segment_text",
    "coverage_report",
]

_BLANK_LINE_RE = re.compile(r"\n[ \t\r]*\n+")
_SENTENCE_END_RE = re.compile(r"[.!?][\"'”’)]*$")
DEFAULT_MAX_SEGMENT_WORDS = 150
DEFAULT_MIN_WINDOW_WORDS = 30
HEADING_MAX_WORDS = 12


@dataclass(frozen=True)
class Segment:
    """One contiguous span of the original text.

    ``start_char``/``end_char`` index into the ORIGINAL input (end exclusive);
    ``original_text[start_char:end_char] == text`` always holds.
    """
    segment_id: int
    text: str
    start_char: int
    end_char: int
    word_count: int
    paragraph_index: int
    kind: str = "paragraph"            # "paragraph" | "window"
    window_index: int | None = None
    n_sentences: int = 0
    hard_split: bool = False           # an over-long sentence had to be cut

    @property
    def words(self) -> list[str]:
        return WORD_RE.findall(self.text)


@dataclass(frozen=True)
class SegmentationPolicy:
    """Explicit, recorded segmentation policy (never an implicit default)."""
    max_segment_words: int = DEFAULT_MAX_SEGMENT_WORDS
    min_window_words: int = DEFAULT_MIN_WINDOW_WORDS
    heading_max_words: int = HEADING_MAX_WORDS
    merge_headings: bool = True

    def as_dict(self) -> dict:
        return {
            "max_segment_words": self.max_segment_words,
            "min_window_words": self.min_window_words,
            "heading_max_words": self.heading_max_words,
            "merge_headings": self.merge_headings,
            "blank_line_split": True,
            "sentence_packing": True,
            "sentence_splitter": "ml.features.extract.split_sentences (v1 "
                                 "heuristic)",
            "overlap": False,
        }


DEFAULT_POLICY = SegmentationPolicy()


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------

def _trimmed_span(text: str, lo: int, hi: int) -> tuple[int, int]:
    """Shrink [lo, hi) so both edges are non-whitespace.

    Uses `str.isspace()`, so invisible-but-not-whitespace characters (e.g. a
    zero-width space used by the RAID zero_width_space attack) stay INSIDE the
    block and are therefore covered by a segment - they are never dropped.
    """
    while lo < hi and text[lo].isspace():
        lo += 1
    while hi > lo and text[hi - 1].isspace():
        hi -= 1
    return lo, hi


def _paragraph_blocks(text: str) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    pos = 0
    for m in _BLANK_LINE_RE.finditer(text):
        blocks.append((pos, m.start()))
        pos = m.end()
    blocks.append((pos, len(text)))
    out: list[tuple[int, int]] = []
    for lo, hi in blocks:
        lo, hi = _trimmed_span(text, lo, hi)
        if hi > lo:
            out.append((lo, hi))
    return out


def _looks_like_heading(block: str, policy: SegmentationPolicy) -> bool:
    if not policy.merge_headings:
        return False
    if word_count(block) > policy.heading_max_words:
        return False
    stripped = block.strip()
    if not stripped:
        return False
    return not _SENTENCE_END_RE.search(stripped.splitlines()[-1].strip())


def _merge_headings(text: str, blocks: list[tuple[int, int]],
                    policy: SegmentationPolicy) -> list[tuple[int, int]]:
    """Absorb runs of heading-like blocks into the following body block."""
    merged: list[tuple[int, int]] = []
    pending_lo: int | None = None
    for lo, hi in blocks:
        if _looks_like_heading(text[lo:hi], policy):
            if pending_lo is None:
                pending_lo = lo
            continue
        merged.append((pending_lo if pending_lo is not None else lo, hi))
        pending_lo = None
    if pending_lo is not None:                     # trailing heading(s)
        merged.append((pending_lo, blocks[-1][1]))
    return merged


def _sentence_spans(block: str, offset: int) -> list[tuple[int, int]]:
    """Absolute (start, end) sentence spans, mapped back with a cursor."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for s in split_sentences(block):
        rel = block.find(s, cursor)
        if rel < 0:
            rel = block.find(s)
        if rel < 0:                                # defensive
            continue
        spans.append((offset + rel, offset + rel + len(s)))
        cursor = rel + len(s)
    # Absorb the separator gap into the PREVIOUS span, then always keep the
    # current sentence. `split_sentences` consumes closing quotes/brackets in
    # its separator (e.g. `...") `), so sentence strings alone can drop
    # characters; absorbing the gap guarantees a lossless, non-overlapping cover.
    merged: list[tuple[int, int]] = []
    for lo, hi in spans:
        if merged and lo > merged[-1][1]:
            merged[-1] = (merged[-1][0], lo)
        merged.append((lo, hi))
    if merged:
        # the first/last sentence can start/end after characters the splitter
        # stripped (a leading or trailing quote, for example); claim them so
        # the window spans cover the whole block
        merged[0] = (offset, merged[0][1])
        merged[-1] = (merged[-1][0], offset + len(block))
    return merged


def _span_words(block: str, offset: int, span: tuple[int, int]) -> int:
    lo, hi = span
    return word_count(block[lo - offset:hi - offset])


def _hard_split(block: str, offset: int, span: tuple[int, int], limit: int
                ) -> list[tuple[int, int]]:
    """Cut one over-long sentence into <= limit-word pieces at word bounds.

    Returns ABSOLUTE (start, end) spans; pieces are non-overlapping and in
    order, so a hard-split sentence never leaks text.
    """
    lo, hi = span
    words = [(offset + m.start(), offset + m.end())
             for m in WORD_RE.finditer(block)
             if lo <= offset + m.start() < hi]
    if not words:
        return []
    out: list[tuple[int, int]] = []
    start, count = words[0][0], 0
    for i, (w0, _w1) in enumerate(words):
        if count == limit and i > 0:
            # cut at the NEXT word's start so the punctuation/space between the
            # two words stays inside the emitted piece (pieces stay contiguous)
            out.append((start, w0))
            start, count = w0, 0
        count += 1
    out.append((start, hi))          # last piece also keeps trailing chars
    return out


def _window_spans(block: str, offset: int, policy: SegmentationPolicy
                  ) -> list[tuple[int, int, bool]]:
    """Sentence-packed, non-overlapping windows: (start, end, hard_split)."""
    limit = max(policy.max_segment_words, 1)
    sents = _sentence_spans(block, offset)
    if not sents:
        sents = [(offset, offset + len(block))]

    packed: list[list[tuple[int, int]]] = [[]]
    words = 0

    def flush() -> None:
        nonlocal packed, words
        packed.append([])
        words = 0

    for span in sents:
        s_words = _span_words(block, offset, span)
        if s_words > limit:
            if packed[-1]:
                flush()
            for piece in _hard_split(block, offset, span, limit):
                packed[-1].append(piece)
                words += _span_words(block, offset, piece)
                if words >= limit:
                    flush()
            continue
        if words + s_words > limit and packed[-1]:
            flush()
        packed[-1].append(span)
        words += s_words

    groups = [g for g in packed if g]
    if not groups:
        return []

    spans: list[tuple[int, int, bool]] = [
        (g[0][0], g[-1][1], False) for g in groups]

    # merge a too-short trailing window backwards (stays non-overlapping)
    if len(spans) > 1:
        last = spans[-1]
        if _span_words(block, offset, (last[0], last[1])) < policy.min_window_words:
            prev = spans[-2]
            spans[-2] = (prev[0], last[1], prev[2] or last[2])
            spans.pop()

    # mark windows that contain a hard-split piece
    hard_starts = set()
    for span in sents:
        if _span_words(block, offset, span) > limit:
            for piece in _hard_split(block, offset, span, limit):
                hard_starts.add(piece[0])
    marked = [(lo, hi, any(lo <= h < hi for h in hard_starts))
              for lo, hi, _ in spans]
    # Close gaps between and around windows: the window cover must be exactly
    # the block, so each window's end is pulled to the next window's start and
    # the first/last reach the block edges. Whatever `split_sentences` stripped
    # (closing quotes, hard-wrap newlines, zero-width characters) stays inside
    # the preceding window instead of vanishing.
    for i in range(len(marked) - 1):
        nxt_lo = marked[i + 1][0]
        if marked[i][1] < nxt_lo:
            marked[i] = (marked[i][0], nxt_lo, marked[i][2])
    if marked:
        marked[0] = (offset, marked[0][1], marked[0][2])
        marked[-1] = (marked[-1][0], offset + len(block), marked[-1][2])
    return marked


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def segment_text(text: str, policy: SegmentationPolicy | None = None
                 ) -> list[Segment]:
    """Deterministically split `text` into ordered, non-overlapping segments."""
    pol = policy or DEFAULT_POLICY
    if not isinstance(text, str) or not text.strip():
        return []

    blocks = _paragraph_blocks(text)
    if not blocks:
        return []
    merged = _merge_headings(text, blocks, pol)

    segments: list[Segment] = []
    for para_idx, (lo, hi) in enumerate(merged):
        block = text[lo:hi]
        wc = word_count(block)
        if wc <= pol.max_segment_words:
            segments.append(Segment(
                segment_id=len(segments), text=block, start_char=lo,
                end_char=hi, word_count=wc, paragraph_index=para_idx,
                kind="paragraph", n_sentences=len(split_sentences(block)) or 1))
            continue
        for w_idx, (w_lo, w_hi, hard) in enumerate(
                _window_spans(block, lo, pol)):
            sub = text[w_lo:w_hi]
            segments.append(Segment(
                segment_id=len(segments), text=sub, start_char=w_lo,
                end_char=w_hi, word_count=word_count(sub),
                paragraph_index=para_idx, kind="window", window_index=w_idx,
                n_sentences=len(split_sentences(sub)) or 1, hard_split=hard))
    return segments


def coverage_report(text: str, segments: list[Segment]) -> dict:
    """Text-preservation check: every non-whitespace char in exactly one segment,
    in order; every segment slice matches the original text."""
    covered = bytearray(len(text))
    for seg in segments:
        if text[seg.start_char:seg.end_char] != seg.text:
            raise AssertionError(f"segment {seg.segment_id} slice mismatch")
        if seg.start_char >= seg.end_char:
            raise AssertionError(f"segment {seg.segment_id} empty span")
        for j in range(seg.start_char, seg.end_char):
            covered[j] = 1
    missing = sum(1 for j, ch in enumerate(text)
                  if not covered[j] and not ch.isspace())
    return {
        "n_segments": len(segments),
        "n_chars": len(text),
        "non_whitespace_missing": missing,
        "total_words": sum(s.word_count for s in segments),
        "paragraphs": sum(1 for s in segments if s.kind == "paragraph"),
        "windows": sum(1 for s in segments if s.kind == "window"),
        "hard_splits": sum(1 for s in segments if s.hard_split),
    }
