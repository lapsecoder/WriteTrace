"""Conservative text normalization for WriteTrace.

Design rule (docs/methodology.md): remove *container* artifacts (encoding issues,
URLs, control characters) while PRESERVING all stylistic signals: case, punctuation,
em dashes, paragraph breaks, markdown markers, spelling variants.

Every transformation is listed in TRANSFORMATIONS and mirrored in the docs.
"""
from __future__ import annotations

import re
import unicodedata

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HSPACE_RE = re.compile(r"[ \t\r\f\v]+")
NEWLINE_RUN_RE = re.compile(r"\n{3,}")
WORD_RE = re.compile(r"\b[\w’']+\b", re.UNICODE)

TRANSFORMATIONS: dict[str, str] = {
    "unicode_nfc": "Unicode NFC normalization (composition form only; no character folding).",
    "newline_unify": "CRLF/CR -> LF.",
    "control_chars": "C0 control characters except \\n and \\t removed.",
    "url_placeholder": "http(s)/www URLs replaced with the literal token [URL].",
    "email_placeholder": "Email addresses replaced with the literal token [EMAIL].",
    "horizontal_whitespace": "Runs of spaces/tabs collapsed to a single space (newlines untouched).",
    "newline_cap": "Runs of 3+ newlines collapsed to a blank line (paragraph structure preserved).",
    "strip": "Leading/trailing whitespace stripped.",
    "PRESERVED": ("Case, punctuation (including em/en dashes and ellipses), quotes, markdown "
                  "markers, contractions, spelling and paragraph lengths are NEVER altered."),
}

# --- Artifact-stripped eval slice (docs/methodology.md §1) -------------------
# Applied ONLY to the secondary text view; the primary training view keeps these.
_ARTIFACT_RULES: list[tuple[str, re.Pattern[str], str]] = [
    ("llm_leadin_sure", re.compile(r"\A\s*(?:sure|certainly|of course|absolutely)(?:,|!|\.)\s+", re.I), ""),
    ("llm_leadin_here", re.compile(r"\A\s*(?:sure|certainly)\s+(?:here'?s|let'?s|i'?ll)\b[^.\n]*[.,]\s*", re.I), ""),
    ("llm_self_reference_comma", re.compile(r"\A\s*as an ai (?:language )?model,\s*", re.I), ""),
    ("llm_self_reference_bare", re.compile(r"\A\s*as an ai (?:language )?model\s+", re.I), ""),
    ("llm_closing_offer", re.compile(
        r"\n+\s*(?:i hope (?:this|that) (?:helps|was helpful)[^.]*\.?\s*"
        r"|let me know if you (?:have|need|want|'d|would)[\s\S]*"
        r"|is there anything (?:else|specific|more)[\s\S]*)\Z", re.I), ""),
]

ARTIFACT_PATTERNS = [
    {"name": n, "pattern": p.pattern, "replacement": r} for n, p, r in _ARTIFACT_RULES
]

STOP_HINTS = {"the", "of", "and", "to", "in", "a", "is", "that", "for", "it",
              "with", "as", "was", "on", "be", "at", "by", "this", "from", "are"}


def normalize_text(text) -> str | None:
    """Conservative normalization. Returns None for invalid input (non-string/empty)."""
    if text is None or not isinstance(text, str):
        return None
    t = unicodedata.normalize("NFC", text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = CTRL_RE.sub("", t)
    t = EMAIL_RE.sub("[EMAIL]", t)
    t = URL_RE.sub("[URL]", t)
    t = HSPACE_RE.sub(" ", t)
    t = NEWLINE_RUN_RE.sub("\n\n", t)
    t = t.strip()
    return t or None


def strip_artifacts(text: str | None) -> str | None:
    """Secondary view: remove common LLM lead-in/closing boilerplate (eval slice only)."""
    if text is None or not isinstance(text, str):
        return None
    t = text
    for _name, pattern, repl in _ARTIFACT_RULES:
        t = pattern.sub(repl, t)
    t = t.strip()
    return t or None


def word_count(text: str | None) -> int:
    if not text:
        return 0
    return len(WORD_RE.findall(text))


def looks_english(text: str | None) -> bool:
    """Conservative English heuristic for en-only corpora (no network, deterministic)."""
    if not text:
        return False
    sample = text[:4000]
    non_ascii = sum(1 for c in sample if ord(c) > 127)
    if non_ascii / max(len(sample), 1) > 0.4:
        return False
    words = [w.lower() for w in WORD_RE.findall(sample)][:300]
    if len(words) < 8:
        return True  # too short to judge; the length filter decides
    hits = sum(1 for w in words if w in STOP_HINTS)
    return hits >= 3
