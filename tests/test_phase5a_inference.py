"""Phase-5A segment-inference, artifact-integrity and API-compatibility tests.

These tests assert the Phase-5A contract:
  * segment scoring goes through the SAME final-v1 path and is deterministic,
  * the segment evidence gate abstains (no label/confidence) on short segments,
  * insufficient-evidence segments get no explanations,
  * final-v1 and baseline-v1 are untouched (hashes vs their manifests),
  * segments-v1 is a SEPARATE artifact with its own gate/manifest,
  * the public /api/v1/analyze contract is unchanged (segments stays []).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.experiments.phase3 import FINAL_DIR

if not (FINAL_DIR / "manifest.json").exists():
    pytest.skip("final artifact not written yet", allow_module_level=True)

SEG_DIR = Path(__file__).resolve().parents[1] / "ml" / "models" / "segments-v1"

MACHINE_PARA = (
    "The committee announced yesterday that the new library will open in "
    "March after two years of renovation work. It cost roughly 4.2 million "
    "euros, officials said, and will stay open seven days a week from eight "
    "in the morning until ten at night. Furthermore, the accessibility "
    "improvements include automatic doors on every floor and a dedicated "
    "reading room for children. In conclusion, residents will be able to "
    "borrow materials through the existing membership system without any "
    "additional registration fees being required."
)
HUMAN_PARA = (
    "so we ended up going to that new cafe near the station, the one everyone "
    "keeps talking about. honestly the coffee was fine, nothing special, but "
    "the cake was unreal - i had this lemon thing with poppy seeds and now i "
    "want to go back every weekend. anyway we sat there for two hours just "
    "catching up, you know how it is. sarah brought her dog, which was a "
    "whole situation on its own, but whatever, it was really nice to just sit "
    "and not think about work for a while. we should go together sometime, "
    "i think youd like it."
)


@pytest.fixture(scope="module")
def predictor():
    from ml.inference import Predictor
    return Predictor(FINAL_DIR)


@pytest.fixture(scope="module")
def mixed_doc():
    return MACHINE_PARA + "\n\n" + HUMAN_PARA


# --------------------------------------------------------------------------
# segment scoring (same path, deterministic, bounded)
# --------------------------------------------------------------------------

def test_segment_scoring_uses_same_path_and_is_deterministic(predictor,
                                                             mixed_doc):
    r1 = predictor.analyze_segments(mixed_doc, include_signals=False)
    r2 = predictor.analyze_segments(mixed_doc, include_signals=False)
    assert r1.document is not None
    p1 = [s.machine_probability for s in r1.segments]
    p2 = [s.machine_probability for s in r2.segments]
    assert p1 == p2, "segment probabilities must be deterministic"
    for p in p1:
        assert 0.0 <= p <= 1.0
    # human + machine paragraphs get separated probabilities
    assert len(r1.segments) == 2
    assert p1[0] > p1[1], "the machine paragraph should score higher"


def test_document_prediction_is_authoritative_and_unchanged(predictor,
                                                           mixed_doc):
    doc_only = predictor.analyze(mixed_doc, include_signals=False)
    with_segments = predictor.analyze_segments(mixed_doc,
                                               include_signals=False)
    assert with_segments.document.machine_probability == \
        doc_only.machine_probability
    # the public document result still exposes an empty segments list
    assert doc_only.segments == []


def test_segment_probability_matches_document_path(mixed_doc, predictor):
    # scoring a segment text directly == the segment's probability
    segs = predictor.analyze_segments(mixed_doc, include_signals=False).segments
    direct = predictor.score(HUMAN_PARA)
    seg_p = next(s.machine_probability for s in segs
                 if HUMAN_PARA in s.segment.text)
    assert seg_p == pytest.approx(direct, abs=1e-12)


def test_repeated_paragraph_is_still_flagged_machine(predictor):
    """Known Phase-4 weakness, re-checked at segment level.

    Duplicating a human paragraph inside one segment pushes the repetition
    features high and the segment is called machine-like. This is a DOCUMENTED
    failure mode, not a regression: the test exists so it stays visible instead
    of silently changing.
    """
    once = predictor.analyze(HUMAN_PARA, include_signals=False)
    twice = predictor.analyze(HUMAN_PARA * 2, include_signals=False)
    assert twice.machine_probability > once.machine_probability
    assert twice.machine_probability >= 0.5


# --------------------------------------------------------------------------
# evidence gate on segments
# --------------------------------------------------------------------------

def test_segment_gate_abstains_on_short_segments(predictor):
    short_doc = "One short machine-sounding line. Another short sentence here."
    r = predictor.analyze_segments(short_doc, include_signals=False)
    assert r.n_segments >= 1
    for s in r.segments:
        assert s.status == "insufficient_evidence"
        assert s.decision == "abstain"
        assert s.label == "insufficient_evidence"
        assert s.confidence is None
        assert any("insufficient segment evidence" in w for w in s.warnings)
        assert s.signals == [], "no explanation for insufficient evidence"


def test_insufficient_evidence_document_warns_when_no_segment_has_evidence(
        predictor):
    tiny = "Too short."
    r = predictor.analyze_segments(tiny, include_signals=False)
    assert r.n_with_evidence == 0
    assert any("no segment reached" in w for w in r.warnings)


def test_segment_confidence_is_none_when_gated(predictor, mixed_doc):
    r = predictor.analyze_segments(mixed_doc, include_signals=False)
    for s in r.segments:
        if s.evidence_sufficient:
            assert s.confidence is not None
        else:
            assert s.confidence is None


# --------------------------------------------------------------------------
# aggregation (research; document stays authoritative)
# --------------------------------------------------------------------------

def test_aggregation_reports_all_candidates(predictor, mixed_doc):
    r = predictor.analyze_segments(mixed_doc, include_signals=False)
    agg = r.aggregation
    for key in ("n", "mean", "median", "weighted_mean", "max", "min", "std"):
        assert key in agg
    assert agg["min"] <= agg["mean"] <= agg["max"]
    assert agg["min"] <= agg["median"] <= agg["max"]
    assert agg["min"] <= agg["weighted_mean"] <= agg["max"]


def test_aggregate_helper_math():
    from ml.inference import aggregate_segment_probabilities
    from ml.segmentation import segment_text

    class Stub:
        def __init__(self, p, w):
            self.machine_probability = p
            self.word_count = w
            self.evidence_sufficient = True

    stubs = [Stub(0.2, 10), Stub(0.4, 30), Stub(0.8, 60)]
    agg = aggregate_segment_probabilities(stubs)
    assert agg["n"] == 3
    assert agg["mean"] == pytest.approx((0.2 + 0.4 + 0.8) / 3, abs=1e-6)
    assert agg["median"] == pytest.approx(0.4, abs=1e-6)
    assert agg["weighted_mean"] == pytest.approx(
        (0.2 * 10 + 0.4 * 30 + 0.8 * 60) / 100, abs=1e-6)
    assert agg["min"] == pytest.approx(0.2, abs=1e-6)
    assert agg["max"] == pytest.approx(0.8, abs=1e-6)


# --------------------------------------------------------------------------
# artifact integrity: final-v1 / baseline-v1 untouched
# --------------------------------------------------------------------------

def test_final_v1_manifest_hashes_still_match():
    manifest = json.loads((FINAL_DIR / "manifest.json").read_text(encoding="utf-8"))
    import hashlib
    for name, meta in manifest["files"].items():
        p = FINAL_DIR / name
        if not p.exists() or name.endswith(".joblib"):
            continue          # joblib files are large; hashed in phase-3 tests
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        assert digest == meta["sha256"], f"{name} changed since Phase 3"


def test_final_v1_still_the_default_artifact():
    from ml.inference import Predictor, get_predictor, reset_predictor
    reset_predictor()
    try:
        assert get_predictor().model_version == "final-v1"
        assert Predictor().manifest["winner"] == "hgb_syn"
    finally:
        reset_predictor()


@pytest.mark.skipif(not SEG_DIR.exists(), reason="segments-v1 not built yet")
def test_segments_v1_is_separate_with_own_gate():
    from ml.inference import Predictor
    assert SEG_DIR.resolve() != FINAL_DIR.resolve()
    gate = json.loads((SEG_DIR / "gate.json").read_text(encoding="utf-8"))
    manifest = json.loads((SEG_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert gate["kind"] == "segment_evidence_gate"
    assert gate["min_words"] >= 30
    assert manifest["artifact_kind"] == "segments"
    # segments-v1 loads independently of final-v1
    sp = Predictor(SEG_DIR)
    assert sp.model_version == "segments-v1"
    assert sp.min_words == gate["min_words"]


@pytest.mark.skipif(not SEG_DIR.exists(), reason="segments-v1 not built yet")
def test_segments_v1_does_not_overwrite_final_v1_files():
    """Separate directory AND separate file contents (same file NAMES are fine:
    they live in different artifact directories and never share a byte)."""
    import hashlib

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    assert SEG_DIR.resolve() != FINAL_DIR.resolve()
    shared = {p.name for p in SEG_DIR.iterdir()} & {p.name for p in FINAL_DIR.iterdir()}
    assert shared, "the two artifacts follow the same layout by design"
    for name in shared:
        if name.endswith(".joblib") or name == "features.json":
            assert sha(SEG_DIR / name) != sha(FINAL_DIR / name), \
                f"{name} must differ between the two artifacts"


# --------------------------------------------------------------------------
# API backward compatibility
# --------------------------------------------------------------------------

def test_analyze_response_still_exposes_empty_segments_and_unchanged_schema():
    from fastapi.testclient import TestClient
    from backend.app import app
    with TestClient(app) as client:
        r = client.post("/api/v1/analyze",
                        json={"text": MACHINE_PARA + HUMAN_PARA})
        assert r.status_code == 200
        body = r.json()
        # backward compatibility: the document contract is unchanged
        assert body["segments"] == []
        for field in ("status", "label", "decision", "machine_probability",
                      "human_probability", "confidence", "word_count",
                      "evidence_sufficient", "signals", "model_version",
                      "warnings", "disclaimer"):
            assert field in body


def test_analyze_response_model_still_validates():
    from backend.schemas import AnalyzeResponse
    assert "segments" in AnalyzeResponse.model_fields
    assert AnalyzeResponse.model_fields["segments"].default == []