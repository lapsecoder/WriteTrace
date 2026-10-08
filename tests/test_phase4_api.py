"""Phase-4 FastAPI backend tests.

Covers: response schema, validation/error handling (safe messages), label
semantics, determinism across requests, load-once behavior, privacy
requirements (no text logging, no paths/stack traces in responses).
"""
from __future__ import annotations

import json
import logging

import pytest

from ml.experiments.phase3 import FINAL_DIR

if not (FINAL_DIR / "manifest.json").exists():
    pytest.skip("final artifact not written yet", allow_module_level=True)

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from backend.app import app  # noqa: E402
from backend.schemas import AnalyzeResponse  # noqa: E402

NORMAL_TEXT = (
    "The committee announced yesterday that the new library will open in "
    "March. It cost roughly 4.2 million euros, officials said, and will "
    "stay open seven days a week. \"We are thrilled,\" the mayor told "
    "reporters. I have to admit the delays frustrated quite a few "
    "residents, but the finished building looks wonderful. Bring your "
    "library cards! The project began in 2019 after a survey found that "
    "residents wanted longer opening hours, better accessibility, and a "
    "dedicated children's area."
)

SHORT_TEXT = "Too short to judge."

UNICODE_TEXT = (
    "Le comité a annoncé hier que la nouvelle bibliothèque ouvrira en mars. "
    "Elle a coûté environ 4,2 millions d'euros, ont dit les responsables. "
    "這個項目是本地運行的，不會把文字發送到任何外部服務。 "
    "I have to admit the delays frustrated quite a few residents, but the "
    "finished building looks wonderful. Bring your library cards! 🎉"
)

MULTILINE_TEXT = (
    "First paragraph of the report describes the renovation timeline.\n\n"
    "Second paragraph lists the contractors and the final budget.\n\n"
    "Third paragraph thanks the residents for their patience during "
    "two years of construction work and noise in the neighbourhood."
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _analyze(client, text, **kw):
    return client.post("/api/v1/analyze", json={"text": text, **kw})


def _assert_no_leak(body_text: str):
    assert "Traceback" not in body_text
    assert "D:\\" not in body_text
    assert "site-packages" not in body_text
    assert "/ml/models" not in body_text and "ml\\models" not in body_text
    assert '"detail"' not in body_text or '"error"' in body_text


# --------------------------------------------------------------------------
# happy path / schema
# --------------------------------------------------------------------------

def test_health_endpoint(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["api_version"] == "v1"
    assert body["model_version"] == "final-v1"
    assert body["model_name"] == "hgb_syn"
    assert body["loaded"] is True
    _assert_no_leak(r.text)


def test_info_endpoint_exposes_metadata_without_paths(client):
    r = client.get("/api/v1/info")
    assert r.status_code == 200
    body = r.json()
    assert body["calibrator"] == "platt"
    assert body["threshold"] == 0.5
    assert body["n_features"] == 89
    assert body["gate"]["min_words"] == 50
    assert body["gate"]["mixed_band"] == [0.4, 0.6]
    assert body["data_policy"]["trained_on"] == "MAGE-train only"
    assert "HC3" in body["data_policy"]["eval_only"]
    assert len(body["global_importance"]) > 0
    assert "does not establish authorship" in body["disclaimer"]
    _assert_no_leak(r.text)


def test_analyze_response_schema(client):
    r = _analyze(client, NORMAL_TEXT)
    assert r.status_code == 200
    body = r.json()
    AnalyzeResponse.model_validate(body)          # exact schema conformance
    expected_keys = set(AnalyzeResponse.model_fields)
    assert set(body) == expected_keys
    assert body["status"] == "ok"
    assert body["evidence_sufficient"] is True
    assert body["label"] in {"likely_human", "mostly_human", "mixed",
                             "mostly_machine", "likely_machine"}
    assert body["decision"] in {"machine", "human"}
    assert body["machine_probability"] + body["human_probability"] == \
        pytest.approx(1.0, abs=1e-9)
    assert 0.0 <= body["machine_probability"] <= 1.0
    assert body["confidence"] is not None
    assert body["word_count"] >= 50
    assert body["text_length"]["words"] == body["word_count"]
    assert body["text_length"]["sentences"] >= 1
    assert body["model_version"] == "final-v1"
    assert body["model_name"] == "hgb_syn"
    assert body["calibrator"] == "platt"
    assert body["threshold"] == 0.5
    assert body["gate"] == {"min_words": 50, "mixed_band": [0.4, 0.6]}
    assert body["segments"] == []
    assert "does not establish authorship" in body["disclaimer"]
    _assert_no_leak(r.text)


def test_signals_in_response(client):
    r = _analyze(client, NORMAL_TEXT, max_signals=5)
    assert r.status_code == 200
    sigs = r.json()["signals"]
    assert 1 <= len(sigs) <= 5
    for s in sigs:
        assert set(s) == {"signal", "label", "family", "value",
                          "reference_median", "contribution", "direction",
                          "interpretation", "caveat"}
        assert s["direction"] in {"machine", "human", "neutral"}
        assert "not proof of authorship" in s["caveat"]


def test_signals_can_be_disabled(client):
    r = _analyze(client, NORMAL_TEXT, include_signals=False)
    assert r.status_code == 200
    assert r.json()["signals"] == []


def test_short_text_returns_insufficient_evidence(client):
    r = _analyze(client, SHORT_TEXT)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "insufficient_evidence"
    assert body["label"] == "insufficient_evidence"
    assert body["decision"] == "abstain"
    assert body["evidence_sufficient"] is False
    assert body["confidence"] is None
    assert body["word_count"] == 4             # "Too short to judge."
    assert body["gate"]["min_words"] == 50
    assert any("insufficient evidence" in w for w in body["warnings"])
    assert 0.0 <= body["machine_probability"] <= 1.0


def test_unicode_and_multiline_inputs(client):
    for text in (UNICODE_TEXT, MULTILINE_TEXT):
        r = _analyze(client, text)
        assert r.status_code == 200
        body = r.json()
        assert math_ok(body)
        assert body["text_length"]["paragraphs"] >= 1


def math_ok(body: dict) -> bool:
    p = body["machine_probability"]
    return (0.0 <= p <= 1.0 and p == p
            and abs(p + body["human_probability"] - 1.0) <= 1e-9)


def test_deterministic_across_requests(client):
    a = _analyze(client, NORMAL_TEXT).json()
    b = _analyze(client, NORMAL_TEXT).json()
    assert a["machine_probability"] == b["machine_probability"]
    assert a["label"] == b["label"]
    assert [s["contribution"] for s in a["signals"]] == \
        [s["contribution"] for s in b["signals"]]


FOREIGN_TEXT = (
    "這個項目是本地運行的，不會把文字發送到任何外部服務，所有的分析都在"
    "本機完成。Der Ausschuss hat gestern mitgeteilt, dass die neue "
    "Bibliothek im März eröffnet wird. Sie hat ungefähr 4,2 Millionen "
    "Euro gekostet und wird sieben Tage die Woche geöffnet sein. Le "
    "comité a annoncé hier que la nouvelle bibliothèque ouvrira en mars."
)


def test_non_english_input_gets_warning(client):
    r = _analyze(client, FOREIGN_TEXT)
    assert r.status_code == 200
    warnings = r.json()["warnings"]
    assert any("English" in w for w in warnings)
    assert r.json()["label"] in {"likely_human", "mostly_human", "mixed",
                                 "mostly_machine", "likely_machine",
                                 "insufficient_evidence"}


# --------------------------------------------------------------------------
# validation & error handling
# --------------------------------------------------------------------------

@pytest.mark.parametrize("payload", [
    {"text": ""},                       # empty
    {"text": "   \n\t  "},              # whitespace only
    {"text": 123},                      # wrong type
    {"text": None},                     # null
    {},                                 # missing field
    {"text": "x" * 30_001},             # over the 30,000-char bound
    {"text": NORMAL_TEXT, "max_signals": 0},   # below field bound
    {"text": NORMAL_TEXT, "max_signals": 999},  # above field bound
    {"text": NORMAL_TEXT, "include_signals": 2},   # wrong bool type
])
def test_validation_errors_are_safe_422(client, payload):
    r = client.post("/api/v1/analyze", json=payload)
    assert r.status_code == 422
    body = r.json()
    assert body["status"] == "error"
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["details"]
    for d in body["error"]["details"]:
        assert d["message"]
        assert "Traceback" not in d["message"]
    _assert_no_leak(r.text)


def test_malformed_json(client):
    r = client.post("/api/v1/analyze", content=b"{not json at all",
                    headers={"content-type": "application/json"})
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["code"] == "validation_error"
    _assert_no_leak(r.text)


def test_validation_error_does_not_echo_input(client):
    canary = "CANARY-" + "unique phrase " * 50
    r = client.post("/api/v1/analyze", json={"text": canary,
                                             "max_signals": -3})
    assert r.status_code == 422
    assert canary[:20] not in r.text


def test_unknown_route_404(client):
    r = client.get("/api/v1/nope")
    assert r.status_code == 404
    body = r.json()
    assert body["status"] == "error"
    assert body["error"]["code"] == "not_found"
    _assert_no_leak(r.text)


def test_wrong_method_405(client):
    r = client.get("/api/v1/analyze")
    assert r.status_code == 405
    assert r.json()["error"]["code"] == "method_not_allowed"


def test_unhandled_exception_handler_hides_details():
    import asyncio
    from backend.app import _unhandled_handler
    resp = asyncio.run(_unhandled_handler(
        None, ValueError(r"boom in D:\secret\path.py")))
    assert resp.status_code == 500
    content = json.loads(bytes(resp.body).decode("utf-8"))
    assert content["error"]["code"] == "internal_error"
    assert "boom" not in resp.body.decode("utf-8")
    assert "secret" not in resp.body.decode("utf-8")


# --------------------------------------------------------------------------
# load-once & privacy
# --------------------------------------------------------------------------

def test_model_not_reloaded_per_request(client, monkeypatch):
    import ml.inference as inf
    calls: list = []
    monkeypatch.setattr(inf, "load_final",
                        lambda d: calls.append(str(d)))
    for _ in range(3):
        assert _analyze(client, NORMAL_TEXT).status_code == 200
    assert client.get("/api/v1/health").status_code == 200
    assert calls == [], f"artifact reloaded {len(calls)} times"


def test_submitted_text_is_never_logged(client, caplog):
    canary = "CANARY-LOG-7f3a9c " + "sentences about a library. " * 10
    with caplog.at_level(logging.DEBUG):
        r = _analyze(client, canary)
    assert r.status_code == 200
    assert canary[:24] not in caplog.text
    assert canary[:24] not in r.text
