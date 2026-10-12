"""Phase-6A tests: public-exposure behaviour of the API.

Covers the exact-origin CORS mechanism (no wildcard, no credentials, opt-in),
input-size enforcement, and the guarantee that error bodies never leak the
submitted text, filesystem paths, or library internals.
"""
from __future__ import annotations

import importlib
import os

import pytest
from fastapi.testclient import TestClient

SECRET = "UNIQUE-MARKER-9f3a2b-SECRET-DO-NOT-LEAK"


@pytest.fixture()
def client():
    from backend.app import app
    with TestClient(app) as c:
        yield c


def test_no_cors_headers_by_default(client, monkeypatch):
    """The shipped default is same-origin: no CORS headers at all."""
    monkeypatch.delenv("WRITETRACE_ALLOWED_ORIGINS", raising=False)
    r = client.post("/api/v1/analyze", json={"text": "hello"},
                    headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}


def test_allowlisted_origin_receives_cors_headers(client, monkeypatch):
    monkeypatch.setenv("WRITETRACE_ALLOWED_ORIGINS",
                       "https://writetrace.example")
    r = client.post("/api/v1/analyze", json={"text": "hello"},
                    headers={"Origin": "https://writetrace.example"})
    assert r.headers.get("access-control-allow-origin") == \
        "https://writetrace.example"
    assert "vary" in {k.lower() for k in r.headers}


def test_non_allowlisted_origin_is_not_granted(client, monkeypatch):
    monkeypatch.setenv("WRITETRACE_ALLOWED_ORIGINS",
                       "https://writetrace.example")
    r = client.post("/api/v1/analyze", json={"text": "hello"},
                    headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}


def test_wildcard_is_refused(client, monkeypatch):
    monkeypatch.setenv("WRITETRACE_ALLOWED_ORIGINS", "*")
    r = client.post("/api/v1/analyze", json={"text": "hello"},
                    headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}


def test_credentials_are_never_enabled(client, monkeypatch):
    monkeypatch.setenv("WRITETRACE_ALLOWED_ORIGINS",
                       "https://writetrace.example")
    r = client.post("/api/v1/analyze", json={"text": "hello"},
                    headers={"Origin": "https://writetrace.example"})
    assert "access-control-allow-credentials" not in \
        {k.lower() for k in r.headers}


def test_wildcard_config_fails_closed(client, monkeypatch):
    """A '*' config is never honoured: no CORS headers, no crash."""
    from backend.app import _allowed_origins
    monkeypatch.setenv("WRITETRACE_ALLOWED_ORIGINS", "https://a.example, *")
    assert _allowed_origins() == []
    r = client.post("/api/v1/analyze", json={"text": "hello"},
                    headers={"Origin": "https://evil.example"})
    assert r.status_code == 200, "must fail closed, not error out"
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}


def test_input_size_is_enforced(client):
    assert client.post("/api/v1/analyze",
                       json={"text": "x" * 30_000}).status_code == 200
    assert client.post("/api/v1/analyze",
                       json={"text": "x" * 30_001}).status_code == 422
    assert client.post("/api/v1/analyze",
                       json={"text": "x" * 200_000}).status_code == 422


@pytest.mark.parametrize("payload", [
    {"text": ""},
    {"text": "   "},
    {"text": 123},
    {"text": None},
    {},
])
def test_invalid_payloads_return_422(client, payload):
    assert client.post("/api/v1/analyze", json=payload).status_code == 422


def test_error_bodies_do_not_echo_submitted_text(client):
    r = client.post("/api/v1/analyze", json={"text": SECRET})
    # 'SECRET' is non-empty so this is a 200; the error path is an oversized
    # body that carries the marker in the payload.
    over = client.post("/api/v1/analyze", json={"text": SECRET + "x" * 40_000})
    assert over.status_code == 422
    assert SECRET not in over.text
    assert "Traceback" not in over.text
    assert "site-packages" not in over.text
    assert "D:\\" not in over.text


def test_segments_are_never_populated(client):
    r = client.post("/api/v1/analyze", json={"text": "The committee " * 80})
    assert r.status_code == 200
    assert r.json()["segments"] == []