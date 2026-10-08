"""Phase-4 API smoke test against a REAL running server.

Starts uvicorn locally, sends real HTTP requests to POST /api/v1/analyze
(short / normal human / machine reference / unicode / repeated / invalid),
measures end-to-end latency and records safe response summaries (no request
text is stored). Writes logs/phase4_smoke.json.

Usage: python scripts/smoke_phase4.py [--port 8017]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from ml.experiments.phase4 import LOGS, sample_texts, write_json  # noqa: E402

UNICODE_TEXT = (
    "Le comité a annoncé hier que la nouvelle bibliothèque ouvrira en mars. "
    "Elle a coûté environ 4,2 millions d'euros, ont dit les responsables, et "
    "elle restera ouverte sept jours sur sept. Nous sommes ravis, a déclaré "
    "la maire. 這個項目是本地運行的，不會把文字發送到任何外部服務。 "
    "I have to admit the delays frustrated quite a few residents, but the "
    "finished building looks wonderful. Bring your library cards! 🎉"
)

REPEATED = " ".join(
    ["The quick brown fox jumps over the lazy dog again and again." * 2] * 4)


def _summarize(payload: dict) -> dict:
    """Safe subset of an analyze response (no user text, no paths)."""
    keep = ("status", "label", "decision", "machine_probability",
            "human_probability", "confidence", "word_count",
            "evidence_sufficient", "mixed_low_confidence", "model_version",
            "model_name", "threshold")
    out = {k: payload.get(k) for k in keep}
    out["n_signals"] = len(payload.get("signals") or [])
    out["n_warnings"] = len(payload.get("warnings") or [])
    out["warnings"] = payload.get("warnings") or []
    out["text_length"] = payload.get("text_length")
    out["gate"] = payload.get("gate")
    return out


def _wait_ready(client: httpx.Client, timeout_s: float = 90.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = client.get("/api/v1/health")
            if r.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase-4 API smoke test")
    ap.add_argument("--port", type=int, default=8017)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    samples = sample_texts()
    base_sentence = ("The committee announced yesterday that the new library "
                     "will open in March after two years of renovation work. "
                     "It cost roughly 4.2 million euros, officials said, and "
                     "will stay open seven days a week from eight in the "
                     "morning until ten at night. ")
    long_text = base_sentence
    while len(long_text) <= 30_000:      # strictly over the API char bound
        long_text += base_sentence

    cases: list[dict] = [
        {"name": "short_under_gate", "expect_status": 200,
         "json": {"text": samples["short_under_gate"]}},
        {"name": "normal_human_text", "expect_status": 200,
         "json": {"text": samples["typical_human"]}},
        {"name": "machine_reference_text", "expect_status": 200,
         "json": {"text": samples["typical_machine"]}},
        {"name": "unicode_mixed_language", "expect_status": 200,
         "json": {"text": UNICODE_TEXT}},
        {"name": "repeated_text", "expect_status": 200,
         "json": {"text": REPEATED}},
        {"name": "empty_text", "expect_status": 422, "json": {"text": ""}},
        {"name": "whitespace_only", "expect_status": 422,
         "json": {"text": "   \n\t  "}},
        {"name": "wrong_type", "expect_status": 422, "json": {"text": 123}},
        {"name": "missing_field", "expect_status": 422, "json": {}},
        {"name": "oversized_text", "expect_status": 422,
         "json": {"text": long_text}},
        {"name": "analyze_via_get", "method": "GET",
         "expect_status": 405, "url": "/api/v1/analyze"},
        {"name": "unknown_route", "expect_status": 404,
         "url": "/api/v1/nope"},
        {"name": "health", "expect_status": 200, "method": "GET",
         "url": "/api/v1/health"},
    ]

    log: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": f"127.0.0.1:{args.port}",
        "endpoint": "POST /api/v1/analyze",
        "cases": [],
    }

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app:app",
         "--host", "127.0.0.1", "--port", str(args.port),
         "--log-level", "warning"],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ok = False
    try:
        with httpx.Client(base_url=base, timeout=120.0) as client:
            if not _wait_ready(client):
                out = proc.communicate(timeout=10)[0] if proc.poll() else b""
                log["server_output_tail"] = out.decode("utf-8", "replace")[-800:]
                log["server_ready"] = False
                write_json(LOGS / "phase4_smoke.json", log)
                print("server failed to become ready")
                return 1
            log["server_ready"] = True

            # malformed JSON (raw body)
            cases.insert(5, {"name": "malformed_json", "expect_status": 422,
                             "raw": b"{not json",
                             "content_type": "application/json"})

            for case in cases:
                method = case.get("method", "POST")
                url = case.get("url", "/api/v1/analyze")
                t0 = time.perf_counter()
                if "raw" in case:
                    r = client.request(method, url, content=case["raw"],
                                       headers={"content-type":
                                                case["content_type"]})
                elif "json" in case:
                    r = client.request(method, url, json=case["json"])
                else:
                    r = client.request(method, url)
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)

                entry = {
                    "name": case["name"],
                    "method": method,
                    "url": url,
                    "expected_status": case["expect_status"],
                    "status": r.status_code,
                    "latency_ms": latency_ms,
                    "status_ok": r.status_code == case["expect_status"],
                }
                if "json" in case and isinstance(case["json"], dict) \
                        and isinstance(case["json"].get("text"), str):
                    txt = case["json"]["text"]
                    entry["text_words"] = len(txt.split())
                    entry["text_sha256_8"] = hashlib.sha256(
                        txt.encode("utf-8")).hexdigest()[:8]
                try:
                    body = r.json()
                except ValueError:
                    body = None
                if isinstance(body, dict) and "label" in body:
                    entry["response"] = _summarize(body)
                    # invariants on every successful analyze response
                    if r.status_code == 200:
                        entry["invariants"] = {
                            "probs_in_0_1":
                                0.0 <= body["machine_probability"] <= 1.0
                                and 0.0 <= body["human_probability"] <= 1.0,
                            "probs_sum_to_1":
                                abs(body["machine_probability"]
                                    + body["human_probability"] - 1.0) <= 1e-9,
                            "no_nan": body["machine_probability"]
                            == body["machine_probability"],
                            "has_disclaimer": bool(body.get("disclaimer")),
                            "no_path_leak":
                                "D:\\" not in json.dumps(body)
                                and "/ml/models" not in json.dumps(body),
                        }
                elif isinstance(body, dict) and "error" in body:
                    entry["error"] = body["error"]
                log["cases"].append(entry)
            ok = all(c["status_ok"] for c in log["cases"])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    log["all_passed"] = bool(ok)
    write_json(LOGS / "phase4_smoke.json", log)
    print(f"smoke -> {LOGS / 'phase4_smoke.json'}")
    for c in log["cases"]:
        lat = c.get("latency_ms")
        extra = ""
        if c.get("response"):
            rr = c["response"]
            extra = (f" {rr['label']} p={rr['machine_probability']} "
                     f"words={rr['word_count']} signals={rr['n_signals']}")
        mark = "PASS" if c["status_ok"] else "FAIL"
        print(f"  [{mark}] {c['name']:22s} -> {c['status']} "
              f"({lat} ms){extra}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
