"""WriteTrace FastAPI backend - inference only.

One authoritative ML path: every request goes through
`ml.inference.Predictor` (the same preprocessing -> features -> model ->
calibrator -> gate code the validated experiments used). This module adds
nothing but transport, validation and presentation mapping.

Privacy / local-first:
  * submitted text is never logged, persisted, or sent anywhere;
  * no database, no auth, no telemetry, no external API calls;
  * responses contain no filesystem paths, stack traces, or debug detail.

Run (from the repository root):

    python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # allow `uvicorn backend.app:app`
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402

from ml.explain import importance_summary  # noqa: E402
from ml.inference import DISCLAIMER, get_predictor  # noqa: E402

from .schemas import (  # noqa: E402
    AnalyzeRequest, AnalyzeResponse, ErrorBody, ErrorResponse, GateInfo,
    HealthResponse, InfoResponse, Signal, TextLength, API_VERSION,
)

SERVICE_NAME = "WriteTrace"

# --------------------------------------------------------------------------
# CORS (Phase 6A: public exposure)
# --------------------------------------------------------------------------
# WriteTrace ships NO CORS policy by default: a same-origin deployment needs
# none, and a wildcard would let any website call the API from a visitor's
# browser. For a split-origin deployment (static frontend on one host, this
# API on another) set WRITETRACE_ALLOWED_ORIGINS to an explicit, comma-separated
# allowlist of EXACT origins, e.g.
#
#     WRITETRACE_ALLOWED_ORIGINS=https://writetrace.example
#
# Behaviour:
#   * unset / empty  -> no CORS headers at all (current, default behaviour)
#   * set            -> only listed origins receive Access-Control-Allow-Origin
#   * "*" is refused -> a wildcard is never honoured, even if configured: the
#                       server fails CLOSED (no CORS headers at all) rather
#                       than emitting a permissive policy or crashing.
#   * credentials are NEVER enabled (allow_credentials stays False): the API is
#     unauthenticated and cookie-based credentialing is not used.
ALLOWED_ORIGINS_ENV = "WRITETRACE_ALLOWED_ORIGINS"


def _allowed_origins() -> list[str]:
    raw = os.environ.get(ALLOWED_ORIGINS_ENV, "")
    origins = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
    if "*" in origins:                      # refuse a wildcard, fail CLOSED:
        return []                           # never emit a permissive policy
    return origins


# --------------------------------------------------------------------------
# load-once lifecycle
# --------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Everything expensive (model, calibrator, feature state, spaCy
    # pipeline, reference stats) is loaded exactly here, once per process,
    # so the first user request never pays a load cost.
    predictor = get_predictor()
    app.state.predictor = predictor
    app.state.spacy_warmup_ms = predictor.warmup()
    app.state.global_importance = importance_summary()
    yield


app = FastAPI(
    title="WriteTrace API",
    version=API_VERSION,
    description=(
        "Statistical estimate of machine-like vs human-like writing "
        "patterns. Not proof of authorship."
    ),
    lifespan=lifespan,
)

EXPECTED_4XX = {
    400: "bad_request", 404: "not_found", 405: "method_not_allowed",
    413: "payload_too_large", 415: "unsupported_media_type",
    422: "validation_error",
}


@app.middleware("http")
async def cors_headers(request: Request, call_next):
    """Emit CORS headers ONLY for explicitly allowlisted exact origins.

    No wildcard is ever emitted and credentials are never enabled. When
    WRITETRACE_ALLOWED_ORIGINS is unset the request is passed through with no
    CORS headers at all, preserving the same-origin default.
    """
    allowed = _allowed_origins()
    origin = request.headers.get("origin")
    if allowed and origin and origin.rstrip("/") in allowed:
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "content-type"
        response.headers["Access-Control-Max-Age"] = "600"
        return response
    # Unset allowlist, no Origin header, or a non-allowlisted origin (including
    # the OPTIONS preflight from such an origin): pass through untouched so the
    # browser blocks the cross-origin read.
    return await call_next(request)


def _predictor(request: Request):
    """The process-wide predictor (loaded at startup; never per request)."""
    pred = getattr(request.app.state, "predictor", None)
    if pred is None:                    # lifespan not run (defensive)
        pred = get_predictor()
        request.app.state.predictor = pred
    return pred


# --------------------------------------------------------------------------
# safe error responses (no paths, no stack traces, no input echo)
# --------------------------------------------------------------------------

def _error(status_code: int, code: str, message: str,
           details: list[dict] | None = None) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message,
                                         details=details or []))
    return JSONResponse(status_code=status_code,
                        content=body.model_dump())


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request,
                              exc: RequestValidationError) -> JSONResponse:
    details = []
    for err in exc.errors():
        loc = [x for x in err.get("loc", []) if x != "body"]
        field = ".".join(str(x) for x in loc if isinstance(x, str)) or None
        details.append({
            "field": field,
            # message only - the offending input value is deliberately NOT
            # echoed back (may contain the user's whole document).
            "message": str(err.get("msg", "invalid value")),
            "type": str(err.get("type", "")) or None,
        })
    return _error(422, "validation_error",
                  "Request validation failed. See details.", details)


@app.exception_handler(StarletteHTTPException)
async def _http_handler(request: Request,
                        exc: StarletteHTTPException) -> JSONResponse:
    code = EXPECTED_4XX.get(exc.status_code, "http_error")
    message = exc.detail if isinstance(exc.detail, str) \
        else "Request could not be completed."
    return _error(exc.status_code, code, message)


@app.exception_handler(Exception)
async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    # Never expose exception text or traceback to the client.
    return _error(500, "internal_error",
                  "An unexpected error occurred while processing the request.")


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------

def _build_response(result, include_signals: bool) -> AnalyzeResponse:
    machine = round(result.machine_probability, 6)
    human = round(1.0 - machine, 6)
    return AnalyzeResponse(
        status=result.status,
        label=result.label,
        decision=result.decision,
        machine_probability=machine,
        human_probability=human,
        confidence=result.confidence,
        word_count=result.word_count,
        text_length=TextLength(
            characters=result.text.characters,
            words=result.text.words,
            sentences=result.text.sentences,
            paragraphs=result.text.paragraphs,
        ),
        evidence_sufficient=result.evidence_sufficient,
        mixed_low_confidence=result.mixed_low_confidence,
        signals=([Signal(signal=s.feature, label=s.phrase, family=s.family,
                         value=s.value, reference_median=s.reference_median,
                         contribution=round(s.contribution, 6),
                         direction=s.direction,
                         interpretation=s.interpretation, caveat=s.caveat)
                  for s in result.signals] if include_signals else []),
        gate=GateInfo(
            min_words=result.gate_min_words,
            mixed_band=list(result.mixed_band) if result.mixed_band else None,
        ),
        model_version=result.model_version,
        model_name=result.model_name,
        calibrator=result.calibrator,
        threshold=result.threshold,
        warnings=list(result.warnings),
        disclaimer=result.disclaimer,
    )


@app.post("/api/v1/analyze", response_model=AnalyzeResponse,
          responses={422: {"model": ErrorResponse},
                     500: {"model": ErrorResponse}})
def analyze(body: AnalyzeRequest, request: Request) -> AnalyzeResponse:
    """Analyse one text: calibrated p(machine), evidence gate, signals."""
    predictor = _predictor(request)
    try:
        result = predictor.analyze(body.text,
                                   include_signals=body.include_signals,
                                   max_signals=body.max_signals)
    except ValueError as exc:            # e.g. nothing left after normalization
        raise HTTPException(status_code=422, detail=str(exc)) from None
    return _build_response(result, body.include_signals)


@app.get("/api/v1/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    predictor = _predictor(request)
    return HealthResponse(
        status="ok",
        api_version=API_VERSION,
        model_version=predictor.model_version,
        model_name=str(predictor.manifest.get("winner", "")),
        loaded=True,
    )


@app.get("/api/v1/info", response_model=InfoResponse)
def info(request: Request) -> InfoResponse:
    predictor = _predictor(request)
    m = predictor.manifest
    imp = getattr(request.app.state, "global_importance", None) \
        or importance_summary()
    top = [{"feature": t["feature"], "label": t["phrase"],
            "delta_pr_auc": t["delta_pr_auc"]}
           for t in (imp.get("top") or [])[:10]]
    return InfoResponse(
        api_version=API_VERSION,
        model_version=predictor.model_version,
        model_name=str(m.get("winner", "")),
        calibrator=str(m.get("calibrator", "")),
        threshold=predictor.threshold,
        n_features=len(predictor.features),
        syntactic_features=predictor.needs_syn,
        gate=GateInfo(min_words=predictor.min_words,
                      mixed_band=list(predictor.mixed_band)
                      if predictor.mixed_band else None),
        pipeline_version=str(m.get("pipeline_version", "")),
        label_vocabulary=["likely_human", "mostly_human", "mixed",
                          "mostly_machine", "likely_machine",
                          "insufficient_evidence"],
        data_policy=m.get("data_policy", {}),
        global_importance=top,
        importance_method=imp.get("method"),
        eval_summary=m.get("eval_summary", {}),
        disclaimer=DISCLAIMER,
    )
