"""WriteTrace API contract (Pydantic v2).

Design notes
------------
* Request validation is strict but kind: wrong types / missing field /
  malformed JSON -> 422 with SAFE messages (no input echo, no paths, no
  stack traces). Empty / whitespace-only text is rejected the same way.
* Long text is bounded (MAX_TEXT_CHARS, from docs/technical-specification.md
  3: 50-30,000 chars) -> 422; text below the Phase-3 evidence threshold is
  NOT rejected, it returns 200 with evidence_sufficient=false.
* Probabilities are computed at full precision inside ml.inference and only
  rounded here for presentation (6 dp); human_probability is derived as
  1 - machine_probability so the pair is always consistent.
* `label` is an INFERENCE-TIME presentation vocabulary (5 levels + the
  insufficient-evidence state). The underlying model is binary; see docs/api.md.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

API_VERSION = "v1"
MAX_TEXT_CHARS = 30_000
MIN_TEXT_CHARS = 1

Label = Literal["likely_human", "mostly_human", "mixed", "mostly_machine",
                "likely_machine", "insufficient_evidence"]
Decision = Literal["machine", "human", "abstain"]
Direction = Literal["machine", "human", "neutral"]


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "examples": [{
                "text": "The committee announced yesterday that the new "
                        "library will open in March...",
                "include_signals": True,
                "max_signals": 8,
            }]
        },
    )

    text: str = Field(..., min_length=MIN_TEXT_CHARS,
                      max_length=MAX_TEXT_CHARS,
                      description="Raw text to analyse (English).")
    include_signals: bool = Field(
        default=True,
        description="Return local statistical signals (slower when true).")
    max_signals: int = Field(default=8, ge=1, le=50,
                             description="Maximum number of signals returned.")

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                "text must contain at least one non-whitespace character")
        return v


class TextLength(BaseModel):
    characters: int
    words: int
    sentences: int
    paragraphs: int


class Signal(BaseModel):
    """One statistical stylistic signal (local reference perturbation)."""
    signal: str
    label: str
    family: str
    value: Optional[float] = None
    reference_median: float
    contribution: float
    direction: Direction
    interpretation: str
    caveat: str


class Segment(BaseModel):
    """Reserved for Phase-5 segment-level analysis (always empty now)."""
    status: str = "not_available"
    count: int = 0
    note: str = ""


class GateInfo(BaseModel):
    min_words: Optional[int] = None
    mixed_band: Optional[list[float]] = None


class AnalyzeResponse(BaseModel):
    status: Literal["ok", "insufficient_evidence"]
    label: Label
    decision: Decision
    machine_probability: float = Field(ge=0.0, le=1.0)
    human_probability: float = Field(ge=0.0, le=1.0)
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="max(machine, human) probability; null when evidence is "
                    "insufficient. Not a separately trained confidence model.")
    word_count: int
    text_length: TextLength
    evidence_sufficient: bool
    mixed_low_confidence: bool
    signals: list[Signal] = []
    segments: list[Segment] = []
    gate: GateInfo
    model_version: str
    model_name: str
    calibrator: str
    threshold: float
    warnings: list[str] = []
    disclaimer: str


class ErrorDetail(BaseModel):
    field: Optional[str] = None
    message: str
    type: Optional[str] = None


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[ErrorDetail] = []


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    error: ErrorBody


class HealthResponse(BaseModel):
    status: Literal["ok"]
    api_version: str
    model_version: str
    model_name: str
    loaded: bool


class InfoResponse(BaseModel):
    api_version: str
    model_version: str
    model_name: str
    calibrator: str
    threshold: float
    n_features: int
    syntactic_features: bool
    gate: GateInfo
    pipeline_version: str
    label_vocabulary: list[str]
    data_policy: dict
    global_importance: list[dict]
    importance_method: Optional[str] = None
    eval_summary: dict
    disclaimer: str
