"""WriteTrace inference engine - the ONE authoritative inference path.

Loads the versioned artifact written by Phase-3 SELECTION + E8:

    ml/models/final-v1/{model,calibrator,feature_state}.joblib,
    features.json, config.json, gate.json, manifest.json, reference.json

Contract (verified against the Phase-3 artifacts, never re-derived here):
  * p(machine) = calibrator(model(x)) - Phase-3 candidates are trained on
    canonical labels (y=1 MACHINE), so there is NO orientation complement.
  * threshold 0.5 (Phase-1 fixed, never tuned); all decisions use the
    FULL-precision probability - rounding happens only at presentation time.
  * evidence gate: inputs shorter than gate.json min_words (50) ABSTAIN
    ("insufficient evidence"); the probability is still reported for
    transparency but flagged as unsuitable for a conclusion.
  * mixed band (gate.json, [0.4, 0.6]): calibrated probabilities inside this
    interval are flagged `mixed_low_confidence` (observed val_cal accuracy
    there was 0.5342).
  * explanations are one-at-a-time reference perturbations - model-behavior
    descriptions, never causal attribution; no authorship percentages.

Structure (Phase-4): TextAnalysis / FeatureEvidence / PredictionResult are
the framework-free result types the API layer maps to Pydantic models.
SegmentAnalysis is a reserved slot for the Phase-5 segment model - the
pipeline never assumes whole-document-only input.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .explain import format_explanation, load_reference, local_contributions, phrase
from .experiments.phase3 import FINAL_DIR, load_final
from .features.extract import FEATURE_FAMILIES, extract_many, split_sentences
from .preprocessing.normalize import looks_english, normalize_text, word_count

MODEL_VERSION_FALLBACK = "final-v1"

DISCLAIMER = (
    "WriteTrace does not establish authorship. It estimates how strongly "
    "observed writing patterns resemble machine-generated or human-written "
    "text under the model's training and evaluation conditions."
)

SIGNAL_CAVEAT = (
    "Statistical stylistic signal measured by the model; not proof of "
    "authorship."
)

# Presentation-only label cutoffs (docs/api.md). The inner boundary IS the
# Phase-3 mixed band read from gate.json; the outer 0.2/0.8 cutoffs are a
# symmetric presentation convention - documented, never fitted or tuned, and
# never used for the binary decision (which stays at threshold 0.5).
OUTER_CUT = 0.2

FAMILY_OF: dict[str, str] = {
    f: fam for fam, names in FEATURE_FAMILIES.items() for f in names
}


def family_of(feature: str) -> str:
    if feature.startswith("syn_"):
        return "syntactic_spacy"
    return FAMILY_OF.get(feature, "other")


# --------------------------------------------------------------------------
# result types
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TextAnalysis:
    """Document-size facts measured on the normalized input (no claims)."""
    characters: int
    words: int
    sentences: int
    paragraphs: int
    normalized: bool = True  # input went through the training normalizer


@dataclass(frozen=True)
class FeatureEvidence:
    """One statistical signal (local reference perturbation)."""
    feature: str
    phrase: str
    family: str
    value: float | None
    reference_median: float
    contribution: float
    direction: str            # "machine" | "human" | "neutral"
    interpretation: str
    caveat: str = SIGNAL_CAVEAT


@dataclass(frozen=True)
class SegmentAnalysis:
    """RESERVED for Phase-5 segment-level scoring. Never populated here.

    Kept in the result shape now so the API contract, the inference engine
    and the future segment model share one pipeline instead of bolting
    segments on later.
    """
    status: str = "not_available"
    count: int = 0
    note: str = ("segment-level analysis is not implemented in this version; "
                 "the same preprocessing/feature/model/calibration path will "
                 "be reused when it is")


@dataclass(frozen=True)
class PredictionResult:
    """Full structured decision for one raw text (framework-free)."""
    status: str                       # "ok" | "insufficient_evidence"
    label: str                        # 5-level presentation label or insufficient_evidence
    decision: str                     # "machine" | "human" | "abstain"
    machine_probability: float        # full precision
    human_probability: float          # 1 - machine_probability
    confidence: float | None          # None when evidence is insufficient
    evidence_sufficient: bool
    word_count: int
    text: TextAnalysis
    signals: list[FeatureEvidence] = field(default_factory=list)
    segments: list[SegmentAnalysis] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    abstain_reason: str = ""
    mixed_low_confidence: bool = False
    model_version: str = MODEL_VERSION_FALLBACK
    model_name: str = ""
    calibrator: str = ""
    threshold: float = 0.5
    gate_min_words: int | None = None
    mixed_band: tuple[float, float] | None = None
    disclaimer: str = DISCLAIMER


# --------------------------------------------------------------------------
# presentation policy (documented; separate from model output)
# --------------------------------------------------------------------------

def presentation_label(p: float, evidence_sufficient: bool,
                       mixed_band: tuple[float, float] | None) -> str:
    """Map a calibrated probability to the 5-level presentation vocabulary.

    Phase-3 defines the binary decision (threshold 0.5) and the inner
    "mixed" band; the outer likely/mostly split (0.2/0.8) is a documented
    presentation convention, not a fitted artifact.
    """
    if not evidence_sufficient:
        return "insufficient_evidence"
    lo, hi = mixed_band if mixed_band else (0.4, 0.6)
    if lo <= p <= hi:
        return "mixed"
    if p < lo:
        return "likely_human" if p < OUTER_CUT else "mostly_human"
    return "mostly_machine" if p <= 1 - OUTER_CUT else "likely_machine"


# --------------------------------------------------------------------------
# predictor
# --------------------------------------------------------------------------

class Predictor:
    """Load-once predictor around the versioned final artifact.

    Everything expensive (model, calibrator, feature state, spaCy pipeline,
    reference stats) is loaded on construction / first use and reused for
    every subsequent call. Inference is deterministic: same text -> same
    features -> same probability.
    """

    def __init__(self, artifact_dir: Path | None = None):
        self.dir = Path(artifact_dir) if artifact_dir else FINAL_DIR
        self.final = load_final(self.dir)
        self.features: list[str] = self.final["features"]
        self.needs_syn = any(f.startswith("syn_") for f in self.features)
        self.model_version = self.dir.name or MODEL_VERSION_FALLBACK
        self._ref: dict | None = None

    # -- metadata ----------------------------------------------------------
    @property
    def manifest(self) -> dict:
        return self.final["manifest"]

    @property
    def gate(self) -> dict:
        g = self.final.get("gate") or {}
        return g if g.get("status") == "active" else {}

    @property
    def threshold(self) -> float:
        return float(self.final["manifest"].get("threshold", 0.5))

    @property
    def mixed_band(self) -> tuple[float, float] | None:
        band = self.gate.get("mixed_band")
        if band and len(band) == 2:
            return (float(band[0]), float(band[1]))
        return None

    @property
    def min_words(self) -> int | None:
        mw = self.gate.get("min_words")
        return int(mw) if mw is not None else None

    def info(self) -> dict:
        """Internal metadata (contains the artifact path - never exposed
        through the API)."""
        m = self.manifest
        return {
            "model": m.get("winner"),
            "calibrator": m.get("calibrator"),
            "threshold": m.get("threshold", 0.5),
            "n_features": len(self.features),
            "syntactic_family": self.needs_syn,
            "gate_min_words": self.min_words,
            "mixed_band": list(self.mixed_band) if self.mixed_band else None,
            "model_version": self.model_version,
            "artifact": str(self.dir),
            "eval_summary": m.get("eval_summary"),
        }

    # -- extraction / scoring ---------------------------------------------
    def _normalize(self, text: str) -> str:
        if text is None or not isinstance(text, str):
            raise ValueError("text must be a string")
        nt = normalize_text(text)
        if not nt:
            raise ValueError("text must contain at least one non-whitespace "
                             "character")
        return nt

    def _extract_row(self, text_norm: str) -> pd.DataFrame:
        feats = extract_many([text_norm], self.final["state"])
        if self.needs_syn:
            from .features.syntactic import extract_syntactic
            feats = pd.concat([feats.reset_index(drop=True),
                               extract_syntactic([text_norm])], axis=1)
        missing = [f for f in self.features if f not in feats.columns]
        if missing:
            raise RuntimeError(f"missing feature columns: {missing[:5]}")
        return feats

    def _score_row(self, row: pd.DataFrame) -> float:
        """Full-precision calibrated p(machine) for one feature row."""
        raw = self.final["model"].predict_proba(
            row[self.features].to_numpy())[:, 1]
        p = float(self.final["calibrator"].predict(raw)[0])
        if not math.isfinite(p):
            raise RuntimeError("model produced a non-finite probability")
        return p

    def score(self, text: str) -> float:
        """Full-precision p(machine) for one raw text (no rounding, no gate)."""
        nt = self._normalize(text)
        return self._score_row(self._extract_row(nt))

    def warmup(self) -> float:
        """Force-load every lazy static resource (spaCy pipeline) once.

        Called at server startup so the first user request never pays the
        model-load cost. Returns elapsed milliseconds.
        """
        t0 = time.perf_counter()
        self._extract_row("Warmup input used to load static pipelines.")
        return round((time.perf_counter() - t0) * 1000.0, 1)

    def _prepare(self, text: str) -> tuple[str, TextAnalysis]:
        nt = self._normalize(text)
        words = word_count(nt)
        sents = split_sentences(nt) or [nt]
        paras = [q for q in nt.split("\n\n") if q.strip()] or [nt]
        ta = TextAnalysis(characters=len(nt), words=words,
                          sentences=len(sents), paragraphs=len(paras))
        return nt, ta

    def _signals(self, row: pd.DataFrame, p: float, k: int) -> list[FeatureEvidence]:
        if self._ref is None:
            self._ref = load_reference(self.dir)["train_reference_stats"]
        contribs = local_contributions(
            self.final["model"], self.final["calibrator"], self.features,
            row[self.features].to_numpy()[0], self._ref, k=k)
        out: list[FeatureEvidence] = []
        for c in contribs:
            contrib = float(c["contribution_to_p_machine"])
            direction = ("machine" if contrib > 1e-6
                         else "human" if contrib < -1e-6 else "neutral")
            if direction == "neutral":
                interp = ("observed value matches the effect of a typical "
                          "training document; no measurable contribution")
            else:
                interp = (f"observed value {c['value']} vs train median "
                          f"{c['reference_median']:g} moved the estimate "
                          f"toward {direction} by {contrib:+.4f}")
            out.append(FeatureEvidence(
                feature=c["feature"], phrase=c["phrase"],
                family=family_of(c["feature"]), value=c["value"],
                reference_median=float(c["reference_median"]),
                contribution=contrib, direction=direction, interpretation=interp))
        return out

    # -- main entry points -------------------------------------------------
    def analyze(self, text: str, include_signals: bool = True,
                max_signals: int = 8) -> PredictionResult:
        """Full structured analysis for one raw text."""
        if max_signals < 0:
            raise ValueError("max_signals must be >= 0")
        nt, ta = self._prepare(text)
        row = self._extract_row(nt)
        p = self._score_row(row)

        min_words = self.min_words
        band = self.mixed_band
        evidence_ok = min_words is None or ta.words >= min_words
        abstain = not evidence_ok
        mixed = bool(band and band[0] <= p <= band[1])
        threshold = self.threshold
        if abstain:
            decision = "abstain"
            reason = (f"insufficient evidence: {ta.words} words < "
                      f"min_words={min_words}")
            status = "insufficient_evidence"
        else:
            decision = "machine" if p >= threshold else "human"
            reason = ""
            status = "ok"

        warnings: list[str] = []
        if abstain:
            warnings.append(
                f"insufficient evidence: the input has {ta.words} words, "
                f"below the {min_words}-word threshold selected in Phase 3; "
                "the probability is shown for transparency and must not "
                "support an authorship conclusion")
        if mixed:
            acc = self.gate.get("mixed_band_accuracy")
            warnings.append(
                f"mixed/low confidence: p(machine) lies inside the Phase-3 "
                f"mixed band [{band[0]}, {band[1]}]"
                + (f" where observed validation accuracy was only {acc}"
                   if acc is not None else ""))
        if not looks_english(nt):
            warnings.append(
                "input does not appear to be English; WriteTrace was trained "
                "and evaluated on English text only")

        signals = (self._signals(row, p, max_signals)
                   if include_signals else [])

        p_m = float(p)
        return PredictionResult(
            status=status,
            label=presentation_label(p_m, evidence_ok, band),
            decision=decision,
            machine_probability=p_m,
            human_probability=float(1.0 - p_m),
            confidence=(None if abstain else round(max(p_m, 1.0 - p_m), 4)),
            evidence_sufficient=evidence_ok,
            word_count=int(ta.words),
            text=ta,
            signals=signals,
            warnings=warnings,
            abstain_reason=reason,
            mixed_low_confidence=mixed,
            model_version=self.model_version,
            model_name=str(self.manifest.get("winner", "")),
            calibrator=str(self.manifest.get("calibrator", "")),
            threshold=threshold,
            gate_min_words=min_words,
            mixed_band=band,
        )

    def _legacy(self, r: "PredictionResult") -> dict:
        """Compact decision record (Phase-3 test-suite contract): the label
        stays binary even when the gate abstains; `decision` carries the gate."""
        binary = "machine" if r.machine_probability >= r.threshold else "human"
        return {
            "p_machine": round(r.machine_probability, 4),
            "label": binary,
            "decision": r.decision,
            "abstain": not r.evidence_sufficient,
            "reason": r.abstain_reason,
            "mixed_low_confidence": r.mixed_low_confidence,
            "n_words": int(r.word_count),
            "threshold": r.threshold,
            "convention": "canonical: y=1 MACHINE, y=0 HUMAN",
            "gate": {"min_words": r.gate_min_words,
                     "mixed_band": list(r.mixed_band) if r.mixed_band else None},
        }

    def predict(self, text: str) -> dict:
        """Legacy compact decision record (kept for the Phase-3 test suite)."""
        return self._legacy(self.analyze(text, include_signals=False))

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]

    # -- explanation -------------------------------------------------------
    def explain(self, text: str, k: int = 8) -> dict:
        r = self.analyze(text, include_signals=True, max_signals=k)
        contribs = [{
            "feature": s.feature, "phrase": s.phrase, "value": s.value,
            "reference_median": s.reference_median,
            "contribution_to_p_machine": round(s.contribution, 4),
            "direction": (f"toward {s.direction}" if s.direction != "neutral"
                          else "no effect"),
        } for s in r.signals]
        return {
            "prediction": self._legacy(r),
            "contributions": contribs,
            "text_rendered": format_explanation(
                round(r.machine_probability, 4),
                "machine" if r.machine_probability >= r.threshold else "human",
                contribs, r.word_count),
            "method": "one-at-a-time reference perturbation (train medians); "
                      "model-behavior description, not causal attribution",
        }


_default: Predictor | None = None


def get_predictor(artifact_dir: Path | None = None) -> Predictor:
    """Process-wide singleton for the default artifact."""
    global _default
    if _default is None or (artifact_dir is not None
                            and Path(artifact_dir) != _default.dir):
        _default = Predictor(artifact_dir)
    return _default


def reset_predictor() -> None:
    """Drop the singleton (tests only)."""
    global _default
    _default = None
