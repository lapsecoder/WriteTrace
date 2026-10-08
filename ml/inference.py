"""Inference API for the WriteTrace final model (Phase-4-reusable).

Loads the versioned artifact written by Phase-3 SELECTION + E8:
    ml/models/final-v1/{model,calibrator,feature_state}.joblib,
    features.json, config.json, gate.json, manifest.json, reference.json.

Contract:
  * p(machine) = calibrator(model(x)) - Phase-3 candidates are trained on
    canonical labels, so there is NO orientation complement here.
  * threshold 0.5 (Phase-1 fixed, never tuned).
  * evidence gate: if gate.json declares min_words and the input is shorter,
    the decision is ABSTAIN ("insufficient evidence"); p is still returned
    and flagged. A mixed/low-confidence band (if declared) flags predictions
    whose calibrated probability falls in the empirically unreliable interval.
  * never emits authorship percentages or claims beyond the binary decision;
    explanations describe model behavior, not authorship causation.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .explain import format_explanation, load_reference, local_contributions
from .experiments.phase3 import FINAL_DIR, load_final
from .features.extract import extract_many
from .preprocessing.normalize import normalize_text, word_count


class Predictor:
    """Load-once predictor around the versioned final artifact."""

    def __init__(self, artifact_dir: Path | None = None):
        self.dir = Path(artifact_dir) if artifact_dir else FINAL_DIR
        self.final = load_final(self.dir)
        self.features: list[str] = self.final["features"]
        self.needs_syn = any(f.startswith("syn_") for f in self.features)
        self._nlp = None
        self._ref = None

    # -- metadata ----------------------------------------------------------
    @property
    def manifest(self) -> dict:
        return self.final["manifest"]

    @property
    def gate(self) -> dict:
        g = self.final.get("gate") or {}
        return g if g.get("status") == "active" else {}

    def info(self) -> dict:
        m = self.manifest
        g = self.gate
        return {
            "model": m.get("winner"),
            "calibrator": m.get("calibrator"),
            "threshold": m.get("threshold", 0.5),
            "n_features": len(self.features),
            "syntactic_family": self.needs_syn,
            "gate_min_words": g.get("min_words"),
            "mixed_band": g.get("mixed_band"),
            "artifact": str(self.dir),
            "eval_summary": m.get("eval_summary"),
        }

    # -- extraction --------------------------------------------------------
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

    # -- prediction --------------------------------------------------------
    def predict(self, text: str) -> dict:
        """Full decision record for one raw text."""
        nt = normalize_text(text)
        nt = nt if nt else ""
        words = word_count(nt)
        row = self._extract_row(nt)
        p = float(self.final["calibrator"].predict(
            self.final["model"].predict_proba(
                row[self.features].to_numpy())[:, 1])[0])
        label = "machine" if p >= 0.5 else "human"
        g = self.gate
        abstain = bool(g.get("min_words") is not None
                       and words < int(g["min_words"]))
        band = g.get("mixed_band")
        mixed = bool(band and band[0] <= p <= band[1])
        if abstain:
            decision = "abstain"
            reason = (f"insufficient evidence: {words} words < min_words="
                      f"{g['min_words']}")
        else:
            decision = label
            reason = ""
        return {
            "p_machine": round(p, 4),
            "label": label,
            "decision": decision,
            "abstain": abstain,
            "reason": reason,
            "mixed_low_confidence": mixed,
            "n_words": int(words),
            "threshold": 0.5,
            "convention": "canonical: y=1 MACHINE, y=0 HUMAN",
            "gate": {"min_words": g.get("min_words"),
                     "mixed_band": band},
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]

    # -- explanation -------------------------------------------------------
    def explain(self, text: str, k: int = 8) -> dict:
        nt = normalize_text(text)
        nt = nt if nt else ""
        row = self._extract_row(nt)
        if self._ref is None:
            self._ref = load_reference(self.dir)["train_reference_stats"]
        contribs = local_contributions(
            self.final["model"], self.final["calibrator"], self.features,
            row[self.features].to_numpy()[0], self._ref, k=k)
        pred = self.predict(text)
        return {
            "prediction": pred,
            "contributions": contribs,
            "text_rendered": format_explanation(
                pred["p_machine"], pred["label"], contribs, pred["n_words"]),
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
