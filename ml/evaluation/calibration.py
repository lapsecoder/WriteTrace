"""Probability calibration helpers (manual Platt / isotonic + reliability metrics).

Implemented manually (no sklearn CalibratedClassifierCV 'prefit', removed in
sklearn >= 1.6) so behaviour is explicit, version-stable and testable.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def clip_p(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


class PlattCalibrator:
    """Logistic (Platt) mapping fit on held-out probabilities."""

    def __init__(self):
        self._lr: LogisticRegression | None = None

    def fit(self, p, y) -> "PlattCalibrator":
        z = np.log(clip_p(p) / (1 - clip_p(p))).reshape(-1, 1)
        self._lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        self._lr.fit(z, np.asarray(y, dtype=int))
        return self

    def predict(self, p) -> np.ndarray:
        z = np.log(clip_p(p) / (1 - clip_p(p))).reshape(-1, 1)
        return self._lr.predict_proba(z)[:, 1]


class IsotonicCalibrator:
    def __init__(self):
        self._ir: IsotonicRegression | None = None

    def fit(self, p, y) -> "IsotonicCalibrator":
        self._ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self._ir.fit(np.asarray(p, dtype=float), np.asarray(y, dtype=int))
        return self

    def predict(self, p) -> np.ndarray:
        return self._ir.predict(np.asarray(p, dtype=float))


def brier_score(p, y) -> float:
    p, y = clip_p(p), np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(p, y, n_bins: int = 10) -> float:
    p, y = clip_p(p), np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & ((p < hi) if i < n_bins - 1 else (p <= hi))
        if not mask.any():
            continue
        ece += mask.mean() * abs(float(p[mask].mean() - y[mask].mean()))
    return float(ece)


def reliability_bins(p, y, n_bins: int = 10) -> list[dict]:
    p, y = clip_p(p), np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & ((p < hi) if i < n_bins - 1 else (p <= hi))
        rows.append({
            "bin_lo": round(float(lo), 2), "bin_hi": round(float(hi), 2),
            "count": int(mask.sum()),
            "mean_p": round(float(p[mask].mean()), 4) if mask.any() else None,
            "frac_positive": round(float(y[mask].mean()), 4) if mask.any() else None,
        })
    return rows


def fit_best_calibrator(p_val, y_val) -> tuple[str, object, dict]:
    """Fit Platt and isotonic on validation; pick the better Brier score."""
    cands = {
        "platt": PlattCalibrator().fit(p_val, y_val),
        "isotonic": IsotonicCalibrator().fit(p_val, y_val),
    }
    scores = {k: brier_score(c.predict(p_val), y_val) for k, c in cands.items()}
    best = min(scores, key=scores.get)
    return best, cands[best], scores
