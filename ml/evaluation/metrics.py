"""Metric computation for WriteTrace baselines (accuracy is never the only metric)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .calibration import expected_calibration_error

LENGTH_BUCKETS = [(0, 50), (50, 150), (150, 400), (400, 10 ** 9)]
BUCKET_NAMES = ["<50w", "50-150w", "150-400w", "400w+"]


def compute_metrics(y_true, p_scores, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(p_scores, dtype=float)
    y_hat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()) if len(y_true) else None,
        "threshold": threshold,
        "accuracy": float((y_hat == y_true).mean()) if len(y_true) else None,
        "precision_machine": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall_machine": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1_machine": float(f1_score(y_true, y_hat, zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_hat, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_hat, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_hat, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, p)) if len(np.unique(y_true)) > 1 else None,
        "brier": float(brier_score_loss(y_true, p)),
        "ece_10": expected_calibration_error(p, y_true, n_bins=10),
        "mean_abs_p_minus_y": float(np.mean(np.abs(p - y_true))) if len(y_true) else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def slice_metrics(y_true, p_scores, words: Sequence[int], domains: Sequence[str]) -> dict:
    """Per length-bucket and per-domain metrics (honest failure visibility)."""
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(p_scores, dtype=float)
    words = np.asarray(words, dtype=int)
    out: dict[str, dict] = {"by_length": {}, "by_domain": {}}
    for (lo, hi), name in zip(LENGTH_BUCKETS, BUCKET_NAMES):
        mask = (words >= lo) & (words < hi)
        if mask.sum() >= 20 and len(np.unique(y_true[mask])) > 1:
            out["by_length"][name] = compute_metrics(y_true[mask], p[mask])
        else:
            out["by_length"][name] = {"n": int(mask.sum()), "note": "too few / single-class"}
    for dom in sorted(set(domains)):
        mask = np.asarray([d == dom for d in domains])
        if mask.sum() >= 20 and len(np.unique(y_true[mask])) > 1:
            out["by_domain"][str(dom)] = compute_metrics(y_true[mask], p[mask])
        else:
            out["by_domain"][str(dom)] = {"n": int(mask.sum()), "note": "too few / single-class"}
    return out


def _pr_auc(y, p) -> float:
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
