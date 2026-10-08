"""Tests for calibration and metric computation."""
import numpy as np

from ml.evaluation.calibration import (IsotonicCalibrator, PlattCalibrator,
                                       brier_score, expected_calibration_error,
                                       fit_best_calibrator, reliability_bins)
from ml.evaluation.metrics import compute_metrics, slice_metrics


def _toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    p = np.clip(rng.uniform(0.02, 0.98, n), 0.01, 0.99)
    y = (rng.uniform(size=n) < p).astype(int)
    return p, y


def test_brier_and_ece_are_probabilities():
    p, y = _toy()
    assert 0.0 <= brier_score(p, y) <= 1.0
    assert 0.0 <= expected_calibration_error(p, y) <= 1.0
    perfect = np.full(50, 0.7)
    y_bin = (np.arange(50) < 35).astype(int)  # exactly 0.7 positive rate
    assert expected_calibration_error(perfect, y_bin) < 1e-9


def test_platt_calibration_is_monotonic_and_bounded():
    p, y = _toy()
    cal = PlattCalibrator().fit(p, y)
    out = cal.predict(np.array([0.05, 0.5, 0.95]))
    assert np.all((out > 0) & (out < 1))
    assert out[0] <= out[1] <= out[2]


def test_isotonic_calibration_is_monotonic_and_bounded():
    p, y = _toy()
    cal = IsotonicCalibrator().fit(p, y)
    out = cal.predict(np.linspace(0.01, 0.99, 50))
    assert np.all((out >= 0) & (out <= 1))
    assert np.all(np.diff(out) >= -1e-9)  # non-decreasing


def test_fit_best_calibrator_picks_by_brier():
    p, y = _toy()
    name, cal, scores = fit_best_calibrator(p, y)
    assert name in ("platt", "isotonic")
    assert set(scores) == {"platt", "isotonic"}
    assert scores[name] <= min(scores.values())


def test_reliability_bins_partition_the_sample():
    p, y = _toy(500)
    bins = reliability_bins(p, y, n_bins=10)
    assert len(bins) == 10
    assert sum(b["count"] for b in bins) == 500


def test_compute_metrics_perfect_prediction():
    y = [0, 0, 1, 1]
    p = [0.1, 0.4, 0.6, 0.9]
    m = compute_metrics(y, p)
    assert m["accuracy"] == 1.0
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0
    assert m["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
    for k in ("precision_machine", "recall_machine", "f1_machine", "f1_macro",
              "brier", "ece_10"):
        assert k in m


def test_compute_metrics_single_class_does_not_crash():
    m = compute_metrics([0, 0, 0], [0.1, 0.4, 0.2])
    assert m["roc_auc"] is None  # undefined, reported honestly
    assert m["accuracy"] is not None


def test_threshold_boundary_is_inclusive():
    m = compute_metrics([1], [0.5])
    assert m["confusion_matrix"]["tp"] == 1


def test_slice_metrics_reports_buckets_and_notes():
    y = np.array([0, 1] * 60)
    p = np.linspace(0.1, 0.9, 120)
    words = np.array([10, 80, 200, 500] * 30)
    domains = ["news" if i % 2 else "forums" for i in range(120)]
    out = slice_metrics(y, p, words, domains)
    assert set(out["by_length"]) == {"<50w", "50-150w", "150-400w", "400w+"}
    assert "news" in out["by_domain"] and "forums" in out["by_domain"]
    tiny = slice_metrics(y[:5], p[:5], words[:5], ["x"] * 5)
    assert "note" in list(tiny["by_domain"].values())[0]
