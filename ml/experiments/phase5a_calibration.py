"""Phase-5A calibration sensitivity for segment-level inference.

The segments-v1 calibrator type was selected on val_cal SEGMENTS by Brier
(isotonic won). This block measures what that choice costs on the two things a
user would rely on:

  * reliability of the probability itself (Brier / ECE / reliability bins), and
  * the probability RANGE: isotonic regression is a step function, so on inputs
    outside its fit range it saturates at 0.0 / 1.0. A saturated segment score
    ("this paragraph is 100% machine-like") is a display hazard even when the
    ranking (AUC) is fine.

Both calibrators are refit on the SAME val_cal segments (the only data a
calibrator may use here) and compared on MAGE-test segments and on genuine
whole-text composite segments. This does NOT re-select the shipped calibrator;
it reports the trade-off so the report can state it honestly.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.evaluation.calibration import IsotonicCalibrator, PlattCalibrator  # noqa: E402
from ml.evaluation.metrics import compute_metrics  # noqa: E402
from ml.experiments.phase5a import human_fpr, load_json, reliability  # noqa: E402
from ml.experiments.phase5a_train import (CACHE_DIR, SEG_DIR, _eval_segment_sets,  # noqa: E402
                                          _extract, load_config)
from ml.segmentation import SegmentationPolicy  # noqa: E402

OUT = CACHE_DIR.parent / "calibration-sensitivity.json"


def saturation_stats(p: np.ndarray) -> dict:
    p = np.asarray(p, dtype=float)
    return {
        "share_p_le_001": round(float((p <= 0.001).mean()), 4),
        "share_p_ge_999": round(float((p >= 0.999).mean()), 4),
        "share_saturated": round(float(((p <= 0.001) | (p >= 0.999)).mean()), 4),
        "share_in_0_05_0_95": round(float(((p > 0.05) & (p < 0.95)).mean()), 4),
        "distinct_values_1e3": int(len(np.unique(np.round(p, 3)))),
    }


def main() -> int:
    cfg = load_config()
    policy = SegmentationPolicy(**cfg["policy"])
    workers = int(cfg.get("workers", 0))

    features = list(load_json(SEG_DIR / "features.json")["features"])
    state = joblib.load(SEG_DIR / "feature_state.joblib")
    model = joblib.load(SEG_DIR / "model.joblib")

    pool = pd.read_parquet(CACHE_DIR / "train_segments.parquet")
    cal = pool[pool["corpus"] == "val_cal"].reset_index(drop=True)
    Xcal = _extract(cal, features, workers, state)[features].to_numpy()
    ycal = cal["y"].to_numpy()
    raw_cal = model.predict_proba(Xcal)[:, 1]

    calibrators = {
        "platt": PlattCalibrator().fit(raw_cal, ycal),
        "isotonic": IsotonicCalibrator().fit(raw_cal, ycal),
    }

    eval_sets = _eval_segment_sets(policy, cfg, workers, state, features)
    out: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact": "ml/models/segments-v1",
        "fit_data": {"corpus": "val_cal segments",
                     "n": int(len(cal)),
                     "note": "same data the shipped calibrator was fit on"},
        "calibrators": {},
    }
    for name, c in calibrators.items():
        block = {"on_val_cal_fit": {
            "brier": round(float(np.mean((c.predict(raw_cal) - ycal) ** 2)), 4),
            **saturation_stats(c.predict(raw_cal))}}
        for set_name in ("mage_test", "composite"):
            df = eval_sets[set_name]
            X = _extract(df, features, workers, state)[features].to_numpy()
            p = c.predict(model.predict_proba(X)[:, 1])
            y = df["y"].to_numpy()
            m = compute_metrics(y, p)
            block[set_name] = {
                "n": int(len(y)),
                "roc_auc": (round(m["roc_auc"], 4) if m["roc_auc"] else None),
                "accuracy": round(m["accuracy"], 4),
                "human_fpr": human_fpr(m["confusion_matrix"]),
                "brier": round(m["brier"], 4),
                "ece_10": round(m["ece_10"], 4),
                "mean_abs_p_minus_y": round(m["mean_abs_p_minus_y"], 4),
                **saturation_stats(p),
                "reliability_bins": reliability(y, p),
            }
        out["calibrators"][name] = block
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"calibration sensitivity -> {OUT}")
    for name, b in out["calibrators"].items():
        comp = b["composite"]
        mt = b["mage_test"]
        print(f"  {name:9s} composite: ece={comp['ece_10']} brier={comp['brier']} "
              f"acc={comp['accuracy']} saturated={comp['share_saturated']} | "
              f"mage_test: ece={mt['ece_10']} saturated={mt['share_saturated']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())