"""Evaluate the trained baseline on the locked test split; write metrics + report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..features.extract import feature_names
from ..preprocessing.build_dataset import PROCESSED, ROOT, load_config
from ..training.train import MODELS_DIR
from .calibration import reliability_bins
from .metrics import compute_metrics, slice_metrics
from .report import render_report


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Evaluate WriteTrace baseline").parse_args(argv)
    cfg = load_config()
    run_dir = MODELS_DIR / cfg["run_name"]
    test = pd.read_parquet(PROCESSED / "features_test.parquet")
    names = feature_names()
    X = test[names].to_numpy()
    y = test["label"].to_numpy()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    val_metrics = json.loads((run_dir / "val_metrics.json").read_text(encoding="utf-8"))
    selected = manifest["selected_model"]
    calibrator = joblib.load(run_dir / "calibrator.joblib")

    probs: dict[str, np.ndarray] = {}
    results: dict = {"selected_model": selected, "calibrator": val_metrics["calibration"]["chosen"],
                     "threshold": 0.5, "test_rows": int(len(test)), "raw": {}}
    for m in ["dummy_prior", "logistic_regression", "hist_gradient_boosting"]:
        model = joblib.load(run_dir / f"model_{m}.joblib")
        p = model.predict_proba(X)[:, 1]
        probs[m] = p
        results["raw"][m] = compute_metrics(y, p)

    p_cal = calibrator.predict(probs[selected])
    results["selected_calibrated"] = compute_metrics(y, p_cal)
    results["slices_calibrated"] = slice_metrics(
        y, p_cal, test["words"].to_numpy(), test["domain"].astype(str).tolist())
    results["reliability_bins_raw"] = reliability_bins(probs[selected], y)
    results["reliability_bins_calibrated"] = reliability_bins(p_cal, y)
    results["class_balance_test"] = {"label_0": int((y == 0).sum()),
                                     "label_1": int((y == 1).sum())}

    (run_dir / "test_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    processing = json.loads((PROCESSED / "processing_report.json").read_text(encoding="utf-8"))
    extraction = json.loads((PROCESSED / "extraction_report.json").read_text(encoding="utf-8"))
    md = render_report(results, processing, extraction, val_metrics, manifest, cfg)
    out_path = ROOT / "docs" / "experiments" / f"{cfg['run_name']}-report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")

    sel = results["selected_calibrated"]
    print(json.dumps({
        "report": str(out_path),
        "selected": selected,
        "test_calibrated": {
            "accuracy": sel["accuracy"], "f1_macro": sel["f1_macro"],
            "roc_auc": sel["roc_auc"], "pr_auc": sel["pr_auc"],
            "brier": sel["brier"], "ece_10": sel["ece_10"]},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
