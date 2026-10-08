from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.configs.phase1 import load_config
from ml.data.download import ROOT
from ml.evaluation.calibration import fit_best_calibrator, brier_score, expected_calibration_error
from ml.evaluation.metrics import _pr_auc
from ml.preprocessing.build_dataset import PROCESSED, load_config as _load_config2
from ml.features.extract import feature_names

MODELS_DIR = ROOT / "ml" / "models"


def load_config():
    cfg_path = ROOT / "ml" / "configs" / "phase1.json"
    with cfg_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)



def _sha_groups(group_ids: pd.Series, seed: int, frac_tune: float) -> pd.Series:
    """Deterministic group-level split of validation into tune/calibration halves."""
    def bucket(g: str) -> bool:
        h = int(hashlib.sha1(f"{g}|{seed}".encode()).hexdigest()[:8], 16) % 1000
        return h < int(frac_tune * 1000)
    return group_ids.map(bucket)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Train WriteTrace baselines").parse_args(argv)
    cfg = load_config()
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    tr = pd.read_parquet(PROCESSED / "features_train.parquet")
    va = pd.read_parquet(PROCESSED / "features_val.parquet")
    names = feature_names()
    X_train, y_train = tr[names].to_numpy(), tr["label"].to_numpy()
    tune_mask = _sha_groups(va["group_id"], seed, cfg["val_tune_fraction"])
    X_tune, y_tune = va.loc[tune_mask, names].to_numpy(), va.loc[tune_mask, "label"].to_numpy()
    X_cal, y_cal = va.loc[~tune_mask, names].to_numpy(), va.loc[~tune_mask, "label"].to_numpy()
    print(f"train={len(tr)} tune={int(tune_mask.sum())} cal={int((~tune_mask).sum())}", flush=True)

    mcfg = cfg["models"]
    fitted: dict[str, object] = {}
    tune_scores: dict[str, float] = {}
    grid_records: list[dict] = []

    dummy = DummyClassifier(strategy="prior", random_state=seed).fit(X_train, y_train)
    fitted["dummy_prior"] = dummy
    tune_scores["dummy_prior"] = _pr_auc(y_tune, dummy.predict_proba(X_tune)[:, 1])
    grid_records.append({"model": "dummy_prior", "params": {"strategy": "prior"},
                         "val_tune_pr_auc": tune_scores["dummy_prior"]})

    best = {"score": -1.0, "C": None, "pipe": None}
    for C in mcfg["lr_C_grid"]:
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=1000, class_weight="balanced",
                                       random_state=seed, solver="lbfgs")),
        ]).fit(X_train, y_train)
        score = _pr_auc(y_tune, pipe.predict_proba(X_tune)[:, 1])
        grid_records.append({"model": "logreg", "params": {"C": C},
                             "val_tune_pr_auc": score})
        if score > best["score"]:
            best = {"score": score, "C": C, "pipe": pipe}
    fitted["logistic_regression"] = best["pipe"]
    tune_scores["logistic_regression"] = best["score"]

    best = {"score": -1.0, "params": None, "clf": None}
    for lr in mcfg["hgb_learning_rate_grid"]:
        for leaves in mcfg["hgb_max_leaf_nodes_grid"]:
            clf = HistGradientBoostingClassifier(
                learning_rate=lr, max_leaf_nodes=leaves, max_iter=mcfg["hgb_max_iter"],
                random_state=seed).fit(X_train, y_train)
            score = _pr_auc(y_tune, clf.predict_proba(X_tune)[:, 1])
            grid_records.append({"model": "hist_gbdt",
                                 "params": {"learning_rate": lr, "max_leaf_nodes": leaves},
                                 "val_tune_pr_auc": score})
            if score > best["score"]:
                best = {"score": score,
                        "params": {"learning_rate": lr, "max_leaf_nodes": leaves},
                        "clf": clf}
    fitted["hist_gradient_boosting"] = best["clf"]
    tune_scores["hist_gradient_boosting"] = best["score"]
    # selection on validation-tune PR-AUC
    selected = max(tune_scores, key=tune_scores.get)

    # calibration on val_cal (group-disjoint from tune); Platt vs isotonic by Brier
    p_cal = fitted[selected].predict_proba(X_cal)[:, 1]
    cal_type, calibrator, cal_scores = fit_best_calibrator(p_cal, y_cal)
    p_cal_cal = calibrator.predict(p_cal)

    out_dir = MODELS_DIR / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, model in fitted.items():
        joblib.dump(model, out_dir / f"model_{name}.joblib")
    joblib.dump(calibrator, out_dir / "calibrator.joblib")
    joblib.dump(joblib.load(PROCESSED / "feature_state.joblib"),
                out_dir / "feature_state.joblib")

    def file_sha(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()

    val_metrics = {
        "selection_metric": cfg["selection_metric"],
        "val_tune_pr_auc": tune_scores,
        "selected_model": selected,
        "calibration": {
            "fitted_on": "val_cal (group-disjoint from val_tune)",
            "chosen": cal_type,
            "brier_platt": cal_scores.get("platt"),
            "brier_isotonic": cal_scores.get("isotonic"),
            "val_cal_raw": {"brier": brier_score(p_cal, y_cal),
                            "ece": expected_calibration_error(p_cal, y_cal)},
            "val_cal_calibrated": {"brier": brier_score(p_cal_cal, y_cal),
                                   "ece": expected_calibration_error(p_cal_cal, y_cal)},
        },
        "grid": grid_records,
        "n_features": len(names),
    }
    (out_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")

    manifest = {
        "run_name": cfg["run_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.time() - t0, 1),
        "seed": seed,
        "config": cfg,
        "feature_version": cfg["feature_version"],
        "feature_names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "data": {
            "dataset_manifest_sha256": file_sha(
                ROOT / "ml/data/manifests/datasets_manifest.json")
            if (ROOT / "ml/data/manifests/datasets_manifest.json").exists() else None,
            "processing_report_sha256": file_sha(PROCESSED / "processing_report.json"),
            "extraction_report_sha256": file_sha(PROCESSED / "extraction_report.json"),
        },
        "rows": {"train": len(tr), "val_total": len(va),
                 "val_tune": int(tune_mask.sum()), "val_cal": int((~tune_mask).sum())},
        "selected_model": selected,
        "versions": {"python": __import__("sys").version.split()[0],
                     "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": sklearn.__version__},
        "artifacts": [p.name for p in sorted(out_dir.iterdir())],
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected, "val_tune_pr_auc": tune_scores,
                      "calibrator": cal_type, "out_dir": str(out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

