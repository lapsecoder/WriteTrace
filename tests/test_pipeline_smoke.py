"""End-to-end smoke test of the ML pipeline on synthetic, separable data.

Exercises: normalize -> dedup -> grouped split -> features -> train -> predict
-> metrics. No real dataset files required.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.evaluation.calibration import fit_best_calibrator
from ml.evaluation.metrics import compute_metrics
from ml.features.extract import FeatureState, extract_many, feature_names
from ml.preprocessing.build_dataset import derive_domain, stratified_sample
from ml.preprocessing.dedup import cluster_duplicates
from ml.preprocessing.normalize import normalize_text
from ml.preprocessing.split import grouped_split, make_stratum

FORMAL = ["the system processes the data efficiently and reports the results clearly",
          "each cycle produces consistent outcomes that are carefully documented",
          "the methodology ensures reliable performance across all test conditions",
          "additionally the framework supports modular components for future extension",
          "furthermore the approach demonstrates robust behavior under load"]
CASUAL = ["ugh, i totally blew this one — didn't sleep at all last night!",
          "so we tried it again and wow, it actually worked?? kinda wild tbh",
          "my boss keeps saying 'trust the process' but honestly? i'm not sold.",
          "late night ramblings: why does coffee never taste the same twice",
          "ok so hear me out — what if we just skipped the meeting entirely"]


def _synthetic(n_per_class=70, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for label, pool in [(1, FORMAL), (0, CASUAL)]:
        for i in range(n_per_class):
            parts = rng.choice(pool, size=6, replace=True)
            text = ". ".join(str(p) for p in parts).replace("  ", " ") + f" variant {i}."
            rows.append({"text_raw": normalize_text(text), "label": label,
                         "src": "synthetic", "source_dataset": "synthetic",
                         "group_id": f"g{label}_{i}", "words": len(text.split())})
    df = pd.DataFrame(rows)
    df["stratum"] = make_stratum(df["label"], df["src"])
    df["domain"] = df["src"].map(derive_domain)
    return df


def test_end_to_end_smoke():
    df = _synthetic()
    labels, _stats = cluster_duplicates(df["text_raw"].tolist(), workers=1)
    df["group_id"] = [f"n{i}" for i in labels]
    df["split"] = grouped_split(df, seed=42)
    assert set(df["split"]) == {"train", "val", "test"}
    g = {s: set(x["group_id"]) for s, x in df.groupby("split")}
    assert not (g["train"] & g["test"]) and not (g["val"] & g["test"])

    train, val, test = (df[df.split == s] for s in ("train", "val", "test"))
    state = FeatureState.fit(train["text_raw"].tolist())
    Xtr = extract_many(train["text_raw"].tolist(), state, workers=0).to_numpy()
    Xte = extract_many(test["text_raw"].tolist(), state, workers=0).to_numpy()
    ytr, yte = train["label"].to_numpy(), test["label"].to_numpy()

    clf = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000)),
    ]).fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    assert np.all((p >= 0) & (p <= 1)) and p.shape == (len(test),)

    _name, cal, _scores = fit_best_calibrator(p, yte)
    p_cal = cal.predict(p)
    assert np.all((p_cal >= 0) & (p_cal <= 1))

    m = compute_metrics(yte, p_cal)
    assert m["roc_auc"] is not None and m["roc_auc"] > 0.7, m
    assert set(m) >= {"accuracy", "pr_auc", "brier", "ece_10", "confusion_matrix"}


def test_stratified_sample_hits_target():
    df = _synthetic(n_per_class=60)
    out = stratified_sample(df, target=60, stratum_col="stratum", seed=1)
    assert len(out) == 60
    out2 = stratified_sample(df, target=60, stratum_col="stratum", seed=1)
    assert sorted(out.index) == sorted(out2.index)


def test_report_renders_with_structured_inputs():
    from ml.evaluation.report import render_report
    y = np.array([0, 0, 1, 1])
    p = np.array([0.2, 0.4, 0.6, 0.8])
    met = compute_metrics(y, p)
    results = {"selected_model": "logistic_regression", "calibrator": "platt",
               "threshold": 0.5, "test_rows": 4, "raw": {"logistic_regression": met},
               "selected_calibrated": met,
               "slices_calibrated": {"by_length": {"<50w": {"n": 4, "note": "too few"}},
                                     "by_domain": {"x": met}},
               "reliability_bins_raw": [], "reliability_bins_calibrated": [],
               "class_balance_test": {"label_0": 2, "label_1": 2}}
    processing = {"stages": {"mage_raw_rows": 10, "exact_duplicates_dropped_total": 0},
                  "duration_seconds": 1.0, "raid_status": "none",
                  "near_dup_stats": {"candidate_pairs": 0, "near_pairs_verified": 0,
                                     "exact_pairs": 0, "clusters": 4,
                                     "rows_in_multirow_clusters": 0, "buckets_overrun": 0,
                                     "params": {"jaccard_threshold": 0.75}},
                  "splits": {"train": {"rows": 2,
                                       "class_distribution": {"label_0": 1, "label_1": 1},
                                       "unique_groups": 2, "median_words": 50.0,
                                       "domains": {"x": 2}}}}
    extraction = {"n_features": 73, "families": {"structure": 18},
                  "feature_version": "features_v1", "fitted_on": "train_split_only",
                  "state_n_docs": 2, "state_vocab": 50, "rare_max_df": 1,
                  "feature_names_sha256": "abc",
                  "splits": {"train": {"all_nan_feature_rows": 0}}}
    val_metrics = {"selection_metric": "pr_auc_val_tune",
                   "val_tune_pr_auc": {"logistic_regression": 0.9},
                   "calibration": {"chosen": "platt", "brier_platt": 0.1,
                                   "brier_isotonic": 0.12,
                                   "val_cal_raw": {"brier": 0.2, "ece": 0.1},
                                   "val_cal_calibrated": {"brier": 0.1, "ece": 0.05}},
                   "grid": [{"model": "logreg", "params": {"C": 1},
                             "val_tune_pr_auc": 0.9}]}
    manifest = {"generated_at": "now", "seed": 42, "feature_version": "features_v1",
                "rows": {"val_tune": 2, "val_cal": 2},
                "feature_names_sha256": "fff",
                "data": {"dataset_manifest_sha256": "a",
                         "processing_report_sha256": "b",
                         "extraction_report_sha256": "c"},
                "versions": {"python": "3.13"}}
    cfg = {"run_name": "t", "seed": 42, "feature_version": "features_v1"}
    md = render_report(results, processing, extraction, val_metrics, manifest, cfg)
    assert "Dataset sizes after every filtering stage" in md
    assert "Confusion matrix" in md
    assert "Errors & limitations" in md
    assert "not a product claim" in md


