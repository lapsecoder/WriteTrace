"""Fit feature state on TRAIN, transform all splits, write feature parquets.

Deterministic given (data, config). Records feature version, family sizes and
column-order hash in extraction_report.json for reproducibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from .extract import FEATURE_FAMILIES, FEATURE_VERSION, FeatureState, extract_many, feature_names
from ..preprocessing.build_dataset import PROCESSED, load_config


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Extract WriteTrace v1 features").parse_args(argv)
    cfg = load_config()
    t0 = time.time()
    train_path = PROCESSED / "train.parquet"
    if not train_path.exists():
        raise SystemExit("processed splits missing - run scripts/build_dataset.py first")

    train = pd.read_parquet(train_path)
    state = FeatureState.fit(train["text_norm"].tolist(), rare_frac=cfg["rare_frac"])
    workers = int(cfg.get("feature_workers", 1))
    print(f"FeatureState fitted on {state.n_docs} train docs "
          f"(vocab={len(state.df)}, rare_max_df={state.rare_max_df})", flush=True)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool": "ml/features/pipeline.py",
        "feature_version": FEATURE_VERSION,
        "fitted_on": "train_split_only",
        "state_n_docs": int(state.n_docs),
        "state_vocab": int(len(state.df)),
        "rare_max_df": int(state.rare_max_df),
        "n_features": len(feature_names()),
        "families": {k: len(v) for k, v in FEATURE_FAMILIES.items()},
        "feature_names_sha256": hashlib.sha256(
            "\n".join(feature_names()).encode("utf-8")).hexdigest(),
        "workers": workers,
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        df = pd.read_parquet(PROCESSED / f"{split}.parquet")
        feats = extract_many(df["text_norm"].tolist(), state, workers=workers)
        all_nan = feats.isna().all(axis=1)
        out = pd.concat(
            [df[["row_id", "label", "words", "domain", "group_id"]].reset_index(drop=True),
             feats.reset_index(drop=True)],
            axis=1)
        out.to_parquet(PROCESSED / f"features_{split}.parquet", index=False)
        report["splits"][split] = {
            "rows": int(len(df)),
            "all_nan_feature_rows": int(all_nan.sum()),
        }
        print(f"  {split}: {len(df)} rows, {int(all_nan.sum())} all-NaN rows", flush=True)

    joblib.dump(state, PROCESSED / "feature_state.joblib")
    report["duration_seconds"] = round(time.time() - t0, 1)
    (PROCESSED / "extraction_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"features written; report -> {PROCESSED / 'extraction_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
