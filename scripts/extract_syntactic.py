"""Pre-extract Phase-3 syntactic (spaCy) features for every evaluation split.

Standalone so it can run in the background; ml/experiments/phase3.py consumes
the cached parquet files. Deterministic; writes one file per split with a meta
hash so re-runs skip completed splits.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "ml" / "experiments" / "phase3" / "cache"
PROCESSED = ROOT / "ml" / "data" / "processed"
SYN_VERSION = "syntactic_spacy_v1"


def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def _done(path: Path, key: dict) -> bool:
    meta = path.with_suffix(path.suffix + ".meta.json")
    if not (path.exists() and meta.exists()):
        return False
    try:
        got = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return False
    return all(got.get(k) == v for k, v in key.items())


def _write(df: pd.DataFrame, path: Path, key: dict) -> None:
    df.to_parquet(path, index=False)
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps({**key, "rows": len(df), "written_at":
                    datetime.now(timezone.utc).isoformat()}, indent=2),
        encoding="utf-8")


def _load_texts(name: str) -> list[str]:
    from ml.preprocessing.normalize import normalize_text
    if name in ("train", "val", "test"):
        df = pd.read_parquet(PROCESSED / f"{name}.parquet", columns=["text_norm"])
        return df["text_norm"].tolist()
    if name in ("ood_gpt", "ood_gpt_para"):
        import csv
        fname = "test_ood_set_gpt.csv" if name == "ood_gpt" else "test_ood_set_gpt_para.csv"
        path = ROOT / "ml" / "data" / "raw" / "mage" / fname
        texts = []
        with path.open("r", encoding="utf-8") as fh:
            r = csv.reader(fh)
            next(r)
            for row in r:
                nt = normalize_text(row[0])
                texts.append(nt if nt else "")
        return texts
    if name == "hc3":
        df = pd.read_parquet(PROCESSED / "eval_hc3.parquet", columns=["text_norm"])
        return df["text_norm"].tolist()
    if name == "raid":
        from ml.experiments.phase2 import raid_frame
        return raid_frame()["text_norm"].tolist()
    raise ValueError(f"unknown split {name}")


def main() -> int:
    from ml.features.syntactic import SYNTACTIC_FEATURES, extract_syntactic
    CACHE.mkdir(parents=True, exist_ok=True)
    splits = ["test", "val", "ood_gpt", "ood_gpt_para", "raid", "hc3", "train"]
    n_process = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    total = time.time()
    for name in splits:
        out = CACHE / f"syn_{name}.parquet"
        try:
            texts = _load_texts(name)
        except Exception as exc:
            print(f"{name}: LOAD FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        key = {"version": SYN_VERSION, "rows": len(texts),
               "model": "en_core_web_sm", "n_features": len(SYNTACTIC_FEATURES),
               "n_process": n_process, "batch_size": batch_size}
        if _done(out, key):
            print(f"{name}: cached ({len(texts)} rows)", flush=True)
            continue
        t0 = time.time()
        try:
            feats = extract_syntactic(texts, n_process=n_process,
                                      batch_size=batch_size)
        except Exception as exc:
            import traceback
            print(f"{name}: FAILED {type(exc).__name__}: {exc}\n"
                  f"{traceback.format_exc()[-1500:]}", flush=True)
            continue
        _write(feats, out, key)
        dt = time.time() - t0
        print(f"{name}: {len(feats)} rows in {dt:.1f}s "
              f"({len(feats) / max(dt, 1e-9):.1f} rows/s)", flush=True)
    print(f"ALL DONE in {time.time() - total:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
