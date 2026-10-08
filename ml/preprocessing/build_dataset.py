"""Build processed train/val/test from raw v0 datasets.

Pipeline (each stage's counts go into processing_report.json):
  raw rows -> valid-string -> English filter -> length filter
  -> stratified subsample (MAGE only) -> normalize
  -> exact dedup (priority: MAGE > HC3) -> MinHash near-dup clustering (groups)
  -> cross-corpus removal from eval pools -> grouped split -> parquet out

HC3 = eval-only pool (never trained on). RAID is recorded as deferred (Phase 2).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .normalize import (ARTIFACT_PATTERNS, TRANSFORMATIONS, looks_english,
                        normalize_text, strip_artifacts, word_count)
from .dedup import cluster_duplicates
from .split import grouped_split, make_stratum

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "ml" / "data" / "raw"
PROCESSED = ROOT / "ml" / "data" / "processed"
CONFIG_PATH = ROOT / "ml" / "configs" / "phase1.json"
HC3_FILES = ["reddit_eli5", "finance", "medicine", "open_qa", "wiki_csai"]


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def derive_domain(src: str) -> str:
    """Best-effort writing-domain label for slice reporting only.

    'xsum_machine_continuation_opt' -> 'xsum'; 'cmv_human' -> 'cmv';
    'sci_gen' / 'open_qa' / 'reddit_eli5' preserved. Never used as a model input.
    """
    s = str(src)
    s = re.sub(r"_(?:machine|human)_(?:continuation|specified|topical)"
               r"(?:_[a-z0-9.\-]+)*$", "", s, flags=re.I)
    s = re.sub(r"_(?:human|chatgpt|gpt4|gpt-4|gpt35|davinci|dolly|llama|opt"
               r"|flan_t5|bloomz|cohere|palm|vicuna|alpaca)$", "", s, flags=re.I)
    return s or "unknown"


def load_mage() -> pd.DataFrame:
    frames = []
    for split in ["train", "valid", "test"]:
        df = pd.read_csv(RAW / "mage" / f"{split}.csv", engine="pyarrow")
        df["official_split"] = split
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"text": "text_raw"})
    out["source_dataset"] = "mage"
    out["group_hint"] = ""
    return out[["text_raw", "label", "src", "source_dataset", "official_split", "group_hint"]]


def load_hc3() -> pd.DataFrame:
    rows = []
    for fname in HC3_FILES:
        path = RAW / "hc3" / f"{fname}.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for qi, line in enumerate(fh):
                rec = json.loads(line)
                qid = f"{fname}:{qi}"
                for ans in (rec.get("human_answers") or []):
                    rows.append({"text_raw": ans, "label": 0, "src": fname,
                                 "source_dataset": "hc3", "official_split": "",
                                 "group_hint": qid})
                for ans in (rec.get("chatgpt_answers") or []):
                    rows.append({"text_raw": ans, "label": 1, "src": fname,
                                 "source_dataset": "hc3", "official_split": "",
                                 "group_hint": qid})
    return pd.DataFrame(rows)


def _prep_row(t):
    n = normalize_text(t)
    if n is None:
        return (None, False, 0)
    return (n, looks_english(n), word_count(n))


def apply_filters(df: pd.DataFrame, cfg: dict, stages: dict, tag: str) -> pd.DataFrame:
    stages[f"{tag}_raw_rows"] = int(len(df))
    prepped = df["text_raw"].map(_prep_row)
    tmp = pd.DataFrame(prepped.tolist(), index=df.index,
                       columns=["text_norm", "looks_english", "words"])
    df = pd.concat([df.drop(columns=["text_raw"]), tmp], axis=1)

    valid = df["text_norm"].notna()
    stages[f"{tag}_dropped_invalid_string"] = int((~valid).sum())
    df = df[valid]

    eng = df["looks_english"]
    stages[f"{tag}_dropped_not_english"] = int((~eng).sum())
    df = df[eng]

    length_ok = df["words"].between(cfg["min_words"], cfg["max_words"])
    stages[f"{tag}_dropped_out_of_length_range"] = int((~length_ok).sum())
    df = df[length_ok]
    stages[f"{tag}_after_filters"] = int(len(df))
    return df.copy()


def stratified_sample(df: pd.DataFrame, target: int, stratum_col: str, seed: int) -> pd.DataFrame:
    """Seeded proportional sample with largest-remainder allocation."""
    if len(df) <= target:
        return df
    counts = df[stratum_col].value_counts()
    exact = counts / counts.sum() * target
    alloc = exact.astype(int)
    remainder = target - int(alloc.sum())
    if remainder > 0:
        frac = (exact - alloc).sort_values(ascending=False)
        for s in frac.index[:remainder]:
            alloc[s] += 1
    parts = []
    for s, n in alloc.items():
        group = df[df[stratum_col] == s]
        take = min(int(n), len(group))
        if take < len(group):
            group = group.sample(n=take, random_state=seed)
        parts.append(group)
    return pd.concat(parts).sort_index()


def _dist(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {"label_0": 0, "label_1": 0}
    vc = df["label"].value_counts()
    return {"label_0": int(vc.get(0, 0)), "label_1": int(vc.get(1, 0))}


EMPTY_COLS = ["text_raw", "label", "src", "source_dataset", "official_split", "group_hint"]


def run(cfg: dict) -> dict:
    t0 = time.time()
    stages: dict[str, int] = {}
    PROCESSED.mkdir(parents=True, exist_ok=True)

    if not (RAW / "mage" / "train.csv").exists():
        raise SystemExit("raw MAGE data missing - run scripts/download_data.py first")
    mage = load_mage()
    hc3 = load_hc3() if (RAW / "hc3").exists() else pd.DataFrame(columns=EMPTY_COLS)
    if len(hc3) == 0:
        hc3 = pd.DataFrame(columns=EMPTY_COLS)

    mage = apply_filters(mage, cfg, stages, "mage")
    hc3 = apply_filters(hc3, cfg, stages, "hc3")

    # stratified subsample (MAGE only, seeded, documented stage)
    mage["stratum"] = make_stratum(mage["label"], mage["src"])
    before = len(mage)
    mage = stratified_sample(mage, cfg["target_sample_rows"], "stratum", cfg["seed"])
    stages["mage_before_stratified_sample"] = before
    stages["mage_after_stratified_sample"] = int(len(mage))
    # note: text_stripped/domain are computed AFTER dedup+split (memory economy)

    # ---- dedup: exact (priority MAGE > HC3) then near-dup grouping ---------
    comb = pd.concat([mage, hc3], ignore_index=True)  # mage first = higher priority
    comb["exact_hash"] = comb["text_norm"].map(
        lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest())
    keep = ~comb.duplicated(subset="exact_hash", keep="first")
    stages["exact_duplicates_dropped_total"] = int((~keep).sum())
    stages["exact_duplicates_dropped_by_dataset"] = {
        str(k): int(v) for k, v in comb.loc[~keep, "source_dataset"].value_counts().items()}
    comb = comb.loc[keep].copy()

    labels, near_stats = cluster_duplicates(
        comb["text_norm"].tolist(),
        workers=int(cfg.get("dedup_workers", 1)),
        seed=int(cfg["dedup_seed"]),
        **{k: v for k, v in cfg["dedup"].items() if k != "seed"})
    comb["dup_group"] = [f"g{i}" for i in labels]
    comb["group_id"] = comb["dup_group"]

    # cross-corpus: drop eval rows that near-duplicate ANY training row
    mage_groups = set(comb.loc[comb["source_dataset"] == "mage", "dup_group"])
    cross = (comb["source_dataset"] == "hc3") & comb["dup_group"].isin(mage_groups)
    stages["cross_corpus_rows_removed_from_eval"] = int(cross.sum())
    comb = comb.loc[~cross].copy()
    # ---- grouped split on MAGE --------------------------------------------
    mage_rows = comb[comb["source_dataset"] == "mage"].copy()
    mage_rows["split"] = grouped_split(
        mage_rows, group_col="group_id", stratum_col="stratum",
        seed=int(cfg["seed"]), fractions=tuple(cfg["split_fractions"]))
    hc3_rows = comb[comb["source_dataset"] == "hc3"].copy()
    hc3_rows["split"] = "eval_hc3"

    # leakage assertions (also covered by tests/test_split.py)
    per_split_groups = {s: set(g["group_id"]) for s, g in mage_rows.groupby("split")}
    keys = sorted(per_split_groups)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            overlap = per_split_groups[a] & per_split_groups[b]
            assert not overlap, f"group leakage between {a}/{b}: {len(overlap)}"
    hs = mage_rows["text_norm"].map(lambda t: hashlib.sha1(t.encode()).hexdigest())
    assert hs.nunique() == len(hs), "exact duplicates survive inside MAGE pool"

    final = pd.concat([mage_rows, hc3_rows], ignore_index=True)
    final["text_stripped"] = final["text_norm"].map(strip_artifacts)
    final["domain"] = final["src"].map(derive_domain)
    final["row_id"] = range(len(final))
    final["label"] = final["label"].astype(int)
    cols = ["row_id", "text_norm", "text_stripped", "label", "source_dataset", "src",
            "domain", "official_split", "group_id", "words", "stratum", "split"]
    final = final[cols]
    for name, part in final.groupby("split", sort=True):
        out_name = "eval_hc3" if name == "eval_hc3" else name
        part.to_parquet(PROCESSED / f"{out_name}.parquet", index=False)

    splits_report = {}
    for name, part in final.groupby("split", sort=True):
        splits_report[str(name)] = {
            "rows": int(len(part)),
            "class_distribution": _dist(part),
            "unique_groups": int(part["group_id"].nunique()),
            "domains": {str(k): int(v) for k, v in part["domain"].value_counts().head(30).items()},
            "median_words": float(part["words"].median()),
        }

    raid_note = "not downloaded"
    manifest_path = ROOT / "ml" / "data" / "manifests" / "datasets_manifest.json"
    if manifest_path.exists():
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        raid_files = man.get("approved", {}).get("raid", {}).get("files", [])
        raid_note = "; ".join(
            f"{f.get('name')}:{f.get('status')}" for f in raid_files) or "no files"
        if (RAW / "raid" / "test.csv").exists():
            raid_note += " (downloaded; normalization/features deferred to Phase 2 - eval-only)"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool": "ml/preprocessing/build_dataset.py",
        "config": cfg,
        "transformations": TRANSFORMATIONS,
        "artifact_patterns": ARTIFACT_PATTERNS,
        "stages": stages,
        "near_dup_stats": near_stats,
        "splits": splits_report,
        "raid_status": raid_note,
        "duration_seconds": round(time.time() - t0, 1),
    }
    (PROCESSED / "processing_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Build WriteTrace processed splits").parse_args(argv)
    report = run(load_config())
    print(json.dumps({k: report[k] for k in
                      ("stages", "splits", "raid_status", "duration_seconds")},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


