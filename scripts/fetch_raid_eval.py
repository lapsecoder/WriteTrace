"""Phase-2 acquisition of RAID-train (evaluation-only, never trained on).

Why: docs/experiments/robustness-plan.md E2 requires RAID adversarial subsets
(attack column, paraphrase family) and E5 requires a cross-corpus supervised
test. Both need LABELLED RAID-train rows; the unlabeled RAID test.csv cannot
support them.

Safety rules enforced here:
  * RAID is evaluation-only. Nothing produced here is ever used to fit a
    model, a calibrator, a threshold or a vocabulary.
  * A stratified reservoir sample is persisted as a derived eval asset
    (raid_eval_subset.csv); the raw file is kept so the sample is
    reproducible and sha256-verifiable.

Phases, both resumable:
  download : stream ml/data/raw/raid/train.csv (11.78 GB) with HTTP Range
             resume into place (only if absent).
  subset   : one csv pass -> deterministic per-(stratum, domain) reservoir
             samples, seed 42 -> raid_eval_subset.csv + provenance json.

Usage:
  python scripts/fetch_raid_eval.py            # download (if needed) + subset
  python scripts/fetch_raid_eval.py --subset-only
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

csv.field_size_limit(2 ** 31 - 1)  # RAID 'books' rows hold full chapter fields

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "ml" / "data" / "raw" / "raid"
TRAIN = RAW / "train.csv"
PART = RAW / "train.csv.part"
SUBSET = RAW / "raid_eval_subset.csv"
STATE = RAW / "raid_subset_state.json"
PROV = ROOT / "ml" / "data" / "manifests" / "raid_phase2_acquisition.json"

URL = "https://huggingface.co/datasets/liamdugan/raid/resolve/main/train.csv"
EXPECTED_BYTES = 11_779_491_051
HF_REVISION = "865cac74188466cb0c3b7574a10204007b57a459"
SEED = 42
CAP_PER_STRATUM_DOMAIN = {  # reservoir cap per (stratum, domain) cell
    "human_clean": 1000,
    "machine_clean": 1000,
    "machine_paraphrase": 1000,
    "human_paraphrase": 500,
    "machine_other_attack": 500,
}
CHECKPOINT_ROWS = 200_000
COLUMNS = ["id", "adv_source_id", "source_id", "model", "decoding",
           "repetition_penalty", "attack", "domain", "generation"]


def _log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def download() -> None:
    """Stream train.csv with Range resume. Idempotent."""
    if TRAIN.exists() and TRAIN.stat().st_size == EXPECTED_BYTES:
        _log("train.csv already complete")
        return
    existing = PART.stat().st_size if PART.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    _log(f"download start (resume offset={existing})")
    t0 = time.time()
    with requests.get(URL, headers=headers, stream=True, timeout=(15, 180)) as r:
        r.raise_for_status()
        if existing and r.status_code != 206:
            existing = 0
            _log("server ignored Range; restarting from 0")
        mode = "ab" if existing else "wb"
        done, last = existing, existing
        with PART.open(mode) as fh:
            for block in r.iter_content(1 << 20):
                fh.write(block)
                done += len(block)
                if done - last >= (1 << 30):
                    rate = (done - existing) / max(1e-6, time.time() - t0) / 1e6
                    _log(f"  {done >> 30} GiB / 11 GiB ({rate:.1f} MB/s)")
                    last = done
    PART.replace(TRAIN)
    _log(f"download complete: {TRAIN.stat().st_size} bytes in {time.time() - t0:.0f}s")


def stratum_of(model: str, attack: str) -> str:
    if model == "human":
        return "human_clean" if attack == "none" else "human_paraphrase"
    if attack == "none":
        return "machine_clean"
    if attack == "paraphrase":
        return "machine_paraphrase"
    return "machine_other_attack"


def _new_state() -> dict:
    return {
        "rows_seen": 0,
        "cell_seen": {},
        "reservoirs": {},
        "rng_state": np.random.default_rng(SEED).bit_generator.state,
        "finished": False,
        "seed": SEED,
    }


def _save_state(st: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st), encoding="utf-8")
    tmp.replace(STATE)


def subset() -> None:
    if not TRAIN.exists():
        raise SystemExit(f"missing {TRAIN}; run download first")
    st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else None
    if st is None:
        st = _new_state()
    if st.get("finished"):
        _log("subset already finished (delete state file to redo)")
        _write_provenance(st)
        return
    rng = np.random.default_rng()
    rng.bit_generator.state = st["rng_state"]
    kept: dict[str, list[dict]] = st.setdefault("reservoirs", {})
    cell_seen: dict[str, int] = st.setdefault("cell_seen", {})
    rows_seen = int(st.get("rows_seen", 0))
    resume_from = rows_seen  # rows already processed in a previous run
    rows_out = 0
    t0 = time.time()
    last_ckpt = rows_seen
    parsed = 0

    with TRAIN.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        rd = csv.reader(fh)
        file_hdr = next(rd)
        idx = {n: file_hdr.index(n) for n in COLUMNS}
        for row in rd:
            if parsed < resume_from:
                parsed += 1
                continue  # already accounted for in a previous run
            parsed += 1
            rows_seen += 1
            if len(row) != len(file_hdr):
                continue
            model = row[idx["model"]]
            attack = row[idx["attack"]]
            domain = row[idx["domain"]]
            stratum = stratum_of(model, attack)
            cap = CAP_PER_STRATUM_DOMAIN.get(stratum, 0)
            if cap == 0:
                continue
            key = f"{stratum}|{domain}"
            cell_seen[key] = cell_seen.get(key, 0) + 1
            bucket = kept.setdefault(key, [])
            item = {c: row[idx[c]] for c in COLUMNS}
            if len(bucket) < cap:
                bucket.append(item)
                rows_out += 1
            else:
                # deterministic reservoir decision: pure function of
                # (cell key, observation index) -> resume-safe + reproducible
                n = cell_seen[key]
                h = int.from_bytes(hashlib.sha1(
                    f"{SEED}|{key}|{n}".encode()).digest()[:8], "big") % n
                if h < cap:
                    bucket[h] = item
            if rows_seen - last_ckpt >= CHECKPOINT_ROWS:
                st["rng_state"] = rng.bit_generator.state
                st["rows_seen"] = rows_seen
                _save_state(st)
                last_ckpt = rows_seen
                _log(f"  checkpoint rows_seen={rows_seen:,} "
                     f"elapsed={time.time() - t0:.0f}s")
    st["rng_state"] = rng.bit_generator.state
    st["rows_seen"] = rows_seen
    st["finished"] = True
    st["rows_sampled"] = sum(len(v) for v in kept.values())
    _save_state(st)

    # deterministic output order: stratum, domain, then file order within cell
    out_rows: list[dict] = []
    for key in sorted(kept):
        out_rows.extend(kept[key])
    with SUBSET.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLUMNS)
        wr.writeheader()
        wr.writerows(out_rows)
    st["subset_sha256"] = sha256_file(SUBSET)
    st["train_sha256"] = sha256_file(TRAIN)
    _save_state(st)
    _log(f"subset rows={len(out_rows):,} -> {SUBSET}")
    _write_provenance(st)


def _write_provenance(st: dict) -> None:
    counts: dict[str, int] = {}
    for key, rows in sorted(st.get("reservoirs", {}).items()):
        stratum = key.split("|", 1)[0]
        counts[stratum] = counts.get(stratum, 0) + len(rows)
    domains: dict[str, int] = {}
    for key, rows in st.get("reservoirs", {}).items():
        d = key.split("|", 1)[1]
        domains[d] = domains.get(d, 0) + len(rows)
    prov = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "hf_id": "liamdugan/raid",
            "file": "train.csv",
            "url": URL,
            "hf_revision": HF_REVISION,
            "license": "MIT",
            "bytes": TRAIN.stat().st_size if TRAIN.exists() else None,
            "sha256": st.get("train_sha256"),
            "rows_seen": st.get("rows_seen"),
        },
        "policy": ("EVALUATION-ONLY (Phase 2). RAID-train rows are never used "
                   "to fit a model, calibrator, threshold or vocabulary. "
                   "RAID remains held out from all training per Phase 0."),
        "acquired_for": ["E2 paraphrase/adversarial evaluation",
                         "E5 cross-corpus supervised evaluation",
                         "E1 per-generator recall on unseen generators"],
        "sampling": {
            "method": "deterministic per-(stratum, domain) reservoir sampling",
            "seed": SEED,
            "caps_per_stratum_domain": CAP_PER_STRATUM_DOMAIN,
        },
        "stratum_counts": counts,
        "domain_counts": domains,
        "asset": {
            "path": "ml/data/raw/raid/raid_eval_subset.csv",
            "bytes": SUBSET.stat().st_size if SUBSET.exists() else None,
            "sha256": st.get("subset_sha256"),
            "rows": sum(counts.values()),
        },
    }
    PROV.parent.mkdir(parents=True, exist_ok=True)
    PROV.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    _log(f"provenance -> {PROV}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset-only", action="store_true",
                    help="skip download; build subset from local train.csv")
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()
    if not args.subset_only:
        download()
    if not args.download_only:
        subset()
