"""Reproducible acquisition of the approved WriteTrace v0 datasets.

Registered sources only. Excluded datasets are recorded in the manifest and are
NEVER downloaded. Large RAID train/extra files are intentionally not downloaded
(never used for training); this is recorded explicitly in the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "ml" / "data" / "raw"
MANIFEST_PATH = ROOT / "ml" / "data" / "manifests" / "datasets_manifest.json"
HF = "https://huggingface.co/datasets"
API = "https://huggingface.co/api/datasets"

APPROVED: dict[str, dict] = {
    "mage": {
        "hf_id": "yaful/MAGE",
        "license": "Apache-2.0",
        "role": "primary-training-corpus",
        "homepage": f"{HF}/yaful/MAGE",
        "files": ["train.csv", "valid.csv", "test.csv",
                  "test_ood_set_gpt.csv", "test_ood_set_gpt_para.csv"],
    },
    "hc3": {
        "hf_id": "Hello-SimpleAI/HC3",
        "license": "CC-BY-SA-4.0 (source-specific licenses may be stricter, see dataset card)",
        "role": "unseen-generator-unseen-domain-eval-only-never-trained-on",
        "homepage": f"{HF}/Hello-SimpleAI/HC3",
        "files": ["reddit_eli5.jsonl", "finance.jsonl", "medicine.jsonl",
                  "open_qa.jsonl", "wiki_csai.jsonl"],
    },
    "raid": {
        "hf_id": "liamdugan/raid",
        "license": "MIT",
        "role": "held-out-robustness-eval-only-never-trained-on",
        "homepage": f"{HF}/liamdugan/raid",
        "files": ["test.csv"],
        "optional": True,
        "not_downloaded": {
            "train.csv": "11779491051 bytes; RAID is never used for training (Phase 0 decision)",
            "extra.csv": "3707337095 bytes; not required for Phase 1",
        },
    },
}

EXCLUDED: dict[str, str] = {
    "m4": "No LICENSE file upstream; mixed provenance (Phase 0: excluded pending license review).",
    "llm-detectaive": "No confirmed upstream data license (Phase 0: reference only).",
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download_file(url: str, dest: Path) -> str:
    """Stream a file with resume support. Returns status."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    existing = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, headers=headers, stream=True, timeout=(15, 120)) as r:
        r.raise_for_status()
        if existing and r.status_code != 206:
            existing = 0  # server ignored resume; restart
        mode = "ab" if existing else "wb"
        done, last_report = existing, existing
        with part.open(mode) as fh:
            for block in r.iter_content(1 << 20):
                fh.write(block)
                done += len(block)
                if done - last_report >= 256 << 20:
                    print(f"    {dest.name}: {done >> 20} MB", flush=True)
                    last_report = done
    part.replace(dest)
    return "resumed-and-completed" if existing else "downloaded"


def hf_revision(hf_id: str) -> str | None:
    try:
        r = requests.get(f"{API}/{hf_id}", timeout=30)
        r.raise_for_status()
        return r.json().get("sha")
    except Exception:
        return None


def count_rows(path: Path) -> int | str | None:
    """Row count. CSV rows via pyarrow (quoted newlines handled); jsonl = lines."""
    if path.suffix == ".jsonl":
        with path.open("rb") as fh:
            return sum(1 for _ in fh)
    try:
        import pyarrow.csv as pacsv
        with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
            first_col = fh.readline().split(",")[0].strip().strip('"')
        reader = pacsv.open_csv(
            str(path),
            read_options=pacsv.ReadOptions(block_size=1 << 26),
            parse_options=pacsv.ParseOptions(newlines_in_values=True),
            convert_options=pacsv.ConvertOptions(include_columns=[first_col]),
        )
        return sum(batch.num_rows for batch in reader)
    except Exception as exc:  # honest failure recording
        return f"count-failed: {type(exc).__name__}"

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download approved WriteTrace v0 datasets")
    ap.add_argument("--skip-raid", action="store_true",
                    help="skip the optional RAID test split (recorded in manifest)")
    args = ap.parse_args(argv)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "manifest_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool": "ml/data/download.py",
        "approved": {},
        "excluded_never_downloaded": EXCLUDED,
    }
    failures_required = 0

    for name, spec in APPROVED.items():
        print(f"[{name}] license={spec['license']}", flush=True)
        entry = {k: v for k, v in spec.items() if k != "files"}
        entry["revision_sha"] = hf_revision(spec["hf_id"])
        entry["files"] = []
        if args.skip_raid and name == "raid":
            entry["status"] = "skipped-by-flag"
            print("    skipped (--skip-raid); recorded in manifest", flush=True)
        else:
            for fname in spec["files"]:
                url = f"{spec['homepage']}/resolve/main/{fname}"
                dest = RAW_DIR / name / fname
                try:
                    if dest.exists() and dest.stat().st_size > 0:
                        status = "already-present"
                    else:
                        status = download_file(url, dest)
                    rec = {
                        "name": fname,
                        "relative_path": f"ml/data/raw/{name}/{fname}",
                        "url": url,
                        "bytes": dest.stat().st_size,
                        "sha256": sha256_file(dest),
                        "rows": count_rows(dest),
                        "downloaded_at": datetime.now(timezone.utc).isoformat(),
                        "status": status,
                    }
                except Exception as exc:
                    rec = {"name": fname, "url": url,
                           "status": f"failed: {type(exc).__name__}: {exc}"}
                    if not spec.get("optional"):
                        failures_required += 1
                    print(f"    FAILED {fname}: {exc}", flush=True)
                entry["files"].append(rec)
        manifest["approved"][name] = entry

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"manifest -> {MANIFEST_PATH}", flush=True)
    if failures_required:
        print(f"ERROR: {failures_required} required file(s) failed", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

