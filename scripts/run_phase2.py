"""Phase 2 runner: E0 label-orientation audit + E1..E7, locked baseline only.

Reads config from ml/configs/phase2.json. Never retrains or re-tunes anything.
Each experiment's result is cached immediately to
ml/experiments/phase2/cache/E*.json so an interrupted run never loses finished
work; re-running resumes from the cache unless --force is given.

Usage: python scripts/run_phase2.py [--only E1,E2,...] [--force] [--cpu]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.experiments.phase2 import EXPERIMENTS, RESULTS_DIR, run_all


def main() -> int:
    ap = argparse.ArgumentParser(description="Run WriteTrace Phase-2 robustness experiments")
    ap.add_argument("--only", default="", help="comma-separated subset, e.g. E0,E1 or E5")
    ap.add_argument("--force", action="store_true", help="re-run even if cached")
    ap.add_argument("--cpu", action="store_true", help="force CPU for local models")
    args = ap.parse_args()
    only = [x.strip().upper() for x in args.only.split(",") if x.strip()] or None
    unknown = [x for x in (only or []) if x not in EXPERIMENTS]
    if unknown:
        print(f"unknown experiments: {unknown}; valid: {sorted(EXPERIMENTS)}")
        return 2
    cfg_path = ROOT / "ml" / "configs" / "phase2.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if args.cpu:
        cfg["device"] = "cpu"
    results = run_all(cfg, only=only, force=args.force)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "phase2-results.json"
    merged = {}
    if out.exists():  # never clobber results from previous partial runs
        try:
            merged = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            merged = {}
    merged.update(results)
    results = merged
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"results -> {out}")
    rc = 0
    for k, v in results.items():
        if not isinstance(v, dict):
            continue
        status = v.get("status", "reference" if k == "reference" else "?")
        suffix = " (cached)" if v.get("cached") else ""
        print(f"  {k}: {status}{suffix}")
        if status == "failed":
            print(f"      {v.get('error', '')}")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
