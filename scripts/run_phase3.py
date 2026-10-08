"""Phase 3 runner: E3 models -> E1 ablation -> E2 syntactic -> E4 calibration
-> E5 generalization/LODO -> E6 leakage/importance -> SELECTION -> E7 errors
-> E8 gate. Reads ml/configs/phase3.json.

Every finished experiment is cached to ml/experiments/phase3/cache/*.json so
an interrupted run never loses finished work; re-running resumes from cache
unless --force. SELECTION always re-runs (cheap, rewrites the final artifact
under ml/models/final-v1); E7/E8 are invalidated automatically when the
selected winner changes.

Usage: python scripts/run_phase3.py [--only E3,E1,...] [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.experiments.phase3 import EXPERIMENTS, RESULTS_DIR, run_all


def main() -> int:
    ap = argparse.ArgumentParser(description="Run WriteTrace Phase-3 experiments")
    ap.add_argument("--only", default="",
                    help="comma-separated subset, e.g. E3,E1 or SELECTION,E8")
    ap.add_argument("--force", action="store_true", help="re-run even if cached")
    args = ap.parse_args()
    only = [x.strip().upper() for x in args.only.split(",") if x.strip()] or None
    unknown = [x for x in (only or []) if x not in EXPERIMENTS]
    if unknown:
        print(f"unknown experiments: {unknown}; valid: {sorted(EXPERIMENTS)}")
        return 2
    cfg_path = ROOT / "ml" / "configs" / "phase3.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    results = run_all(cfg, only=only, force=args.force)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "phase3-results.json"
    merged = {}
    if out.exists():
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
        status = v.get("status", "meta")
        suffix = " (cached)" if v.get("cached") else ""
        print(f"  {k}: {status}{suffix}")
        if status == "failed":
            print(f"      {v.get('error', '')}")
            rc = 1
    sel = results.get("SELECTION", {})
    if isinstance(sel, dict) and sel.get("winner"):
        print(f"  final model: {sel['winner']} -> {sel.get('artifact_dir')}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
