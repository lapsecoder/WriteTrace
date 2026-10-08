"""Phase-4 reference parity check: Phase-4 inference vs Phase-3 numbers.

Writes logs/phase4_reference.json (read by the Phase-4 report generator).
Usage: python scripts/verify_phase4.py [--n 5] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.experiments.phase4 import LOGS, reference_check, write_json


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase-4 reference parity check")
    ap.add_argument("--n", type=int, default=5, help="rows per class")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    res = reference_check(n_per_class=args.n, seed=args.seed)
    out = write_json(LOGS / "phase4_reference.json", res)
    print(f"reference check -> {out}")
    print(f"  rows: {res['n_rows']}  max_abs_delta: {res['max_abs_delta']:.3e} "
          f"(tolerance {res['tolerance']})  "
          f"within tolerance: {res['all_within_tolerance']}")
    ec = res.get("e7_cross_check") or {}
    if ec:
        print(f"  E7 cross-check: recorded {ec['e7_recorded_p_machine']} / "
              f"phase3 recompute {ec['recomputed_p_machine']} / "
              f"phase4 {ec['phase4_p_machine']} -> match={ec['match']}")
    return 0 if (res["all_within_tolerance"] and ec.get("match", True)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
