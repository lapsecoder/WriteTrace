"""Phase-4 latency benchmark (real measurements, CPU).

Writes logs/phase4_bench.json (read by the Phase-4 report generator).
Usage: python scripts/bench_phase4.py [--runs 20]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.experiments.phase4 import LOGS, bench, write_json


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase-4 inference latency bench")
    ap.add_argument("--runs", type=int, default=20)
    args = ap.parse_args()

    res = bench(n_runs=args.runs)
    out = write_json(LOGS / "phase4_bench.json", res)
    print(f"bench -> {out}")
    print(f"  cold load: {res['cold_load_ms']} ms | startup warm-up "
          f"(incl. spaCy load): {res['warmup_including_spacy_load_ms']} ms")
    for name, blk in res["warm"].items():
        w = blk["analyze_with_signals_ms"]
        n = blk["analyze_no_signals_ms"]
        print(f"  {name:20s} words={blk['words']:<5d} "
              f"with signals med {w['median_ms']} ms (p95 {w['p95_ms']}) | "
              f"no signals med {n['median_ms']} ms (p95 {n['p95_ms']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
