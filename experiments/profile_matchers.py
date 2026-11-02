"""T13: per-matcher profiling (cost, escalation rate, precision, recall) on the stratified sample.

    python experiments/profile_matchers.py abt_buy [--sample-size 300] [--llm-sample 60] [--no-llm]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from planer.optimizer.profiling import profile_matchers, summarize  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--sample-size", type=int, default=300)
    ap.add_argument("--llm-sample", type=int, default=60)
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    matchers = ("rules", "fs", "rf") if args.no_llm else ("rules", "fs", "rf", "llm")
    rows = []
    for ds in args.datasets:
        prof = profile_matchers(ds, args.sample_size, matchers, args.llm_sample)
        for name, s in summarize(prof).items():
            rows.append({"dataset": ds, "matcher": name, **s, "profile_secs": prof.secs})
    df = pd.DataFrame(rows)
    out = ROOT / "experiments" / "results" / "matcher_profile.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nwritten to {out}")
