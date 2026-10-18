"""T7: profile every access path on every loaded dataset and write a CSV summary.

    python experiments/profile_blocking.py [dataset ...] [--ks 5,10,20]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from planer.blocking.profile import profile_all  # noqa: E402
from planer.datasets import DATASETS  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--ks", default="5,10,20")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    rows = []
    for ds in args.datasets:
        for r in profile_all(ds, ks=tuple(int(k) for k in args.ks.split(","))):
            summary = (r.pop("explain", None) or {}).get("summary", {})
            rows.append(r | {"indexes_used": ",".join(summary.get("indexes_used", []))})
    out = ROOT / "experiments" / "results" / "blocking_profile.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).drop(columns=["_id"], errors="ignore")
    df.to_csv(out, index=False)
    cols = ["dataset", "path", "k", "n_candidates", "pair_completeness", "reduction_ratio", "secs", "indexes_used"]
    print(df[[c for c in cols if c in df]].to_string(index=False))
    print(f"\nwritten to {out}")
