"""Demo D6 (T11, AC-5): an uncertain pair goes to the local LLM; re-submitting it is answered
from llm_cache with zero model calls.

    python experiments/demo_llm_cache.py abt_buy [--pairs 5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from planer import db as dbm  # noqa: E402
from planer.blocking.profile import latest_profiles  # noqa: E402
from planer.datasets import get_dataset  # noqa: E402
from planer.matchers import PairBatch, RuleMatcher  # noqa: E402
from planer.matchers.llm import OllamaMatcher  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--pairs", type=int, default=5)
    args = ap.parse_args()

    db = dbm.get_db()
    spec = get_dataset(args.dataset)
    prof = next(p for p in latest_profiles(args.dataset, db) if p["path"] == "vector")
    cands = [(c["a_id"], c["b_id"]) for c in db[dbm.CANDIDATES].find({"req_id": prof["req_id"]}).limit(2000)]
    batch = PairBatch.load(db, spec, cands)
    rule = RuleMatcher().score(batch)
    uncertain = [i for i in (abs(rule - 0.5)).argsort()[: args.pairs]]   # the most uncertain pairs
    sub = batch.subset(uncertain)

    llm = OllamaMatcher()
    for attempt in (1, 2):
        llm.reset_stats()
        scores = llm.score(sub)
        print(f"run {attempt}: model calls={llm.stats['calls']}  cache hits={llm.stats['cache_hits']}  "
              f"model time={llm.stats['secs']:.1f}s")
    for i, s in enumerate(scores):
        a = sub.a_docs[i].get(spec.a.title_attr)
        b = sub.b_docs[i].get(spec.b.title_attr)
        print(f"  rule={rule[uncertain[i]]:.2f}  LLM={'match' if s >= 0.5 else 'no_match':8}  {a!r}  <->  {b!r}")
