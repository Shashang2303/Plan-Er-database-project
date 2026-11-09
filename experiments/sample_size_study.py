"""How does the optimizer's estimation error shrink with the labeled sample size? (T14, AC-4, D.2.1 risk)

For a fixed set of plans without an LLM tier, the observed F1 and cost are measured once by executing
each plan; the plans are then re-estimated from stratified samples of several sizes and seeds.
Profiles built here are not stored, so the optimizer's latest profiles stay untouched.

    python experiments/sample_size_study.py dblp_acm abt_buy [--sizes 150,300,600,1200] [--seeds 3]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from planer.executor import run_fixed_plan  # noqa: E402
from planer.optimizer import estimate_plan, profile_matchers  # noqa: E402
from planer.request import Plan  # noqa: E402

PLANS = [
    Plan("btree", ["rf"]),
    Plan("vector", ["rules", "rf"], [(0.3, 0.8)], k=10),
    Plan("search", ["rules", "fs", "rf"], [(0.2, 0.9), (0.3, 0.7)], k=10),
    Plan("multikey", ["fs", "rf"], [(0.1, 0.9)]),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--sizes", default="150,300,600,1200")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    sizes = [int(s) for s in args.sizes.split(",")]

    rows = []
    for ds in args.datasets:
        observed = {}
        for plan in PLANS:
            rep = run_fixed_plan(ds, plan)
            observed[plan.describe()] = (rep.f1_pair, rep.cost_usd)
            print(f"{ds} observed  F1={rep.f1_pair:.3f}  cost=${rep.cost_usd:.2f}  {plan.describe()}", flush=True)
        for size in sizes:
            for seed in range(1, args.seeds + 1):
                prof = profile_matchers(ds, size, ("rules", "fs", "rf"), seed=seed, store=False)
                for plan in PLANS:
                    est = estimate_plan(plan, prof)
                    f1, cost = observed[plan.describe()]
                    # sample labels are charged once per optimization, so compare like with like
                    est_cost = est.cost_usd - prof.sample.label_cost_usd
                    rows.append({"dataset": ds, "sample_size": size, "seed": seed, "plan": plan.describe(),
                                 "pred_f1": est.f1, "obs_f1": f1, "f1_abs_error": abs(est.f1 - f1),
                                 "cost_rel_error": abs(est_cost - cost) / cost if cost else np.nan})
            part = pd.DataFrame([r for r in rows if r["dataset"] == ds and r["sample_size"] == size])
            print(f"{ds} n={size:5}  mean |F1 err|={part.f1_abs_error.mean():.3f}  "
                  f"share within 0.05={(part.f1_abs_error <= 0.05).mean():.0%}  "
                  f"mean cost err={part.cost_rel_error.mean():.1%}", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "experiments" / "results" / "sample_size_study.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summary = df.groupby(["dataset", "sample_size"]).agg(
        mean_f1_error=("f1_abs_error", "mean"), within_005=("f1_abs_error", lambda s: (s <= 0.05).mean()),
        mean_cost_error=("cost_rel_error", "mean")).reset_index()
    print("\n" + summary.to_string(index=False))
    print(f"\nwritten to {out}")
