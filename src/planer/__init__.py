"""PlanER: cost-based optimization of hybrid classical-LLM entity resolution plans in MongoDB.

Python API (D.4.2 section 8, NFR-9):
    req_id = planer.submit(request)            # store a declarative ER request
    result = planer.optimize(req_id)           # chosen plan P = (b, M, tau, kappa), Pareto set, predictions
    report = planer.execute(req_id, plan)      # run a physical plan, log observed cost and F1
    est    = planer.estimate(dataset, plan)    # predicted cost and F1 of any plan
"""
from __future__ import annotations

__version__ = "1.0.0"

from .request import ERRequest, Objective, Plan, submit


def execute(req_id, plan: Plan, predicted: dict | None = None, sample_label_cost: float = 0.0):
    from .executor import execute as _execute

    return _execute(req_id, plan, predicted=predicted, sample_label_cost=sample_label_cost)


def estimate(dataset: str, plan: Plan, use_cache_stats: bool = True):
    from .optimizer import estimate_plan, latest_profile, llm_cached_fraction

    frac = llm_cached_fraction(dataset, plan) if use_cache_stats else 0.0
    return estimate_plan(plan, latest_profile(dataset), llm_cached_fraction=frac)


def optimize(req_id):
    from .optimizer import optimize as _optimize

    return _optimize(req_id)


__all__ = ["ERRequest", "Objective", "Plan", "submit", "execute", "estimate", "optimize"]
