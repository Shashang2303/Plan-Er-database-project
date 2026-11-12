"""T15 / FR-14, FR-15: plan enumeration, Pareto pruning and plan selection.

The plan space is  access path (incl. its k)  x  cost-ordered matcher subset  x  threshold grid.
Because matchers have a fixed cost order, a subset fixes the cascade order, and every non-final tier
picks (l, u) from a small grid. Each plan is costed by estimate_plan() (Eq. 1) on the labeled sample;
plans dominated in both estimated cost and F1 are pruned, and the objective selects from the frontier:
    theta:  cheapest plan with F1 >= theta        (if none, the most accurate plan, flagged infeasible)
    budget: most accurate plan with cost*(1+margin) <= B  (if none, the cheapest plan, flagged)
The safety margin keeps observed cost under the budget despite estimation error (NFR-11).
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

from ..request import MATCHER_ORDER, Plan
from .estimate import PlanEstimate, estimate_plan
from .profiling import OptimizerProfile

log = logging.getLogger(__name__)

LOWER_GRID = (0.1, 0.2, 0.3, 0.4)
UPPER_GRID = (0.6, 0.7, 0.8, 0.9)
THRESHOLD_GRID = tuple((lo, hi) for lo in LOWER_GRID for hi in UPPER_GRID)   # 16 pairs per tier
BUDGET_MARGIN = 0.10
F1_TOLERANCE = 0.005   # estimated F1 differences below this are not meaningful at a ~300-pair sample


@dataclass
class PlanPoint:
    plan: Plan
    est: PlanEstimate

    @property
    def cost(self) -> float:
        return self.est.cost_usd

    @property
    def f1(self) -> float:
        return self.est.f1

    def to_doc(self) -> dict:
        return {"plan": self.plan.to_doc(), "plan_text": self.plan.describe(), "cost_usd": self.cost, "f1": self.f1,
                "precision": self.est.precision, "recall": self.est.recall,
                "pair_completeness": self.est.pair_completeness, "n_candidates": self.est.n_candidates,
                "tier_n": self.est.tier_n, "tier_escalation": self.est.tier_escalation,
                "cost_breakdown": self.est.cost_breakdown}

    @classmethod
    def from_doc(cls, d: dict) -> "PlanPoint":
        plan = Plan.from_doc(d["plan"])
        est = PlanEstimate(plan=d["plan_text"], cost_usd=d["cost_usd"], f1=d["f1"], precision=d["precision"],
                           recall=d["recall"], recall_within_candidates=0.0,
                           pair_completeness=d["pair_completeness"], n_candidates=d["n_candidates"],
                           tier_n=d.get("tier_n", []), tier_escalation=d.get("tier_escalation", []),
                           cost_breakdown=d.get("cost_breakdown", {}))
        return cls(plan, est)


def enumerate_plans(prof: OptimizerProfile, llm_cached: dict[str, float] | None = None,
                    matchers: tuple[str, ...] | None = None, grid=THRESHOLD_GRID) -> list[PlanPoint]:
    """Estimate every plan in the space. llm_cached maps path_key -> cached fraction of LLM verdicts."""
    llm_cached = llm_cached or {}
    available = [m for m in (matchers or MATCHER_ORDER) if m in prof.matchers]
    points: list[PlanPoint] = []
    for key, path in prof.paths.items():
        if key not in prof.sample.membership or not prof.sample.membership[key].any():
            log.debug("skipping %s: no sampled pair", key)
            continue
        for r in range(1, len(available) + 1):
            for subset in itertools.combinations(available, r):       # combinations keep the cost order
                for taus in itertools.product(grid, repeat=r - 1):
                    plan = Plan(path["path"], list(subset), list(taus), k=path.get("k"))
                    frac = llm_cached.get(key, 0.0) if "llm" in subset else 0.0
                    points.append(PlanPoint(plan, estimate_plan(plan, prof, llm_cached_fraction=frac)))
    return points


def expected_llm_calls(p: PlanPoint) -> float:
    """New (uncached) LLM calls the plan is expected to make."""
    llm_tiers = [i for i, m in enumerate(p.plan.matchers) if m.startswith("llm")]
    if not llm_tiers:
        return 0.0
    cached = p.est.cost_breakdown.get("llm_cached_fraction", 0.0)
    return sum(p.est.tier_n[i] for i in llm_tiers) * (1 - cached)


def pareto_front(points: list[PlanPoint]) -> list[PlanPoint]:
    """Plans not dominated in (lower cost, higher F1), sorted by cost."""
    front: list[PlanPoint] = []
    best_f1 = -1.0
    for p in sorted(points, key=lambda p: (p.cost, -p.f1)):
        if p.f1 > best_f1 + 1e-12:
            front.append(p)
            best_f1 = p.f1
    return front


def select_plan(front: list[PlanPoint], objective_type: str, value: float,
                margin: float = BUDGET_MARGIN) -> tuple[PlanPoint, bool]:
    """Returns (chosen plan, feasible)."""
    if not front:
        raise ValueError("empty Pareto front")
    if objective_type == "theta":
        ok = [p for p in front if p.f1 >= value]
        if ok:
            return min(ok, key=lambda p: p.cost), True
        # infeasible: the cheapest plan whose F1 is within F1_TOLERANCE of the best achievable one,
        # instead of paying several times more for a difference the estimate cannot resolve
        best = max(p.f1 for p in front)
        return min((p for p in front if p.f1 >= best - F1_TOLERANCE), key=lambda p: p.cost), False
    if objective_type == "budget":
        ok = [p for p in front if p.cost * (1 + margin) <= value]
        return (max(ok, key=lambda p: p.f1), True) if ok else (min(front, key=lambda p: p.cost), False)
    raise ValueError(f"unknown objective type {objective_type!r}")
