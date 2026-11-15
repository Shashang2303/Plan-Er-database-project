"""planer.optimize / T16: choose a plan for a stored request and (optionally) execute it.

optimize()              enumerate -> Pareto-prune -> select; stores the plan and the Pareto set
                        in the request document and logs the decision in run_log.
reselect()              re-applies a new objective to the stored Pareto set without re-estimating
                        anything, so the demo updates in well under 5 s (AC-8, NFR-8).
optimize_and_execute()  runs the chosen plan and logs predicted vs observed cost and F1 (FR-19, FR-20).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from bson import ObjectId
from pymongo.database import Database

from .. import db as dbm
from ..request import Objective, Plan, load_request
from ..config import settings
from .enumerate import BUDGET_MARGIN, PlanPoint, enumerate_plans, expected_llm_calls, pareto_front, select_plan
from .estimate import llm_cached_fraction
from .profiling import OptimizerProfile, latest_profile

log = logging.getLogger(__name__)

DEFAULT_CLUSTERING = "cc_repair"   # repair costs no model call, so the optimizer always enables it


@dataclass
class OptimizeResult:
    req_id: str
    dataset: str
    objective: dict
    chosen: PlanPoint
    feasible: bool
    pareto: list[PlanPoint]
    n_plans: int
    secs: float
    sample_label_cost_usd: float

    @property
    def predicted(self) -> dict:
        return {"cost_usd": float(self.chosen.cost), "f1": float(self.chosen.f1),
                "precision": float(self.chosen.est.precision), "recall": float(self.chosen.est.recall)}

    def summary(self) -> dict:
        return {"req_id": self.req_id, "dataset": self.dataset, "objective": self.objective,
                "plan": self.chosen.plan.describe(), "feasible": self.feasible, "predicted": self.predicted,
                "plans_enumerated": self.n_plans, "pareto_size": len(self.pareto), "secs": self.secs}


def _oid(req_id):
    return ObjectId(req_id) if isinstance(req_id, str) else req_id


def optimize(req_id, db: Database | None = None, prof: OptimizerProfile | None = None) -> OptimizeResult:
    db = db if db is not None else dbm.get_db()
    oid = _oid(req_id)
    req = load_request(oid, db)
    dataset = req.dataset.key
    t0 = time.perf_counter()
    prof = prof or latest_profile(dataset, db)

    llm_cached: dict[str, float] = {}
    if "llm" in prof.matchers:
        for key, path in prof.paths.items():
            llm_cached[key] = llm_cached_fraction(dataset, Plan(path["path"], ["llm"], k=path.get("k")), db=db)

    all_points = enumerate_plans(prof, llm_cached)
    points = [p for p in all_points if expected_llm_calls(p) <= settings.llm_max_calls]
    if len(points) < len(all_points):
        log.info("latency guard: %d of %d plans exceed %d expected LLM calls and were skipped",
                 len(all_points) - len(points), len(all_points), settings.llm_max_calls)
    front = pareto_front(points)
    chosen, feasible = select_plan(front, req.objective.type, req.objective.value)
    chosen.plan.clustering = DEFAULT_CLUSTERING
    secs = time.perf_counter() - t0
    result = OptimizeResult(str(oid), dataset, {"type": req.objective.type, "value": req.objective.value},
                            chosen, feasible, front, len(points), secs, prof.sample.label_cost_usd)

    db[dbm.REQUESTS].update_one({"_id": oid}, {"$set": {
        "status": "optimized", "plan": chosen.plan.to_doc(),
        "optimizer": {"ts": datetime.now(timezone.utc), "predicted": result.predicted, "feasible": feasible,
                      "plans_enumerated": len(points), "secs": secs, "profile_ts": prof.ts,
                      "sample_label_cost_usd": prof.sample.label_cost_usd, "budget_margin": BUDGET_MARGIN,
                      "pareto": [p.to_doc() for p in front]},
    }})
    db[dbm.RUN_LOG].insert_one({"kind": "optimization", "ts": datetime.now(timezone.utc), "req_id": oid,
                                "dataset": dataset, **{k: v for k, v in result.summary().items() if k != "req_id"}})
    log.info("optimizer: %d plans, %d on the Pareto front, chose %s (pred cost $%.4f, F1 %.3f%s) in %.1fs",
             len(points), len(front), chosen.plan.describe(), chosen.cost, chosen.f1,
             "" if feasible else ", objective infeasible", secs)
    return result


def stored_pareto(req_id, db: Database | None = None) -> list[PlanPoint]:
    db = db if db is not None else dbm.get_db()
    doc = db[dbm.REQUESTS].find_one({"_id": _oid(req_id)}, {"optimizer.pareto": 1})
    if not doc or not doc.get("optimizer"):
        raise RuntimeError(f"request {req_id} has not been optimized yet")
    return [PlanPoint.from_doc(d) for d in doc["optimizer"]["pareto"]]


def reselect(req_id, objective: dict, db: Database | None = None) -> tuple[PlanPoint, bool]:
    """Apply a new objective to the cached Pareto set (no re-estimation)."""
    obj = Objective(objective["type"], float(objective["value"]))
    point, feasible = select_plan(stored_pareto(req_id, db), obj.type, obj.value)
    point.plan.clustering = DEFAULT_CLUSTERING
    return point, feasible


def optimize_and_execute(req_id, db: Database | None = None):
    """T16: run the plan the optimizer selects and log predicted against observed cost and F1."""
    from ..executor import execute

    db = db if db is not None else dbm.get_db()
    res = optimize(req_id, db)
    report = execute(req_id, res.chosen.plan, predicted=res.predicted,
                     sample_label_cost=res.sample_label_cost_usd, db=db)
    return res, report
