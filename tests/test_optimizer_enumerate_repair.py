"""Optimizer enumeration and Pareto pruning (T15), plan selection under theta and budget, the LLM-call
latency guard, transitivity repair (T17) and BSON-safe prediction-error logging (regression test)."""
from planer.clustering import count_transitivity_violations, repair_clusters
from planer.optimizer.enumerate import (
    THRESHOLD_GRID, PlanPoint, enumerate_plans, expected_llm_calls, pareto_front, select_plan,
)
from planer.optimizer.estimate import PlanEstimate
from planer.request import Plan

from .test_optimizer_estimate import _profile


def _pt(cost: float, f1: float, path: str = "btree") -> PlanPoint:
    return PlanPoint(Plan(path, ["rules"]), PlanEstimate("p", cost, f1, f1, f1, f1, 1.0, 100))


def test_enumeration_covers_the_plan_space():
    points = enumerate_plans(_profile())
    # one profiled path; subsets {rules}, {llm}, {rules, llm} with 16 threshold pairs for rules
    assert len(points) == 1 + 1 + len(THRESHOLD_GRID)
    assert all(p.plan.matchers in (["rules"], ["llm"], ["rules", "llm"]) for p in points)


def test_pareto_front_drops_dominated_plans():
    pts = [_pt(1.0, 0.5), _pt(2.0, 0.4), _pt(2.0, 0.7), _pt(3.0, 0.7), _pt(5.0, 0.9)]
    front = pareto_front(pts)
    assert [(p.cost, p.f1) for p in front] == [(1.0, 0.5), (2.0, 0.7), (5.0, 0.9)]


def test_select_theta_and_budget():
    front = pareto_front([_pt(1.0, 0.5), _pt(2.0, 0.7), _pt(5.0, 0.9)])
    chosen, ok = select_plan(front, "theta", 0.6)
    assert ok and chosen.cost == 2.0                        # cheapest plan meeting the target
    chosen, ok = select_plan(front, "theta", 0.95)
    assert not ok and chosen.f1 == 0.9                      # infeasible: best available
    chosen, ok = select_plan(front, "budget", 2.3)
    assert ok and chosen.f1 == 0.7                          # 2.0 * 1.1 <= 2.3
    chosen, ok = select_plan(front, "budget", 2.1)
    assert ok and chosen.f1 == 0.5                          # safety margin excludes 2.0 * 1.1 = 2.2
    chosen, ok = select_plan(front, "budget", 0.5)
    assert not ok and chosen.cost == 1.0


def test_expected_llm_calls_respects_cache():
    pts = enumerate_plans(_profile(), llm_cached={"vector:10": 0.5})
    two_tier = next(p for p in pts if p.plan.matchers == ["rules", "llm"] and p.plan.thresholds[0] == (0.1, 0.9))
    assert expected_llm_calls(two_tier) == two_tier.est.tier_n[1] * 0.5
    assert expected_llm_calls(next(p for p in pts if p.plan.matchers == ["rules"])) == 0


def test_transitivity_repair_cuts_weakest_edge():
    # a1-b1 strong, a1-b2 weak, a2-b2 strong: one component {a1, a2, b1, b2};
    # the cascade said a2-b1 is NOT a match -> violation, repaired by removing the weak edge a1-b2
    matches = {("a1", "b1"): 0.95, ("a1", "b2"): 0.55, ("a2", "b2"): 0.9}
    nonmatches = {("a2", "b1")}
    docs, stats = repair_clusters(matches, nonmatches, "A", "B")
    clusters = sorted(d["members"] for d in docs)
    assert clusters == [["A/a1", "B/b1"], ["A/a2", "B/b2"]]
    assert stats["violations_before"] == 1 and stats["edges_removed"] == 1 and stats["llm_calls"] == 0
    assert count_transitivity_violations([d["members"] for d in docs], nonmatches, "A", "B") == 0


def test_repair_leaves_consistent_clusters_alone():
    docs, stats = repair_clusters({("a1", "b1"): 0.9, ("a2", "b1"): 0.8}, {("a3", "b9")}, "A", "B")
    assert [d["members"] for d in docs] == [["A/a1", "A/a2", "B/b1"]]
    assert stats["edges_removed"] == 0


def test_prediction_error_is_bson_encodable():
    import bson
    import numpy as np

    from planer.executor import prediction_error

    err = prediction_error({"cost_usd": np.float64(10.0), "f1": np.float64(0.8)}, 11.0, 0.78)
    bson.encode(err)                                   # raises on NumPy scalars such as np.bool_
    assert err["cost_within_15pct"] is True and abs(err["f1_abs_error"] - 0.02) < 1e-9
    assert prediction_error(None, 1.0, 0.5) is None
