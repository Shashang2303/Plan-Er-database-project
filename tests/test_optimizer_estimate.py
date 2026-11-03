import numpy as np

from planer.optimizer.estimate import estimate_plan
from planer.optimizer.profiling import MatcherProfile, OptimizerProfile
from planer.optimizer.sampling import LabeledSample, allocate
from planer.request import Plan


def test_allocate_equal_and_capped():
    assert allocate({0: 1000, 1: 1000, 2: 10}, 90) == {0: 40, 1: 40, 2: 10}
    assert sum(allocate({0: 5, 1: 3}, 100).values()) == 8


def _profile():
    # 4 sampled pairs, weights 10 each; pairs 0 and 1 are true matches
    sample = LabeledSample(
        dataset="toy", pairs=[("a0", "b0"), ("a1", "b1"), ("a2", "b2"), ("a3", "b3")],
        sim=np.array([0.9, 0.5, 0.4, 0.1]), stratum=np.array([5, 4, 3, 0]),
        weight=np.full(4, 10.0), labels=np.array([True, True, False, False]),
        membership={"vector:10": np.array([True, True, True, True])},
    )
    matchers = {
        "rules": MatcherProfile("rules", False, np.array([0.9, 0.5, 0.5, 0.1]), 1e-6, 0.01),
        "llm": MatcherProfile("llm:x", True, np.array([1.0, 1.0, 0.0, np.nan]), 1e-3, 1.0, tpr=1.0, fpr=0.0),
    }
    paths = {"vector:10": {"path": "vector", "k": 10, "n_candidates": 1000, "pc": 0.8, "secs": 1.0,
                           "block_cost_usd": 1e-4}}
    return OptimizerProfile("toy", sample, matchers, 1e-7, paths)


def test_estimate_eq1_cost_and_pc_capped_recall():
    est = estimate_plan(Plan("vector", ["rules", "llm"], [(0.2, 0.8)], k=10), _profile())
    # rules decides pair0 (match) and pair3 (non-match); pairs 1, 2 (half the weight) escalate
    assert est.tier_escalation[0] == 0.5
    assert est.tier_n == [1000, 500]
    assert est.precision == 1.0 and est.recall_within_candidates == 1.0
    assert abs(est.recall - 0.8) < 1e-9                    # capped by pair completeness
    assert abs(est.f1 - 2 * 0.8 / 1.8) < 1e-9
    expected = 1e-4 + 1000 * 1e-7 + 1000 * 1e-6 + 500 * 1e-3 + est.cost_breakdown["labels"]
    assert abs(est.cost_usd - expected) < 1e-12


def test_cached_llm_pairs_are_free():
    base = estimate_plan(Plan("vector", ["rules", "llm"], [(0.2, 0.8)], k=10), _profile())
    cached = estimate_plan(Plan("vector", ["rules", "llm"], [(0.2, 0.8)], k=10), _profile(), llm_cached_fraction=1.0)
    assert abs((base.cost_usd - cached.cost_usd) - 500 * 1e-3) < 1e-12
