"""Cost-based optimizer.

profiling.py   T13: per-matcher cost, escalation, precision, recall on the labeled sample (FR-13)
sampling.py    T14: stratified labeled sample with inverse-probability weights (FR-13)
estimate.py    T14: plan cost and F1 under Eq. (1)
enumerate.py   T15: plan enumeration, Pareto pruning, plan selection (FR-14, FR-15)
optimize.py    T15/T16: optimize a stored request, re-select from the cached Pareto set, execute (FR-19, FR-20)
"""
from .enumerate import PlanPoint, enumerate_plans, pareto_front, select_plan
from .estimate import PlanEstimate, estimate_plan, llm_cached_fraction
from .optimize import OptimizeResult, optimize, optimize_and_execute, reselect, stored_pareto
from .profiling import OptimizerProfile, latest_profile, profile_matchers
from .sampling import LabeledSample, stratified_sample

__all__ = [
    "PlanEstimate", "estimate_plan", "llm_cached_fraction", "OptimizerProfile", "latest_profile",
    "profile_matchers", "LabeledSample", "stratified_sample", "PlanPoint", "enumerate_plans",
    "pareto_front", "select_plan", "OptimizeResult", "optimize", "optimize_and_execute", "reselect",
    "stored_pareto",
]
