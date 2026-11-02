"""Cost-based optimizer (Phase 3).

Done:        T13 matcher profiling (profiling.py), T14 stratified sampling + Eq. (1) estimation
             (sampling.py, estimate.py).
Remaining:   T15 plan enumeration with Pareto pruning and plan selection (FR-14, FR-15),
             T16 optimizer-driven execution with predicted-vs-observed logging (FR-19, FR-20).
"""
from .estimate import PlanEstimate, estimate_plan, llm_cached_fraction
from .profiling import OptimizerProfile, latest_profile, profile_matchers
from .sampling import LabeledSample, stratified_sample

__all__ = [
    "PlanEstimate", "estimate_plan", "llm_cached_fraction", "OptimizerProfile", "latest_profile",
    "profile_matchers", "LabeledSample", "stratified_sample",
]
