"""T14 (part 2): estimate a plan's cost and F1 from the optimizer profile (Eq. 1).

    C(P) = C_block(b) + C_feat * N_1 + sum_i N_i * c_i + C_label(P),     N_{i+1} = N_i * e_i(l_i, u_i)

The cascade is replayed on the weighted labeled sample restricted to the pairs that access path b
produces. Weighted counts give escalation rates e_i and the expected TP / FP / FN among the
candidates. Recall is capped by the pair completeness of b, since no matcher can recover a match
that blocking discarded. LLM tiers are charged only for pairs that miss llm_cache.
Plan enumeration and Pareto pruning (T15) will call estimate_plan() for every candidate plan.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..config import settings
from ..request import Plan
from .profiling import OptimizerProfile
from .sampling import path_key


@dataclass
class PlanEstimate:
    plan: str
    cost_usd: float
    f1: float
    precision: float
    recall: float
    recall_within_candidates: float
    pair_completeness: float
    n_candidates: int
    tier_n: list[float] = field(default_factory=list)          # N_i
    tier_escalation: list[float] = field(default_factory=list)  # e_i
    cost_breakdown: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _profile_key(matcher_spec: str) -> str:
    return "llm" if matcher_spec.startswith("llm") else matcher_spec


def estimate_plan(plan: Plan, prof: OptimizerProfile, llm_cached_fraction: float = 0.0) -> PlanEstimate:
    s = prof.sample
    key = path_key(plan.access_path, plan.k)
    if key not in prof.paths or key not in s.membership:
        raise KeyError(f"access path {key} was not profiled; available: {sorted(prof.paths)}")
    path = prof.paths[key]
    warnings: list[str] = []

    in_path = np.where(s.membership[key])[0]
    if len(in_path) == 0:
        raise ValueError(f"no sampled pair comes from {key}; increase the sample size")
    w_total = s.weight[in_path].sum()
    n1 = path["n_candidates"]

    pending = in_path
    tier_n, tier_e = [], []
    tp = fp = 0.0
    match_cost = 0.0
    for i, (spec, (lo, hi)) in enumerate(zip(plan.matchers, plan.thresholds)):
        mp = prof.matchers.get(_profile_key(spec))
        if mp is None:
            raise KeyError(f"matcher {spec} was not profiled; available: {sorted(prof.matchers)}")
        if mp.is_llm and spec not in ("llm", mp.name):
            warnings.append(f"plan uses {spec} but the profile measured {mp.name}")
        last = i == len(plan.matchers) - 1
        w_in = s.weight[pending].sum()
        n_i = n1 * w_in / w_total
        tier_n.append(n_i)
        c_i = mp.cost_per_pair_usd * ((1 - llm_cached_fraction) if mp.is_llm else 1.0)
        match_cost += n_i * c_i

        sc = mp.scores[pending]
        y = s.labels[pending]
        w = s.weight[pending]
        known = ~np.isnan(sc)
        if not last:
            if not known.all():
                warnings.append(f"{spec} is not the last tier but has unscored sample pairs; treating them as escalated")
            is_match = known & (sc >= hi)
            decided = known & ((sc >= hi) | (sc <= lo))
        else:
            is_match = known & (sc >= 0.5)
            decided = np.ones(len(pending), dtype=bool)
            if not known.all():  # LLM sub-sample: expected verdicts from its TPR / FPR
                tpr = mp.tpr if mp.tpr is not None else 0.9
                fpr = mp.fpr if mp.fpr is not None else 0.1
                unk = ~known
                tp += float((w[unk] * y[unk]).sum() * tpr)
                fp += float((w[unk] * ~y[unk]).sum() * fpr)
        tp += float((w * (is_match & y)).sum())
        fp += float((w * (is_match & ~y)).sum())
        w_esc = w[~decided].sum()
        tier_e.append(float(w_esc / w_in) if w_in else 0.0)
        pending = pending[~decided]

    w_pos = float((s.weight[in_path] * s.labels[in_path]).sum())
    if w_pos == 0:
        warnings.append("no true match in this path's sample; F1 estimate is unreliable")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall_within = tp / w_pos if w_pos else 0.0
    pc = path["pc"]
    recall = recall_within * pc
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    feature_cost = n1 * prof.feature_cost_per_pair_usd if not plan.matchers[0].startswith("llm") else 0.0
    rf_labels = sum(prof.matchers[_profile_key(m)].train_labels for m in plan.matchers if m == "rf")
    label_cost = s.label_cost_usd + rf_labels * settings.usd_per_label
    block_cost = path["block_cost_usd"]
    total = block_cost + feature_cost + match_cost + label_cost
    return PlanEstimate(
        plan=plan.describe(), cost_usd=total, f1=f1, precision=precision, recall=recall,
        recall_within_candidates=recall_within, pair_completeness=pc, n_candidates=n1,
        tier_n=tier_n, tier_escalation=tier_e,
        cost_breakdown={"block": block_cost, "features": feature_cost, "matchers": match_cost,
                        "labels": label_cost, "llm_cached_fraction": llm_cached_fraction},
        warnings=warnings,
    )


def llm_cached_fraction(dataset: str, plan: Plan, model: str | None = None, db=None) -> float:
    """Fraction of the path's profiled candidates whose verdict is already in llm_cache."""
    from .. import db as dbm
    from ..blocking.profile import profile_req_id
    from ..matchers.llm import MongoVerdictCache, cache_key

    db = db if db is not None else dbm.get_db()
    llm = next((m for m in plan.matchers if m.startswith("llm")), None)
    if llm is None:
        return 0.0
    model = model or (llm.partition(":")[2] or settings.llm_models[0])
    req_id = profile_req_id(dataset, plan.access_path, plan.k)
    keys = [cache_key(model, c["a_id"], c["b_id"]) for c in db[dbm.CANDIDATES].find({"req_id": req_id}, {"a_id": 1, "b_id": 1})]
    return MongoVerdictCache(db).count(keys) / len(keys) if keys else 0.0
