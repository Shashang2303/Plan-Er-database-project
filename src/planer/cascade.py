"""Matcher cascade (FR-10, design section 7.2).

Tier i scores the pairs still pending: s >= u_i -> match, s <= l_i -> non-match, otherwise the pair
is escalated to tier i+1. The last tier decides every remaining pair at 0.5.
LLM tiers read and write llm_cache inside OllamaMatcher.score(), so cached pairs cost nothing.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .config import settings
from .matchers.base import Matcher
from .matchers.pairs import PairBatch


@dataclass
class TierStats:
    matcher: str
    lower: float
    upper: float
    n_in: int = 0
    n_match: int = 0
    n_nonmatch: int = 0
    n_escalated: int = 0
    secs: float = 0.0
    cost_usd: float = 0.0
    llm_calls: int = 0
    cache_hits: int = 0

    @property
    def escalation_rate(self) -> float:
        return self.n_escalated / self.n_in if self.n_in else 0.0

    def to_doc(self) -> dict:
        return asdict(self) | {"escalation_rate": self.escalation_rate}


@dataclass
class CascadeResult:
    verdict: np.ndarray          # bool per pair
    score: np.ndarray            # score of the deciding tier
    tier: np.ndarray             # index of the deciding tier
    tiers: list[TierStats] = field(default_factory=list)
    tier_scores: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)  # (pair idx, scores)

    @property
    def cost_usd(self) -> float:
        return sum(t.cost_usd for t in self.tiers)

    @property
    def llm_calls(self) -> int:
        return sum(t.llm_calls for t in self.tiers)

    @property
    def cache_hits(self) -> int:
        return sum(t.cache_hits for t in self.tiers)


def decide(scores: np.ndarray, lower: float, upper: float, last: bool) -> tuple[np.ndarray, np.ndarray]:
    """Returns (is_match, is_decided) masks for one tier."""
    if last:
        return scores >= 0.5, np.ones(len(scores), dtype=bool)
    is_match = scores >= upper
    is_nonmatch = scores <= lower
    return is_match, is_match | is_nonmatch


def run_cascade(batch: PairBatch, matchers: list[Matcher], thresholds: list[tuple[float, float]]) -> CascadeResult:
    n = len(batch)
    verdict = np.zeros(n, dtype=bool)
    final_score = np.zeros(n)
    tier_of = np.full(n, -1)
    result = CascadeResult(verdict, final_score, tier_of)
    pending = np.arange(n)

    for i, (m, (lo, hi)) in enumerate(zip(matchers, thresholds)):
        last = i == len(matchers) - 1
        stats = TierStats(m.name, lo, hi, n_in=len(pending))
        if len(pending) == 0:
            result.tiers.append(stats)
            continue
        sub = batch.subset(pending) if len(pending) < n else batch
        if m.is_llm:
            m.reset_stats()
        t0 = time.perf_counter()
        s = np.asarray(m.score(sub), dtype=float)
        stats.secs = time.perf_counter() - t0
        if m.is_llm:
            stats.llm_calls, stats.cache_hits = m.stats["calls"], m.stats["cache_hits"]
            stats.cost_usd = m.stats["cost_usd"]
        else:
            # tier 1 always sees the full batch, so the shared features are computed (and charged) there
            stats.cost_usd = stats.secs * settings.usd_per_compute_second
        is_match, decided = decide(s, lo, hi, last)
        idx_decided = pending[decided]
        verdict[idx_decided] = is_match[decided]
        final_score[idx_decided] = s[decided]
        tier_of[idx_decided] = i
        stats.n_match = int((is_match & decided).sum())
        stats.n_nonmatch = int((~is_match & decided).sum())
        stats.n_escalated = int((~decided).sum())
        result.tier_scores.append((pending.copy(), s))
        result.tiers.append(stats)
        pending = pending[~decided]
    return result
