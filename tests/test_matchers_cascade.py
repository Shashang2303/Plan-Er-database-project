import numpy as np

from planer.cascade import run_cascade
from planer.matchers import FellegiSunterMatcher, RandomForestMatcher, RuleMatcher
from planer.matchers.base import Matcher
from planer.matchers.llm import DictVerdictCache, OllamaMatcher, cache_key, parse_verdict

from .conftest import GOLD


def test_features_and_rules_rank_true_matches_first(toy_batch):
    scores = RuleMatcher().score(toy_batch)
    assert scores.shape == (len(toy_batch),)
    assert ((0 <= scores) & (scores <= 1)).all()
    labels = toy_batch.labels(GOLD)
    assert scores[labels].min() > np.median(scores[~labels])


def test_fellegi_sunter_em_separates_matches(toy_batch):
    fs = FellegiSunterMatcher().fit(toy_batch)
    s = fs.score(toy_batch)
    labels = toy_batch.labels(GOLD)
    assert s[labels].mean() > s[~labels].mean() + 0.3
    params = fs.parameters()
    assert 0 < params["p"] < 1 and len(params["m"]) == len(params["fields"])


def test_random_forest_learns(toy_batch):
    labels = toy_batch.labels(GOLD)
    rf = RandomForestMatcher(n_estimators=50).fit(toy_batch, labels)
    s = rf.score(toy_batch)
    assert s[labels].mean() > 0.5 > s[~labels].mean()


class Fixed(Matcher):
    def __init__(self, name, scores, is_llm=False):
        self.name, self._s, self.is_llm = name, np.asarray(scores, dtype=float), is_llm
        self.seen = []

    def score(self, batch):
        idx = [int(a) for a in batch.a_ids]
        self.seen.append(idx)
        return self._s[idx]


def test_cascade_escalates_only_uncertain_pairs():
    from planer.matchers.pairs import PairBatch

    from .conftest import SPEC

    n = 6
    batch = PairBatch([str(i) for i in range(n)], [str(i) for i in range(n)], [{}] * n, [{}] * n, SPEC, {})
    t1 = Fixed("t1", [0.95, 0.05, 0.5, 0.5, 0.9, 0.1])
    t2 = Fixed("t2", [0, 0, 0.9, 0.2, 0, 0])
    res = run_cascade(batch, [t1, t2], [(0.2, 0.8), (0.5, 0.5)])
    assert t2.seen == [[2, 3]]                       # only the two uncertain pairs escalate
    assert res.verdict.tolist() == [True, False, True, False, True, False]
    assert res.tier.tolist() == [0, 0, 1, 1, 0, 0]
    assert res.tiers[0].n_escalated == 2 and res.tiers[0].escalation_rate == 2 / 6


def test_llm_cache_means_never_pay_twice(toy_batch, monkeypatch):
    cache = DictVerdictCache()
    llm = OllamaMatcher(model="fake-model", cache=cache)
    calls = []

    def fake_judge(prompt):
        calls.append(prompt)
        return {"verdict": "match", "score": 1.0, "tokens": 10, "secs": 0.01, "cost_usd": 1e-6}

    monkeypatch.setattr(llm, "judge", fake_judge)
    sub = toy_batch.subset(range(5))
    llm.score(sub)
    assert llm.stats["calls"] == 5 and len(cache.store) == 5
    llm.reset_stats()
    llm.score(sub)                                    # identical re-run (NFR-3, AC-5)
    assert llm.stats["calls"] == 0 and llm.stats["cache_hits"] == 5
    assert len(calls) == 5
    assert cache_key("m", "a", "b") != cache_key("m", "b", "a")


def test_parse_verdict():
    assert parse_verdict("Yes.") is True and parse_verdict(" no\n") is False
    assert parse_verdict('{"verdict": "match"}') is True
    assert parse_verdict('{"verdict": "no_match"}') is False
    assert parse_verdict("garbage no_match") is False
