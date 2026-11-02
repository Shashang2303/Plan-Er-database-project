"""T13: per-matcher profiling on the labeled sample (FR-13).

For every matcher: per-pair cost c_i (USD), scores on the sample, and from them precision, recall
and escalation rate for any threshold pair (l, u). The LLM is profiled on a smaller stratified
sub-sample to respect NFR-1 (estimation < 5 minutes); its true/false positive rates fill in pairs
it did not score. The full profile is stored in run_log (kind "optimizer_profile") so that plan
estimation (estimate.py) and later plan enumeration (T15) never re-run the matchers.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from pymongo.database import Database

from .. import db as dbm
from ..blocking.profile import latest_profiles
from ..config import settings
from ..datasets import get_dataset
from ..matchers import PairBatch, make_matcher
from ..matchers.llm import OllamaMatcher, cache_key
from ..metrics import in_split, load_gold
from .sampling import LabeledSample, path_key, stratified_sample

log = logging.getLogger(__name__)

FS_FIT_PAIRS = 5000      # unsupervised EM is fitted on a random slice of the candidate population


@dataclass
class MatcherProfile:
    name: str
    is_llm: bool
    scores: np.ndarray                 # NaN where the matcher did not score the pair
    cost_per_pair_usd: float
    secs_per_pair: float
    fit_secs: float = 0.0
    train_labels: int = 0              # labels consumed by a supervised matcher (-> C_label)
    tpr: float | None = None           # LLM only: P(match verdict | true match)
    fpr: float | None = None           # LLM only: P(match verdict | non-match)

    def to_doc(self) -> dict:
        return {
            "name": self.name, "is_llm": self.is_llm,
            "scores": [None if np.isnan(s) else float(s) for s in self.scores],
            "cost_per_pair_usd": self.cost_per_pair_usd, "secs_per_pair": self.secs_per_pair,
            "fit_secs": self.fit_secs, "train_labels": self.train_labels, "tpr": self.tpr, "fpr": self.fpr,
        }

    @classmethod
    def from_doc(cls, d: dict) -> "MatcherProfile":
        return cls(
            name=d["name"], is_llm=d["is_llm"],
            scores=np.array([np.nan if s is None else s for s in d["scores"]], dtype=float),
            cost_per_pair_usd=d["cost_per_pair_usd"], secs_per_pair=d["secs_per_pair"],
            fit_secs=d.get("fit_secs", 0.0), train_labels=d.get("train_labels", 0),
            tpr=d.get("tpr"), fpr=d.get("fpr"),
        )


@dataclass
class OptimizerProfile:
    dataset: str
    sample: LabeledSample
    matchers: dict[str, MatcherProfile]
    feature_cost_per_pair_usd: float
    paths: dict[str, dict]             # path_key -> {n_candidates, pc, secs, block_cost_usd}
    secs: float = 0.0
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_doc(self) -> dict:
        return {
            "kind": "optimizer_profile", "ts": self.ts, "dataset": self.dataset, "secs": self.secs,
            "sample": self.sample.to_doc(), "label_cost_usd": self.sample.label_cost_usd,
            "matchers": {k: v.to_doc() for k, v in self.matchers.items()},
            "feature_cost_per_pair_usd": self.feature_cost_per_pair_usd, "paths": self.paths,
        }

    @classmethod
    def from_doc(cls, d: dict) -> "OptimizerProfile":
        return cls(
            dataset=d["dataset"], sample=LabeledSample.from_doc(d["sample"]),
            matchers={k: MatcherProfile.from_doc(v) for k, v in d["matchers"].items()},
            feature_cost_per_pair_usd=d["feature_cost_per_pair_usd"], paths=d["paths"],
            secs=d.get("secs", 0.0), ts=d.get("ts"),
        )


def _sub_sample(sample: LabeledSample, n: int, seed: int = 11) -> np.ndarray:
    """Indices of a stratified sub-sample (same strata, proportional to sample counts)."""
    if n >= len(sample):
        return np.arange(len(sample))
    rng = np.random.default_rng(seed)
    out = []
    strata = np.unique(sample.stratum)
    per = max(1, n // len(strata))
    for h in strata:
        idx = np.where(sample.stratum == h)[0]
        out.extend(rng.choice(idx, size=min(per, len(idx)), replace=False).tolist())
    return np.array(sorted(out[:n]))


def _population_pairs(dataset: str, db: Database, split: str) -> list[tuple[str, str]]:
    pairs = set()
    for prof in latest_profiles(dataset, db):
        for c in db[dbm.CANDIDATES].find({"req_id": prof["req_id"]}, {"a_id": 1, "b_id": 1}):
            if in_split(c["a_id"], split):
                pairs.add((c["a_id"], c["b_id"]))
    return sorted(pairs)


def profile_matchers(
    dataset: str,
    sample_size: int = 300,
    matchers: tuple[str, ...] = ("rules", "fs", "rf", "llm"),
    llm_sample: int = 60,
    seed: int = 7,
    db: Database | None = None,
) -> OptimizerProfile:
    db = db if db is not None else dbm.get_db()
    t_start = time.perf_counter()
    spec = get_dataset(dataset)
    rate = settings.usd_per_compute_second

    sample = stratified_sample(dataset, sample_size, seed, db)
    batch = PairBatch.load(db, spec, sample.pairs)
    n = len(batch)
    log.info("sample: %d pairs, %d matches (strata sizes %s)", n, sample.labels.sum(), sample.population)

    t0 = time.perf_counter()
    batch.features()
    feature_cost = (time.perf_counter() - t0) / max(1, n) * rate

    profiles: dict[str, MatcherProfile] = {}
    for name in matchers:
        m = make_matcher(name)
        fit_secs, train_labels = 0.0, 0
        if m.supervised:
            t0 = time.perf_counter()
            train_pairs = _population_pairs(dataset, db, "train")
            train = PairBatch.load(db, spec, train_pairs)
            y = train.labels(load_gold(dataset, db, split="train"))
            m.fit(train, y)
            fit_secs, train_labels = time.perf_counter() - t0, len(y)
        elif name == "fs":
            t0 = time.perf_counter()
            pop = _population_pairs(dataset, db, "estimation")
            rng = np.random.default_rng(seed)
            if len(pop) > FS_FIT_PAIRS:
                pop = [pop[i] for i in rng.choice(len(pop), FS_FIT_PAIRS, replace=False)]
            m.fit(PairBatch.load(db, spec, pop))
            fit_secs = time.perf_counter() - t0

        scores = np.full(n, np.nan)
        if m.is_llm:
            assert isinstance(m, OllamaMatcher)
            idx = _sub_sample(sample, llm_sample, seed)
            m.reset_stats()
            t0 = time.perf_counter()
            scores[idx] = m.score(batch.subset(idx))
            elapsed = time.perf_counter() - t0
            # per-pair cost of a cache MISS, from what each verdict originally cost
            keys = [cache_key(m.model, *sample.pairs[i]) for i in idx]
            docs = list(db[dbm.LLM_CACHE].find({"_id": {"$in": keys}}, {"cost_usd": 1, "secs": 1}))
            c = float(np.mean([d["cost_usd"] for d in docs])) if docs else elapsed / max(1, len(idx)) * rate
            s = float(np.mean([d["secs"] for d in docs])) if docs else elapsed / max(1, len(idx))
            lab = sample.labels[idx]
            pred = scores[idx] >= 0.5
            tpr = float(pred[lab].mean()) if lab.any() else None
            fpr = float(pred[~lab].mean()) if (~lab).any() else None
            profiles[m.name if name != "llm" else "llm"] = MatcherProfile(
                name=m.name, is_llm=True, scores=scores, cost_per_pair_usd=c, secs_per_pair=s,
                tpr=tpr, fpr=fpr,
            )
            log.info("%s: %d new calls, %d cache hits, $%.6f/pair, TPR=%s FPR=%s",
                     m.name, m.stats["calls"], m.stats["cache_hits"], c, tpr, fpr)
            continue

        t0 = time.perf_counter()
        scores[:] = m.score(batch)
        secs = (time.perf_counter() - t0) / max(1, n)
        profiles[name] = MatcherProfile(
            name=name, is_llm=False, scores=scores, cost_per_pair_usd=secs * rate,
            secs_per_pair=secs, fit_secs=fit_secs, train_labels=train_labels,
        )

    paths = {}
    for prof in latest_profiles(dataset, db):
        if "error" in prof:
            continue
        paths[path_key(prof["path"], prof.get("k"))] = {
            "path": prof["path"], "k": prof.get("k"), "n_candidates": prof["n_candidates"],
            "pc": prof.get("pc_estimation", prof["pair_completeness"]),
            "secs": prof["secs"], "block_cost_usd": prof["secs"] * rate,
        }

    result = OptimizerProfile(
        dataset=dataset, sample=sample, matchers=profiles, feature_cost_per_pair_usd=feature_cost,
        paths=paths, secs=time.perf_counter() - t_start,
    )
    db[dbm.RUN_LOG].insert_one(result.to_doc() | {"summary": summarize(result)})
    log.info("optimizer profile for %s built in %.1fs", dataset, result.secs)
    return result


def summarize(prof: OptimizerProfile, lower: float = 0.3, upper: float = 0.7) -> dict:
    """Weighted precision / recall / F1 at 0.5 and escalation rate at (l, u), per matcher."""
    s = prof.sample
    out = {}
    for name, mp in prof.matchers.items():
        ok = ~np.isnan(mp.scores)
        w, y, sc = s.weight[ok], s.labels[ok], mp.scores[ok]
        pred = sc >= 0.5
        tp = float((w * (pred & y)).sum())
        fp = float((w * (pred & ~y)).sum())
        fn = float((w * (~pred & y)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        esc = float((w * ((sc > lower) & (sc < upper))).sum() / w.sum()) if w.sum() else 0.0
        out[name] = {
            "n_scored": int(ok.sum()), "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            f"escalation@({lower},{upper})": esc, "cost_per_pair_usd": mp.cost_per_pair_usd,
            "secs_per_pair": mp.secs_per_pair,
        }
    return out


def latest_profile(dataset: str, db: Database | None = None) -> OptimizerProfile:
    db = db if db is not None else dbm.get_db()
    doc = db[dbm.RUN_LOG].find_one({"kind": "optimizer_profile", "dataset": dataset}, sort=[("ts", -1)])
    if doc is None:
        raise RuntimeError(f"no optimizer profile for {dataset}; run `planer profile-matchers {dataset}` first")
    return OptimizerProfile.from_doc(doc)
