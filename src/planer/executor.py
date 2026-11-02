"""T12: execute a physical plan end to end (planer.execute).

    blocking (aggregation pipeline) -> candidates
    -> matcher cascade (llm_cache for LLM tiers) -> decisions, matches
    -> connected components -> clusters
    -> observed cost, time, LLM calls, pairwise and cluster F1, label cost -> run_log (FR-19)

Quality is reported on the held-out `test` split only (NFR-7): the random forest trains on the
`train` split and the optimizer samples from the `estimation` split.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
from bson import ObjectId
from pymongo import InsertOne
from pymongo.database import Database

from . import db as dbm
from .blocking import get_path
from .blocking.profile import store_candidates
from .cascade import run_cascade
from .clustering import connected_components, count_transitivity_violations
from .config import settings
from .matchers import PairBatch, make_matcher
from .matchers.llm import cache_key
from .metrics import in_split, load_gold, pair_completeness, pairs_from_clusters, prf
from .request import ERRequest, Plan, default_request, load_request, submit

log = logging.getLogger(__name__)


@dataclass
class RunReport:
    req_id: str
    dataset: str
    plan: dict
    n_candidates: int
    n_matches: int
    n_clusters: int
    cost_usd: float
    block_cost_usd: float
    match_cost_usd: float
    label_cost_usd: float
    secs: float
    llm_calls: int
    cache_hits: int
    pc_test: float
    precision: float
    recall: float
    f1_pair: float
    f1_cluster: float
    transitivity_violations: int
    tiers: list[dict] = field(default_factory=list)
    predicted: dict | None = None

    @property
    def cache_hit_rate(self) -> float:
        total = self.llm_calls + self.cache_hits
        return self.cache_hits / total if total else 1.0

    def as_dict(self) -> dict:
        return asdict(self) | {"cache_hit_rate": self.cache_hit_rate}


def _fit_matchers(matchers, batch: PairBatch, gold: set) -> tuple[dict[str, float], int]:
    """Fit every tier. Returns per-matcher fit seconds and the number of training labels used."""
    fit_secs: dict[str, float] = {}
    n_labels = 0
    for m in matchers:
        t0 = time.perf_counter()
        if m.supervised:
            train_idx = [i for i, a in enumerate(batch.a_ids) if in_split(a, "train")]
            train = batch.subset(train_idx)
            labels = train.labels(gold)
            m.fit(train, labels)
            n_labels += len(train_idx)
            log.info("%s trained on %d pairs (%d matches) from the train split", m.name, len(train_idx), labels.sum())
        elif not m.is_llm:
            m.fit(batch)
        fit_secs[m.name] = time.perf_counter() - t0
    return fit_secs, n_labels


def execute(req_id, plan: Plan, predicted: dict | None = None, db: Database | None = None) -> RunReport:
    db = db if db is not None else dbm.get_db()
    req_oid = ObjectId(req_id) if isinstance(req_id, str) else req_id
    req: ERRequest = load_request(req_oid, db)
    spec = req.dataset
    db[dbm.REQUESTS].update_one({"_id": req_oid}, {"$set": {"status": "running", "plan": plan.to_doc()}})
    t_start = time.perf_counter()

    # 1. blocking -------------------------------------------------------------------------
    access_path = get_path(plan.access_path, db)
    block = access_path.run(req, plan.k)
    store_candidates(db, req_oid, block)
    cand_ids = {(c["a_id"], c["b_id"]): c["_id"] for c in db[dbm.CANDIDATES].find({"req_id": req_oid}, {"a_id": 1, "b_id": 1})}
    log.info("blocking %s produced %d candidates in %.1fs", plan.access_path, len(block), block.secs)

    # 2. matcher cascade ------------------------------------------------------------------
    gold = load_gold(spec.key, db)
    batch = PairBatch.load(db, spec, block.pairs, req.attr_map)
    matchers = [make_matcher(m) for m in plan.matchers]
    fit_secs, n_train_labels = _fit_matchers(matchers, batch, gold)
    result = run_cascade(batch, matchers, plan.thresholds)
    for t in result.tiers:  # fitting is part of a tier's cost
        t.secs += fit_secs.get(t.matcher, 0.0)
        if not t.matcher.startswith("llm"):
            t.cost_usd = t.secs * settings.usd_per_compute_second

    # 3. decisions and matches ------------------------------------------------------------
    ops = []
    for tier_idx, (idx, scores) in enumerate(result.tier_scores):
        m = matchers[tier_idx]
        lo, hi = plan.thresholds[tier_idx]
        last = tier_idx == len(matchers) - 1
        per_pair = result.tiers[tier_idx].cost_usd / max(1, len(idx))
        for i, s in zip(idx, scores):
            a, b = batch.a_ids[i], batch.b_ids[i]
            if last:
                verdict = "match" if s >= 0.5 else "non_match"
            else:
                verdict = "match" if s >= hi else ("non_match" if s <= lo else "escalate")
            doc = {"req_id": req_oid, "pair_id": cand_ids.get((a, b)), "a_id": a, "b_id": b,
                   "matcher": m.name, "score": float(s), "verdict": verdict, "cost": per_pair}
            if m.is_llm:
                doc["cache_key"] = cache_key(m.model, a, b)
            ops.append(InsertOne(doc))
    db[dbm.DECISIONS].delete_many({"req_id": req_oid})
    for i in range(0, len(ops), 5000):
        db[dbm.DECISIONS].bulk_write(ops[i : i + 5000], ordered=False)

    match_idx = np.where(result.verdict)[0]
    matches = [(batch.a_ids[i], batch.b_ids[i]) for i in match_idx]
    db[dbm.MATCHES].delete_many({"req_id": req_oid})
    if matches:
        db[dbm.MATCHES].insert_many([
            {"req_id": req_oid, "a_id": batch.a_ids[i], "b_id": batch.b_ids[i],
             "conf": float(result.score[i]), "tier": matchers[result.tier[i]].name}
            for i in match_idx
        ])

    # 4. clustering -----------------------------------------------------------------------
    clusters = connected_components(matches, req.source_a, req.source_b)
    db[dbm.CLUSTERS].delete_many({"req_id": req_oid})
    if clusters:
        db[dbm.CLUSTERS].insert_many([
            {"req_id": req_oid, "cluster_id": n, "members": members, "size": len(members), "violations_repaired": 0}
            for n, members in enumerate(clusters)
        ])
    nonmatches = {(batch.a_ids[i], batch.b_ids[i]) for i in np.where(~result.verdict)[0]}
    violations = count_transitivity_violations(clusters, nonmatches, req.source_a, req.source_b)
    secs = time.perf_counter() - t_start

    # 5. observed quality on the test split ------------------------------------------------
    gold_test = {p for p in gold if in_split(p[0], "test")}
    pred_test = {p for p in matches if in_split(p[0], "test")}
    pair_q = prf(pred_test, gold_test)
    cluster_pairs = {p for p in pairs_from_clusters(clusters, req.source_a, req.source_b) if in_split(p[0], "test")}
    cluster_q = prf(cluster_pairs, gold_test)
    pc_test = pair_completeness({p for p in block.pairs if in_split(p[0], "test")}, gold_test)

    block_cost = block.secs * settings.usd_per_compute_second
    match_cost = sum(t.cost_usd for t in result.tiers)
    label_cost = n_train_labels * settings.usd_per_label
    report = RunReport(
        req_id=str(req_oid), dataset=spec.key, plan=plan.to_doc(),
        n_candidates=len(block), n_matches=len(matches), n_clusters=len(clusters),
        cost_usd=block_cost + match_cost + label_cost, block_cost_usd=block_cost,
        match_cost_usd=match_cost, label_cost_usd=label_cost, secs=secs,
        llm_calls=result.llm_calls, cache_hits=result.cache_hits, pc_test=pc_test,
        precision=pair_q["precision"], recall=pair_q["recall"], f1_pair=pair_q["f1"],
        f1_cluster=cluster_q["f1"], transitivity_violations=violations,
        tiers=[t.to_doc() for t in result.tiers], predicted=predicted,
    )

    db[dbm.RUN_LOG].insert_one({
        "kind": "execution", "ts": datetime.now(timezone.utc), "req_id": req_oid, "dataset": spec.key,
        "plan": plan.to_doc(), "plan_text": plan.describe(), "pred": predicted,
        "obs": {
            "cost_usd": report.cost_usd, "block_cost_usd": block_cost, "match_cost_usd": match_cost,
            "label_cost": label_cost, "secs": secs, "llm_calls": report.llm_calls,
            "cache_hits": report.cache_hits, "cache_hit_rate": report.cache_hit_rate,
            "f1_pair": report.f1_pair, "precision": report.precision, "recall": report.recall,
            "f1_cluster": report.f1_cluster, "pc_test": pc_test, "n_candidates": len(block),
            "n_matches": len(matches), "n_clusters": len(clusters), "transitivity_violations": violations,
        },
        "tiers": report.tiers, "explain": block.explain, "search_backend": settings.search_backend,
    })
    db[dbm.REQUESTS].update_one({"_id": req_oid}, {"$set": {"status": "done", "finished": datetime.now(timezone.utc)}})
    log.info("run %s: F1=%.3f cost=$%.4f LLM calls=%d (cache hits %d) in %.1fs",
             req_oid, report.f1_pair, report.cost_usd, report.llm_calls, report.cache_hits, secs)
    return report


def run_fixed_plan(dataset: str, plan: Plan, objective: dict | None = None, predicted: dict | None = None,
                   db: Database | None = None) -> RunReport:
    """Submit the default request for a dataset and execute a hand-chosen plan (the T12 prototype)."""
    db = db if db is not None else dbm.get_db()
    req_id = submit(default_request(dataset, objective), db)
    return execute(req_id, plan, predicted=predicted, db=db)
