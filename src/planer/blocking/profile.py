"""T7: profile every access path (candidate count, pair completeness, reduction ratio, latency,
index size) and write the results with explain() output to run_log (FR-13, FR-16).

Profiled candidates are kept in `candidates` under req_id "profile:<dataset>:<path>:<k>"; the
optimizer's stratified sample is drawn from them (T13/T14).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from pymongo import InsertOne
from pymongo.database import Database

from .. import db as dbm
from ..config import settings
from ..metrics import in_split, load_gold, pair_completeness, reduction_ratio
from ..request import ERRequest, default_request
from . import get_path
from .base import BlockingResult

log = logging.getLogger(__name__)

INDEX_FOR_PATH = {"btree": "bkey_btree", "multikey": "tokens_multikey", "search": "title_norm_textidx"}


def profile_req_id(dataset: str, path: str, k: int | None) -> str:
    return f"profile:{dataset}:{path}:{k if k else '-'}"


def store_candidates(db: Database, req_id, res: BlockingResult, batch: int = 5000) -> int:
    coll = db[dbm.CANDIDATES]
    coll.delete_many({"req_id": req_id})
    ops = [
        InsertOne({"req_id": req_id, "a_id": a, "b_id": b, "path": res.path, "k": res.k,
                   "sim": s, "path_score": ps, "rank": r})
        for (a, b), s, ps, r in zip(res.pairs, res.sim, res.path_score, res.rank)
    ]
    for i in range(0, len(ops), batch):
        coll.bulk_write(ops[i : i + batch], ordered=False)
    return len(ops)


def index_size_bytes(db: Database, source: str, path: str) -> int | None:
    """Size of the B-tree / multikey (or local text) index backing a path; None for mongot indexes."""
    name = INDEX_FOR_PATH.get(path)
    if name is None or (path == "search" and settings.search_backend != "local"):
        return None
    try:
        stats = next(db[source].aggregate([{"$collStats": {"storageStats": {}}}]))
        return stats["storageStats"]["indexSizes"].get(name)
    except Exception:
        return None


def profile_path(dataset: str, path: str, k: int | None = None, db: Database | None = None) -> dict:
    db = db if db is not None else dbm.get_db()
    req = ERRequest.from_dict(default_request(dataset))
    ap = get_path(path, db)
    res = ap.run(req, k if ap.has_k or path == "multikey" else None)

    gold = load_gold(dataset, db)
    gold_est = {p for p in gold if in_split(p[0], "estimation")}
    cand = set(res.pairs)
    n_a = db[req.source_a].estimated_document_count()
    n_b = db[req.source_b].estimated_document_count()
    entry = {
        "kind": "blocking_profile",
        "ts": datetime.now(timezone.utc),
        "req_id": profile_req_id(dataset, path, res.k),
        "dataset": dataset,
        "path": path,
        "k": res.k,
        "backend": ap.backend,
        "n_a": n_a,
        "n_b": n_b,
        "n_candidates": len(res),
        "pair_completeness": pair_completeness(cand, gold),
        # PC measured on the estimation split only: what the optimizer may use without leaking test labels
        "pc_estimation": pair_completeness({p for p in cand if in_split(p[0], "estimation")}, gold_est),
        "reduction_ratio": reduction_ratio(len(res), n_a, n_b),
        "secs": res.secs,
        "index_size_bytes": index_size_bytes(db, req.source_b, path),
        "explain": res.explain,
    }
    store_candidates(db, entry["req_id"], res)
    db[dbm.RUN_LOG].insert_one(dict(entry))
    log.info(
        "%s %-8s k=%-4s candidates=%-7d PC=%.3f RR=%.4f %.1fs",
        dataset, path, res.k, len(res), entry["pair_completeness"], entry["reduction_ratio"], res.secs,
    )
    return entry


def profile_all(dataset: str, ks: tuple[int, ...] = (5, 10, 20), paths: tuple[str, ...] | None = None,
                db: Database | None = None) -> list[dict]:
    """Profile the fixed paths once and the top-k paths once per k."""
    paths = paths or ("btree", "multikey", "search", "vector")
    out = []
    for path in paths:
        for k in (ks if path in ("search", "vector") else (None,)):
            try:
                out.append(profile_path(dataset, path, k, db))
            except Exception as exc:
                log.error("profiling %s/%s k=%s failed: %s", dataset, path, k, exc)
                out.append({"dataset": dataset, "path": path, "k": k, "error": str(exc)})
    return out


def latest_profiles(dataset: str, db: Database | None = None) -> list[dict]:
    """Most recent profile per (path, k)."""
    db = db if db is not None else dbm.get_db()
    seen, out = set(), []
    for e in db[dbm.RUN_LOG].find({"kind": "blocking_profile", "dataset": dataset}, {"explain": 0}).sort("ts", -1):
        key = (e["path"], e.get("k"))
        if key not in seen:
            seen.add(key)
            out.append(e)
    return sorted(out, key=lambda e: (e["path"], e.get("k") or 0))
