"""Common interface of the blocking access paths (NFR-5, NFR-9).

Every path implements `candidates(req, k)` and yields (a_id, b_id, path_score). `run()` adds timing,
token-Jaccard similarity (a uniform `sim` across paths, used for stratified sampling) and the
explain("executionStats") output of the path's aggregation pipeline (FR-16).
"""
from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field

from pymongo.database import Database

from .. import db as dbm
from ..request import ERRequest

log = logging.getLogger(__name__)

Candidate = tuple[str, str, float]


@dataclass
class BlockingResult:
    path: str
    k: int | None
    pairs: list[tuple[str, str]]
    path_score: list[float]
    sim: list[float]
    rank: list[int]
    secs: float
    explain: dict | None
    backend: str
    notes: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.pairs)


class AccessPath(ABC):
    name: str = ""
    has_k: bool = False          # True for top-k paths: k is part of the physical plan
    backend: str = "mongod"

    def __init__(self, db: Database | None = None):
        self.db = db if db is not None else dbm.get_db()

    @abstractmethod
    def candidates(self, req: ERRequest, k: int | None = None) -> Iterator[Candidate]:
        """Yield (a_id, b_id, path_score). Path scores are only comparable within one path."""

    @abstractmethod
    def explain(self, req: ERRequest, k: int | None = None) -> dict | None:
        """explain('executionStats') of the path's pipeline (for one probe record if per-record)."""

    # ----------------------------------------------------------------------------------------
    def run(self, req: ERRequest, k: int | None = None, with_explain: bool = True) -> BlockingResult:
        t0 = time.perf_counter()
        best: dict[tuple[str, str], float] = {}
        dedup = req.source_a == req.source_b
        for a_id, b_id, score in self.candidates(req, k):
            if dedup:
                if a_id == b_id:
                    continue
                a_id, b_id = min(a_id, b_id), max(a_id, b_id)
            key = (a_id, b_id)
            if score > best.get(key, float("-inf")):
                best[key] = score
        secs = time.perf_counter() - t0

        # rank of each pair among the candidates of its a-record (1 = best path score)
        by_a: dict[str, list[tuple[float, str]]] = {}
        for (a, b), s in best.items():
            by_a.setdefault(a, []).append((s, b))
        rank: dict[tuple[str, str], int] = {}
        for a, lst in by_a.items():
            for r, (_, b) in enumerate(sorted(lst, key=lambda x: -x[0]), start=1):
                rank[(a, b)] = r

        pairs = list(best)
        sims = token_jaccard(self.db, req, pairs)
        explain = None
        if with_explain:
            try:
                explain = compact_explain(self.explain(req, k))
            except Exception as exc:  # explain must never break a run
                explain = {"error": str(exc)}
        return BlockingResult(
            path=self.name, k=k if self.has_k else None, pairs=pairs,
            path_score=[best[p] for p in pairs], sim=sims, rank=[rank[p] for p in pairs],
            secs=secs, explain=explain, backend=self.backend,
        )


def _tokens(db: Database, source: str, ids: set[str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    id_list = list(ids)
    for i in range(0, len(id_list), 20000):
        for d in db[source].find({"_id": {"$in": id_list[i : i + 20000]}}, {"tokens": 1}):
            out[d["_id"]] = set(d.get("tokens") or [])
    return out


def token_jaccard(db: Database, req: ERRequest, pairs: list[tuple[str, str]]) -> list[float]:
    if not pairs:
        return []
    ta = _tokens(db, req.source_a, {a for a, _ in pairs})
    tb = _tokens(db, req.source_b, {b for _, b in pairs})
    out = []
    for a, b in pairs:
        sa, sb = ta.get(a, set()), tb.get(b, set())
        union = len(sa | sb)
        out.append(len(sa & sb) / union if union else 0.0)
    return out


def explain_aggregate(db: Database, collection: str, pipeline: list[dict]) -> dict:
    return db.command(
        "explain", {"aggregate": collection, "pipeline": pipeline, "cursor": {}}, verbosity="executionStats"
    )


_KEEP = {
    "stage", "inputStage", "inputStages", "indexName", "keyPattern", "isMultiKey", "direction",
    "nReturned", "executionTimeMillis", "executionTimeMillisEstimate", "totalKeysExamined",
    "totalDocsExamined", "keysExamined", "docsExamined", "indexesUsed", "collectionScans",
    "indexedLoopJoin", "strategy", "winningPlan", "executionStats", "queryPlanner", "stages",
    "$cursor", "$lookup", "$search", "$vectorSearch", "$_internalSearchMongotRemote",
    "$_internalSearchIdLookup", "$vectorSearch", "planSummary", "explain", "queryPlan",
    "totalDocsExamined", "namespace", "from", "localField", "foreignField", "limit", "index",
    "numCandidates", "mongotQuery", "$match", "$unwind", "$group", "$project", "$sort", "$limit",
}


def compact_explain(explain: dict | None, max_bytes: int = 200_000) -> dict | None:
    """Keep the parts of explain() that show index use and work done, and cap the size."""
    if explain is None:
        return None

    def walk(node, depth=0):
        if depth > 12:
            return "..."
        if isinstance(node, dict):
            return {k: walk(v, depth + 1) for k, v in node.items() if k in _KEEP or k.startswith("$")}
        if isinstance(node, list):
            return [walk(v, depth + 1) for v in node[:20]]
        if isinstance(node, (str, int, float, bool)) or node is None:
            return node if not isinstance(node, str) or len(node) < 500 else node[:500] + "..."
        return str(node)

    compact = walk(explain)
    text = json.dumps(compact, default=str)
    if len(text) > max_bytes:
        return {"truncated": True, "summary": summarize_explain(explain)}
    compact["summary"] = summarize_explain(explain)
    return compact


def summarize_explain(explain: dict) -> dict:
    """Which indexes were used, and whether any stage scanned a whole collection."""
    used: set[str] = set()
    found = {"COLLSCAN": False, "IXSCAN": False, "lookup_collscans": 0}

    def walk(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("indexName", "index") and isinstance(value, str):
                    used.add(value)                     # find stages; $search / $vectorSearch index
                elif key == "indexesUsed" and isinstance(value, list):
                    used.update(v for v in value if isinstance(v, str))   # $lookup stages
                elif key == "collectionScans" and isinstance(value, int):
                    found["lookup_collscans"] += value
                elif key == "stage" and value in ("COLLSCAN", "IXSCAN"):
                    found[value] = True
                walk(value)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(explain)
    text = json.dumps(explain, default=str)
    return {
        "indexes_used": sorted(used),
        "has_COLLSCAN": found["COLLSCAN"],          # the outer scan of source A is expected
        "has_IXSCAN": found["IXSCAN"] or bool(used),
        "lookup_collection_scans": found["lookup_collscans"],
        "uses_mongot": "$_internalSearchMongotRemote" in text or "mongotQuery" in text,
    }
