"""T5 / FR-5: top-k lexical neighbours with $search (BM25, served by mongot).

For every A record, one $search query over B with the record's normalized title; k is part of the
physical plan. With SEARCH_BACKEND=local the same path uses MongoDB's classic $text index instead.
"""
from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

from ..config import settings
from ..request import ERRequest
from ..search_indexes import TEXT_INDEX, search_db
from .base import AccessPath, Candidate, explain_aggregate

MAX_QUERY_CHARS = 512


class FullTextSearchPath(AccessPath):
    name = "search"
    has_k = True

    def __init__(self, db=None, workers: int = 8):
        super().__init__(db)
        self.workers = workers
        self.backend = settings.search_backend
        self.sdb = search_db()

    def query_pipeline(self, text: str, k: int) -> list[dict]:
        text = text[:MAX_QUERY_CHARS] or "-"
        if self.backend == "local":
            return [
                {"$match": {"$text": {"$search": text}}},
                {"$sort": {"score": {"$meta": "textScore"}}},
                {"$limit": k},
                {"$project": {"_id": 1, "score": {"$meta": "textScore"}}},
            ]
        return [
            {"$search": {"index": TEXT_INDEX, "text": {"query": text, "path": "title_norm"}}},
            {"$limit": k},
            {"$project": {"_id": 1, "score": {"$meta": "searchScore"}}},
        ]

    def _probe(self, req: ERRequest, a_id: str, text: str, k: int) -> list[Candidate]:
        if not text.strip():
            return []
        hits = list(self.sdb[req.source_b].aggregate(self.query_pipeline(text, k)))
        return [(a_id, h["_id"], float(h["score"])) for h in hits]

    def candidates(self, req: ERRequest, k: int | None = None) -> Iterator[Candidate]:
        k = k or req.dataset.default_k
        records = list(self.db[req.source_a].find({}, {"title_norm": 1}))
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for hits in pool.map(lambda d: self._probe(req, d["_id"], d.get("title_norm") or "", k), records):
                yield from hits

    def explain(self, req: ERRequest, k: int | None = None) -> dict:
        k = k or req.dataset.default_k
        probe = self.db[req.source_a].find_one({"title_norm": {"$nin": ["", None]}}, {"title_norm": 1})
        out = explain_aggregate(self.sdb, req.source_b, self.query_pipeline(probe["title_norm"], k))
        out["probe_a_id"] = probe["_id"]
        return out
