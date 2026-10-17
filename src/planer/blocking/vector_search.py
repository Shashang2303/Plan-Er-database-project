"""T6 / FR-6: top-k semantic neighbours with $vectorSearch (HNSW, cosine, served by mongot).

For every A record, one $vectorSearch over B's 384-d embeddings. With SEARCH_BACKEND=local the
path computes exact cosine k-NN in Python instead (same interface, no approximate index).
"""
from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..config import settings
from ..request import ERRequest
from ..search_indexes import VECTOR_INDEX, search_db
from .base import AccessPath, Candidate, explain_aggregate


class VectorSearchPath(AccessPath):
    name = "vector"
    has_k = True

    def __init__(self, db=None, workers: int = 8, candidate_factor: int = 10):
        super().__init__(db)
        self.workers = workers
        self.candidate_factor = candidate_factor
        self.backend = settings.search_backend
        self.sdb = search_db()

    def query_pipeline(self, vec: list[float], k: int) -> list[dict]:
        return [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX,
                    "path": "embedding",
                    "queryVector": vec,
                    "numCandidates": max(100, k * self.candidate_factor),
                    "limit": k,
                }
            },
            {"$project": {"_id": 1, "score": {"$meta": "vectorSearchScore"}}},
        ]

    def _probe(self, req: ERRequest, a_id: str, vec: list[float], k: int) -> list[Candidate]:
        hits = self.sdb[req.source_b].aggregate(self.query_pipeline(vec, k))
        return [(a_id, h["_id"], float(h["score"])) for h in hits]

    def candidates(self, req: ERRequest, k: int | None = None) -> Iterator[Candidate]:
        k = k or req.dataset.default_k
        if self.backend == "local":
            yield from self._exact_knn(req, k)
            return
        records = list(self.db[req.source_a].find({}, {"embedding": 1}))
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for hits in pool.map(lambda d: self._probe(req, d["_id"], d["embedding"], k), records):
                yield from hits

    def _exact_knn(self, req: ERRequest, k: int, chunk: int = 512) -> Iterator[Candidate]:
        a_docs = list(self.db[req.source_a].find({}, {"embedding": 1}))
        b_docs = list(self.db[req.source_b].find({}, {"embedding": 1}))
        A = np.asarray([d["embedding"] for d in a_docs], dtype=np.float32)
        B = np.asarray([d["embedding"] for d in b_docs], dtype=np.float32)
        b_ids = [d["_id"] for d in b_docs]
        k = min(k, len(b_ids))
        for i in range(0, len(A), chunk):
            sims = A[i : i + chunk] @ B.T
            top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            for row, idxs in enumerate(top):
                a_id = a_docs[i + row]["_id"]
                for j in idxs:
                    # same scale as Atlas/mongot vectorSearchScore for cosine: (1 + cos) / 2
                    yield a_id, b_ids[j], float((1 + sims[row, j]) / 2)

    def explain(self, req: ERRequest, k: int | None = None) -> dict | None:
        if self.backend == "local":
            return {"backend": "local", "note": "exact k-NN computed in Python; no MongoDB pipeline to explain"}
        k = k or req.dataset.default_k
        probe = self.db[req.source_a].find_one({}, {"embedding": 1})
        out = explain_aggregate(self.sdb, req.source_b, self.query_pipeline(probe["embedding"], k))
        out["probe_a_id"] = probe["_id"]
        return out
