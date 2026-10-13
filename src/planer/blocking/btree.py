"""T3 / FR-3: equality on the blocking key through a B-tree index and $lookup.

The pipeline runs over source A; $lookup probes the bkey_btree index of source B once per record.
This path has no k knob: its recall is fixed by how well the blocking key fits the data.
"""
from __future__ import annotations

from collections.abc import Iterator

from ..request import ERRequest
from .base import AccessPath, Candidate, explain_aggregate

JACCARD_EXPR = {
    "$let": {
        "vars": {
            "i": {"$size": {"$setIntersection": ["$tokens", "$m.tokens"]}},
            "u": {"$size": {"$setUnion": ["$tokens", "$m.tokens"]}},
        },
        "in": {"$cond": [{"$gt": ["$$u", 0]}, {"$divide": ["$$i", "$$u"]}, 0]},
    }
}


class BTreeKeyPath(AccessPath):
    name = "btree"
    has_k = False

    def pipeline(self, req: ERRequest) -> list[dict]:
        return [
            {"$match": {"bkey": {"$nin": ["", None]}}},
            {"$project": {"bkey": 1, "tokens": 1}},
            {
                "$lookup": {
                    "from": req.source_b,
                    "localField": "bkey",
                    "foreignField": "bkey",
                    "pipeline": [{"$project": {"tokens": 1}}],
                    "as": "m",
                }
            },
            {"$unwind": "$m"},
            {"$project": {"_id": 0, "a_id": "$_id", "b_id": "$m._id", "score": JACCARD_EXPR}},
        ]

    def candidates(self, req: ERRequest, k: int | None = None) -> Iterator[Candidate]:
        cursor = self.db[req.source_a].aggregate(self.pipeline(req), allowDiskUse=True)
        for d in cursor:
            yield d["a_id"], d["b_id"], float(d["score"])

    def explain(self, req: ERRequest, k: int | None = None) -> dict:
        return explain_aggregate(self.db, req.source_a, self.pipeline(req))
