"""T4 / FR-4: shared rare tokens through the multikey index on `tokens`.

A token is "rare" when its document frequency over A and B together is at most `max_df`
and it occurs on both sides. The pipeline unwinds each A record's rare tokens and $lookups
B through the tokens_multikey index; pairs are ranked by the number of rare tokens they share,
and k (optional) keeps the top-k B records per A record.
"""
from __future__ import annotations

from collections.abc import Iterator

from ..request import ERRequest
from .base import AccessPath, Candidate, explain_aggregate


class MultikeyTokenPath(AccessPath):
    name = "multikey"
    has_k = False   # k is optional here; by default every pair sharing a rare token is kept

    def __init__(self, db=None, max_df: int | None = None, max_df_frac: float | None = None):
        super().__init__(db)
        self.max_df = max_df
        self.max_df_frac = max_df_frac
        self._rare_cache: dict[tuple, list[str]] = {}

    def _cutoff(self, req: ERRequest) -> int:
        if self.max_df:
            return self.max_df
        frac = self.max_df_frac if self.max_df_frac is not None else req.dataset.multikey_max_df_frac
        n = self.db[req.source_a].estimated_document_count() + self.db[req.source_b].estimated_document_count()
        return max(5, int(n * frac))

    def rare_tokens(self, req: ERRequest) -> list[str]:
        """Document frequency computed inside MongoDB with $unwind / $group / $unionWith."""
        cutoff = self._cutoff(req)
        key = (req.source_a, req.source_b, cutoff)
        if key in self._rare_cache:
            return self._rare_cache[key]

        def side(tag: str) -> list[dict]:
            return [
                {"$project": {"tokens": 1}},
                {"$unwind": "$tokens"},
                {"$group": {"_id": "$tokens", tag: {"$sum": 1}}},
            ]

        pipeline = side("na") + [
            {"$unionWith": {"coll": req.source_b, "pipeline": side("nb")}},
            {"$group": {"_id": "$_id", "na": {"$sum": "$na"}, "nb": {"$sum": "$nb"}}},
            {"$match": {"na": {"$gte": 1}, "nb": {"$gte": 1}, "$expr": {"$lte": [{"$add": ["$na", "$nb"]}, cutoff]}}},
            {"$project": {"_id": 1}},
        ]
        if req.source_a == req.source_b:  # deduplication: one collection
            pipeline = side("n") + [{"$match": {"n": {"$gte": 2, "$lte": cutoff}}}, {"$project": {"_id": 1}}]
        rare = [d["_id"] for d in self.db[req.source_a].aggregate(pipeline, allowDiskUse=True)]
        self._rare_cache[key] = rare
        return rare

    def pipeline(self, req: ERRequest, k: int | None = None) -> list[dict]:
        rare = self.rare_tokens(req)
        pipe: list[dict] = [
            {"$project": {"rt": {"$setIntersection": ["$tokens", rare]}}},
            {"$unwind": "$rt"},
            {
                "$lookup": {
                    "from": req.source_b,
                    "localField": "rt",
                    "foreignField": "tokens",
                    "pipeline": [{"$project": {"_id": 1}}],
                    "as": "m",
                }
            },
            {"$unwind": "$m"},
            {"$group": {"_id": {"a": "$_id", "b": "$m._id"}, "shared": {"$sum": 1}}},
        ]
        if k:
            pipe += [
                {"$sort": {"_id.a": 1, "shared": -1}},
                {"$group": {"_id": "$_id.a", "bs": {"$push": {"b": "$_id.b", "shared": "$shared"}}}},
                {"$project": {"bs": {"$slice": ["$bs", k]}}},
                {"$unwind": "$bs"},
                {"$project": {"_id": 0, "a_id": "$_id", "b_id": "$bs.b", "shared": "$bs.shared"}},
            ]
        else:
            pipe.append({"$project": {"_id": 0, "a_id": "$_id.a", "b_id": "$_id.b", "shared": 1}})
        return pipe

    def candidates(self, req: ERRequest, k: int | None = None) -> Iterator[Candidate]:
        for d in self.db[req.source_a].aggregate(self.pipeline(req, k), allowDiskUse=True):
            yield d["a_id"], d["b_id"], float(d["shared"])

    def explain(self, req: ERRequest, k: int | None = None) -> dict:
        return explain_aggregate(self.db, req.source_a, self.pipeline(req, k))
