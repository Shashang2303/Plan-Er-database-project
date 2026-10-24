"""A batch of candidate pairs with their records loaded from MongoDB, plus lazily computed features."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pymongo.database import Database

from ..datasets import DatasetSpec
from .features import FeatureSet, compute_features


@dataclass
class PairBatch:
    a_ids: list[str]
    b_ids: list[str]
    a_docs: list[dict]
    b_docs: list[dict]
    spec: DatasetSpec
    attr_map: dict[str, str]
    _features: FeatureSet | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.a_ids)

    @property
    def pairs(self) -> list[tuple[str, str]]:
        return list(zip(self.a_ids, self.b_ids))

    @classmethod
    def load(cls, db: Database, spec: DatasetSpec, pairs: list[tuple[str, str]],
             attr_map: dict[str, str] | None = None) -> "PairBatch":
        a_src, b_src = spec.a.name, spec.b.name

        def fetch(source: str, ids: set[str]) -> dict[str, dict]:
            out, id_list = {}, list(ids)
            for i in range(0, len(id_list), 20000):
                for d in db[source].find({"_id": {"$in": id_list[i : i + 20000]}}):
                    out[d["_id"]] = d
            return out

        da = fetch(a_src, {a for a, _ in pairs})
        dbb = fetch(b_src, {b for _, b in pairs})
        kept = [(a, b) for a, b in pairs if a in da and b in dbb]  # NFR-4: ids must exist
        return cls(
            a_ids=[a for a, _ in kept], b_ids=[b for _, b in kept],
            a_docs=[da[a] for a, _ in kept], b_docs=[dbb[b] for _, b in kept],
            spec=spec, attr_map=dict(attr_map or spec.attr_map),
        )

    def features(self) -> FeatureSet:
        if self._features is None:
            self._features = compute_features(self.a_docs, self.b_docs, self.attr_map, self.spec)
        return self._features

    def subset(self, idx) -> "PairBatch":
        idx = np.asarray(idx, dtype=int)
        sub = PairBatch(
            a_ids=[self.a_ids[i] for i in idx], b_ids=[self.b_ids[i] for i in idx],
            a_docs=[self.a_docs[i] for i in idx], b_docs=[self.b_docs[i] for i in idx],
            spec=self.spec, attr_map=self.attr_map,
        )
        if self._features is not None:
            sub._features = self._features.subset(idx)
        return sub

    def labels(self, gold: set[tuple[str, str]]) -> np.ndarray:
        return np.fromiter(((a, b) in gold for a, b in zip(self.a_ids, self.b_ids)), dtype=bool, count=len(self))
