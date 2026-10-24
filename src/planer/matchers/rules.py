"""T8 / FR-7: string-similarity rule matcher (RapidFuzz).

A fixed weighted combination of title similarities, model-code overlap, embedding similarity
and the other mapped attributes. Missing features drop out and their weight is redistributed.
"""
from __future__ import annotations

import numpy as np

from .base import Matcher, weighted_mean
from .pairs import PairBatch


class RuleMatcher(Matcher):
    name = "rules"

    def score(self, batch: PairBatch) -> np.ndarray:
        F = batch.features()
        title = batch.spec.a.title_attr
        cols: list[tuple[np.ndarray, float]] = [
            (F.col(f"{title}_tset"), 0.25),
            (F.col(f"{title}_ratio"), 0.10),
            (F.col(f"{title}_jac"), 0.15),
            (F.col("title_partial"), 0.05),
            (F.col("code_jac"), 0.15),
            (F.col("emb_cos"), 0.15),
        ]
        others = [a for a in batch.attr_map if a != title]
        for a in others:
            name = f"{a}_num" if F.has(f"{a}_num") else f"{a}_tset"
            cols.append((F.col(name), 0.15 / max(1, len(others))))
        return weighted_mean(cols)
