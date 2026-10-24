"""Matcher interface (NFR-9): every tier implements fit() and score(batch) -> scores in [0, 1]."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .pairs import PairBatch


class Matcher(ABC):
    name: str = ""
    is_llm: bool = False
    supervised: bool = False   # needs labeled training pairs (their cost goes to C_label)

    def fit(self, batch: PairBatch, labels: np.ndarray | None = None) -> "Matcher":
        """Prepare the matcher. Unsupervised matchers ignore labels."""
        return self

    @abstractmethod
    def score(self, batch: PairBatch) -> np.ndarray:
        """Calibrated match scores in [0, 1], one per pair."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


def weighted_mean(columns: list[tuple[np.ndarray, float]]) -> np.ndarray:
    """Row-wise weighted mean that ignores NaN entries (their weight is redistributed)."""
    vals = np.column_stack([c for c, _ in columns])
    w = np.array([w for _, w in columns], dtype=float)
    mask = ~np.isnan(vals)
    weights = mask * w
    total = weights.sum(axis=1)
    out = np.where(total > 0, np.nansum(np.where(mask, vals, 0) * w, axis=1) / np.where(total > 0, total, 1), 0.0)
    return np.clip(out, 0.0, 1.0)
