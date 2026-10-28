"""T10 / FR-9: random forest over the similarity features.

Trained only on pairs from the `train` split (NFR-7); its labels are charged to C_label.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base import Matcher
from .pairs import PairBatch


class RandomForestMatcher(Matcher):
    name = "rf"
    supervised = True

    def __init__(self, n_estimators: int = 200, seed: int = 7):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=seed, n_jobs=-1,
        )
        self.n_train = 0
        self.n_train_pos = 0
        self._constant: float | None = None

    def fit(self, batch: PairBatch, labels: np.ndarray | None = None) -> "RandomForestMatcher":
        if labels is None:
            raise ValueError("RandomForestMatcher needs labeled training pairs")
        y = np.asarray(labels, dtype=int)
        self.n_train, self.n_train_pos = len(y), int(y.sum())
        if len(np.unique(y)) < 2:  # degenerate training set: fall back to the base rate
            self._constant = float(y.mean()) if len(y) else 0.0
            return self
        self.model.fit(batch.features().filled(-1.0), y)
        self._constant = None
        return self

    def score(self, batch: PairBatch) -> np.ndarray:
        if self._constant is not None:
            return np.full(len(batch), self._constant)
        if not hasattr(self.model, "classes_"):
            raise RuntimeError("RandomForestMatcher.fit() must be called first")
        if len(batch) == 0:
            return np.zeros(0)
        return self.model.predict_proba(batch.features().filled(-1.0))[:, 1]

    def feature_importance(self, names: list[str]) -> dict[str, float]:
        if not hasattr(self.model, "feature_importances_"):
            return {}
        return dict(sorted(zip(names, self.model.feature_importances_.round(4).tolist()), key=lambda x: -x[1]))
