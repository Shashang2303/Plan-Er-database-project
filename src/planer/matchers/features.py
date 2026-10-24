"""Similarity features per mapped attribute (RapidFuzz), shared by rules, Fellegi-Sunter and RF.

Feature names:
  <attr>_tset, <attr>_ratio, <attr>_jac   string attributes (token-set ratio, ratio, token Jaccard)
  <attr>_num                               numeric attributes (year equality, relative price gap)
  title_partial                            partial ratio of the title attribute
  code_jac                                 Jaccard of "code" tokens (contain a digit, length >= 3)
  emb_cos                                  cosine of the 384-d title embeddings, rescaled to [0, 1]
Missing values are NaN.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz

from ..datasets import DatasetSpec
from ..normalize import normalize_text, tokenize

MAX_CHARS = 300  # long descriptions are truncated before fuzzy comparison


@dataclass
class FeatureSet:
    names: list[str]
    X: np.ndarray  # shape (n_pairs, n_features), NaN = missing

    def col(self, name: str) -> np.ndarray:
        return self.X[:, self.names.index(name)]

    def has(self, name: str) -> bool:
        return name in self.names

    def subset(self, idx) -> "FeatureSet":
        return FeatureSet(self.names, self.X[np.asarray(idx, dtype=int)])

    def filled(self, value: float = -1.0) -> np.ndarray:
        return np.where(np.isnan(self.X), value, self.X)


def _codes(tokens: list[str]) -> set[str]:
    return {t for t in tokens if len(t) >= 3 and any(c.isdigit() for c in t)}


def _jac(a: set, b: set) -> float:
    u = len(a | b)
    return len(a & b) / u if u else np.nan


def _num_sim(attr: str, x, y) -> float:
    if x is None or y is None or (isinstance(x, float) and np.isnan(x)) or (isinstance(y, float) and np.isnan(y)):
        return np.nan
    x, y = float(x), float(y)
    if attr == "year":
        d = abs(x - y)
        return 1.0 if d == 0 else (0.5 if d == 1 else 0.0)
    if x <= 0 or y <= 0:  # price 0 means "unknown" in Amazon-Google
        return np.nan
    return 1.0 - abs(x - y) / max(x, y)


def feature_names(attr_map: dict[str, str], spec: DatasetSpec) -> list[str]:
    names = []
    for a_attr in attr_map:
        if a_attr in spec.numeric_attrs:
            names.append(f"{a_attr}_num")
        else:
            names += [f"{a_attr}_tset", f"{a_attr}_ratio", f"{a_attr}_jac"]
    return names + ["title_partial", "code_jac", "emb_cos"]


def compute_features(a_docs: list[dict], b_docs: list[dict], attr_map: dict[str, str], spec: DatasetSpec) -> FeatureSet:
    names = feature_names(attr_map, spec)
    X = np.full((len(a_docs), len(names)), np.nan, dtype=np.float64)
    title_a, title_b = spec.a.title_attr, spec.b.title_attr
    norm_cache: dict[tuple[int, str], str] = {}

    def norm(doc: dict, attr: str) -> str:
        key = (id(doc), attr)
        if key not in norm_cache:
            norm_cache[key] = normalize_text(doc.get(attr))[:MAX_CHARS]
        return norm_cache[key]

    for i, (da, db_) in enumerate(zip(a_docs, b_docs)):
        col = 0
        for a_attr, b_attr in attr_map.items():
            if a_attr in spec.numeric_attrs:
                X[i, col] = _num_sim(a_attr, da.get(a_attr), db_.get(b_attr))
                col += 1
                continue
            sa, sb = norm(da, a_attr), norm(db_, b_attr)
            if sa and sb:
                X[i, col] = fuzz.token_set_ratio(sa, sb) / 100.0
                X[i, col + 1] = fuzz.ratio(sa, sb) / 100.0
                X[i, col + 2] = _jac(set(tokenize(sa)), set(tokenize(sb)))
            col += 3
        ta, tb = da.get("title_norm") or norm(da, title_a), db_.get("title_norm") or norm(db_, title_b)
        if ta and tb:
            X[i, col] = fuzz.partial_ratio(ta, tb) / 100.0
        ca, cb = _codes(da.get("tokens") or []), _codes(db_.get("tokens") or [])
        if ca and cb:
            X[i, col + 1] = _jac(ca, cb)
        ea, eb = da.get("embedding"), db_.get("embedding")
        if ea is not None and eb is not None:
            cos = float(np.dot(ea, eb))
            X[i, col + 2] = min(1.0, max(0.0, (cos - 0.2) / 0.8))
    return FeatureSet(names, X)
