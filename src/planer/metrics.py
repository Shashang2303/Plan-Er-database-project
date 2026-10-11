"""Quality metrics and the disjoint train / estimation / test split (NFR-7)."""
from __future__ import annotations

import hashlib
from collections.abc import Iterable

from pymongo.database import Database

from . import db as dbm

Pair = tuple[str, str]

# Records of source A are assigned to one split by a stable hash of their id, so the three splits
# never share a record (and therefore never share a labeled pair).
SPLITS = {"train": (0, 30), "estimation": (30, 50), "test": (50, 100)}


def split_of(a_id: str) -> str:
    bucket = int(hashlib.sha1(str(a_id).encode()).hexdigest()[:8], 16) % 100
    for name, (lo, hi) in SPLITS.items():
        if lo <= bucket < hi:
            return name
    raise AssertionError("unreachable")


def in_split(a_id: str, split: str | None) -> bool:
    return split is None or split_of(a_id) == split


def load_gold(dataset: str, db: Database | None = None, split: str | None = None) -> set[Pair]:
    db = db if db is not None else dbm.get_db()
    pairs = {(g["a_id"], g["b_id"]) for g in db[dbm.GOLD].find({"dataset": dataset}, {"a_id": 1, "b_id": 1})}
    return {p for p in pairs if in_split(p[0], split)}


def pair_completeness(candidates: Iterable[Pair], gold: set[Pair]) -> float:
    """Fraction of true matches that survive blocking (recall of the access path)."""
    if not gold:
        return 0.0
    return len(gold.intersection(candidates)) / len(gold)


def reduction_ratio(n_candidates: int, n_a: int, n_b: int) -> float:
    """1 - |candidates| / |A x B|."""
    total = n_a * n_b
    return 1.0 - n_candidates / total if total else 0.0


def prf(predicted: Iterable[Pair], gold: set[Pair]) -> dict[str, float]:
    predicted = set(predicted)
    tp = len(predicted & gold)
    p = tp / len(predicted) if predicted else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": len(predicted) - tp, "fn": len(gold) - tp}


def pairs_from_clusters(clusters: Iterable[list[str]], source_a: str, source_b: str) -> set[Pair]:
    """All (a, b) pairs implied by clusters whose members are "<source>/<id>" strings."""
    out: set[Pair] = set()
    pa, pb = source_a + "/", source_b + "/"
    for members in clusters:
        a_ids = [m[len(pa):] for m in members if m.startswith(pa)]
        b_ids = [m[len(pb):] for m in members if m.startswith(pb)]
        out.update((a, b) for a in a_ids for b in b_ids)
    return out
