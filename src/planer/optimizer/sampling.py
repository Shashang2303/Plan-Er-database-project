"""T14 (part 1): stratified labeled sample (FR-13).

Population: the union of the candidates produced by every profiled access path, restricted to the
`estimation` split. Strata are bins of token-Jaccard similarity, so rare high-similarity pairs
(where most matches live) are over-represented. Every sampled pair carries the inverse-probability
weight N_h / n_h, so weighted sums over the sample estimate totals over the population.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pymongo.database import Database

from .. import db as dbm
from ..blocking.profile import latest_profiles
from ..config import settings
from ..metrics import in_split, load_gold

STRATA_EDGES = (0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 1.000001)


def path_key(path: str, k: int | None) -> str:
    return f"{path}:{k if k else '-'}"


@dataclass
class LabeledSample:
    dataset: str
    pairs: list[tuple[str, str]]
    sim: np.ndarray
    stratum: np.ndarray
    weight: np.ndarray
    labels: np.ndarray
    membership: dict[str, np.ndarray]          # path_key -> bool mask over the sample
    population: dict[int, int] = field(default_factory=dict)   # stratum -> N_h
    path_population: dict[str, int] = field(default_factory=dict)  # path_key -> |candidates in estimation split|

    @property
    def label_cost_usd(self) -> float:
        return len(self.pairs) * settings.usd_per_label

    def __len__(self) -> int:
        return len(self.pairs)

    def to_doc(self) -> dict:
        return {
            "dataset": self.dataset,
            "pairs": [list(p) for p in self.pairs],
            "sim": self.sim.tolist(),
            "stratum": self.stratum.tolist(),
            "weight": self.weight.tolist(),
            "labels": self.labels.astype(int).tolist(),
            "membership": {k: v.astype(int).tolist() for k, v in self.membership.items()},
            "population": {str(k): v for k, v in self.population.items()},
            "path_population": self.path_population,
        }

    @classmethod
    def from_doc(cls, d: dict) -> "LabeledSample":
        return cls(
            dataset=d["dataset"], pairs=[tuple(p) for p in d["pairs"]],
            sim=np.asarray(d["sim"], dtype=float), stratum=np.asarray(d["stratum"], dtype=int),
            weight=np.asarray(d["weight"], dtype=float), labels=np.asarray(d["labels"], dtype=bool),
            membership={k: np.asarray(v, dtype=bool) for k, v in d["membership"].items()},
            population={int(k): v for k, v in d.get("population", {}).items()},
            path_population=d.get("path_population", {}),
        )


def allocate(sizes: dict[int, int], n: int) -> dict[int, int]:
    """Equal allocation across strata, capped by stratum size; leftovers go to strata with room."""
    alloc = {h: 0 for h in sizes}
    remaining = n
    open_strata = [h for h in sizes if sizes[h] > 0]
    while remaining > 0 and open_strata:
        share = max(1, remaining // len(open_strata))
        for h in list(open_strata):
            take = min(share, sizes[h] - alloc[h], remaining)
            alloc[h] += take
            remaining -= take
            if alloc[h] >= sizes[h]:
                open_strata.remove(h)
            if remaining == 0:
                break
    return alloc


def stratified_sample(dataset: str, size: int = 300, seed: int = 7, db: Database | None = None) -> LabeledSample:
    db = db if db is not None else dbm.get_db()
    profiles = [p for p in latest_profiles(dataset, db) if "error" not in p]
    if not profiles:
        raise RuntimeError(f"no blocking profiles for {dataset}; run `planer profile-blocking {dataset}` first")

    # population: union of profiled candidates in the estimation split
    pop: dict[tuple[str, str], float] = {}
    members: dict[str, set[tuple[str, str]]] = {}
    for prof in profiles:
        key = path_key(prof["path"], prof.get("k"))
        pairs = set()
        for c in db[dbm.CANDIDATES].find({"req_id": prof["req_id"]}, {"a_id": 1, "b_id": 1, "sim": 1}):
            if in_split(c["a_id"], "estimation"):
                p = (c["a_id"], c["b_id"])
                pairs.add(p)
                pop[p] = c["sim"]
        members[key] = pairs
    if not pop:
        raise RuntimeError(f"no candidates of {dataset} fall in the estimation split")

    pop_pairs = list(pop)
    pop_sim = np.array([pop[p] for p in pop_pairs])
    pop_stratum = np.digitize(pop_sim, STRATA_EDGES[1:-1])
    sizes = {h: int((pop_stratum == h).sum()) for h in range(len(STRATA_EDGES) - 1)}
    alloc = allocate(sizes, min(size, len(pop_pairs)))

    rng = np.random.default_rng(seed)
    chosen, weights = [], []
    for h, n_h in alloc.items():
        if n_h == 0:
            continue
        idx = np.where(pop_stratum == h)[0]
        pick = rng.choice(idx, size=n_h, replace=False)
        chosen.extend(pick.tolist())
        weights.extend([sizes[h] / n_h] * n_h)

    gold = load_gold(dataset, db, split="estimation")  # the "labeler": benchmark ground truth
    pairs = [pop_pairs[i] for i in chosen]
    return LabeledSample(
        dataset=dataset,
        pairs=pairs,
        sim=pop_sim[chosen],
        stratum=pop_stratum[chosen],
        weight=np.asarray(weights),
        labels=np.array([p in gold for p in pairs], dtype=bool),
        membership={k: np.array([p in s for p in pairs], dtype=bool) for k, s in members.items()},
        population=sizes,
        path_population={k: len(s) for k, s in members.items()},
    )
