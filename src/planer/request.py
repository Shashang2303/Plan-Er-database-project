"""Declarative ER request (FR-1) and physical plan P = (b, M, tau, kappa)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo.database import Database

from . import db as dbm
from .datasets import DatasetSpec, dataset_for_sources

ACCESS_PATHS = ("btree", "multikey", "search", "vector")
TOPK_PATHS = ("search", "vector")
# Fixed cost order of the matcher tiers (D.2.1 section V). "llm" means the first configured local model.
MATCHER_ORDER = ("rules", "fs", "rf", "llm")


@dataclass
class Objective:
    type: str            # "theta" (min cost s.t. F1 >= value) or "budget" (max F1 s.t. cost <= value)
    value: float

    def __post_init__(self) -> None:
        if self.type not in {"theta", "budget"}:
            raise ValueError("objective.type must be 'theta' or 'budget'")
        if self.type == "theta" and not 0 < self.value <= 1:
            raise ValueError("theta must be in (0, 1]")
        if self.type == "budget" and self.value < 0:
            raise ValueError("budget must be >= 0")


@dataclass
class ERRequest:
    sources: list[str]
    attr_map: dict[str, str]
    objective: Objective
    sample_size: int = 300
    _id: Any = None

    @classmethod
    def from_dict(cls, d: dict) -> "ERRequest":
        obj = d.get("objective") or {"type": "theta", "value": 0.85}
        return cls(
            sources=list(d["sources"]),
            attr_map=dict(d.get("attr_map") or {}),
            objective=obj if isinstance(obj, Objective) else Objective(obj["type"], float(obj["value"])),
            sample_size=int(d.get("sample_size", 300)),
            _id=d.get("_id"),
        )

    @property
    def dataset(self) -> DatasetSpec:
        return dataset_for_sources(self.sources)

    @property
    def source_a(self) -> str:
        return self.sources[0]

    @property
    def source_b(self) -> str:
        return self.sources[1] if len(self.sources) > 1 else self.sources[0]

    def to_doc(self) -> dict:
        doc = {
            "sources": self.sources,
            "attr_map": self.attr_map,
            "objective": asdict(self.objective),
            "sample_size": self.sample_size,
        }
        return doc


@dataclass
class Plan:
    """Physical plan P = (b, M, tau, kappa); k is the top-k parameter of the search / vector paths."""

    access_path: str
    matchers: list[str]
    thresholds: list[tuple[float, float]] = field(default_factory=list)
    k: int | None = None
    clustering: str = "connected_components"

    def __post_init__(self) -> None:
        if self.access_path not in ACCESS_PATHS:
            raise ValueError(f"access_path must be one of {ACCESS_PATHS}")
        if not self.matchers:
            raise ValueError("a plan needs at least one matcher")
        order = [MATCHER_ORDER.index(m.split(":")[0]) for m in self.matchers]
        if order != sorted(order) or len(set(order)) != len(order):
            raise ValueError(f"matchers must be distinct and in cost order {MATCHER_ORDER}")
        if not self.thresholds:
            self.thresholds = [(0.3, 0.7)] * (len(self.matchers) - 1) + [(0.5, 0.5)]
        if len(self.thresholds) == len(self.matchers) - 1:
            self.thresholds = list(self.thresholds) + [(0.5, 0.5)]  # last tier decides everything
        if len(self.thresholds) != len(self.matchers):
            raise ValueError("need one (l, u) threshold pair per matcher")
        self.thresholds = [(float(lo), float(hi)) for lo, hi in self.thresholds]
        for lo, hi in self.thresholds:
            if not 0 <= lo <= hi <= 1:
                raise ValueError(f"invalid thresholds ({lo}, {hi}); need 0 <= l <= u <= 1")
        if self.access_path in TOPK_PATHS and not self.k:
            self.k = 10

    def to_doc(self) -> dict:
        return {
            "access_path": self.access_path,
            "k": self.k,
            "matchers": self.matchers,
            "thresholds": [list(t) for t in self.thresholds],
            "clustering": self.clustering,
        }

    @classmethod
    def from_doc(cls, d: dict) -> "Plan":
        return cls(
            access_path=d["access_path"],
            matchers=list(d["matchers"]),
            thresholds=[tuple(t) for t in d.get("thresholds", [])],
            k=d.get("k"),
            clustering=d.get("clustering", "connected_components"),
        )

    def describe(self) -> str:
        path = self.access_path + (f"(k={self.k})" if self.k else "")
        tiers = " -> ".join(
            f"{m}[{lo:.2f},{hi:.2f}]" if i < len(self.matchers) - 1 else f"{m}[final]"
            for i, (m, (lo, hi)) in enumerate(zip(self.matchers, self.thresholds))
        )
        return f"{path} | {tiers} | {self.clustering}"


def submit(request: dict | ERRequest, db: Database | None = None) -> ObjectId:
    """Store a declarative request in er_requests and return its id (planer.submit)."""
    db = db if db is not None else dbm.get_db()
    req = request if isinstance(request, ERRequest) else ERRequest.from_dict(request)
    req.dataset  # validates that the sources are a registered dataset pair
    doc = req.to_doc() | {"status": "submitted", "created": datetime.now(timezone.utc), "plan": None}
    return db[dbm.REQUESTS].insert_one(doc).inserted_id


def load_request(req_id, db: Database | None = None) -> ERRequest:
    db = db if db is not None else dbm.get_db()
    doc = db[dbm.REQUESTS].find_one({"_id": ObjectId(req_id) if isinstance(req_id, str) else req_id})
    if doc is None:
        raise KeyError(f"no er_request with _id {req_id}")
    return ERRequest.from_dict(doc)


def default_request(dataset_key: str, objective: dict | None = None) -> dict:
    from .datasets import get_dataset

    spec = get_dataset(dataset_key)
    return {
        "sources": list(spec.sources),
        "attr_map": dict(spec.attr_map),
        "objective": objective or {"type": "theta", "value": 0.85},
        "sample_size": 300,
    }
