"""Blocking access paths (T3-T6) and their profiling (T7)."""
from __future__ import annotations

from pymongo.database import Database

from .base import AccessPath, BlockingResult
from .btree import BTreeKeyPath
from .multikey import MultikeyTokenPath
from .text_search import FullTextSearchPath
from .vector_search import VectorSearchPath

PATHS: dict[str, type[AccessPath]] = {
    "btree": BTreeKeyPath,
    "multikey": MultikeyTokenPath,
    "search": FullTextSearchPath,
    "vector": VectorSearchPath,
}


def get_path(name: str, db: Database | None = None) -> AccessPath:
    try:
        return PATHS[name](db)
    except KeyError:
        raise KeyError(f"unknown access path {name!r}; choose from {sorted(PATHS)}") from None


__all__ = ["AccessPath", "BlockingResult", "PATHS", "get_path"]
