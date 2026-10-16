"""Full-text and vector index management for the $search / $vectorSearch access paths (FR-5, FR-6).

Three interchangeable backends behind the same functions (NFR-5):
  mongot  MongoDB Community Search in Docker (default)
  atlas   MongoDB Atlas free tier; source collections are copied there first
  local   no mongot: a classic MongoDB text index (full-text) and exact k-NN in Python (vector)
"""
from __future__ import annotations

import logging
import time

from pymongo import TEXT
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.operations import SearchIndexModel

from . import db as dbm
from .config import EMBED_DIM, settings

log = logging.getLogger(__name__)

TEXT_INDEX = "title_text"
VECTOR_INDEX = "embedding_vector"


def search_db() -> Database:
    """Database that serves $search / $vectorSearch for the configured backend."""
    if settings.search_backend == "atlas":
        return dbm.get_client(settings.atlas_uri)[settings.mongo_db]
    return dbm.get_db()


def _text_model() -> SearchIndexModel:
    return SearchIndexModel(
        name=TEXT_INDEX,
        type="search",
        definition={
            "analyzer": "lucene.standard",
            "mappings": {"dynamic": False, "fields": {"title_norm": {"type": "string"}}},
        },
    )


def _vector_model() -> SearchIndexModel:
    return SearchIndexModel(
        name=VECTOR_INDEX,
        type="vectorSearch",
        definition={
            "fields": [{"type": "vector", "path": "embedding", "numDimensions": EMBED_DIM, "similarity": "cosine"}]
        },
    )


def list_search_indexes(coll: Collection) -> dict[str, dict]:
    try:
        return {ix["name"]: ix for ix in coll.list_search_indexes()}
    except Exception as exc:  # mongot not reachable
        log.debug("list_search_indexes failed: %s", exc)
        return {}


def copy_to_atlas(source: str) -> int:
    """Copy one enriched source collection to Atlas (atlas backend only)."""
    src = dbm.get_db()[source]
    dst = search_db()[source]
    dst.drop()
    docs = list(src.find({}))
    for i in range(0, len(docs), 1000):
        dst.insert_many(docs[i : i + 1000], ordered=False)
    return len(docs)


def create_search_indexes(source: str, wait: bool = True, timeout_s: float = 600) -> dict:
    """Build the text and vector index for one source collection; returns their status."""
    backend = settings.search_backend
    if backend == "local":
        dbm.get_db()[source].create_index([("title_norm", TEXT)], name="title_norm_textidx")
        return {"backend": "local", TEXT_INDEX: "classic $text index", VECTOR_INDEX: "exact k-NN in Python"}

    if backend == "atlas" and search_db()[source].estimated_document_count() == 0:
        log.info("copying %s to Atlas", source)
        copy_to_atlas(source)

    coll = search_db()[source]
    if coll.estimated_document_count() == 0:
        # mongot needs the collection to exist; an empty one is fine but probably a mistake
        log.warning("collection %s is empty; did you run ingest first?", source)
    existing = list_search_indexes(coll)
    to_create = [m for m in (_text_model(), _vector_model()) if m.document["name"] not in existing]
    if to_create:
        coll.create_search_indexes(to_create)
    if wait:
        wait_until_queryable(coll, [TEXT_INDEX, VECTOR_INDEX], timeout_s)
    status = list_search_indexes(coll)
    return {"backend": backend, **{n: status.get(n, {}).get("status", "missing") for n in (TEXT_INDEX, VECTOR_INDEX)}}


def wait_until_queryable(coll: Collection, names: list[str], timeout_s: float = 600) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        status = list_search_indexes(coll)
        ready = [n for n in names if status.get(n, {}).get("queryable")]
        if len(ready) == len(names):
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"search indexes on {coll.name} not queryable after {timeout_s}s: {status}")
        time.sleep(2)


def search_backend_available() -> bool:
    """True when $search / $vectorSearch can be served by the configured backend."""
    if settings.search_backend == "local":
        return True
    try:
        search_db().command("ping")
        list(search_db()["__planer_probe__"].list_search_indexes())
        return True
    except Exception as exc:
        log.debug("search backend unavailable: %s", exc)
        return False
