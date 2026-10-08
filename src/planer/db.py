"""MongoDB access: client, collection names and the B-tree / multikey indexes of the data model.

Collections (D.4.2 section 6):
  <source>      one per data source (e.g. abt, buy): original attributes + title_norm, tokens,
                bkey, embedding[384]. Called source_a / source_b in the design document.
  gold          ground-truth matches of each benchmark (dataset, a_id, b_id)
  er_requests   declarative ER requests and their plans
  candidates    candidate pairs produced by a blocking access path
  decisions     per-pair matcher scores and verdicts
  llm_cache     LLM verdicts keyed by sha256(model, prompt_ver, a_id, b_id)
  matches, clusters   matcher output and entity clusters
  run_log       observed (and later predicted) cost, time, F1, explain() output
"""
from __future__ import annotations

from functools import lru_cache

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.database import Database

from .config import settings

GOLD = "gold"
REQUESTS = "er_requests"
CANDIDATES = "candidates"
DECISIONS = "decisions"
LLM_CACHE = "llm_cache"
MATCHES = "matches"
CLUSTERS = "clusters"
RUN_LOG = "run_log"


@lru_cache(maxsize=4)
def get_client(uri: str | None = None) -> MongoClient:
    return MongoClient(uri or settings.mongo_uri, serverSelectionTimeoutMS=5000, tz_aware=True)


def get_db(name: str | None = None) -> Database:
    return get_client()[name or settings.mongo_db]


def ping() -> bool:
    try:
        get_client().admin.command("ping")
        return True
    except Exception:
        return False


def ensure_system_indexes(db: Database) -> None:
    """Indexes on the PlanER bookkeeping collections (D.4.2, data-model table)."""
    db[GOLD].create_index([("dataset", ASCENDING), ("a_id", ASCENDING), ("b_id", ASCENDING)], unique=True)
    db[CANDIDATES].create_index([("req_id", ASCENDING), ("a_id", ASCENDING), ("b_id", ASCENDING)], unique=True)
    db[DECISIONS].create_index([("pair_id", ASCENDING), ("matcher", ASCENDING)])
    db[DECISIONS].create_index([("req_id", ASCENDING)])
    db[MATCHES].create_index([("req_id", ASCENDING)])
    db[CLUSTERS].create_index([("req_id", ASCENDING)])
    db[CLUSTERS].create_index([("members", ASCENDING)])  # multikey
    db[RUN_LOG].create_index([("req_id", ASCENDING), ("ts", DESCENDING)])
    db[RUN_LOG].create_index([("kind", ASCENDING), ("ts", DESCENDING)])


def ensure_source_indexes(db: Database, source: str) -> None:
    """B-tree on bkey (FR-3) and multikey on tokens (FR-4). Search indexes live in search_indexes.py."""
    coll = db[source]
    coll.create_index([("bkey", ASCENDING)], name="bkey_btree")
    coll.create_index([("tokens", ASCENDING)], name="tokens_multikey")
