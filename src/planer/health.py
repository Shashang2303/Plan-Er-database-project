"""Status of the stack (demo D1): MongoDB, mongot search indexes, Ollama models, loaded data."""
from __future__ import annotations

import requests

from . import db as dbm
from .config import settings
from .datasets import DATASETS


def status() -> dict:
    out: dict = {"mongo_uri": settings.mongo_uri.split("@")[-1], "search_backend": settings.search_backend}
    out["mongodb"] = dbm.ping()
    if out["mongodb"]:
        db = dbm.get_db()
        out["mongodb_version"] = dbm.get_client().server_info().get("version")
        try:
            out["replica_set"] = dbm.get_client().admin.command("hello").get("setName")
        except Exception:
            out["replica_set"] = None
        loaded = {}
        names = set(db.list_collection_names())
        for key, spec in DATASETS.items():
            if spec.a.name in names and spec.b.name in names:
                loaded[key] = {
                    spec.a.name: db[spec.a.name].estimated_document_count(),
                    spec.b.name: db[spec.b.name].estimated_document_count(),
                    "gold": db[dbm.GOLD].count_documents({"dataset": key}),
                }
                if settings.search_backend != "local":
                    from .search_indexes import list_search_indexes, search_db

                    loaded[key]["search_indexes"] = {
                        src: {n: ix.get("status") for n, ix in list_search_indexes(search_db()[src]).items()}
                        for src in spec.sources
                    }
        out["datasets"] = loaded
        out["llm_cache_entries"] = db[dbm.LLM_CACHE].estimated_document_count()
    if settings.search_backend == "mongot":
        try:
            out["mongot_health"] = requests.get("http://localhost:8080/health", timeout=3).text.strip()
        except requests.RequestException:
            out["mongot_health"] = "unreachable (fine inside the app container)"
    try:
        from .matchers.llm import ollama_models

        out["ollama_models"] = ollama_models()
        out["ollama_missing"] = [m for m in settings.llm_models if m not in out["ollama_models"]]
    except Exception as exc:
        out["ollama_models"] = f"unreachable: {exc.__class__.__name__}"
    return out
