"""T11 / FR-10, FR-12: local LLM tier served by Ollama, with the llm_cache verdict cache.

Reproducibility (NFR-7): greedy decoding (temperature 0, fixed seed) and a fixed, versioned prompt.
Never pay twice (NFR-3): every verdict is stored in llm_cache under
    _id = sha256(model | prompt_version | a_id | b_id)
and looked up before any model call, so an identical re-run makes zero new LLM calls.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Protocol

import numpy as np
import requests
from pymongo.database import Database

from .. import db as dbm
from ..config import settings
from .base import Matcher
from .pairs import PairBatch

log = logging.getLogger(__name__)

# Bump PROMPT_VERSION whenever the template changes: it is part of the llm_cache key, so verdicts
# produced by an older prompt are never reused.
# v3 asks a plain yes/no question. Measured on this hardware: qwen2.5:1.5b answered every
# true match "no_match" under the JSON-verdict prompt (v1/v2), but is accurate when asked yes/no.
# The answer is mapped to the design's {"verdict": "match" | "no_match"}.
PROMPT_VERSION = "v3"
PROMPT_TEMPLATE = """Do these two {entity} records refer to the same real-world {entity}? They come from different sources, so ignore differences in formatting, word order, abbreviations, punctuation or hyphens in codes (e.g. "XK200B" and "XK-200B" are the same code), and missing fields. Different model numbers, editions or versions mean different {entity}s.
A: {a}
B: {b}
Answer yes or no."""
MAX_ATTR_CHARS = 200


class VerdictCache(Protocol):
    def get_many(self, keys: list[str]) -> dict[str, dict]: ...
    def put_many(self, docs: list[dict]) -> None: ...


class MongoVerdictCache:
    """llm_cache collection: a materialized view of LLM results, persistent across requests."""

    def __init__(self, db: Database | None = None):
        self.coll = (db if db is not None else dbm.get_db())[dbm.LLM_CACHE]

    def get_many(self, keys: list[str]) -> dict[str, dict]:
        out = {}
        for i in range(0, len(keys), 5000):
            for d in self.coll.find({"_id": {"$in": keys[i : i + 5000]}}):
                out[d["_id"]] = d
        return out

    def put_many(self, docs: list[dict]) -> None:
        from pymongo import ReplaceOne

        if docs:
            self.coll.bulk_write([ReplaceOne({"_id": d["_id"]}, d, upsert=True) for d in docs], ordered=False)

    def count(self, keys: list[str]) -> int:
        n = 0
        for i in range(0, len(keys), 5000):
            n += self.coll.count_documents({"_id": {"$in": keys[i : i + 5000]}})
        return n


class DictVerdictCache:
    """In-memory cache with the same interface (tests, dry runs)."""

    def __init__(self):
        self.store: dict[str, dict] = {}

    def get_many(self, keys):
        return {k: self.store[k] for k in keys if k in self.store}

    def put_many(self, docs):
        for d in docs:
            self.store[d["_id"]] = d


def cache_key(model: str, a_id: str, b_id: str, prompt_version: str = PROMPT_VERSION) -> str:
    return hashlib.sha256(f"{model}|{prompt_version}|{a_id}|{b_id}".encode()).hexdigest()


def render_record(doc: dict, attrs: list[str]) -> str:
    parts = []
    for attr in attrs:
        value = doc.get(attr)
        if value is None or value == "" or (isinstance(value, float) and np.isnan(value)):
            continue
        text = str(value).replace("\n", " ").strip()
        if len(text) > MAX_ATTR_CHARS:
            text = text[:MAX_ATTR_CHARS] + "..."
        parts.append(f"{attr}: {text}")
    return "; ".join(parts) or "(empty record)"


def parse_verdict(text: str) -> bool:
    """True for a match. Accepts a yes/no answer (prompt v3) or a JSON verdict (v1/v2)."""
    low = text.strip().lower()
    if low.startswith("yes"):
        return True
    if low.startswith("no"):
        return False
    try:
        return json.loads(text).get("verdict", "").strip().lower() == "match"
    except (json.JSONDecodeError, AttributeError):
        return "no_match" not in low and "no match" not in low and "match" in low


class OllamaMatcher(Matcher):
    is_llm = True

    def __init__(self, model: str | None = None, cache: VerdictCache | None = None,
                 url: str | None = None, timeout_s: float | None = None):
        self.model = model or settings.llm_models[0]
        self.name = f"llm:{self.model}"
        self.url = url or settings.ollama_url
        self.timeout_s = timeout_s or settings.llm_timeout_s
        self.cache = cache if cache is not None else MongoVerdictCache()
        self.reset_stats()

    def reset_stats(self) -> None:
        self.stats = {"calls": 0, "cache_hits": 0, "tokens": 0, "secs": 0.0, "cost_usd": 0.0, "errors": 0}

    def prompt(self, batch: PairBatch, i: int) -> str:
        entity = "product" if batch.spec.kind == "product" else "publication"
        a_attrs = list(batch.attr_map.keys())
        b_attrs = list(batch.attr_map.values())
        return PROMPT_TEMPLATE.format(
            entity=entity, a=render_record(batch.a_docs[i], a_attrs), b=render_record(batch.b_docs[i], b_attrs)
        )

    def judge(self, prompt: str) -> dict:
        """One model call. Returns verdict, tokens, wall seconds and dollar-equivalent cost."""
        t0 = time.perf_counter()
        resp = requests.post(
            f"{self.url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "seed": 42, "top_k": 1, "num_predict": 3},
                "keep_alive": "15m",
            },
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        body = resp.json()
        secs = time.perf_counter() - t0
        tokens = int(body.get("prompt_eval_count", 0)) + int(body.get("eval_count", 0))
        match = parse_verdict(body.get("response", ""))
        return {
            "verdict": "match" if match else "no_match",
            "score": 1.0 if match else 0.0,
            "tokens": tokens,
            "secs": secs,
            # local model: charge compute time at the configured rate
            "cost_usd": secs * settings.usd_per_compute_second,
        }

    def score(self, batch: PairBatch) -> np.ndarray:
        """Scores for the batch, reading llm_cache first and writing every new verdict back."""
        n = len(batch)
        scores = np.zeros(n)
        keys = [cache_key(self.model, a, b) for a, b in zip(batch.a_ids, batch.b_ids)]
        cached = self.cache.get_many(keys) if n else {}
        new_docs = []
        for i, key in enumerate(keys):
            if key in cached:
                scores[i] = cached[key]["score"]
                self.stats["cache_hits"] += 1
                continue
            try:
                v = self.judge(self.prompt(batch, i))
            except requests.RequestException as exc:
                # an unreachable model must not silently turn into "no match" decisions
                raise RuntimeError(f"Ollama call failed for model {self.model}: {exc}") from exc
            scores[i] = v["score"]
            self.stats["calls"] += 1
            self.stats["tokens"] += v["tokens"]
            self.stats["secs"] += v["secs"]
            self.stats["cost_usd"] += v["cost_usd"]
            new_docs.append({
                "_id": key, "model": self.model, "prompt_ver": PROMPT_VERSION,
                "a_id": batch.a_ids[i], "b_id": batch.b_ids[i], "dataset": batch.spec.key,
                "verdict": v["verdict"], "score": v["score"], "tokens": v["tokens"],
                "secs": v["secs"], "cost_usd": v["cost_usd"], "ts": datetime.now(timezone.utc),
            })
            if len(new_docs) >= 20:  # persist incrementally so an interrupted run keeps its verdicts
                self.cache.put_many(new_docs)
                new_docs = []
        self.cache.put_many(new_docs)
        return scores


# ------------------------------------------------------------------------------- Ollama admin
def ollama_models(url: str | None = None) -> list[str]:
    resp = requests.get(f"{url or settings.ollama_url}/api/tags", timeout=5)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]


def ollama_pull(model: str, url: str | None = None) -> None:
    """Download a model into the Ollama container (streams progress to the log)."""
    with requests.post(f"{url or settings.ollama_url}/api/pull", json={"model": model}, stream=True, timeout=3600) as r:
        r.raise_for_status()
        last = None
        for line in r.iter_lines():
            if not line:
                continue
            status = json.loads(line).get("status")
            if status != last:
                log.info("%s: %s", model, status)
                last = status
