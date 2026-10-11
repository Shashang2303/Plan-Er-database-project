"""T2: download, normalize and load a benchmark into MongoDB (FR-2).

CSV -> enriched documents (original attributes + title_norm, tokens, bkey, embedding[384])
-> one collection per source, plus the ground truth in `gold`.
"""
from __future__ import annotations

import csv
import io
import logging
import time
import zipfile
from pathlib import Path

import requests
from pymongo import InsertOne
from pymongo.database import Database

from . import db as dbm
from .config import settings
from .datasets import LEIPZIG_BASE_URL, DatasetSpec, SourceSpec, get_dataset
from .embeddings import embed
from .normalize import blocking_key, normalize_text, parse_number, tokenize

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- download
def download(spec: DatasetSpec, data_dir: Path | None = None, force: bool = False) -> Path:
    """Fetch and unzip the benchmark archive into data/raw/<folder>/. Returns that folder."""
    data_dir = Path(data_dir or settings.data_dir)
    target = data_dir / spec.folder
    if target.exists() and not force and (target / spec.gold_file).exists():
        return target
    url = f"{LEIPZIG_BASE_URL}/{spec.zip_name}"
    log.info("downloading %s", url)
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for member in zf.infolist():
            name = Path(member.filename).name  # flatten, and never write outside target
            if not name or member.is_dir():
                continue
            (target / name).write_bytes(zf.read(member))
    return target


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a benchmark CSV. The archives mix UTF-8 (with BOM) and Latin-1 files."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return list(csv.DictReader(io.StringIO(text, newline="")))


# ---------------------------------------------------------------------------- enrich
def enrich(row: dict[str, str], src: SourceSpec, spec: DatasetSpec) -> dict:
    """Original attributes plus the derived fields of the D.4.2 data model."""
    doc: dict = {"_id": str(row["id"]).strip(), "dataset": spec.key}
    for col, value in row.items():
        if col == "id" or col is None:
            continue
        value = (value or "").strip()
        if col in spec.numeric_attrs:
            doc[col] = parse_number(value)
        else:
            doc[col] = value or None
    title_norm = normalize_text(doc.get(src.title_attr))
    doc["title_norm"] = title_norm
    doc["tokens"] = tokenize(title_norm)
    doc["bkey"] = blocking_key(title_norm, spec.bkey_rule)
    return doc


def rekey(key: str, rule: str | None = None, db: Database | None = None) -> dict:
    """Recompute bkey in place for both sources of a dataset (no re-download, no re-embedding)."""
    from pymongo import UpdateOne

    spec = get_dataset(key)
    db = db if db is not None else dbm.get_db()
    rule = rule or spec.bkey_rule
    out = {}
    for src in (spec.a, spec.b):
        ops = [UpdateOne({"_id": d["_id"]}, {"$set": {"bkey": blocking_key(d.get("title_norm") or "", rule)}})
               for d in db[src.name].find({}, {"title_norm": 1})]
        for i in range(0, len(ops), 5000):
            db[src.name].bulk_write(ops[i : i + 5000], ordered=False)
        out[src.name] = len(ops)
    return {"dataset": key, "rule": rule, "updated": out}


def compare_bkey_rules(key: str, db: Database | None = None) -> list[dict]:
    """Candidate count and pair completeness of every blocking-key rule (computed in memory)."""
    from collections import defaultdict

    from .metrics import load_gold
    from .normalize import BKEY_RULES

    spec = get_dataset(key)
    db = db if db is not None else dbm.get_db()
    gold = load_gold(key, db)
    a = {d["_id"]: d.get("title_norm") or "" for d in db[spec.a.name].find({}, {"title_norm": 1})}
    b = {d["_id"]: d.get("title_norm") or "" for d in db[spec.b.name].find({}, {"title_norm": 1})}
    rows = []
    for rule in BKEY_RULES:
        blocks: dict[str, list[str]] = defaultdict(list)
        for bid, t in b.items():
            if k := blocking_key(t, rule):
                blocks[k].append(bid)
        n_cand, hit = 0, 0
        for aid, t in a.items():
            bs = blocks.get(blocking_key(t, rule), []) if t else []
            n_cand += len(bs)
            hit += sum((aid, bid) in gold for bid in bs)
        rows.append({"rule": rule, "candidates": n_cand, "pair_completeness": hit / len(gold) if gold else 0.0,
                     "reduction_ratio": 1 - n_cand / (len(a) * len(b)), "current": rule == spec.bkey_rule})
    return rows


def _load_source(db: Database, src: SourceSpec, spec: DatasetSpec, folder: Path, batch: int) -> int:
    rows = read_csv(folder / src.file)
    docs, seen = [], set()
    for row in rows:
        doc = enrich(row, src, spec)
        if doc["_id"] and doc["_id"] not in seen:  # a few benchmark files repeat an id
            seen.add(doc["_id"])
            docs.append(doc)
    coll = db[src.name]
    coll.drop()
    t0 = time.perf_counter()
    for i in range(0, len(docs), batch):
        chunk = docs[i : i + batch]
        vecs = embed([d["title_norm"] for d in chunk])
        for d, v in zip(chunk, vecs):
            d["embedding"] = v.tolist()
        coll.bulk_write([InsertOne(d) for d in chunk], ordered=False)
        log.info("%s: %d/%d records", src.name, min(i + batch, len(docs)), len(docs))
    dbm.ensure_source_indexes(db, src.name)
    log.info("%s loaded in %.1fs", src.name, time.perf_counter() - t0)
    return len(docs)


def _load_gold(db: Database, spec: DatasetSpec, folder: Path) -> int:
    rows = read_csv(folder / spec.gold_file)
    a_ids = set(db[spec.a.name].distinct("_id"))
    b_ids = set(db[spec.b.name].distinct("_id"))
    col_a, col_b = spec.gold_cols
    pairs = {(str(r[col_a]).strip(), str(r[col_b]).strip()) for r in rows}
    # NFR-4: only keep ground-truth pairs whose records exist in the source collections
    kept = [p for p in pairs if p[0] in a_ids and p[1] in b_ids]
    if len(kept) < len(pairs):
        log.warning("%s: %d gold pairs reference missing records and were skipped", spec.key, len(pairs) - len(kept))
    db[dbm.GOLD].delete_many({"dataset": spec.key})
    if kept:
        db[dbm.GOLD].insert_many([{"dataset": spec.key, "a_id": a, "b_id": b} for a, b in kept], ordered=False)
    return len(kept)


def ingest_dataset(key: str, db: Database | None = None, batch: int = 512) -> dict:
    """Download (if needed) and load one benchmark. Returns record and gold counts."""
    spec = get_dataset(key)
    db = db if db is not None else dbm.get_db()
    dbm.ensure_system_indexes(db)
    folder = download(spec)
    n_a = _load_source(db, spec.a, spec, folder, batch)
    n_b = _load_source(db, spec.b, spec, folder, batch)
    n_gold = _load_gold(db, spec, folder)
    summary = {"dataset": key, spec.a.name: n_a, spec.b.name: n_b, "gold_pairs": n_gold, "possible_pairs": n_a * n_b}
    log.info("ingested %s", summary)
    return summary
