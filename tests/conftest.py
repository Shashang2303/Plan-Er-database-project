"""Shared fixtures: a tiny product dataset that needs no download, and an isolated test database."""
from __future__ import annotations

import zlib

import numpy as np
import pytest

from planer.datasets import DatasetSpec, SourceSpec
from planer.normalize import blocking_key, normalize_text, tokenize

SPEC = DatasetSpec(
    key="toy", zip_name="toy.zip",
    a=SourceSpec("toy_a", "a.csv", "name"), b=SourceSpec("toy_b", "b.csv", "name"),
    gold_file="gold.csv", gold_cols=("a", "b"),
    attr_map={"name": "name", "price": "price"}, kind="product", numeric_attrs=("price",),
)

A_ROWS = [
    ("a1", "Apple iPhone 15 Pro 128GB Black", 999.0),
    ("a2", "Sony Turntable PSLX350H", 149.0),
    ("a3", "Linksys EtherFast EZXS88W Switch", 39.0),
    ("a4", "Bose Acoustimass 5 Speaker AM53BK", 399.0),
    ("a5", "Canon PowerShot SD1100 Camera", 199.0),
]
B_ROWS = [
    ("b1", "Apple iPhone15 Pro (Blk) 128 GB", 989.0),
    ("b2", "Sony PS-LX350H Belt Drive Turntable", 139.0),
    ("b3", "Linksys EZXS88W 8-Port Switch", 35.0),
    ("b4", "Bose Acoustimass 5 Series III AM53BK", 389.0),
    ("b5", "Samsung Galaxy S24 Ultra 256GB", 1199.0),
    ("b6", "Canon EOS Rebel T7 DSLR", 479.0),
]
GOLD = {("a1", "b1"), ("a2", "b2"), ("a3", "b3"), ("a4", "b4")}


def make_doc(rec_id: str, name: str, price: float) -> dict:
    norm = normalize_text(name)
    toks = tokenize(norm)
    rng = np.random.default_rng(zlib.crc32(toks[0].encode()))  # same brand -> same toy embedding
    vec = rng.normal(size=384)
    vec /= np.linalg.norm(vec)
    return {"_id": rec_id, "dataset": "toy", "name": name, "price": price, "title_norm": norm,
            "tokens": toks, "bkey": blocking_key(norm, "product"), "embedding": vec.tolist()}


@pytest.fixture
def toy_docs():
    return [make_doc(*r) for r in A_ROWS], [make_doc(*r) for r in B_ROWS]


@pytest.fixture
def toy_batch(toy_docs):
    from planer.matchers.pairs import PairBatch

    a_docs, b_docs = toy_docs
    pairs = [(a, b) for a in a_docs for b in b_docs]
    return PairBatch(
        a_ids=[a["_id"] for a, _ in pairs], b_ids=[b["_id"] for _, b in pairs],
        a_docs=[a for a, _ in pairs], b_docs=[b for _, b in pairs], spec=SPEC, attr_map=dict(SPEC.attr_map),
    )


@pytest.fixture
def mongo_db(toy_docs):
    """Isolated database with the toy collections; skips the test when MongoDB is not running."""
    from planer import db as dbm

    if not dbm.ping():
        pytest.skip("MongoDB is not running")
    name = "planer_test"
    db = dbm.get_client()[name]
    dbm.get_client().drop_database(name)
    a_docs, b_docs = toy_docs
    db["toy_a"].insert_many(a_docs)
    db["toy_b"].insert_many(b_docs)
    dbm.ensure_source_indexes(db, "toy_a")
    dbm.ensure_source_indexes(db, "toy_b")
    dbm.ensure_system_indexes(db)
    db[dbm.GOLD].insert_many([{"dataset": "toy", "a_id": a, "b_id": b} for a, b in GOLD])
    yield db
    dbm.get_client().drop_database(name)
