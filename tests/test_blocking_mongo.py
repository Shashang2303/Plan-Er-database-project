"""Blocking paths against a live MongoDB (skipped when MongoDB is not running)."""
import pytest

from planer.blocking import BTreeKeyPath, MultikeyTokenPath
from planer.metrics import pair_completeness
from planer.request import ERRequest

from .conftest import GOLD


def _req():
    return ERRequest.from_dict({"sources": ["toy_a", "toy_b"], "attr_map": {"name": "name"}})


@pytest.fixture(autouse=True)
def _toy_dataset(monkeypatch):
    from planer import request

    from .conftest import SPEC

    monkeypatch.setattr(request, "dataset_for_sources", lambda sources: SPEC)


def test_btree_path_uses_index(mongo_db):
    res = BTreeKeyPath(mongo_db).run(_req())
    assert ("a1", "b1") in res.pairs                     # both keys are "apple|iph"
    assert res.explain["summary"]["indexes_used"], res.explain
    assert "bkey_btree" in res.explain["summary"]["indexes_used"]


def test_multikey_rare_tokens(mongo_db):
    path = MultikeyTokenPath(mongo_db, max_df=3)
    rare = set(path.rare_tokens(_req()))
    assert "ezxs88w" in rare and "am53bk" in rare
    res = path.run(_req())
    assert pair_completeness(res.pairs, GOLD) >= 0.5
    assert "tokens_multikey" in res.explain["summary"]["indexes_used"]
