from planer.clustering import connected_components, count_transitivity_violations
from planer.metrics import SPLITS, pair_completeness, pairs_from_clusters, prf, reduction_ratio, split_of
from planer.normalize import blocking_key, normalize_text, parse_number, tokenize
from planer.request import Plan


def test_normalize_and_tokens():
    norm = normalize_text("iPhone15 Pro (Blk) 128 GB — Café")
    assert norm == "iphone15 pro blk 128 gb cafe"
    assert tokenize(norm) == ["iphone15", "pro", "blk", "128", "gb", "cafe"]
    assert tokenize("the a of x data") == ["data"]  # stopwords and 1-char tokens dropped


def test_blocking_key_matches_design_example():
    assert blocking_key("apple iphone 15 pro", "product") == "apple|iph"
    assert blocking_key("towards a cooperative transaction model", "bibliographic") == "coop|tran"  # "towards" is a stopword
    assert blocking_key("", "product") == ""


def test_parse_number():
    assert parse_number("$1,299.00") == 1299.0
    assert parse_number("") is None
    assert parse_number("n/a") is None


def test_splits_are_disjoint_and_stable():
    ids = [f"id{i}" for i in range(3000)]
    seen = {split_of(i) for i in ids}
    assert seen == set(SPLITS)
    assert all(split_of(i) == split_of(i) for i in ids[:50])
    frac_test = sum(split_of(i) == "test" for i in ids) / len(ids)
    assert 0.45 < frac_test < 0.55


def test_metrics():
    gold = {("a", "1"), ("b", "2"), ("c", "3")}
    assert pair_completeness({("a", "1"), ("b", "9")}, gold) == 1 / 3
    assert reduction_ratio(10, 10, 10) == 0.9
    q = prf({("a", "1"), ("b", "2"), ("x", "y")}, gold)
    assert q["tp"] == 2 and abs(q["precision"] - 2 / 3) < 1e-9 and abs(q["recall"] - 2 / 3) < 1e-9


def test_connected_components_and_violations():
    clusters = connected_components([("a1", "b1"), ("a2", "b1"), ("a3", "b3")], "A", "B")
    assert clusters[0] == ["A/a1", "A/a2", "B/b1"]
    assert pairs_from_clusters(clusters, "A", "B") == {("a1", "b1"), ("a2", "b1"), ("a3", "b3")}
    # a2-b1 matched via the component, but the cascade said a2-b1 is a non-match -> one violation
    assert count_transitivity_violations(clusters, {("a2", "b1")}, "A", "B") == 1


def test_plan_validation():
    p = Plan("vector", ["rules", "fs", "llm"], [(0.2, 0.8), (0.3, 0.7)])
    assert p.k == 10 and p.thresholds[-1] == (0.5, 0.5)
    assert "vector(k=10)" in p.describe()
    for bad in (lambda: Plan("vector", ["fs", "rules"]), lambda: Plan("nope", ["rules"]),
                lambda: Plan("btree", ["rules", "fs"], [(0.8, 0.2)])):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("invalid plan accepted")
