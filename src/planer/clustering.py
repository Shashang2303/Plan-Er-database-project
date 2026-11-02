"""FR-17: entity clusters as connected components of the match graph (union-find).

Members are "<source>/<_id>" strings so every cluster references records that exist in the
source collections (NFR-4). Transitivity repair (FR-18, T17) is scheduled for Phase 4; this
module already counts violations so the repair step has its baseline.
"""
from __future__ import annotations

from collections import defaultdict


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        self.size.setdefault(x, 1)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def groups(self) -> list[list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for x in self.parent:
            out[self.find(x)].append(x)
        return [sorted(v) for v in out.values()]


def node(source: str, rec_id: str) -> str:
    return f"{source}/{rec_id}"


def connected_components(matches: list[tuple[str, str]], source_a: str, source_b: str) -> list[list[str]]:
    """Clusters with at least two members, largest first."""
    uf = UnionFind()
    for a, b in matches:
        uf.union(node(source_a, a), node(source_b, b))
    return sorted((g for g in uf.groups() if len(g) > 1), key=len, reverse=True)


def count_transitivity_violations(clusters: list[list[str]], confident_nonmatches: set[tuple[str, str]],
                                  source_a: str, source_b: str) -> int:
    """Pairs inside one cluster that the cascade confidently decided were NOT a match."""
    pa, pb = source_a + "/", source_b + "/"
    n = 0
    for members in clusters:
        a_ids = [m[len(pa):] for m in members if m.startswith(pa)]
        b_ids = {m[len(pb):] for m in members if m.startswith(pb)}
        n += sum(1 for a in a_ids for b in b_ids if (a, b) in confident_nonmatches)
    return n
