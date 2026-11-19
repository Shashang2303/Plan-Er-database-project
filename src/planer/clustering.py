"""FR-17: entity clusters as connected components of the match graph (union-find).

Members are "<source>/<_id>" strings so every cluster references records that exist in the
source collections (NFR-4). Plans choose plain connected components (clustering
"connected_components") or components followed by transitivity repair ("cc_repair", FR-18).
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


def repair_clusters(matches: dict[tuple[str, str], float], nonmatches: set[tuple[str, str]],
                    source_a: str, source_b: str) -> tuple[list[dict], dict]:
    """FR-18 / T17: split components that contain a confident non-match, without any model call.

    A violation is a pair (a, b) inside one component that the cascade decided is NOT a match
    (A-B and B-C matched, A-C not). While a component holds a violation, the lowest-confidence match
    edge on the path between the two records is removed and the component re-split (design 7.5).
    Returns cluster docs {members, violations_repaired} and run statistics.
    """
    from collections import deque

    adj: dict[str, dict[str, float]] = defaultdict(dict)
    for (a, b), conf in matches.items():
        u, v = node(source_a, a), node(source_b, b)
        adj[u][v] = conf
        adj[v][u] = conf
    bad: dict[str, set[str]] = defaultdict(set)
    for a, b in nonmatches:
        u, v = node(source_a, a), node(source_b, b)
        if u in adj and v in adj:      # only records that are in some cluster can violate
            bad[u].add(v)
            bad[v].add(u)

    def components(nodes: set[str]) -> list[set[str]]:
        seen, out = set(), []
        for start in nodes:
            if start in seen:
                continue
            comp, stack = set(), [start]
            seen.add(start)
            while stack:
                x = stack.pop()
                comp.add(x)
                for y in adj[x]:
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
            out.append(comp)
        return out

    def path(u: str, v: str, comp: set[str]) -> list[str] | None:
        prev = {u: None}
        q = deque([u])
        while q:
            x = q.popleft()
            if x == v:
                out = [v]
                while prev[out[-1]] is not None:
                    out.append(prev[out[-1]])
                return out[::-1]
            for y in adj[x]:
                if y in comp and y not in prev:
                    prev[y] = x
                    q.append(y)
        return None

    def violations_in(comp: set[str]) -> list[tuple[str, str]]:
        return [(u, v) for u in comp for v in bad.get(u, ()) if v in comp and u < v]

    initial = components(set(adj))
    before = sum(len(violations_in(c)) for c in initial)
    out: list[dict] = []
    removed_total = 0
    for root in initial:
        removed_here = 0
        stack = [root]
        done: list[set[str]] = []
        while stack:
            comp = stack.pop()
            viol = violations_in(comp)
            if not viol:
                done.append(comp)
                continue
            u, v = viol[0]
            p = path(u, v, comp)
            if p is None:            # already separated: just re-split
                stack.extend(components(comp))
                continue
            x, y = min(zip(p, p[1:]), key=lambda e: adj[e[0]][e[1]])   # weakest edge on the path
            del adj[x][y]
            del adj[y][x]
            removed_here += 1
            stack.extend(components(comp))
        removed_total += removed_here
        out.extend({"members": sorted(c), "violations_repaired": removed_here} for c in done if len(c) > 1)
    out.sort(key=lambda d: len(d["members"]), reverse=True)
    stats = {"violations_before": before, "violations_after": 0, "edges_removed": removed_total, "llm_calls": 0}
    return out, stats


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
