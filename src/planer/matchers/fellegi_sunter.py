"""T9 / FR-8: unsupervised Fellegi-Sunter record linkage fitted by expectation-maximization.

Each pair gets a comparison vector gamma: one discretized agreement level per field
(level 0..L-1, or missing). Under conditional independence,
    P(gamma | M) = prod_k m_k(gamma_k),   P(gamma | U) = prod_k u_k(gamma_k)
EM estimates p = P(M), m and u without labels. The score is the posterior
    P(M | gamma) = sigmoid( logit(p) + sum_k log(m_k / u_k) ),
i.e. the Fellegi-Sunter match weight calibrated to [0, 1]. Missing fields contribute nothing.
"""
from __future__ import annotations

import logging

import numpy as np

from .base import Matcher
from .pairs import PairBatch

log = logging.getLogger(__name__)

LEVEL_CUTS = (0.5, 0.75, 0.9)   # similarity -> agreement level 0..3
N_LEVELS = len(LEVEL_CUTS) + 1


def discretize(values: np.ndarray, cuts=LEVEL_CUTS) -> np.ndarray:
    """Similarity in [0,1] -> level 0..len(cuts); NaN -> -1 (missing)."""
    levels = np.digitize(np.nan_to_num(values, nan=0.0), cuts)
    return np.where(np.isnan(values), -1, levels).astype(int)


class FellegiSunterMatcher(Matcher):
    name = "fs"

    def __init__(self, max_iter: int = 200, tol: float = 1e-6, smoothing: float = 1e-3, p_init: float = 0.1):
        self.max_iter, self.tol, self.smoothing, self.p_init = max_iter, tol, smoothing, p_init
        self.fields: list[str] = []
        self.m: np.ndarray | None = None
        self.u: np.ndarray | None = None
        self.p: float = p_init
        self.n_iter = 0

    # ------------------------------------------------------------------ comparison vectors
    def _fields(self, batch: PairBatch) -> list[str]:
        F = batch.features()
        title = batch.spec.a.title_attr
        fields = [f"{title}_tset", "code_jac", "emb_cos"]
        for a in batch.attr_map:
            if a == title:
                continue
            fields.append(f"{a}_num" if F.has(f"{a}_num") else f"{a}_tset")
        return [f for f in fields if F.has(f)]

    def gamma(self, batch: PairBatch) -> np.ndarray:
        F = batch.features()
        return np.column_stack([discretize(F.col(f)) for f in self.fields])

    # ------------------------------------------------------------------ EM
    def fit(self, batch: PairBatch, labels: np.ndarray | None = None) -> "FellegiSunterMatcher":
        self.fields = self._fields(batch)
        G = self.gamma(batch)
        n, K = G.shape
        onehot = np.zeros((n, K, N_LEVELS))
        for k in range(K):
            ok = G[:, k] >= 0
            onehot[np.where(ok)[0], k, G[ok, k]] = 1.0  # missing fields stay all-zero

        # initial guess: matches agree strongly, non-matches weakly
        m = np.tile(np.linspace(0.05, 0.6, N_LEVELS), (K, 1))
        u = np.tile(np.linspace(0.6, 0.05, N_LEVELS), (K, 1))
        m /= m.sum(axis=1, keepdims=True)
        u /= u.sum(axis=1, keepdims=True)
        p = self.p_init
        prev_ll = -np.inf
        for it in range(1, self.max_iter + 1):
            # E-step
            log_m = np.einsum("nkl,kl->n", onehot, np.log(m))
            log_u = np.einsum("nkl,kl->n", onehot, np.log(u))
            a = np.log(p) + log_m
            b = np.log1p(-p) + log_u
            mx = np.maximum(a, b)
            ll = float(np.sum(mx + np.log(np.exp(a - mx) + np.exp(b - mx))))
            g = 1.0 / (1.0 + np.exp(b - a))
            # M-step (Laplace smoothing keeps every level's probability > 0)
            p = float(np.clip(g.mean(), 1e-4, 1 - 1e-4))
            m = np.einsum("n,nkl->kl", g, onehot) + self.smoothing
            u = np.einsum("n,nkl->kl", 1 - g, onehot) + self.smoothing
            m /= m.sum(axis=1, keepdims=True)
            u /= u.sum(axis=1, keepdims=True)
            self.n_iter = it
            if abs(ll - prev_ll) < self.tol * max(1.0, abs(ll)):
                break
            prev_ll = ll

        # Guard against label switching: the "match" class must agree more on the first field.
        levels = np.arange(N_LEVELS)
        if (m[0] * levels).sum() < (u[0] * levels).sum():
            m, u, p = u, m, 1 - p
        self.m, self.u, self.p = m, u, p
        log.debug("FS fitted in %d iterations, p=%.4f", self.n_iter, p)
        return self

    def match_weights(self, batch: PairBatch) -> np.ndarray:
        if self.m is None:
            raise RuntimeError("FellegiSunterMatcher.fit() must be called first")
        G = self.gamma(batch)
        w = np.log(self.m / self.u)  # (K, L)
        total = np.zeros(len(batch))
        for k in range(G.shape[1]):
            ok = G[:, k] >= 0
            total[ok] += w[k, G[ok, k]]
        return total

    def score(self, batch: PairBatch) -> np.ndarray:
        if self.m is None:
            self.fit(batch)
        z = np.log(self.p / (1 - self.p)) + self.match_weights(batch)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

    def parameters(self) -> dict:
        return {
            "fields": self.fields, "p": self.p, "iterations": self.n_iter,
            "m": None if self.m is None else self.m.round(4).tolist(),
            "u": None if self.u is None else self.u.round(4).tolist(),
        }
