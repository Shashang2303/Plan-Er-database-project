"""Matcher tiers in fixed cost order: rules -> Fellegi-Sunter -> random forest -> local LLM."""
from __future__ import annotations

from .base import Matcher
from .fellegi_sunter import FellegiSunterMatcher
from .llm import OllamaMatcher, VerdictCache
from .pairs import PairBatch
from .random_forest import RandomForestMatcher
from .rules import RuleMatcher


def make_matcher(spec: str, cache: VerdictCache | None = None) -> Matcher:
    """'rules' | 'fs' | 'rf' | 'llm' | 'llm:<ollama model>'"""
    kind, _, arg = spec.partition(":")
    if kind == "rules":
        return RuleMatcher()
    if kind == "fs":
        return FellegiSunterMatcher()
    if kind == "rf":
        return RandomForestMatcher()
    if kind == "llm":
        return OllamaMatcher(model=arg or None, cache=cache)
    raise KeyError(f"unknown matcher {spec!r}; use rules, fs, rf, llm or llm:<model>")


__all__ = [
    "Matcher", "PairBatch", "RuleMatcher", "FellegiSunterMatcher", "RandomForestMatcher",
    "OllamaMatcher", "make_matcher",
]
