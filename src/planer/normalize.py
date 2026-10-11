"""Record normalization (FR-2): normalized text, token array and blocking key.

These are pure functions so they can be unit-tested without MongoDB.
"""
from __future__ import annotations

import re
import unicodedata

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_PRICE = re.compile(r"[-+]?\d[\d,]*\.?\d*")

STOPWORDS = frozenset(
    """a an and are as at be by for from in into is it of on or the to with without via its this that
    over under using towards toward new""".split()
)


def normalize_text(value: object) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return _NON_ALNUM.sub(" ", text.lower()).strip()


def tokenize(norm_text: str) -> list[str]:
    """Distinct tokens in order of first appearance; stopwords and 1-character tokens dropped."""
    seen: dict[str, None] = {}
    for tok in norm_text.split():
        if len(tok) > 1 and tok not in STOPWORDS:
            seen.setdefault(tok, None)
    return list(seen)


BKEY_RULES = ("brand_prefix3", "first_token", "title_prefix")


def blocking_key(norm_title: str, rule: str) -> str:
    """Scalar key for the B-tree access path (FR-3). The rule is set per dataset (datasets.py).

    brand_prefix3:  first token (usually the brand) + '|' + first 3 chars of the next token,
                    e.g. "apple|iph" as in the D.4.2 example record. Precise but brittle.
    first_token:    first token only (the brand for product titles). Larger blocks, higher recall.
    title_prefix:   first 4 chars of the first two content words, e.g. "coop|tran".
    ("product" and "bibliographic" are accepted as aliases of brand_prefix3 and title_prefix.)
    """
    rule = {"product": "brand_prefix3", "bibliographic": "title_prefix"}.get(rule, rule)
    toks = tokenize(norm_title)
    if not toks:
        return ""
    if rule == "brand_prefix3":
        return toks[0] + ("|" + toks[1][:3] if len(toks) > 1 else "")
    if rule == "first_token":
        return toks[0]
    if rule == "title_prefix":
        return "|".join(t[:4] for t in toks[:2])
    raise ValueError(f"unknown blocking-key rule {rule!r}; choose from {BKEY_RULES}")


def parse_number(value: object) -> float | None:
    """'$1,299.00' -> 1299.0; '' / None / unparsable -> None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = _PRICE.search(str(value))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None
