"""Deterministic search ranking for code-mode catalog entries."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from tess_vetter.code_mode.catalog import CatalogEntry

_TIER_BIAS: dict[str, int] = {
    "golden_path": 30,
    "primitive": 20,
    "internal": 10,
}

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True, slots=True)
class SearchMatch:
    """Single ranked search result with deterministic score and reasons."""

    entry: CatalogEntry
    score: int
    why_matched: tuple[str, ...]


def _tokenize(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(value.lower()))


def _text_relevance(query_tokens: tuple[str, ...], entry: CatalogEntry) -> tuple[int, tuple[str, ...]]:
    if not query_tokens:
        return 0, ()

    id_tokens = set(_tokenize(entry.id))
    title_tokens = set(_tokenize(entry.title))
    desc_tokens = set(_tokenize(entry.description))

    score = 0
    reasons: list[str] = []

    for token in query_tokens:
        if token in id_tokens:
            score += 6
            reasons.append(f"text:id:{token}")
        elif token in title_tokens:
            score += 4
            reasons.append(f"text:title:{token}")
        elif token in desc_tokens:
            score += 2
            reasons.append(f"text:description:{token}")

    return score, tuple(reasons)


def search_catalog(
    entries: Iterable[CatalogEntry],
    *,
    query: str,
    limit: int = 10,
    tags: Iterable[str] | None = None,
) -> tuple[SearchMatch, ...]:
    """Search catalog entries with deterministic ranking.

    Ranking precedence:
    1) tier bias
    2) tag matches
    3) text relevance
    4) lexical id tie-break
    """
    if limit < 1:
        return ()

    query_tokens = _tokenize(query)
    requested_tags = {tag.strip().lower() for tag in (tags or ()) if tag and tag.strip()}

    matches: list[SearchMatch] = []
    for entry in entries:
        tier_bias = _TIER_BIAS.get(entry.tier, 0)
        tag_matches = len(requested_tags.intersection(entry.tags))
        text_score, text_reasons = _text_relevance(query_tokens, entry)

        why: list[str] = [f"tier:{entry.tier}:{tier_bias}"]
        if tag_matches > 0:
            why.append(f"tags:{tag_matches}")
        why.extend(text_reasons)

        score = tier_bias * 1000 + tag_matches * 100 + text_score
        matches.append(SearchMatch(entry=entry, score=score, why_matched=tuple(why)))

    matches.sort(
        key=lambda m: (
            -_TIER_BIAS.get(m.entry.tier, 0),
            -len(requested_tags.intersection(m.entry.tags)),
            -_text_relevance(query_tokens, m.entry)[0],
            m.entry.id,
        )
    )
    return tuple(matches[:limit])


__all__ = ["SearchMatch", "search_catalog"]
