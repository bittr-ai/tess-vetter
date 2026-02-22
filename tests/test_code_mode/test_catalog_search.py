from __future__ import annotations

from copy import deepcopy
from random import Random

import pytest

from tess_vetter.code_mode.catalog import build_catalog
from tess_vetter.code_mode.search import search_catalog


def _sample_entries() -> list[dict[str, object]]:
    return [
        {
            "id": "z_internal_cache",
            "tier": "internal",
            "title": "Internal cache",
            "description": "Internal memoized cache lifecycle",
            "tags": ["cache", "infra"],
            "schema": {"type": "object", "properties": {"ttl": {"type": "number"}}},
        },
        {
            "id": "a_primitive_fold",
            "tier": "primitive",
            "title": "Fold curve",
            "description": "Primitive fold utility for transit diagnostics",
            "tags": ["lightcurve", "fold"],
            "schema": {"type": "object", "properties": {"period": {"type": "number"}}},
        },
        {
            "id": "b_golden_report",
            "tier": "golden_path",
            "title": "Golden report",
            "description": "Main report generation flow",
            "tags": ["report", "pipeline"],
            "schema": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "max_rows": {"type": "integer"},
                },
            },
        },
    ]


def test_build_catalog_repeatable_hash_and_order() -> None:
    entries = _sample_entries()

    build_a = build_catalog(entries)
    build_b = build_catalog(entries)

    assert build_a.catalog_version_hash == build_b.catalog_version_hash
    assert build_a.canonical_lines == build_b.canonical_lines
    assert [entry.id for entry in build_a.entries] == [
        "b_golden_report",
        "a_primitive_fold",
        "z_internal_cache",
    ]


def test_build_catalog_shuffled_input_invariant() -> None:
    base = _sample_entries()
    shuffled = deepcopy(base)
    Random(42).shuffle(shuffled)

    build_base = build_catalog(base)
    build_shuffled = build_catalog(shuffled)

    assert build_base.catalog_version_hash == build_shuffled.catalog_version_hash
    assert build_base.canonical_lines == build_shuffled.canonical_lines


def test_schema_change_changes_catalog_hash() -> None:
    original = _sample_entries()
    changed = deepcopy(original)

    changed[0]["schema"] = {
        "type": "object",
        "properties": {
            "ttl": {"type": "number"},
            "max_size": {"type": "integer"},
        },
    }

    build_original = build_catalog(original)
    build_changed = build_catalog(changed)

    assert build_original.catalog_version_hash != build_changed.catalog_version_hash


def test_non_finite_schema_number_rejected() -> None:
    broken = _sample_entries()
    broken[0]["schema"] = {"type": "object", "default": float("nan")}

    with pytest.raises(ValueError, match="Non-finite"):
        build_catalog(broken)


def test_search_rank_and_why_matched() -> None:
    catalog = build_catalog(_sample_entries())

    matches = search_catalog(
        catalog.entries,
        query="report",
        tags=["report", "pipeline"],
        limit=3,
    )

    assert [match.entry.id for match in matches] == [
        "b_golden_report",
        "a_primitive_fold",
        "z_internal_cache",
    ]
    assert any(reason.startswith("tier:golden_path") for reason in matches[0].why_matched)
    assert "tags:2" in matches[0].why_matched
    assert any(reason.startswith("text:") for reason in matches[0].why_matched)


def test_search_tiebreak_by_id_for_equal_rank() -> None:
    entries = build_catalog(
        [
            {
                "id": "z_same_rank",
                "tier": "primitive",
                "title": "Transit helper",
                "description": "No query terms",
                "tags": ["infra"],
                "schema": {"type": "object", "properties": {}},
            },
            {
                "id": "a_same_rank",
                "tier": "primitive",
                "title": "Transit helper",
                "description": "No query terms",
                "tags": ["infra"],
                "schema": {"type": "object", "properties": {}},
            },
        ]
    )

    matches = search_catalog(entries.entries, query="report", tags=["pipeline"], limit=2)

    assert [match.entry.id for match in matches] == ["a_same_rank", "z_same_rank"]
    assert matches[0].score == matches[1].score


def test_search_limit_is_deterministic_across_input_order() -> None:
    base = [
        {
            "id": "c_rank",
            "tier": "primitive",
            "title": "Transit helper",
            "description": "No query terms",
            "tags": ["infra"],
            "schema": {"type": "object", "properties": {}},
        },
        {
            "id": "a_rank",
            "tier": "primitive",
            "title": "Transit helper",
            "description": "No query terms",
            "tags": ["infra"],
            "schema": {"type": "object", "properties": {}},
        },
        {
            "id": "b_rank",
            "tier": "primitive",
            "title": "Transit helper",
            "description": "No query terms",
            "tags": ["infra"],
            "schema": {"type": "object", "properties": {}},
        },
    ]
    expected = ["a_rank", "b_rank"]

    for seed in (1, 7, 21):
        shuffled = deepcopy(base)
        Random(seed).shuffle(shuffled)
        catalog = build_catalog(shuffled)

        matches = search_catalog(catalog.entries, query="report", tags=["pipeline"], limit=2)
        assert [match.entry.id for match in matches] == expected


def test_search_score_respects_tag_over_text_precedence() -> None:
    catalog = build_catalog(
        [
            {
                "id": "high_text_one_tag",
                "tier": "primitive",
                "title": "alpha beta gamma delta epsilon zeta eta theta iota kappa",
                "description": "alpha beta gamma delta epsilon zeta eta theta iota kappa",
                "tags": ["only"],
                "schema": {"type": "object", "properties": {}},
            },
            {
                "id": "low_text_two_tags",
                "tier": "primitive",
                "title": "alpha",
                "description": "alpha",
                "tags": ["only", "extra"],
                "schema": {"type": "object", "properties": {}},
            },
        ]
    )

    matches = search_catalog(
        catalog.entries,
        query="alpha beta gamma delta epsilon zeta eta theta iota kappa",
        tags=["only", "extra"],
        limit=2,
    )

    assert [match.entry.id for match in matches] == ["low_text_two_tags", "high_text_one_tag"]
    assert matches[0].score > matches[1].score
