from __future__ import annotations

from copy import deepcopy
from random import Random
from typing import Any

import pytest
from pydantic import BaseModel

import tess_vetter.api as api
from tess_vetter.api import primitives as api_primitives
from tess_vetter.api.contracts import opaque_object_schema
from tess_vetter.code_mode.catalog import build_catalog, extract_required_input_paths
from tess_vetter.code_mode.mcp_adapter import SearchRequest, make_default_mcp_adapter
from tess_vetter.code_mode.search import search_catalog


def _sample_entries() -> list[dict[str, object]]:
    return [
        {
            "id": "z_internal_cache",
            "tier": "internal",
            "title": "Internal cache",
            "description": "Internal memoized cache lifecycle",
            "tags": ["cache", "infra"],
            "availability": "available",
            "schema": {"type": "object", "properties": {"ttl": {"type": "number"}}},
        },
        {
            "id": "a_primitive_fold",
            "tier": "primitive",
            "title": "Fold curve",
            "description": "Primitive fold utility for transit diagnostics",
            "tags": ["lightcurve", "fold"],
            "availability": "available",
            "status": "active",
            "schema": {"type": "object", "properties": {"period": {"type": "number"}}},
        },
        {
            "id": "b_golden_report",
            "tier": "golden_path",
            "title": "Golden report",
            "description": "Main report generation flow",
            "tags": ["report", "pipeline"],
            "availability": "unavailable",
            "status": "deprecated",
            "deprecated": True,
            "replacement": "a_primitive_fold",
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


def test_build_catalog_normalizes_legacy_tier_labels_for_order_and_hash() -> None:
    legacy = [
        {
            "id": "legacy_golden",
            "tier": "golden",
            "title": "Legacy golden",
            "description": "Legacy tier label",
            "tags": ["legacy"],
            "schema": {"type": "object", "properties": {}},
        },
        {
            "id": "legacy_api",
            "tier": "api",
            "title": "Legacy api",
            "description": "Legacy tier label",
            "tags": ["legacy"],
            "schema": {"type": "object", "properties": {}},
        },
        {
            "id": "new_internal",
            "tier": "internal",
            "title": "Internal",
            "description": "Expanded tier label",
            "tags": ["legacy"],
            "schema": {"type": "object", "properties": {}},
        },
    ]
    expanded = deepcopy(legacy)
    expanded[0]["tier"] = "golden_path"
    expanded[1]["tier"] = "primitive"

    build_legacy = build_catalog(legacy)
    build_expanded = build_catalog(expanded)

    assert [entry.tier for entry in build_legacy.entries] == [
        "golden_path",
        "primitive",
        "internal",
    ]
    assert [entry.id for entry in build_legacy.entries] == [
        "legacy_golden",
        "legacy_api",
        "new_internal",
    ]
    assert build_legacy.canonical_lines == build_expanded.canonical_lines
    assert build_legacy.catalog_version_hash == build_expanded.catalog_version_hash


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


def test_catalog_entry_metadata_is_preserved() -> None:
    catalog = build_catalog(_sample_entries())
    by_id = {entry.id: entry for entry in catalog.entries}

    assert by_id["b_golden_report"].availability == "unavailable"
    assert by_id["b_golden_report"].status == "deprecated"
    assert by_id["b_golden_report"].deprecated is True
    assert by_id["b_golden_report"].replacement == "a_primitive_fold"

    assert by_id["z_internal_cache"].availability == "available"
    assert by_id["z_internal_cache"].status == "active"
    assert by_id["z_internal_cache"].deprecated is False
    assert by_id["z_internal_cache"].replacement is None


def test_deprecation_and_replacement_change_catalog_hash() -> None:
    original = _sample_entries()
    changed_deprecated = deepcopy(original)
    changed_replacement = deepcopy(original)

    changed_deprecated[2]["deprecated"] = False
    changed_replacement[2]["replacement"] = "z_internal_cache"

    build_original = build_catalog(original)
    build_changed_deprecated = build_catalog(changed_deprecated)
    build_changed_replacement = build_catalog(changed_replacement)

    assert build_original.catalog_version_hash != build_changed_deprecated.catalog_version_hash
    assert build_original.catalog_version_hash != build_changed_replacement.catalog_version_hash


def test_availability_change_changes_catalog_hash() -> None:
    original = _sample_entries()
    changed = deepcopy(original)

    changed[1]["availability"] = "unavailable"

    build_original = build_catalog(original)
    build_changed = build_catalog(changed)

    assert build_original.catalog_version_hash != build_changed.catalog_version_hash


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
    assert "availability:unavailable" in matches[0].why_matched
    assert "status:deprecated" in matches[0].why_matched
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


def test_search_legacy_and_expanded_tiers_rank_identically() -> None:
    legacy_catalog = build_catalog(
        [
            {
                "id": "legacy_golden",
                "tier": "golden",
                "title": "Golden path legacy",
                "description": "Flow",
                "tags": ["catalog"],
                "schema": {"type": "object", "properties": {}},
            },
            {
                "id": "legacy_api",
                "tier": "api",
                "title": "Primitive legacy",
                "description": "Flow",
                "tags": ["catalog"],
                "schema": {"type": "object", "properties": {}},
            },
            {
                "id": "native_internal",
                "tier": "internal",
                "title": "Internal native",
                "description": "Flow",
                "tags": ["catalog"],
                "schema": {"type": "object", "properties": {}},
            },
        ]
    )
    expanded_catalog = build_catalog(
        [
            {
                "id": "legacy_golden",
                "tier": "golden_path",
                "title": "Golden path legacy",
                "description": "Flow",
                "tags": ["catalog"],
                "schema": {"type": "object", "properties": {}},
            },
            {
                "id": "legacy_api",
                "tier": "primitive",
                "title": "Primitive legacy",
                "description": "Flow",
                "tags": ["catalog"],
                "schema": {"type": "object", "properties": {}},
            },
            {
                "id": "native_internal",
                "tier": "internal",
                "title": "Internal native",
                "description": "Flow",
                "tags": ["catalog"],
                "schema": {"type": "object", "properties": {}},
            },
        ]
    )

    legacy_matches = search_catalog(legacy_catalog.entries, query="", tags=[], limit=3)
    expanded_matches = search_catalog(expanded_catalog.entries, query="", tags=[], limit=3)

    assert [match.entry.id for match in legacy_matches] == [
        "legacy_golden",
        "legacy_api",
        "native_internal",
    ]
    assert [match.entry.id for match in expanded_matches] == [
        "legacy_golden",
        "legacy_api",
        "native_internal",
    ]
    assert [match.score for match in legacy_matches] == [match.score for match in expanded_matches]
    assert any(reason == "tier:golden_path:30" for reason in legacy_matches[0].why_matched)
    assert any(reason == "tier:primitive:20" for reason in legacy_matches[1].why_matched)


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


def test_wrapper_backed_entry_exposes_static_input_output_schema_snippets() -> None:
    class WrapperInput(BaseModel):
        candidate_id: str
        max_rows: int = 10

    class WrapperOutput(BaseModel):
        accepted: bool
        score: float

    catalog = build_catalog(
        [
            {
                "id": "wrapper_schema_op",
                "tier": "primitive",
                "title": "Wrapper schema op",
                "description": "Wrapper-backed operation",
                "tags": ["wrapper"],
                "schema": {"type": "object"},
                "input_model": WrapperInput,
                "output_model": WrapperOutput,
            }
        ]
    )

    entry = catalog.entries[0]
    assert isinstance(entry.schema, dict)
    wrapper_schemas = entry.schema.get("wrapper_schemas")
    assert isinstance(wrapper_schemas, dict)
    assert "input" in wrapper_schemas
    assert "output" in wrapper_schemas

    input_schema = wrapper_schemas["input"]
    output_schema = wrapper_schemas["output"]

    assert input_schema["type"] == "object"
    assert set(input_schema["properties"]) == {"candidate_id", "max_rows"}
    assert input_schema["properties"]["candidate_id"]["type"] == "string"
    assert input_schema["properties"]["max_rows"]["type"] == "integer"
    assert input_schema["required"] == ["candidate_id"]
    assert "title" not in input_schema

    assert output_schema["type"] == "object"
    assert set(output_schema["properties"]) == {"accepted", "score"}
    assert output_schema["properties"]["accepted"]["type"] == "boolean"
    assert output_schema["properties"]["score"]["type"] == "number"
    assert output_schema["required"] == ["accepted", "score"]
    assert "title" not in output_schema


def test_wrapper_schema_snippet_hash_is_stable_for_equivalent_model_shapes() -> None:
    class InputShapeA(BaseModel):
        candidate_id: str
        max_rows: int = 10

    class OutputShapeA(BaseModel):
        accepted: bool

    class InputShapeB(BaseModel):
        candidate_id: str
        max_rows: int = 10

    class OutputShapeB(BaseModel):
        accepted: bool

    base_entry = {
        "id": "wrapper_shape_stability",
        "tier": "primitive",
        "title": "Wrapper shape stability",
        "description": "Wrapper-backed operation",
        "tags": ["wrapper"],
        "schema": {"type": "object"},
    }

    build_a = build_catalog([{**base_entry, "input_model": InputShapeA, "output_model": OutputShapeA}])
    build_b = build_catalog([{**base_entry, "input_model": InputShapeB, "output_model": OutputShapeB}])

    assert build_a.catalog_version_hash == build_b.catalog_version_hash
    assert build_a.canonical_lines == build_b.canonical_lines
    assert build_a.entries[0].schema == build_b.entries[0].schema


def test_search_rank_is_unchanged_by_wrapper_schema_metadata() -> None:
    class WrapperInput(BaseModel):
        candidate_id: str

    class WrapperOutput(BaseModel):
        accepted: bool

    catalog = build_catalog(
        [
            {
                "id": "a_schema",
                "tier": "primitive",
                "title": "Transit helper",
                "description": "No query terms",
                "tags": ["infra"],
                "schema": {"type": "object"},
                "input_model": WrapperInput,
                "output_model": WrapperOutput,
            },
            {
                "id": "b_no_schema",
                "tier": "primitive",
                "title": "Transit helper",
                "description": "No query terms",
                "tags": ["infra"],
                "schema": {"type": "object"},
            },
        ]
    )

    matches = search_catalog(catalog.entries, query="report", tags=["pipeline"], limit=2)
    assert [match.entry.id for match in matches] == ["a_schema", "b_no_schema"]
    assert matches[0].score == matches[1].score


@pytest.mark.parametrize(
    ("operation_id", "fn"),
    (
        ("code_mode.golden_path.vet_candidate", api.vet_candidate),
        ("code_mode.golden_path.run_periodogram", api.run_periodogram),
        ("code_mode.primitive.fold", api_primitives.fold),
        ("code_mode.primitive.median_detrend", api_primitives.median_detrend),
    ),
)
def test_manual_seed_schema_snippet_matches_upstream_opaque_contract(
    operation_id: str,
    fn: Any,
) -> None:
    del fn
    response = make_default_mcp_adapter().search(SearchRequest(query="", limit=1_000, tags=[]))
    assert response.error is None
    by_id = {row.id: row for row in response.results}
    row = by_id.get(operation_id)
    assert row is not None, f"Missing search row for {operation_id}"

    schema_snippet = row.metadata.get("schema_snippet")
    assert isinstance(schema_snippet, dict), f"Missing schema_snippet for {operation_id}"
    actual_input = schema_snippet.get("input")
    assert isinstance(actual_input, dict), f"Missing schema_snippet.input for {operation_id}"

    assert actual_input == opaque_object_schema()


_HIGH_TRAFFIC_INTERNAL_OPERATION_IDS: tuple[str, ...] = (
    "code_mode.internal.check_v01_odd_even_depth",
    "code_mode.internal.check_v02_secondary_eclipse",
    "code_mode.internal.check_v03_duration_consistency",
    "code_mode.internal.check_v04_depth_stability",
    "code_mode.internal.check_v05_v_shape",
    "code_mode.internal.check_v06_nearby_eb_search",
    "code_mode.internal.check_v07_exofop_toi_lookup",
    "code_mode.internal.check_v08_centroid_shift",
    "code_mode.internal.check_v09_difference_image",
    "code_mode.internal.check_v10_aperture_dependence",
    "code_mode.internal.check_v11_modshift",
    "code_mode.internal.check_v12_sweet",
    "code_mode.internal.check_v13_data_gaps",
    "code_mode.internal.check_v15_transit_asymmetry",
    "code_mode.internal.run_check",
    "code_mode.internal.vet_catalog",
    "code_mode.internal.calculate_fpp",
)


def _tranche_high_traffic_internal_search_rows() -> list[dict[str, Any]]:
    response = make_default_mcp_adapter().search(SearchRequest(query="", limit=1_000, tags=[]))
    assert response.error is None
    by_id = {row.id: row for row in response.results}
    tranche_rows: list[dict[str, Any]] = []
    for operation_id in _HIGH_TRAFFIC_INTERNAL_OPERATION_IDS:
        row = by_id.get(operation_id)
        if row is None:
            continue
        metadata = row.metadata
        if metadata.get("operation_tier") != "internal":
            continue
        if metadata.get("availability") != "available":
            continue
        if metadata.get("status") != "active":
            continue
        tranche_rows.append({"operation_id": row.id, "metadata": metadata})
    return tranche_rows


def _missing_high_traffic_internal_operation_ids() -> list[str]:
    response = make_default_mcp_adapter().search(SearchRequest(query="", limit=1_000, tags=[]))
    assert response.error is None
    by_id = {row.id: row for row in response.results}
    missing_ids: list[str] = []
    for operation_id in _HIGH_TRAFFIC_INTERNAL_OPERATION_IDS:
        row = by_id.get(operation_id)
        if row is None:
            missing_ids.append(operation_id)
            continue
        metadata = row.metadata
        if metadata.get("operation_tier") != "internal":
            missing_ids.append(operation_id)
            continue
        if metadata.get("availability") != "available":
            missing_ids.append(operation_id)
            continue
        if metadata.get("status") != "active":
            missing_ids.append(operation_id)
            continue
    return missing_ids


@pytest.mark.parametrize(
    "row",
    _tranche_high_traffic_internal_search_rows(),
    ids=lambda row: row["operation_id"],
)
def test_tranche_internal_required_paths_match_schema_truth(row: dict[str, Any]) -> None:
    operation_id = row["operation_id"]
    metadata = row["metadata"]
    callability = metadata.get("operation_callability")
    schema_snippet = metadata.get("schema_snippet")

    assert isinstance(callability, dict), f"Missing operation_callability for operation id: {operation_id}"
    assert isinstance(schema_snippet, dict), f"Missing schema_snippet for operation id: {operation_id}"

    input_schema = schema_snippet.get("input")
    assert isinstance(input_schema, dict), f"Missing schema_snippet.input for operation id: {operation_id}"

    expected = sorted(set(extract_required_input_paths(input_schema)))
    actual_raw = callability.get("required_paths")
    assert isinstance(actual_raw, list) and all(isinstance(path, str) for path in actual_raw), (
        "Missing/invalid required_paths for operation id: " f"{operation_id}"
    )
    actual = sorted(set(actual_raw))

    assert actual == expected, (
        "required_paths mismatch for operation id "
        f"{operation_id}: expected={expected}, actual={actual}"
    )


def test_tranche_internal_required_paths_summary_lists_failing_operation_ids() -> None:
    failing: list[str] = []

    missing_ids = _missing_high_traffic_internal_operation_ids()
    failing.extend(f"{operation_id}:operation_missing_or_inactive" for operation_id in missing_ids)

    for row in _tranche_high_traffic_internal_search_rows():
        operation_id = row["operation_id"]
        metadata = row["metadata"]
        callability = metadata.get("operation_callability")
        schema_snippet = metadata.get("schema_snippet")
        if not isinstance(callability, dict):
            failing.append(f"{operation_id}:operation_callability_missing")
            continue
        if not isinstance(schema_snippet, dict):
            failing.append(f"{operation_id}:schema_snippet_missing")
            continue
        input_schema = schema_snippet.get("input")
        if not isinstance(input_schema, dict):
            failing.append(f"{operation_id}:schema_input_missing")
            continue
        expected = sorted(set(extract_required_input_paths(input_schema)))
        actual_raw = callability.get("required_paths")
        if not isinstance(actual_raw, list) or any(not isinstance(path, str) for path in actual_raw):
            failing.append(f"{operation_id}:required_paths_missing_or_invalid")
            continue
        actual = sorted(set(actual_raw))
        if actual != expected:
            failing.append(f"{operation_id}:required_paths_mismatch")

    assert not failing, f"High-traffic internal required_paths failures: {sorted(set(failing))}"
