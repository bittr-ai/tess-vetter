from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace
from typing import Any

import pytest

import tess_vetter.api.contracts as api_contracts
import tess_vetter.code_mode.adapters.manual as manual_adapters
from tess_vetter.code_mode.adapters.base import OperationAdapter
from tess_vetter.code_mode.catalog import CatalogEntry
from tess_vetter.code_mode.mcp_adapter import (
    ErrorPayload,
    ExecuteRequest,
    ExecuteResponse,
    MCPAdapter,
    SearchRequest,
    SearchResponse,
    SearchResult,
    _callability_score_and_flags,
    make_default_mcp_adapter,
)
from tess_vetter.code_mode.operation_spec import OperationSpec, SafetyClass, SafetyRequirements


def test_manual_seed_adapters_import_upstream_typing_utility() -> None:
    assert (
        manual_adapters.callable_input_schema_from_signature
        is api_contracts.callable_input_schema_from_signature
    )


def test_execute_request_contract_is_strict_plan_code_context_and_catalog_hash() -> None:
    assert [field.name for field in fields(ExecuteRequest)] == [
        "plan_code",
        "context",
        "catalog_version_hash",
    ]

    request = ExecuteRequest(
        plan_code="async def execute_plan(ops, context):\n    return {'ok': True}\n",
        context={"candidate": "tic-123"},
        catalog_version_hash="catalog-v7",
    )

    assert request.plan_code.startswith("async def execute_plan")
    assert request.context == {"candidate": "tic-123"}
    assert request.catalog_version_hash == "catalog-v7"


def test_search_preserves_callability_metadata_for_strict_execute_contract() -> None:
    def _search_handler(_request: SearchRequest) -> SearchResponse:
        return SearchResponse(
            results=[
                SearchResult(
                    id="code_mode.golden_path.vet_candidate",
                    title="vet_candidate",
                    snippet="Run candidate vetting.",
                    score=0.95,
                    metadata={
                        "operation_id": "code_mode.golden_path.vet_candidate",
                        "operation_version": "1.2.3",
                        "operation_tier": "golden_path",
                        "operation_tags": ["vet"],
                        "operation_requirements": {},
                        "operation_safety_class": "sandboxed",
                        "operation_callability": {
                            "tool": "execute",
                            "request_type": "ExecuteRequest",
                            "operation_field": "plan_code",
                            "payload_field": "context",
                            "sandbox_field": "catalog_version_hash",
                            "hash_field": "catalog_version_hash",
                            "sandbox_required": True,
                            "operation": "code_mode.golden_path.vet_candidate",
                            "required_paths": [
                                "plan_code",
                                "context",
                                "catalog_version_hash",
                            ],
                            "minimal_payload_example": {
                                "plan_code": "async def execute_plan(ops, context):\n    return {'ok': True}\n",
                                "context": {
                                    "candidate": "tic-123",
                                },
                                "catalog_version_hash": "catalog-v2",
                            },
                            "callability_score": 1.0,
                            "reason_flags": ["direct_execute_ready"],
                        },
                        "callability_score": 1.0,
                        "callability_reason_flags": ["direct_execute_ready"],
                    },
                )
            ],
            total=1,
            cursor=None,
            catalog_version_hash="catalog-v2",
            error=None,
        )

    adapter = MCPAdapter(search_handler=_search_handler)

    response = adapter.search(SearchRequest(query="vet", limit=10, tags=[]))

    assert response.error is None
    assert response.catalog_version_hash == "catalog-v2"
    metadata = response.results[0].metadata
    assert metadata["operation_id"] == "code_mode.golden_path.vet_candidate"
    assert metadata["operation_version"] == "1.2.3"
    assert metadata["operation_tier"] == "golden_path"
    assert metadata["operation_callability"]["tool"] == "execute"
    assert metadata["operation_callability"]["request_type"] == "ExecuteRequest"
    assert metadata["operation_callability"]["operation_field"] == "plan_code"
    assert metadata["operation_callability"]["payload_field"] == "context"
    assert metadata["operation_callability"]["sandbox_field"] == "catalog_version_hash"
    assert metadata["operation_callability"]["hash_field"] == "catalog_version_hash"
    assert metadata["operation_callability"]["required_paths"] == [
        "plan_code",
        "context",
        "catalog_version_hash",
    ]
    assert metadata["operation_callability"]["minimal_payload_example"]["context"] == {
        "candidate": "tic-123"
    }
    assert metadata["operation_callability"]["callability_score"] == 1.0
    assert metadata["operation_callability"]["reason_flags"] == ["direct_execute_ready"]
    assert metadata["callability_score"] == 1.0
    assert metadata["callability_reason_flags"] == ["direct_execute_ready"]


def test_default_search_metadata_exposes_truthful_execute_callability_contract() -> None:
    adapter = make_default_mcp_adapter()

    response = adapter.search(SearchRequest(query="vet", limit=1, tags=[]))

    assert response.error is None
    assert response.results
    metadata = response.results[0].metadata
    callability = metadata["operation_callability"]

    assert isinstance(callability["required_paths"], list)
    assert all(isinstance(path, str) for path in callability["required_paths"])
    assert callability["request_required_paths"] == [
        "plan_code",
        "context",
        "catalog_version_hash",
    ]
    example = callability["minimal_payload_example"]
    assert isinstance(example, dict)
    assert set(example) == {"plan_code", "context", "catalog_version_hash"}
    assert isinstance(example["plan_code"], str)
    assert "async def execute_plan(ops, context):" in example["plan_code"]
    assert response.results[0].id in example["plan_code"]
    assert isinstance(example["context"], dict)
    assert example["context"]["operation_kwargs"]["example"]
    assert isinstance(example["catalog_version_hash"], str)
    assert callability["callability_score"] == metadata["callability_score"]
    assert callability["reason_flags"] == metadata["callability_reason_flags"]


def test_callability_score_prefers_direct_execute_constructor_and_policy_ready() -> None:
    entry = CatalogEntry(
        id="code_mode.golden_path.unit_ready",
        tier="golden_path",
        title="Unit Ready",
        description="Synthetic operation for callability scoring.",
        tags=(),
        schema={},
        schema_fingerprint="fp-ready",
        availability="available",
        status="active",
        deprecated=False,
        replacement=None,
    )
    adapter = OperationAdapter(
        spec=OperationSpec(
            id=entry.id,
            name="Unit Ready",
            description="Synthetic operation for callability scoring.",
            safety_class=SafetyClass.SAFE,
            safety_requirements=SafetyRequirements(),
            input_json_schema={"type": "object", "properties": {}},
            output_json_schema={"type": "object"},
        ),
        fn=lambda **kwargs: kwargs,
    )

    score, flags = _callability_score_and_flags(entry=entry, adapter=adapter)

    assert score == 1.0
    assert flags == ("direct_execute_ready", "constructor_ready", "policy_ready")


def test_callability_score_penalizes_constructor_and_policy_blockers() -> None:
    entry = CatalogEntry(
        id="code_mode.golden_path.unit_blocked",
        tier="golden_path",
        title="Unit Blocked",
        description="Synthetic operation for callability scoring.",
        tags=(),
        schema={},
        schema_fingerprint="fp-blocked",
        availability="available",
        status="active",
        deprecated=False,
        replacement=None,
    )
    adapter = OperationAdapter(
        spec=OperationSpec(
            id=entry.id,
            name="Unit Blocked",
            description="Synthetic operation for callability scoring.",
            safety_class=SafetyClass.GUARDED,
            safety_requirements=SafetyRequirements(
                needs_network=True,
                needs_secrets=True,
                requires_human_review=True,
            ),
            input_json_schema={
                "type": "object",
                "required": ["tic_id", "sector", "camera"],
                "properties": {
                    "tic_id": {"type": "integer"},
                    "sector": {"type": "integer"},
                    "camera": {"type": "integer"},
                },
            },
            output_json_schema={"type": "object"},
        ),
        fn=lambda **kwargs: kwargs,
    )

    score, flags = _callability_score_and_flags(entry=entry, adapter=adapter)

    assert score < 0.5
    assert "direct_execute_ready" not in flags
    assert "constructor_inputs_required" in flags
    assert "policy_requires_network" in flags
    assert "policy_requires_human_review" in flags
    assert "policy_requires_secrets" in flags


def test_search_request_shape_validation_returns_schema_violation_input() -> None:
    adapter = MCPAdapter(search_handler=lambda _request: SearchResponse())

    response = adapter.search(SimpleNamespace(query=1, limit=10, tags=[]))

    assert response.error is not None
    assert response.error.code == "SCHEMA_VIOLATION_INPUT"
    assert "query" in response.error.message
    assert response.error.details["reason"] == "shape_validation_failed"


def test_execute_request_shape_validation_rejects_non_string_plan_code() -> None:
    adapter = MCPAdapter(execute_handler=lambda _request: ExecuteResponse(status="ok", result={"ok": True}))

    response = adapter.execute(
        SimpleNamespace(
            plan_code=123,
            context={"candidate": "tic-123"},
            catalog_version_hash="catalog-v2",
        )
    )

    assert response.status == "failed"
    assert response.error is not None
    assert response.error.code == "SCHEMA_VIOLATION_INPUT"
    assert response.error.retryable is False
    assert "plan_code" in response.error.message
    assert response.error.details["contract"] == "ExecuteRequest"


def test_execute_request_shape_validation_rejects_non_dict_context() -> None:
    adapter = MCPAdapter(execute_handler=lambda _request: ExecuteResponse(status="ok", result={"ok": True}))

    response = adapter.execute(
        SimpleNamespace(
            plan_code="async def execute_plan(ops, context):\n    return {'ok': True}\n",
            context=[],
            catalog_version_hash="catalog-v2",
        )
    )

    assert response.status == "failed"
    assert response.error is not None
    assert response.error.code == "SCHEMA_VIOLATION_INPUT"
    assert "context" in response.error.message


def test_execute_request_shape_validation_rejects_non_string_catalog_hash() -> None:
    adapter = MCPAdapter(execute_handler=lambda _request: ExecuteResponse(status="ok", result={"ok": True}))

    response = adapter.execute(
        SimpleNamespace(
            plan_code="async def execute_plan(ops, context):\n    return {'ok': True}\n",
            context={"candidate": "tic-123"},
            catalog_version_hash=7,
        )
    )

    assert response.status == "failed"
    assert response.error is not None
    assert response.error.code == "SCHEMA_VIOLATION_INPUT"
    assert "catalog_version_hash" in response.error.message


def test_search_to_execute_hash_pinning_and_drift_rejection() -> None:
    pinned_hash = "catalog-v9"

    def _search_handler(_request: SearchRequest) -> SearchResponse:
        return SearchResponse(
            results=[
                SearchResult(
                    id="code_mode.golden_path.vet_candidate",
                    title="vet_candidate",
                    snippet="Run candidate vetting.",
                    score=0.99,
                    metadata={"operation_version": "1.2.3", "operation_tier": "golden_path"},
                )
            ],
            total=1,
            cursor=None,
            catalog_version_hash=pinned_hash,
            error=None,
        )

    def _execute_handler(request: ExecuteRequest) -> ExecuteResponse:
        if request.catalog_version_hash != pinned_hash:
            return ExecuteResponse(
                status="failed",
                result=None,
                error=ErrorPayload(
                    code="CATALOG_DRIFT",
                    message="Catalog hash mismatch; refresh catalog before execution.",
                    retryable=False,
                    details={
                        "expected_catalog_version_hash": pinned_hash,
                        "received_catalog_version_hash": request.catalog_version_hash,
                    },
                ),
                trace={"short_circuit": True},
                catalog_version_hash=pinned_hash,
            )
        return ExecuteResponse(
            status="ok",
            result={"status": "ok", "hash": request.catalog_version_hash},
            error=None,
            trace={"short_circuit": False},
            catalog_version_hash=pinned_hash,
        )

    adapter = MCPAdapter(search_handler=_search_handler, execute_handler=_execute_handler)

    search_response = adapter.search(SearchRequest(query="vet", limit=10, tags=[]))
    assert search_response.error is None
    assert search_response.catalog_version_hash == pinned_hash

    execute_response = adapter.execute(
        ExecuteRequest(
            plan_code="async def execute_plan(ops, context):\n    return {'ok': True}\n",
            context={"candidate": "tic-123"},
            catalog_version_hash=search_response.catalog_version_hash,
        )
    )
    assert execute_response.status == "ok"
    assert execute_response.result["hash"] == pinned_hash

    drift_response = adapter.execute(
        ExecuteRequest(
            plan_code="async def execute_plan(ops, context):\n    return {'ok': True}\n",
            context={"candidate": "tic-123"},
            catalog_version_hash="catalog-v10",
        )
    )
    assert drift_response.status == "failed"
    assert drift_response.error is not None
    assert drift_response.error.code == "CATALOG_DRIFT"
    assert drift_response.error.details["expected_catalog_version_hash"] == pinned_hash
    assert drift_response.error.details["received_catalog_version_hash"] == "catalog-v10"


def test_error_serialization_uses_retryable_key_not_retriable() -> None:
    adapter = MCPAdapter(execute_handler=lambda _request: (_ for _ in ()).throw(RuntimeError("boom")))

    response = adapter.execute(
        ExecuteRequest(
            plan_code="async def execute_plan(ops, context):\n    return {'ok': True}\n",
            context={"candidate": "tic-123"},
            catalog_version_hash="catalog-v2",
        )
    )

    assert response.status == "failed"
    serialized = response.to_dict()
    assert serialized["error"] is not None
    assert "retryable" in serialized["error"]
    assert "retriable" not in serialized["error"]


def test_execute_preflight_reports_schema_policy_and_no_execution_calls() -> None:
    adapter = make_default_mcp_adapter()

    response = adapter.execute(
        ExecuteRequest(
            plan_code="""
async def execute_plan(ops, context):
    return await ops["code_mode.internal.check_v07_exofop_toi_lookup"](tic_id="not-an-int")
""",
            context={"mode": "preflight"},
            catalog_version_hash=None,
        )
    )

    assert response.status == "failed"
    assert response.error is not None
    assert response.error.code == "SCHEMA_VIOLATION_INPUT"
    assert isinstance(response.result, dict)
    assert response.result["mode"] == "preflight"
    assert response.result["ready"] is False
    blockers = response.result["blockers"]
    missing_fields = blockers["missing_fields"]
    assert [row["field"] for row in missing_fields] == ["candidate", "lc"]
    assert blockers["type_mismatches"] == [
        {
            "operation_id": "code_mode.internal.check_v07_exofop_toi_lookup",
            "field": "tic_id",
            "expected_types": ["integer"],
            "received_type": "str",
            "call_site": {"line": 3, "column": 17},
        }
    ]
    assert blockers["policy_blockers"] == [
        {
            "type": "needs_network",
            "operation_id": "code_mode.internal.check_v07_exofop_toi_lookup",
            "requirement": "needs_network",
            "policy_profile": "readonly_local",
            "call_site": {"line": 3, "column": 17},
        }
    ]
    assert blockers["dependency_blockers"] == []
    assert blockers["constructor_blockers"] == [
        {
            "type": "constructor_missing",
            "operation_id": "code_mode.internal.check_v07_exofop_toi_lookup",
            "field": "candidate",
            "reason": "constructor_missing",
            "call_site": {"line": 3, "column": 17},
        },
        {
            "type": "constructor_missing",
            "operation_id": "code_mode.internal.check_v07_exofop_toi_lookup",
            "field": "lc",
            "reason": "constructor_missing",
            "call_site": {"line": 3, "column": 17},
        },
    ]
    assert response.trace is not None
    assert response.trace["call_budget"]["used_calls"] == 0
    assert response.trace["call_events"] == []
    assert response.trace["metadata"]["mode"] == "preflight"


def test_execute_preflight_reports_dependency_blocker_for_unknown_operation() -> None:
    adapter = make_default_mcp_adapter()

    response = adapter.execute(
        ExecuteRequest(
            plan_code="""
async def execute_plan(ops, context):
    return await ops["code_mode.internal.not_real"]()
""",
            context={"mode": "preflight"},
            catalog_version_hash=None,
        )
    )

    assert response.status == "failed"
    assert response.error is not None
    assert response.error.code == "DEPENDENCY_MISSING"
    assert response.error.retryable is True
    assert isinstance(response.result, dict)
    assert response.result["mode"] == "preflight"
    assert response.result["ready"] is False
    blockers = response.result["blockers"]
    assert blockers["missing_fields"] == []
    assert blockers["type_mismatches"] == []
    assert blockers["policy_blockers"] == []
    assert blockers["dependency_blockers"] == [
        {
            "type": "operation_not_found",
            "operation_id": "code_mode.internal.not_real",
            "reason": "operation_not_found",
            "call_site": {"line": 3, "column": 17},
        }
    ]
    assert blockers["constructor_blockers"] == []
    assert response.trace is not None
    assert response.trace["call_budget"]["used_calls"] == 0
    assert response.trace["metadata"]["mode"] == "preflight"


def test_execute_preflight_ok_when_no_blockers() -> None:
    adapter = make_default_mcp_adapter()

    response = adapter.execute(
        ExecuteRequest(
            plan_code="""
async def execute_plan(ops, context):
    return {"ok": True}
""",
            context={"mode": "preflight"},
            catalog_version_hash=None,
        )
    )

    assert response.status == "ok"
    assert response.error is None
    assert response.result == {
        "mode": "preflight",
        "ready": True,
        "operation_ids": [],
        "blockers": {
            "missing_fields": [],
            "type_mismatches": [],
            "policy_blockers": [],
            "dependency_blockers": [],
            "constructor_blockers": [],
        },
    }
    assert response.trace is not None
    assert response.trace["call_budget"]["used_calls"] == 0
    assert response.trace["call_events"] == []
    assert response.trace["metadata"]["mode"] == "preflight"


def test_execute_preflight_constructor_blockers_are_deterministic_and_retryable_dependency() -> None:
    adapter = make_default_mcp_adapter()

    response = adapter.execute(
        ExecuteRequest(
            plan_code="""
async def execute_plan(ops, context):
    await ops["code_mode.internal.check_v07_exofop_toi_lookup"](tic_id=101)
    return await ops["code_mode.internal.check_v06_nearby_eb_search"](ra_deg=1.2, dec_deg=3.4)
""",
            context={"mode": "preflight", "policy_profile": "network_allowed"},
            catalog_version_hash=None,
        )
    )

    assert response.status == "failed"
    assert response.error is not None
    assert response.error.code == "SCHEMA_VIOLATION_INPUT"
    assert response.error.retryable is False
    assert isinstance(response.result, dict)

    blockers = response.result["blockers"]
    assert blockers["dependency_blockers"] == []
    assert blockers["policy_blockers"] == []
    assert blockers["type_mismatches"] == []
    assert [row["operation_id"] for row in blockers["constructor_blockers"]] == [
        "code_mode.internal.check_v06_nearby_eb_search",
        "code_mode.internal.check_v06_nearby_eb_search",
        "code_mode.internal.check_v07_exofop_toi_lookup",
        "code_mode.internal.check_v07_exofop_toi_lookup",
    ]
    assert [row["field"] for row in blockers["constructor_blockers"]] == [
        "candidate",
        "lc",
        "candidate",
        "lc",
    ]


def _tranche_active_operation_cases() -> tuple[dict[str, Any], ...]:
    high_traffic_internal_operation_ids = {
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
    }

    response = make_default_mcp_adapter().search(SearchRequest(query="", limit=1_000, tags=[]))
    assert response.error is None

    cases: list[dict[str, Any]] = []
    for row in response.results:
        metadata = row.metadata
        if row.id not in high_traffic_internal_operation_ids:
            continue
        if metadata.get("operation_tier") != "internal":
            continue
        if metadata.get("availability") != "available":
            continue
        if metadata.get("status") != "active":
            continue
        cases.append(
            {
                "operation_id": row.id,
                "schema": metadata.get("schema_snippet", {}).get("input", {}),
                "callability": metadata.get("operation_callability", {}),
            }
        )
    return tuple(sorted(cases, key=lambda item: item["operation_id"]))


def _preflight_blockers(schema: dict[str, Any], payload: Any, prefix: str = "") -> tuple[list[str], list[str]]:
    missing_fields: list[str] = []
    type_mismatches: list[str] = []

    if not isinstance(schema, dict):
        return missing_fields, type_mismatches

    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(payload, dict):
            location = prefix or "$"
            type_mismatches.append(f"{location}: expected object")
            return missing_fields, type_mismatches

        required = schema.get("required")
        properties = schema.get("properties")
        if isinstance(required, list):
            for raw_key in required:
                if not isinstance(raw_key, str):
                    continue
                path = f"{prefix}.{raw_key}" if prefix else raw_key
                if raw_key not in payload:
                    missing_fields.append(path)
                    continue
                nested_schema = properties.get(raw_key, {}) if isinstance(properties, dict) else {}
                nested_missing, nested_mismatches = _preflight_blockers(
                    nested_schema,
                    payload[raw_key],
                    prefix=path,
                )
                missing_fields.extend(nested_missing)
                type_mismatches.extend(nested_mismatches)
        return missing_fields, type_mismatches

    type_map = {
        "array": list,
        "boolean": bool,
        "integer": int,
        "number": (int, float),
        "string": str,
    }
    expected = type_map.get(schema_type)
    if expected is None:
        return missing_fields, type_mismatches
    if schema_type == "integer":
        valid = isinstance(payload, int) and not isinstance(payload, bool)
    elif schema_type == "number":
        valid = isinstance(payload, (int, float)) and not isinstance(payload, bool)
    else:
        valid = isinstance(payload, expected)
    if not valid:
        location = prefix or "$"
        type_mismatches.append(f"{location}: expected {schema_type}")
    return missing_fields, type_mismatches


def _minimal_value_for_schema(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return {}

    schema_type = schema.get("type")
    if schema_type == "object":
        properties = schema.get("properties")
        required = schema.get("required")
        payload: dict[str, Any] = {}
        if isinstance(required, list):
            for raw_key in required:
                if not isinstance(raw_key, str):
                    continue
                nested_schema = properties.get(raw_key, {}) if isinstance(properties, dict) else {}
                payload[raw_key] = _minimal_value_for_schema(nested_schema)
        return payload
    if schema_type == "array":
        items = schema.get("items")
        if isinstance(items, dict):
            return [_minimal_value_for_schema(items)]
        return []
    if schema_type == "string":
        return "example"
    if schema_type == "integer":
        return 1
    if schema_type == "number":
        return 1.0
    if schema_type == "boolean":
        return False
    if isinstance(schema.get("oneOf"), list):
        for branch in schema["oneOf"]:
            if isinstance(branch, dict):
                return _minimal_value_for_schema(branch)
    if isinstance(schema.get("anyOf"), list):
        for branch in schema["anyOf"]:
            if isinstance(branch, dict):
                return _minimal_value_for_schema(branch)
    return {}


def _missing_tranche_operation_ids() -> list[str]:
    expected = {
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
    }
    present = {case["operation_id"] for case in _tranche_active_operation_cases()}
    return sorted(expected - present)


# Parametric coverage section.
@pytest.mark.parametrize("case", _tranche_active_operation_cases(), ids=lambda case: case["operation_id"])
def test_tranche_minimal_payload_examples_are_preflight_passable(case: dict[str, Any]) -> None:
    callability = case["callability"]
    schema = case["schema"] if isinstance(case["schema"], dict) else {}
    minimal_payload_example = callability.get("minimal_payload_example") if isinstance(callability, dict) else None
    assert isinstance(minimal_payload_example, dict), (
        "Missing/invalid minimal_payload_example envelope for operation id: " f"{case['operation_id']}"
    )
    assert isinstance(minimal_payload_example.get("context"), dict), (
        "minimal_payload_example.context missing/invalid for operation id: " f"{case['operation_id']}"
    )
    assert "operation_kwargs" in minimal_payload_example["context"], (
        "minimal_payload_example.context.operation_kwargs missing for operation id: " f"{case['operation_id']}"
    )

    minimal_payload = _minimal_value_for_schema(schema)
    assert isinstance(minimal_payload, dict), (
        "Unable to synthesize minimal operation kwargs payload for operation id: " f"{case['operation_id']}"
    )

    missing_fields, type_mismatches = _preflight_blockers(schema, minimal_payload)
    assert not missing_fields and not type_mismatches, (
        "Preflight blockers for operation id "
        f"{case['operation_id']}: missing_fields={missing_fields}, type_mismatches={type_mismatches}"
    )


def test_tranche_preflight_summary_lists_failing_operation_ids() -> None:
    failing_ops: list[str] = []
    failing_ops.extend(f"{operation_id}:operation_missing_or_inactive" for operation_id in _missing_tranche_operation_ids())

    for case in _tranche_active_operation_cases():
        callability = case["callability"]
        schema = case["schema"] if isinstance(case["schema"], dict) else {}
        minimal_payload_example = callability.get("minimal_payload_example") if isinstance(callability, dict) else None
        if not isinstance(minimal_payload_example, dict):
            failing_ops.append(f"{case['operation_id']}:missing_minimal_payload_example")
            continue
        context = minimal_payload_example.get("context")
        if not isinstance(context, dict):
            failing_ops.append(f"{case['operation_id']}:missing_example_context")
            continue
        if "operation_kwargs" not in context:
            failing_ops.append(f"{case['operation_id']}:missing_example_operation_kwargs")
            continue
        minimal_payload = _minimal_value_for_schema(schema)
        if not isinstance(minimal_payload, dict):
            failing_ops.append(f"{case['operation_id']}:unable_to_synthesize_minimal_payload")
            continue
        missing_fields, type_mismatches = _preflight_blockers(schema, minimal_payload)
        if missing_fields or type_mismatches:
            failing_ops.append(
                f"{case['operation_id']}:preflight_blockers(missing={missing_fields},types={type_mismatches})"
            )

    assert not failing_ops, f"Preflight-nonpassable tranche operation ids: {sorted(set(failing_ops))}"
