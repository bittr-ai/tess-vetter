from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

from tess_vetter.code_mode.mcp_adapter import (
    ErrorPayload,
    ExecuteRequest,
    ExecuteResponse,
    MCPAdapter,
    SearchRequest,
    SearchResponse,
    SearchResult,
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
                        },
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
    assert drift_response.error.details == {
        "expected_catalog_version_hash": pinned_hash,
        "received_catalog_version_hash": "catalog-v10",
    }


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
