from __future__ import annotations

import asyncio
import json
import sys
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest

from tess_vetter.code_mode.mcp_adapter import ExecuteRequest, SearchRequest
from tess_vetter.code_mode.runtime import execute
from tess_vetter.code_mode.trace import build_runtime_trace_metadata

_FIXTURE_SPEC = spec_from_file_location("security_fixtures", Path(__file__).with_name("fixtures.py"))
assert _FIXTURE_SPEC is not None and _FIXTURE_SPEC.loader is not None
_fixtures = module_from_spec(_FIXTURE_SPEC)
sys.modules[_FIXTURE_SPEC.name] = _fixtures
_FIXTURE_SPEC.loader.exec_module(_fixtures)

SecurityScenario = _fixtures.SecurityScenario
build_matrix_adapter = _fixtures.build_matrix_adapter


@pytest.mark.parametrize(
    ("scenario", "method"),
    [
        pytest.param(
            SecurityScenario(
                name="search_query_type_violation",
                request=SimpleNamespace(query=123, limit=10, tags=None),
                expected_error_code="SCHEMA_VIOLATION_INPUT",
            ),
            "search",
            id="search-query-type-violation",
        ),
        pytest.param(
            SecurityScenario(
                name="execute_plan_code_type_violation",
                request=SimpleNamespace(
                    plan_code=123,
                    context={"candidate": "tic-123"},
                    catalog_version_hash="catalog-v1",
                ),
                expected_error_code="SCHEMA_VIOLATION_INPUT",
            ),
            "execute",
            id="execute-plan-code-type-violation",
        ),
        pytest.param(
            SecurityScenario(
                name="execute_context_type_violation",
                request=SimpleNamespace(
                    plan_code="async def execute_plan(ops, context):\n    return {'ok': True}\n",
                    context=[],
                    catalog_version_hash="catalog-v1",
                ),
                expected_error_code="SCHEMA_VIOLATION_INPUT",
            ),
            "execute",
            id="execute-context-type-violation",
        ),
    ],
)
def test_security_matrix_negative_paths(scenario: SecurityScenario, method: str) -> None:
    adapter = build_matrix_adapter()
    response = getattr(adapter, method)(scenario.request)

    assert response.error is not None
    assert response.error.code == scenario.expected_error_code


def test_search_shape_violation_remains_explicit_and_stable() -> None:
    adapter = build_matrix_adapter()

    response = adapter.search(SimpleNamespace(query=1, limit=10, tags=None))

    assert response.error is not None
    assert response.error.code == "SCHEMA_VIOLATION_INPUT"
    assert "query" in response.error.message


def test_search_response_includes_catalog_hash_and_operation_metadata_fields() -> None:
    adapter = build_matrix_adapter()
    response = adapter.search(SearchRequest(query="run", limit=5, tags=[]))

    assert response.error is None
    assert response.catalog_version_hash == "catalog-v1"
    assert response.total == 1
    assert response.results
    metadata = response.results[0].metadata
    assert metadata["operation_id"] == "ops.run"
    assert metadata["operation_version"] == "1.2.3"
    assert metadata["operation_tier"] == "golden_path"
    assert "operation_tags" in metadata
    assert "operation_requirements" in metadata
    assert "operation_safety_class" in metadata
    assert metadata["is_callable"] is True

    serialized = response.to_dict()
    assert serialized["catalog_version_hash"] == "catalog-v1"


def test_sandbox_escape_attempt_denied_by_ast_policy() -> None:
    async def _run() -> dict:
        return await execute(
            """
import os

async def execute_plan(ops, context):
    return {"cwd": os.getcwd()}
""",
            ops={},
            context={},
            catalog_version_hash="hash-v1",
        )

    result = asyncio.run(_run())

    assert result["status"] == "failed"
    assert result["error"]["code"] == "POLICY_DENIED"
    assert result["error"]["details"]["node_type"] == "Import"


def test_runtime_readonly_local_denies_transitive_urllib_boundary() -> None:
    class _Ops:
        def fetch_remote(self) -> dict:
            import urllib.request

            urllib.request.urlopen("https://example.com")  # pragma: no cover - guarded path
            return {"ok": True}

    async def _run() -> dict:
        return await execute(
            """
async def execute_plan(ops, context):
    return await ops.fetch_remote()
""",
            ops=_Ops(),
            context={"policy_profile": "readonly_local"},
            catalog_version_hash="hash-v1",
        )

    result = asyncio.run(_run())

    assert result["status"] == "failed"
    assert result["error"]["code"] == "POLICY_DENIED"
    assert result["error"]["details"] == {
        "policy_profile": "readonly_local",
        "boundary": "urllib.request.urlopen",
        "target": "https://example.com",
    }


def test_trace_metadata_determinism_basic_check() -> None:
    kwargs = {
        "trace_id": "trace-security-matrix",
        "policy_profile": "readonly_local",
        "network_ok": False,
        "catalog_version_hash": "catalog-v1",
        "timestamp": 1_706_000_000.0,
    }

    trace_a = build_runtime_trace_metadata(**kwargs)
    trace_b = build_runtime_trace_metadata(**kwargs)

    assert trace_a == trace_b
    assert sha256(json.dumps(trace_a, sort_keys=True).encode("utf-8")).hexdigest() == sha256(
        json.dumps(trace_b, sort_keys=True).encode("utf-8")
    ).hexdigest()


def test_output_payload_limit_enforced_e2e() -> None:
    async def _run() -> dict:
        return await execute(
            """
async def execute_plan(ops, context):
    return {"payload": "x" * 1024}
""",
            ops={},
            context={"budget": {"max_output_bytes": 64}},
            catalog_version_hash="hash-v1",
        )

    result = asyncio.run(_run())

    assert result["status"] == "failed"
    assert result["error"]["code"] == "OUTPUT_LIMIT_EXCEEDED"
    assert result["error"]["details"]["max_output_bytes"] == 64
    assert result["error"]["details"]["actual_output_bytes"] > 64
    assert result["trace"]["call_budget"]["max_output_bytes"] == 64


def test_search_to_execute_hash_pinning_accepts_match_and_rejects_drift() -> None:
    adapter = build_matrix_adapter()
    search_response = adapter.search(SearchRequest(query="run", limit=5, tags=[]))

    assert search_response.error is None
    assert search_response.catalog_version_hash == "catalog-v1"

    called = False

    class _Ops:
        def ping(self) -> dict:
            nonlocal called
            called = True
            return {"ok": True}

    plan_code = """
async def execute_plan(ops, context):
    return await ops.ping()
"""

    pinned = asyncio.run(
        execute(
            plan_code,
            ops=_Ops(),
            context={"catalog_version_hash": search_response.catalog_version_hash},
            catalog_version_hash=search_response.catalog_version_hash,
        )
    )

    assert pinned["status"] == "ok"
    assert pinned["result"] == {"ok": True}
    assert called is True

    called = False
    drifted = asyncio.run(
        execute(
            plan_code,
            ops=_Ops(),
            context={"catalog_version_hash": search_response.catalog_version_hash},
            catalog_version_hash="catalog-v2",
        )
    )

    assert drifted["status"] == "failed"
    assert drifted["error"]["code"] == "CATALOG_DRIFT"
    assert drifted["error"]["details"] == {
        "expected_catalog_version_hash": "catalog-v2",
        "received_catalog_version_hash": "catalog-v1",
    }
    assert called is False


def test_retryable_error_key_is_consistent_across_runtime_and_adapter_serialization() -> None:
    runtime_parse_error = asyncio.run(
        execute(
            "def not_python(:\n    pass",
            ops={},
            context={},
            catalog_version_hash="catalog-v1",
        )
    )
    assert runtime_parse_error["status"] == "failed"
    assert "retryable" in runtime_parse_error["error"]
    assert "retriable" not in runtime_parse_error["error"]

    adapter = build_matrix_adapter()
    adapter_shape_error = adapter.search(SimpleNamespace(query=1, limit=10, tags=None))
    assert adapter_shape_error.error is not None

    serialized = adapter_shape_error.to_dict()
    assert serialized["error"] is not None
    assert "retryable" in serialized["error"]
    assert "retriable" not in serialized["error"]


def test_adapter_execute_response_shape_matches_runtime_style() -> None:
    adapter = build_matrix_adapter()

    response = adapter.execute(
        ExecuteRequest(
            plan_code="async def execute_plan(ops, context):\n    return {'ok': True}\n",
            context={"candidate": "tic-123"},
            catalog_version_hash="catalog-v1",
        )
    )

    assert response.status == "ok"
    assert response.error is None
    assert response.catalog_version_hash == "catalog-v1"
