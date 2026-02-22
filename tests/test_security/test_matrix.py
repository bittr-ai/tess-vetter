from __future__ import annotations

import asyncio
import json
import sys
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

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
canonical_payload_hash = _fixtures.canonical_payload_hash


@pytest.mark.parametrize(
    ("scenario", "method"),
    [
        pytest.param(
            SecurityScenario(
                name="search_network_denied",
                request=SearchRequest(query="tic 123", allow_network=False),
                expected_error_code="network_denied",
            ),
            "search",
            id="search-network-denied",
        ),
        pytest.param(
            SecurityScenario(
                name="execute_sandbox_guard",
                request=ExecuteRequest(operation="run", payload={"x": 1}, sandboxed=False),
                expected_error_code="POLICY_DENIED",
            ),
            "execute",
            id="execute-sandbox-required",
        ),
        pytest.param(
            SecurityScenario(
                name="execute_hash_mismatch",
                request=ExecuteRequest(
                    operation="run",
                    payload={"x": 1},
                    sandboxed=True,
                    expected_payload_sha256=canonical_payload_hash({"x": 2}),
                ),
                expected_error_code="SCHEMA_VIOLATION_INPUT",
            ),
            "execute",
            id="execute-hash-mismatch",
        ),
    ],
)
def test_security_matrix_negative_paths(scenario: SecurityScenario, method: str) -> None:
    adapter = build_matrix_adapter()
    response = getattr(adapter, method)(scenario.request)

    assert response.ok is False
    assert response.error is not None
    assert response.error.code == scenario.expected_error_code


def test_network_denial_scenario_remains_explicit_and_stable() -> None:
    adapter = build_matrix_adapter()

    response = adapter.search(SearchRequest(query="gaia dr3", allow_network=False))

    assert response.ok is False
    assert response.error is not None
    assert response.error.code == "network_denied"


def test_hash_mismatch_error_includes_expected_and_actual_hashes() -> None:
    adapter = build_matrix_adapter()
    req = ExecuteRequest(
        operation="run",
        payload={"a": 1, "b": 2},
        sandboxed=True,
        expected_payload_sha256=canonical_payload_hash({"a": 1, "b": 3}),
    )

    response = adapter.execute(req)
    assert response.ok is False
    assert response.error is not None
    assert response.error.code == "SCHEMA_VIOLATION_INPUT"
    assert response.error.details["legacy_code"] == "hash_mismatch"
    assert response.error.details["expected_payload_sha256"] == req.expected_payload_sha256
    assert response.error.details["actual_payload_sha256"] == canonical_payload_hash(req.payload)


def test_search_response_includes_catalog_hash_and_operation_metadata_fields() -> None:
    adapter = build_matrix_adapter()
    response = adapter.search(SearchRequest(query="run", allow_network=True))

    assert response.ok is True
    assert response.catalog_version_hash == "catalog-v1"
    assert response.results
    metadata = response.results[0].metadata
    assert metadata["operation_id"] == "ops.run"
    assert metadata["operation_version"] == "1.2.3"
    assert metadata["operation_tier"] == "golden_path"
    assert "operation_tags" in metadata
    assert "operation_requirements" in metadata
    assert "operation_safety_class" in metadata

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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "TODO(runtime-capability): no deterministic per-step memory/CPU telemetry is emitted, "
        "so resource-bomb enforcement cannot be verified e2e yet."
    ),
)
def test_todo_resource_bomb_enforcement_e2e() -> None:
    raise AssertionError("Waiting on deterministic runtime resource telemetry.")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "TODO(runtime-capability): runtime lacks deterministic scheduler hooks and fairness counters "
        "across concurrent plans for budget-fairness e2e validation."
    ),
)
def test_todo_budget_fairness_across_concurrent_plans_e2e() -> None:
    raise AssertionError("Waiting on deterministic fairness instrumentation.")
