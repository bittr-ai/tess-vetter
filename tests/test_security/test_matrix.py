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


def test_catalog_hash_drift_denied_before_plan_execution() -> None:
    called = False

    class _Ops:
        def ping(self) -> dict:
            nonlocal called
            called = True
            return {"ok": True}

    async def _run() -> dict:
        return await execute(
            """
async def execute_plan(ops, context):
    return await ops.ping()
""",
            ops=_Ops(),
            context={"catalog_version_hash": "catalog-request-v1"},
            catalog_version_hash="catalog-runtime-v1",
        )

    result = asyncio.run(_run())

    assert result["status"] == "failed"
    assert result["error"]["code"] == "CATALOG_DRIFT"
    assert result["error"]["details"] == {
        "expected_catalog_version_hash": "catalog-runtime-v1",
        "received_catalog_version_hash": "catalog-request-v1",
    }
    assert result["trace"]["call_budget"]["used_calls"] == 0
    assert result["trace"]["call_events"] == []
    assert called is False


def test_budget_fairness_across_concurrent_plans_e2e() -> None:
    class _CoordinatedOps:
        def __init__(self, allow_a_second_call: asyncio.Event) -> None:
            self.allow_a_second_call = allow_a_second_call
            self.a_started = asyncio.Event()
            self.allow_b_first = asyncio.Event()
            self.a_second_done = asyncio.Event()
            self.allow_b_second = asyncio.Event()

        async def a_first(self) -> dict:
            self.a_started.set()
            self.allow_b_first.set()
            return {"step": "a_first"}

        async def a_second(self) -> dict:
            self.a_second_done.set()
            self.allow_b_second.set()
            return {"step": "a_second"}

        async def b_first(self) -> dict:
            await self.allow_b_first.wait()
            self.allow_a_second_call.set()
            await self.a_second_done.wait()
            return {"step": "b_first"}

        async def b_second(self) -> dict:
            await self.allow_b_second.wait()
            return {"step": "b_second"}

    async def _run() -> tuple[dict, dict]:
        allow_a_second_call = asyncio.Event()
        ops = _CoordinatedOps(allow_a_second_call)

        task_a = asyncio.create_task(
            execute(
                """
async def execute_plan(ops, context):
    await ops.a_first()
    await context["allow_a_second_call"].wait()
    await ops.a_second()
    return {"plan": "a"}
""",
                ops=ops,
                context={"allow_a_second_call": allow_a_second_call},
                catalog_version_hash="hash-v1",
            )
        )
        await ops.a_started.wait()
        task_b = asyncio.create_task(
            execute(
                """
async def execute_plan(ops, context):
    await ops.b_first()
    await ops.b_second()
    return {"plan": "b"}
""",
                ops=ops,
                context={"allow_a_second_call": allow_a_second_call},
                catalog_version_hash="hash-v1",
            )
        )

        return await asyncio.gather(task_a, task_b)

    result_a, result_b = asyncio.run(_run())

    assert result_a["status"] == "ok"
    assert result_b["status"] == "ok"

    events_a = result_a["trace"]["call_events"]
    events_b = result_b["trace"]["call_events"]
    tickets = {
        events_a[0]["operation_id"]: events_a[0]["fairness_ticket"],
        events_b[0]["operation_id"]: events_b[0]["fairness_ticket"],
        events_a[1]["operation_id"]: events_a[1]["fairness_ticket"],
        events_b[1]["operation_id"]: events_b[1]["fairness_ticket"],
    }
    expected = [
        tickets["a_first"],
        tickets["b_first"],
        tickets["a_second"],
        tickets["b_second"],
    ]
    assert len(set(expected)) == 4
    assert expected == sorted(expected)

    plan_id_a = result_a["trace"]["metadata"]["fairness"]["plan_instance_id"]
    plan_id_b = result_b["trace"]["metadata"]["fairness"]["plan_instance_id"]
    assert plan_id_a < plan_id_b
