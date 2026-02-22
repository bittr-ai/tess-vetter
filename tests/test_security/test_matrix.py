from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from tess_vetter.code_mode.mcp_adapter import ExecuteRequest, SearchRequest

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
                expected_error_code="sandbox_required",
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
                expected_error_code="hash_mismatch",
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
    assert response.error.code == "hash_mismatch"
    assert response.error.details["expected_payload_sha256"] == req.expected_payload_sha256
    assert response.error.details["actual_payload_sha256"] == canonical_payload_hash(req.payload)
