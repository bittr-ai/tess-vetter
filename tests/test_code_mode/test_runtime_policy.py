from __future__ import annotations

import asyncio

from tess_vetter.code_mode.policy import (
    DEFAULT_PROFILE_NAME,
    POLICY_PROFILE_READONLY_LOCAL,
    PROFILE_TABLE,
)
from tess_vetter.code_mode.runtime import execute


def test_default_profile_limits() -> None:
    async def _run() -> dict:
        return await execute(
            """
async def execute_plan(ops, context):
    return {"ok": True}
""",
            ops={},
            context={},
            catalog_version_hash="hash-v1",
        )

    result = asyncio.run(_run())

    assert DEFAULT_PROFILE_NAME == POLICY_PROFILE_READONLY_LOCAL
    assert result["status"] == "ok"
    assert result["trace"]["policy_profile"] == POLICY_PROFILE_READONLY_LOCAL
    assert result["trace"]["call_budget"]["max_calls"] == 20
    assert result["trace"]["call_budget"]["max_output_bytes"] == 262_144
    assert result["trace"]["call_budget"]["plan_timeout_ms"] == 90_000


def test_profile_override_clamping() -> None:
    async def _run() -> dict:
        return await execute(
            """
async def execute_plan(ops, context):
    return {"ok": True}
""",
            ops={},
            context={
                "policy_profile": POLICY_PROFILE_READONLY_LOCAL,
                "budget": {
                    "max_calls": 99,
                    "max_output_bytes": 999_999,
                    "plan_timeout_ms": 999_999,
                    "per_call_timeout_ms": 99_999,
                },
            },
            catalog_version_hash="hash-v1",
        )

    result = asyncio.run(_run())
    readonly = PROFILE_TABLE[POLICY_PROFILE_READONLY_LOCAL]

    assert result["status"] == "ok"
    assert result["trace"]["call_budget"]["max_calls"] == readonly.max_calls
    assert result["trace"]["call_budget"]["max_output_bytes"] == readonly.max_output_bytes
    assert result["trace"]["call_budget"]["plan_timeout_ms"] == readonly.plan_timeout_ms
    assert result["trace"]["call_budget"]["per_call_timeout_ms"] == readonly.per_call_timeout_ms


def test_disallowed_profile_rejected() -> None:
    async def _run() -> dict:
        return await execute(
            """
async def execute_plan(ops, context):
    return {"ok": True}
""",
            ops={},
            context={"policy_profile": "unknown_profile"},
            catalog_version_hash="hash-v1",
        )

    result = asyncio.run(_run())

    assert result["status"] == "failed"
    assert result["error"]["code"] == "POLICY_DENIED"
    assert result["trace"]["call_events"] == []


def test_hash_mismatch_short_circuit() -> None:
    state = {"called": False}

    async def _op() -> dict:
        state["called"] = True
        return {"ok": True}

    class _Ops:
        async def ping(self) -> dict:
            return await _op()

    async def _run() -> dict:
        return await execute(
            """
async def execute_plan(ops, context):
    return await ops.ping()
""",
            ops=_Ops(),
            context={"catalog_version_hash": "client-hash"},
            catalog_version_hash="server-hash",
        )

    result = asyncio.run(_run())

    assert result["status"] == "failed"
    assert result["error"]["code"] == "CATALOG_DRIFT"
    assert result["trace"]["call_budget"]["used_calls"] == 0
    assert state["called"] is False
