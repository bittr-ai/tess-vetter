from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from tess_vetter.code_mode.policy import (
    DEFAULT_PROFILE_NAME,
    PolicyProfile,
    resolve_profile,
    safe_builtins,
    validate_plan_ast,
)

ERROR_CATALOG_DRIFT = "CATALOG_DRIFT"


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    max_calls: int
    max_output_bytes: int
    plan_timeout_ms: int
    per_call_timeout_ms: int

    @classmethod
    def from_context(cls, context: Mapping[str, Any], profile: PolicyProfile) -> BudgetConfig:
        budget = context.get("budget") if isinstance(context.get("budget"), Mapping) else {}
        return cls(
            max_calls=_clamp_budget_value(
                _first_int(budget, "max_calls", "calls"),
                profile.max_calls,
            ),
            max_output_bytes=_clamp_budget_value(
                _first_int(budget, "max_output_bytes", "output_bytes"),
                profile.max_output_bytes,
            ),
            plan_timeout_ms=_clamp_budget_value(
                _first_int(budget, "plan_timeout_ms", "timeout_ms"),
                profile.plan_timeout_ms,
            ),
            per_call_timeout_ms=_clamp_budget_value(
                _first_int(budget, "per_call_timeout_ms"),
                profile.per_call_timeout_ms,
            ),
        )


def apply_budget_clamp(requested: Mapping[str, Any], profile: PolicyProfile) -> BudgetConfig:
    return BudgetConfig.from_context({"budget": dict(requested)}, profile)


def validate_catalog_hash(request_hash: str | None, current_hash: str | None) -> dict[str, Any] | None:
    if request_hash is None or current_hash is None:
        return None
    if request_hash == current_hash:
        return None
    return _error(
        ERROR_CATALOG_DRIFT,
        "Catalog hash mismatch; refresh catalog before execution.",
        {
            "expected_catalog_version_hash": current_hash,
            "received_catalog_version_hash": request_hash,
        },
    )


async def execute(
    plan_code: str,
    ops: Any,
    context: Mapping[str, Any] | None = None,
    *,
    catalog_version_hash: str | None = None,
) -> dict[str, Any]:
    request_context = dict(context or {})

    try:
        profile = resolve_profile(_pick_profile_name(request_context))
    except ValueError as exc:
        profile_name = str(exc)
        return _failure_response(
            _error(
                "POLICY_DENIED",
                "Requested policy profile is not allowed.",
                {"requested_profile": profile_name},
            ),
            profile_name=profile_name,
            profile=None,
            budget=None,
            call_events=[],
            catalog_hash=catalog_version_hash,
            used_calls=0,
        )

    budget = BudgetConfig.from_context(request_context, profile)
    call_events: list[dict[str, Any]] = []
    used_calls = 0

    hash_error = validate_catalog_hash(request_context.get("catalog_version_hash"), catalog_version_hash)
    if hash_error is not None:
        return _failure_response(
            hash_error,
            profile_name=profile.name,
            profile=profile,
            budget=budget,
            call_events=call_events,
            catalog_hash=catalog_version_hash,
            used_calls=used_calls,
        )

    ast_error = validate_plan_ast(plan_code)
    if ast_error is not None:
        return _failure_response(
            ast_error,
            profile_name=profile.name,
            profile=profile,
            budget=budget,
            call_events=call_events,
            catalog_hash=catalog_version_hash,
            used_calls=used_calls,
        )

    plan_fn_or_error = _load_execute_plan(plan_code)
    if isinstance(plan_fn_or_error, dict):
        return _failure_response(
            plan_fn_or_error,
            profile_name=profile.name,
            profile=profile,
            budget=budget,
            call_events=call_events,
            catalog_hash=catalog_version_hash,
            used_calls=used_calls,
        )

    execute_plan = plan_fn_or_error

    async def _invoke_with_per_call_timeout(op: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        timeout_seconds = budget.per_call_timeout_ms / 1000.0
        if _is_async_callable(op):
            result = await asyncio.wait_for(op(*args, **kwargs), timeout=timeout_seconds)
        else:
            result = await asyncio.wait_for(
                asyncio.to_thread(op, *args, **kwargs),
                timeout=timeout_seconds,
            )

        if inspect.isawaitable(result):
            return await asyncio.wait_for(result, timeout=timeout_seconds)
        return result

    async def invoke_op(op_name: str, op: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        nonlocal used_calls
        if used_calls >= budget.max_calls:
            raise _RuntimeExecutionError(
                _error(
                    "CALL_LIMIT_EXCEEDED",
                    "Operation call limit exceeded.",
                    {"max_calls": budget.max_calls, "used_calls": used_calls},
                )
            )

        used_calls += 1
        started = time.perf_counter()
        event: dict[str, Any] = {"operation_id": op_name, "status": "ok"}
        try:
            result = await _invoke_with_per_call_timeout(op, *args, **kwargs)
            event["duration_ms"] = int((time.perf_counter() - started) * 1000)
            return result
        except TimeoutError as exc:
            event["status"] = "timeout"
            event["duration_ms"] = int((time.perf_counter() - started) * 1000)
            event["error_code"] = "TIMEOUT_EXCEEDED"
            raise _RuntimeExecutionError(
                _error(
                    "TIMEOUT_EXCEEDED",
                    "Operation timed out.",
                    {
                        "operation_id": op_name,
                        "per_call_timeout_ms": budget.per_call_timeout_ms,
                    },
                )
            ) from exc
        except _RuntimeExecutionError:
            event["status"] = "failed"
            event["duration_ms"] = int((time.perf_counter() - started) * 1000)
            raise
        except Exception as exc:
            event["status"] = "failed"
            event["duration_ms"] = int((time.perf_counter() - started) * 1000)
            event["error_code"] = "OPERATION_RUNTIME_ERROR"
            raise _RuntimeExecutionError(
                _error(
                    "OPERATION_RUNTIME_ERROR",
                    "Operation raised an exception.",
                    {
                        "operation_id": op_name,
                        "exception_type": type(exc).__name__,
                        "exception_text": str(exc),
                    },
                )
            ) from exc
        finally:
            call_events.append(event)

    proxy_ops = _OpsProxy(ops, invoke_op)

    try:
        started = time.perf_counter()
        result = await asyncio.wait_for(
            execute_plan(proxy_ops, MappingProxyType(request_context)),
            timeout=budget.plan_timeout_ms / 1000.0,
        )
        plan_duration_ms = int((time.perf_counter() - started) * 1000)
    except TimeoutError:
        return _failure_response(
            _error(
                "TIMEOUT_EXCEEDED",
                "Plan timeout exceeded.",
                {"plan_timeout_ms": budget.plan_timeout_ms},
            ),
            profile_name=profile.name,
            profile=profile,
            budget=budget,
            call_events=call_events,
            catalog_hash=catalog_version_hash,
            used_calls=used_calls,
        )
    except _RuntimeExecutionError as exc:
        return _failure_response(
            exc.error,
            profile_name=profile.name,
            profile=profile,
            budget=budget,
            call_events=call_events,
            catalog_hash=catalog_version_hash,
            used_calls=used_calls,
        )

    serialized = json.dumps(result, sort_keys=True, default=str).encode("utf-8")
    if len(serialized) > budget.max_output_bytes:
        return _failure_response(
            _error(
                "OUTPUT_LIMIT_EXCEEDED",
                "Output payload exceeds configured limit.",
                {
                    "max_output_bytes": budget.max_output_bytes,
                    "actual_output_bytes": len(serialized),
                },
            ),
            profile_name=profile.name,
            profile=profile,
            budget=budget,
            call_events=call_events,
            catalog_hash=catalog_version_hash,
            used_calls=used_calls,
        )

    return {
        "status": "ok",
        "result": result,
        "error": None,
        "catalog_version_hash": catalog_version_hash,
        "trace": {
            "policy_profile": profile.name,
            "policy_limits": _policy_limits(profile),
            "call_budget": {
                "max_calls": budget.max_calls,
                "used_calls": used_calls,
                "max_output_bytes": budget.max_output_bytes,
                "plan_timeout_ms": budget.plan_timeout_ms,
                "per_call_timeout_ms": budget.per_call_timeout_ms,
            },
            "call_events": call_events,
            "metadata": {"plan_duration_ms": plan_duration_ms},
        },
    }


class _RuntimeExecutionError(Exception):
    def __init__(self, error: dict[str, Any]) -> None:
        super().__init__(error["message"])
        self.error = error


class _OpsProxy:
    def __init__(self, raw_ops: Any, invoker: Callable[..., Any]) -> None:
        self._raw_ops = raw_ops
        self._invoker = invoker

    def __getattr__(self, item: str) -> Any:
        raw = getattr(self._raw_ops, item)
        if callable(raw):
            return _CallableOp(item, raw, self._invoker)
        return raw

    def __getitem__(self, item: str) -> Any:
        raw = self._raw_ops[item]
        if callable(raw):
            return _CallableOp(str(item), raw, self._invoker)
        return raw


class _CallableOp:
    def __init__(self, op_name: str, op: Callable[..., Any], invoker: Callable[..., Any]) -> None:
        self._op_name = op_name
        self._op = op
        self._invoker = invoker

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return await self._invoker(self._op_name, self._op, *args, **kwargs)


def _load_execute_plan(plan_code: str) -> Callable[..., Any] | dict[str, Any]:
    try:
        compiled = compile(plan_code, filename="<code_mode_plan>", mode="exec")
    except SyntaxError as exc:
        return _error(
            "PLAN_PARSE_ERROR",
            "Plan code could not be compiled.",
            {"line": exc.lineno, "column": exc.offset, "error_text": exc.msg},
        )

    globals_map: dict[str, Any] = {"__builtins__": safe_builtins()}
    locals_map: dict[str, Any] = {}
    try:
        exec(compiled, globals_map, locals_map)
    except Exception as exc:
        return _error(
            "PLAN_PARSE_ERROR",
            "Plan failed during module initialization.",
            {
                "exception_type": type(exc).__name__,
                "exception_text": str(exc),
            },
        )

    plan_fn = locals_map.get("execute_plan", globals_map.get("execute_plan"))
    if plan_fn is None:
        return _error(
            "PLAN_PARSE_ERROR",
            "Plan must define async execute_plan(ops, context).",
            {},
        )
    if not inspect.iscoroutinefunction(plan_fn):
        return _error(
            "PLAN_PARSE_ERROR",
            "execute_plan must be declared as async.",
            {"function_type": type(plan_fn).__name__},
        )

    return plan_fn


def _is_async_callable(op: Callable[..., Any]) -> bool:
    if inspect.iscoroutinefunction(op):
        return True
    if not callable(op):
        return False
    return inspect.iscoroutinefunction(type(op).__call__)


def _pick_profile_name(context: Mapping[str, Any]) -> str:
    if isinstance(context.get("policy_profile"), str):
        return str(context["policy_profile"])
    if isinstance(context.get("safety_profile"), str):
        return str(context["safety_profile"])
    return DEFAULT_PROFILE_NAME


def _first_int(data: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key in data:
            value = data[key]
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return None
            if parsed > 0:
                return parsed
            return None
    return None


def _clamp_budget_value(requested: int | None, ceiling: int) -> int:
    if requested is None:
        return ceiling
    return min(requested, ceiling)


def _error(code: str, message: str, details: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "retryable": False,
        "details": dict(details),
    }


def _policy_limits(profile: PolicyProfile) -> dict[str, Any]:
    return {
        "max_calls": profile.max_calls,
        "max_output_bytes": profile.max_output_bytes,
        "plan_timeout_ms": profile.plan_timeout_ms,
        "per_call_timeout_ms": profile.per_call_timeout_ms,
        "allow_network": profile.allow_network,
    }


def _failure_response(
    error: dict[str, Any],
    *,
    profile_name: str,
    profile: PolicyProfile | None,
    budget: BudgetConfig | None,
    call_events: list[dict[str, Any]],
    catalog_hash: str | None,
    used_calls: int,
) -> dict[str, Any]:
    limits = _policy_limits(profile) if profile is not None else None
    if budget is None:
        call_budget = None
    else:
        call_budget = {
            "max_calls": budget.max_calls,
            "used_calls": used_calls,
            "max_output_bytes": budget.max_output_bytes,
            "plan_timeout_ms": budget.plan_timeout_ms,
            "per_call_timeout_ms": budget.per_call_timeout_ms,
        }

    return {
        "status": "failed",
        "result": None,
        "error": error,
        "catalog_version_hash": catalog_hash,
        "trace": {
            "policy_profile": profile_name,
            "policy_limits": limits,
            "call_budget": call_budget,
            "call_events": call_events,
            "metadata": {"short_circuit": True},
        },
    }


__all__ = [
    "BudgetConfig",
    "ERROR_CATALOG_DRIFT",
    "apply_budget_clamp",
    "execute",
    "validate_catalog_hash",
]
