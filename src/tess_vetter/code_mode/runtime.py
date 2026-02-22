from __future__ import annotations

import ast
import asyncio
import inspect
import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import count
from types import MappingProxyType
from typing import Any

from tess_vetter.code_mode.policy import (
    DEFAULT_PROFILE_NAME,
    NetworkBoundaryViolationError,
    PolicyProfile,
    network_boundary_guard,
    resolve_profile,
    safe_builtins,
    validate_plan_ast,
)

ERROR_CATALOG_DRIFT = "CATALOG_DRIFT"
MODE_PREFLIGHT = "preflight"
_FAIRNESS_LOCK = threading.Lock()
_FAIRNESS_PLAN_SEQUENCE = count(1)
_FAIRNESS_CALL_SEQUENCE = count(1)
_PREFLIGHT_BLOCKER_KEYS: tuple[str, ...] = (
    "missing_fields",
    "type_mismatches",
    "policy_blockers",
    "dependency_blockers",
    "constructor_blockers",
)
_CONSTRUCTOR_FIELDS: frozenset[str] = frozenset({"candidate", "lc", "stellar", "tpf"})


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
    plan_instance_id = _next_fairness_plan_id()

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
            plan_instance_id=plan_instance_id,
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
            plan_instance_id=plan_instance_id,
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
            plan_instance_id=plan_instance_id,
        )

    if _is_preflight_mode(request_context):
        blockers_payload = _collect_preflight_blockers(
            plan_code=plan_code,
            context=request_context,
            profile=profile,
        )
        preflight_result = {
            "mode": MODE_PREFLIGHT,
            "ready": _is_preflight_ready(blockers_payload),
            "operation_ids": blockers_payload["operation_ids"],
            "blockers": blockers_payload["blockers"],
        }
        if not preflight_result["ready"]:
            return {
                "status": "failed",
                "result": preflight_result,
                "error": _select_preflight_error(blockers_payload["blockers"]),
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
                    "metadata": {
                        "short_circuit": True,
                        "mode": MODE_PREFLIGHT,
                        "fairness": {"plan_instance_id": plan_instance_id},
                    },
                },
            }
        return {
            "status": "ok",
            "result": preflight_result,
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
                "metadata": {
                    "short_circuit": True,
                    "mode": MODE_PREFLIGHT,
                    "fairness": {"plan_instance_id": plan_instance_id},
                },
            },
        }

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
            plan_instance_id=plan_instance_id,
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

        plan_call_index = used_calls + 1
        used_calls += 1
        fairness_ticket = _next_fairness_call_ticket()
        started = time.perf_counter()
        event: dict[str, Any] = {
            "operation_id": op_name,
            "status": "ok",
            "plan_call_index": plan_call_index,
            "fairness_ticket": fairness_ticket,
        }
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
        except NetworkBoundaryViolationError as exc:
            event["status"] = "failed"
            event["duration_ms"] = int((time.perf_counter() - started) * 1000)
            event["error_code"] = "POLICY_DENIED"
            raise _RuntimeExecutionError(
                _error(
                    "POLICY_DENIED",
                    "Network access is not allowed under current policy profile.",
                    {
                        "policy_profile": profile.name,
                        "boundary": exc.boundary,
                        "target": exc.target,
                    },
                )
            ) from exc
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
        with network_boundary_guard(allow_network=profile.allow_network):
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
            plan_instance_id=plan_instance_id,
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
            plan_instance_id=plan_instance_id,
        )
    except NetworkBoundaryViolationError as exc:
        return _failure_response(
            _error(
                "POLICY_DENIED",
                "Network access is not allowed under current policy profile.",
                {
                    "policy_profile": profile.name,
                    "boundary": exc.boundary,
                    "target": exc.target,
                },
            ),
            profile_name=profile.name,
            profile=profile,
            budget=budget,
            call_events=call_events,
            catalog_hash=catalog_version_hash,
            used_calls=used_calls,
            plan_instance_id=plan_instance_id,
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
            plan_instance_id=plan_instance_id,
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
            "metadata": {
                "plan_duration_ms": plan_duration_ms,
                "fairness": {"plan_instance_id": plan_instance_id},
            },
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


def _is_preflight_mode(context: Mapping[str, Any]) -> bool:
    return str(context.get("mode", "")).strip().lower() == MODE_PREFLIGHT


def _is_preflight_ready(payload: Mapping[str, Any]) -> bool:
    blockers = payload.get("blockers")
    if not isinstance(blockers, Mapping):
        return True
    return not any(bool(blockers.get(key)) for key in _PREFLIGHT_BLOCKER_KEYS)


def _select_preflight_error(blockers: Mapping[str, Any]) -> dict[str, Any]:
    if blockers.get("missing_fields") or blockers.get("type_mismatches"):
        return _error(
            "SCHEMA_VIOLATION_INPUT",
            "Preflight detected schema blockers.",
            {"mode": MODE_PREFLIGHT, "blockers": dict(blockers)},
        )
    if blockers.get("policy_blockers"):
        return _error(
            "POLICY_DENIED",
            "Preflight detected policy blockers.",
            {"mode": MODE_PREFLIGHT, "blockers": dict(blockers)},
            retryable=True,
        )
    if blockers.get("constructor_blockers") or blockers.get("dependency_blockers"):
        return _error(
            "DEPENDENCY_MISSING",
            "Preflight detected dependency blockers.",
            {"mode": MODE_PREFLIGHT, "blockers": dict(blockers)},
            retryable=True,
        )
    return _error(
        "DEPENDENCY_MISSING",
        "Preflight detected dependency blockers.",
        {"mode": MODE_PREFLIGHT, "blockers": dict(blockers)},
        retryable=True,
    )


def _collect_preflight_blockers(
    *,
    plan_code: str,
    context: Mapping[str, Any],
    profile: PolicyProfile,
) -> dict[str, Any]:
    try:
        tree = ast.parse(plan_code, mode="exec")
    except SyntaxError:
        return {
            "operation_ids": [],
            "blockers": _empty_preflight_blockers(),
        }

    operation_catalog = context.get("preflight_operation_catalog")
    if isinstance(operation_catalog, Mapping):
        raw_catalog: Mapping[str, Any] = operation_catalog
    else:
        raw_catalog = {}

    call_sites = _extract_operation_call_sites(tree)
    missing_fields: list[dict[str, Any]] = []
    type_mismatches: list[dict[str, Any]] = []
    policy_blockers: list[dict[str, Any]] = []
    dependency_blockers: list[dict[str, Any]] = []
    constructor_blockers: list[dict[str, Any]] = []

    for call_site in call_sites:
        operation_id = call_site["operation_id"]
        call_ref = {"line": call_site["line"], "column": call_site["column"]}
        raw_entry = raw_catalog.get(operation_id)
        entry = dict(raw_entry) if isinstance(raw_entry, Mapping) else None

        if entry is None:
            dependency_blockers.append(
                {
                    "type": "operation_not_found",
                    "operation_id": operation_id,
                    "reason": "operation_not_found",
                    "call_site": call_ref,
                }
            )
            continue

        availability = entry.get("availability")
        if availability != "available":
            dependency_blockers.append(
                {
                    "type": "operation_unavailable",
                    "operation_id": operation_id,
                    "reason": "operation_unavailable",
                    "availability": availability,
                    "call_site": call_ref,
                }
            )

        requirements = entry.get("safety_requirements")
        if isinstance(requirements, Mapping):
            if bool(requirements.get("needs_network")) and not profile.allow_network:
                policy_blockers.append(
                    {
                        "type": "needs_network",
                        "operation_id": operation_id,
                        "requirement": "needs_network",
                        "policy_profile": profile.name,
                        "call_site": call_ref,
                    }
                )
            if bool(requirements.get("requires_human_review")) and not bool(context.get("human_review_approved")):
                policy_blockers.append(
                    {
                        "type": "requires_human_review",
                        "operation_id": operation_id,
                        "requirement": "requires_human_review",
                        "policy_profile": profile.name,
                        "call_site": call_ref,
                    }
                )
            if bool(requirements.get("needs_secrets")) and not bool(context.get("secrets_available")):
                policy_blockers.append(
                    {
                        "type": "needs_secrets",
                        "operation_id": operation_id,
                        "requirement": "needs_secrets",
                        "policy_profile": profile.name,
                        "call_site": call_ref,
                    }
                )
            if bool(requirements.get("needs_filesystem")) and not bool(context.get("filesystem_available", True)):
                policy_blockers.append(
                    {
                        "type": "needs_filesystem",
                        "operation_id": operation_id,
                        "requirement": "needs_filesystem",
                        "policy_profile": profile.name,
                        "call_site": call_ref,
                    }
                )

        if call_site["has_dynamic_kwargs"]:
            continue

        args = call_site["args"]
        if not isinstance(args, Mapping):
            continue

        required_fields = entry.get("required_fields")
        required: tuple[str, ...]
        if isinstance(required_fields, list):
            required = tuple(str(field) for field in required_fields)
        else:
            required = ()

        for field_name in required:
            if field_name not in args:
                missing_fields.append(
                    {
                        "operation_id": operation_id,
                        "field": field_name,
                        "call_site": call_ref,
                    }
                )
                if field_name in _CONSTRUCTOR_FIELDS:
                    constructor_blockers.append(
                        {
                            "type": "constructor_missing",
                            "operation_id": operation_id,
                            "field": field_name,
                            "reason": "constructor_missing",
                            "call_site": call_ref,
                        }
                    )

        field_types = entry.get("field_types")
        if isinstance(field_types, Mapping):
            for field_name, expected_raw in sorted(field_types.items()):
                if field_name not in args:
                    continue
                value = args[field_name]
                expected_types = _normalize_expected_types(expected_raw)
                if not expected_types:
                    continue
                if _value_matches_expected_types(value, expected_types):
                    continue
                type_mismatches.append(
                    {
                        "operation_id": operation_id,
                        "field": field_name,
                        "expected_types": list(expected_types),
                        "received_type": type(value).__name__,
                        "call_site": call_ref,
                    }
                )

    return {
        "operation_ids": sorted({call["operation_id"] for call in call_sites}),
        "blockers": {
            "missing_fields": sorted(
                missing_fields,
                key=lambda row: (
                    str(row.get("operation_id")),
                    str(row.get("field")),
                    int(row.get("call_site", {}).get("line", 0)),
                    int(row.get("call_site", {}).get("column", 0)),
                ),
            ),
            "type_mismatches": sorted(
                type_mismatches,
                key=lambda row: (
                    str(row.get("operation_id")),
                    str(row.get("field")),
                    int(row.get("call_site", {}).get("line", 0)),
                    int(row.get("call_site", {}).get("column", 0)),
                ),
            ),
            "policy_blockers": sorted(
                policy_blockers,
                key=lambda row: (
                    str(row.get("operation_id")),
                    str(row.get("type")),
                    str(row.get("requirement")),
                    int(row.get("call_site", {}).get("line", 0)),
                    int(row.get("call_site", {}).get("column", 0)),
                ),
            ),
            "dependency_blockers": sorted(
                dependency_blockers,
                key=lambda row: (
                    str(row.get("operation_id")),
                    str(row.get("type")),
                    str(row.get("reason")),
                    int(row.get("call_site", {}).get("line", 0)),
                    int(row.get("call_site", {}).get("column", 0)),
                ),
            ),
            "constructor_blockers": sorted(
                constructor_blockers,
                key=lambda row: (
                    str(row.get("operation_id")),
                    str(row.get("field")),
                    str(row.get("type")),
                    int(row.get("call_site", {}).get("line", 0)),
                    int(row.get("call_site", {}).get("column", 0)),
                ),
            ),
        },
    }


def _empty_preflight_blockers() -> dict[str, list[dict[str, Any]]]:
    return {key: [] for key in _PREFLIGHT_BLOCKER_KEYS}


def _extract_operation_call_sites(tree: ast.AST) -> list[dict[str, Any]]:
    call_sites: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        operation_id = _resolve_operation_id(node.func)
        if operation_id is None:
            continue
        kwargs, has_dynamic_kwargs = _extract_literal_kwargs(node)
        call_sites.append(
            {
                "operation_id": operation_id,
                "args": kwargs,
                "has_dynamic_kwargs": has_dynamic_kwargs,
                "line": int(getattr(node, "lineno", 0)),
                "column": int(getattr(node, "col_offset", 0)),
            }
        )
    return sorted(call_sites, key=lambda row: (str(row["operation_id"]), int(row["line"]), int(row["column"])))


def _resolve_operation_id(func: ast.AST) -> str | None:
    if isinstance(func, ast.Subscript) and isinstance(func.value, ast.Name) and func.value.id == "ops":
        key_value = _literal_eval_or_none(func.slice)
        if isinstance(key_value, str):
            return key_value
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "ops":
        return str(func.attr)
    return None


def _extract_literal_kwargs(node: ast.Call) -> tuple[dict[str, Any], bool]:
    kwargs: dict[str, Any] = {}
    has_dynamic_kwargs = False
    for keyword in node.keywords:
        if keyword.arg is None:
            has_dynamic_kwargs = True
            continue
        literal = _literal_eval_or_none(keyword.value)
        if literal is None and not _is_none_literal(keyword.value):
            has_dynamic_kwargs = True
            continue
        kwargs[keyword.arg] = literal
    return kwargs, has_dynamic_kwargs


def _is_none_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _literal_eval_or_none(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _normalize_expected_types(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, list):
        return tuple(sorted({str(item) for item in raw if isinstance(item, str)}))
    return ()


def _value_matches_expected_types(value: Any, expected_types: tuple[str, ...]) -> bool:
    for expected in expected_types:
        if expected == "string" and isinstance(value, str):
            return True
        if expected == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if expected == "number" and ((isinstance(value, int) and not isinstance(value, bool)) or isinstance(value, float)):
            return True
        if expected == "boolean" and isinstance(value, bool):
            return True
        if expected == "array" and isinstance(value, list):
            return True
        if expected == "object" and isinstance(value, dict):
            return True
        if expected == "null" and value is None:
            return True
    return False


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


def _error(
    code: str,
    message: str,
    details: Mapping[str, Any],
    *,
    retryable: bool = False,
) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "retryable": retryable,
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
    plan_instance_id: int,
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
            "metadata": {
                "short_circuit": True,
                "fairness": {"plan_instance_id": plan_instance_id},
            },
        },
    }


def _next_fairness_plan_id() -> int:
    with _FAIRNESS_LOCK:
        return next(_FAIRNESS_PLAN_SEQUENCE)


def _next_fairness_call_ticket() -> int:
    with _FAIRNESS_LOCK:
        return next(_FAIRNESS_CALL_SEQUENCE)


__all__ = [
    "BudgetConfig",
    "ERROR_CATALOG_DRIFT",
    "apply_budget_clamp",
    "execute",
    "validate_catalog_hash",
]
