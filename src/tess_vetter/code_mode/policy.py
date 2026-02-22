from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

POLICY_PROFILE_READONLY_LOCAL = "readonly_local"
POLICY_PROFILE_NETWORK_ALLOWED = "network_allowed"
DEFAULT_PROFILE_NAME = POLICY_PROFILE_READONLY_LOCAL

DEFAULT_PER_CALL_TIMEOUT_MS = 25_000
READONLY_LOCAL_MAX_CALLS = 20
READONLY_LOCAL_MAX_OUTPUT_BYTES = 262_144
READONLY_LOCAL_PLAN_TIMEOUT_MS = 90_000
NETWORK_ALLOWED_MAX_CALLS = 40
NETWORK_ALLOWED_MAX_OUTPUT_BYTES = 524_288
NETWORK_ALLOWED_PLAN_TIMEOUT_MS = 150_000

SAFE_BUILTINS_ALLOWLIST = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

_BANNED_NAME_IDS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "globals",
    "input",
    "locals",
    "open",
    "vars",
}


@dataclass(frozen=True, slots=True)
class PolicyProfile:
    name: str
    max_calls: int
    max_output_bytes: int
    plan_timeout_ms: int
    per_call_timeout_ms: int
    allow_network: bool


PROFILE_TABLE: dict[str, PolicyProfile] = {
    POLICY_PROFILE_READONLY_LOCAL: PolicyProfile(
        name=POLICY_PROFILE_READONLY_LOCAL,
        max_calls=READONLY_LOCAL_MAX_CALLS,
        max_output_bytes=READONLY_LOCAL_MAX_OUTPUT_BYTES,
        plan_timeout_ms=READONLY_LOCAL_PLAN_TIMEOUT_MS,
        per_call_timeout_ms=DEFAULT_PER_CALL_TIMEOUT_MS,
        allow_network=False,
    ),
    POLICY_PROFILE_NETWORK_ALLOWED: PolicyProfile(
        name=POLICY_PROFILE_NETWORK_ALLOWED,
        max_calls=NETWORK_ALLOWED_MAX_CALLS,
        max_output_bytes=NETWORK_ALLOWED_MAX_OUTPUT_BYTES,
        plan_timeout_ms=NETWORK_ALLOWED_PLAN_TIMEOUT_MS,
        per_call_timeout_ms=DEFAULT_PER_CALL_TIMEOUT_MS,
        allow_network=True,
    ),
}


def resolve_profile(profile_name: str | None) -> PolicyProfile:
    selected = profile_name or DEFAULT_PROFILE_NAME
    if selected not in PROFILE_TABLE:
        raise ValueError(selected)
    return PROFILE_TABLE[selected]


def safe_builtins() -> dict[str, Any]:
    return dict(SAFE_BUILTINS_ALLOWLIST)


def validate_plan_ast(plan_code: str) -> dict[str, Any] | None:
    try:
        tree = ast.parse(plan_code, mode="exec")
    except SyntaxError as exc:
        return {
            "code": "PLAN_PARSE_ERROR",
            "message": "Plan code is not valid Python.",
            "retryable": False,
            "details": {
                "line": exc.lineno,
                "column": exc.offset,
                "error_text": exc.msg,
            },
        }

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
            return {
                "code": "POLICY_DENIED",
                "message": "Plan contains a restricted Python construct.",
                "retryable": False,
                "details": {"node_type": node.__class__.__name__},
            }
        if isinstance(node, ast.Name) and node.id in _BANNED_NAME_IDS:
            return {
                "code": "POLICY_DENIED",
                "message": "Plan references a restricted builtin.",
                "retryable": False,
                "details": {"name": node.id},
            }
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return {
                "code": "POLICY_DENIED",
                "message": "Plan references a restricted attribute.",
                "retryable": False,
                "details": {"attribute": node.attr},
            }
    return None


__all__ = [
    "DEFAULT_PER_CALL_TIMEOUT_MS",
    "DEFAULT_PROFILE_NAME",
    "NETWORK_ALLOWED_MAX_CALLS",
    "NETWORK_ALLOWED_MAX_OUTPUT_BYTES",
    "NETWORK_ALLOWED_PLAN_TIMEOUT_MS",
    "POLICY_PROFILE_NETWORK_ALLOWED",
    "POLICY_PROFILE_READONLY_LOCAL",
    "PROFILE_TABLE",
    "PolicyProfile",
    "READONLY_LOCAL_MAX_CALLS",
    "READONLY_LOCAL_MAX_OUTPUT_BYTES",
    "READONLY_LOCAL_PLAN_TIMEOUT_MS",
    "resolve_profile",
    "safe_builtins",
    "validate_plan_ast",
]
