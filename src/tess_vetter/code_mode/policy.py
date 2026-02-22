from __future__ import annotations

import ast
import socket
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
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


@dataclass(frozen=True, slots=True)
class NetworkBoundaryViolationError(Exception):
    boundary: str
    target: str

    def __str__(self) -> str:
        return f"{self.boundary}:{self.target}"


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


@contextmanager
def network_boundary_guard(*, allow_network: bool) -> Iterator[None]:
    if allow_network:
        yield
        return

    original_socket = socket.socket
    original_create_connection = socket.create_connection
    original_urlopen = urllib.request.urlopen
    original_opener_open = urllib.request.OpenerDirector.open

    def _deny(boundary: str, target: Any) -> None:
        raise NetworkBoundaryViolationError(boundary=boundary, target=str(target))

    class _GuardedSocket(original_socket):
        def connect(self, address: Any) -> None:
            _deny("socket.connect", address)

        def connect_ex(self, address: Any) -> int:
            _deny("socket.connect_ex", address)

    def _guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        _deny("socket.create_connection", address)

    def _guarded_urlopen(url: Any, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        _deny("urllib.request.urlopen", url)

    def _guarded_opener_open(self: Any, fullurl: Any, *args: Any, **kwargs: Any) -> Any:
        del self, args, kwargs
        _deny("urllib.request.OpenerDirector.open", fullurl)

    socket.socket = _GuardedSocket
    socket.create_connection = _guarded_create_connection
    urllib.request.urlopen = _guarded_urlopen
    urllib.request.OpenerDirector.open = _guarded_opener_open

    try:
        yield
    finally:
        socket.socket = original_socket
        socket.create_connection = original_create_connection
        urllib.request.urlopen = original_urlopen
        urllib.request.OpenerDirector.open = original_opener_open


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
    "NetworkBoundaryViolationError",
    "POLICY_PROFILE_NETWORK_ALLOWED",
    "POLICY_PROFILE_READONLY_LOCAL",
    "PROFILE_TABLE",
    "PolicyProfile",
    "READONLY_LOCAL_MAX_CALLS",
    "READONLY_LOCAL_MAX_OUTPUT_BYTES",
    "READONLY_LOCAL_PLAN_TIMEOUT_MS",
    "network_boundary_guard",
    "resolve_profile",
    "safe_builtins",
    "validate_plan_ast",
]
