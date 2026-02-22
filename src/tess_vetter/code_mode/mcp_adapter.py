"""Typed MCP adapter surface for strict Code Mode contracts."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict, cast, get_args

from tess_vetter.code_mode.catalog import CatalogBuildResult, CatalogEntry, build_catalog
from tess_vetter.code_mode.ops_library import (
    OperationAdapter,
    OpsLibrary,
    make_default_ops_library,
    required_input_paths_for_adapter,
)
from tess_vetter.code_mode.runtime import execute as runtime_execute
from tess_vetter.code_mode.search import search_catalog

ErrorCode = Literal[
    "PLAN_PARSE_ERROR",
    "SCHEMA_VIOLATION_INPUT",
    "SCHEMA_VIOLATION_OUTPUT",
    "POLICY_DENIED",
    "OPERATION_NOT_FOUND",
    "DEPENDENCY_MISSING",
    "TRANSIENT_EXHAUSTION",
    "TIMEOUT_EXCEEDED",
    "CALL_LIMIT_EXCEEDED",
    "OUTPUT_LIMIT_EXCEEDED",
    "CATALOG_DRIFT",
    "OPERATION_RUNTIME_ERROR",
]


class ErrorPayloadDict(TypedDict):
    """Serialized error payload contract."""

    code: ErrorCode
    message: str
    retryable: bool
    details: dict[str, Any]


class SearchResultDict(TypedDict, total=False):
    """Serialized search result contract."""

    id: str
    title: str
    snippet: str
    score: float
    metadata: dict[str, Any]


class SearchResponseDict(TypedDict):
    """Serialized search response contract."""

    results: list[SearchResultDict]
    total: int
    cursor: str | None
    catalog_version_hash: str | None
    error: ErrorPayloadDict | None


class ExecuteResponseDict(TypedDict):
    """Serialized execute response contract that mirrors runtime output."""

    status: Literal["ok", "failed"]
    result: Any
    error: ErrorPayloadDict | None
    trace: dict[str, Any] | None
    catalog_version_hash: str | None


@dataclass(frozen=True, slots=True)
class ErrorPayload:
    """Typed error payload returned by adapter calls."""

    code: ErrorCode
    message: str
    retryable: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> ErrorPayloadDict:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """Strict request contract for MCP search operations."""

    query: str
    limit: int = 20
    tags: list[str] | None = None


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Single search hit."""

    id: str
    title: str
    snippet: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> SearchResultDict:
        payload: SearchResultDict = {
            "id": self.id,
            "title": self.title,
            "snippet": self.snippet,
            "score": self.score,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """Strict response contract for MCP search operations."""

    results: list[SearchResult] = field(default_factory=list)
    total: int = 0
    cursor: str | None = None
    catalog_version_hash: str | None = None
    error: ErrorPayload | None = None

    def to_dict(self) -> SearchResponseDict:
        return {
            "results": [row.to_dict() for row in self.results],
            "total": self.total,
            "cursor": self.cursor,
            "catalog_version_hash": self.catalog_version_hash,
            "error": None if self.error is None else self.error.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ExecuteRequest:
    """Strict request contract for MCP execute operations."""

    plan_code: str
    context: dict[str, Any] = field(default_factory=dict)
    catalog_version_hash: str | None = None


@dataclass(frozen=True, slots=True)
class ExecuteResponse:
    """Strict response contract for MCP execute operations."""

    status: Literal["ok", "failed"]
    result: Any = None
    error: ErrorPayload | None = None
    trace: dict[str, Any] | None = None
    catalog_version_hash: str | None = None

    def to_dict(self) -> ExecuteResponseDict:
        return {
            "status": self.status,
            "result": self.result,
            "error": None if self.error is None else self.error.to_dict(),
            "trace": None if self.trace is None else dict(self.trace),
            "catalog_version_hash": self.catalog_version_hash,
        }


SearchHandler = Callable[[SearchRequest], SearchResponse]
ExecuteHandler = Callable[[ExecuteRequest], ExecuteResponse]
_ERROR_CODES: frozenset[str] = frozenset(cast(tuple[str, ...], get_args(ErrorCode)))


def _search_error(
    code: ErrorCode,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    retryable: bool = False,
) -> SearchResponse:
    return SearchResponse(
        results=[],
        total=0,
        cursor=None,
        catalog_version_hash=None,
        error=ErrorPayload(
            code=code,
            message=message,
            retryable=retryable,
            details=dict(details or {}),
        ),
    )


def _execute_error(
    code: ErrorCode,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    retryable: bool = False,
    catalog_version_hash: str | None = None,
) -> ExecuteResponse:
    return ExecuteResponse(
        status="failed",
        result=None,
        error=ErrorPayload(
            code=code,
            message=message,
            retryable=retryable,
            details=dict(details or {}),
        ),
        trace=None,
        catalog_version_hash=catalog_version_hash,
    )


class MCPAdapter:
    """Small strict adapter surface for host MCP integrations."""

    def __init__(
        self,
        *,
        search_handler: SearchHandler | None = None,
        execute_handler: ExecuteHandler | None = None,
    ) -> None:
        self._search_handler = search_handler
        self._execute_handler = execute_handler

    def search(self, request: SearchRequest) -> SearchResponse:
        """Run a strict search request through the configured handler."""
        preflight_error = _validate_search_request_shape(request)
        if preflight_error is not None:
            return _search_error(
                "SCHEMA_VIOLATION_INPUT",
                preflight_error,
                details={
                    "contract": "SearchRequest",
                    "reason": "shape_validation_failed",
                    "validation_error": preflight_error,
                },
            )
        if self._search_handler is None:
            return _search_error(
                "OPERATION_NOT_FOUND",
                "No search handler configured.",
                details={"handler": "search", "reason": "handler_missing"},
            )

        try:
            response = self._search_handler(request)
            return _normalize_search_response(response)
        except Exception as exc:  # pragma: no cover - defensive wrapper
            return _search_error(
                "OPERATION_RUNTIME_ERROR",
                str(exc),
                details={"handler": "search", "exception_type": type(exc).__name__},
            )

    def execute(self, request: ExecuteRequest) -> ExecuteResponse:
        """Run a strict execute request through the configured handler."""
        preflight_error = _validate_execute_request_shape(request)
        if preflight_error is not None:
            return _execute_error(
                "SCHEMA_VIOLATION_INPUT",
                preflight_error,
                details={
                    "contract": "ExecuteRequest",
                    "reason": "shape_validation_failed",
                    "validation_error": preflight_error,
                },
                catalog_version_hash=getattr(request, "catalog_version_hash", None),
            )
        if self._execute_handler is None:
            return _execute_error(
                "OPERATION_NOT_FOUND",
                "No execute handler configured.",
                details={"handler": "execute", "reason": "handler_missing"},
                catalog_version_hash=request.catalog_version_hash,
            )

        try:
            response = self._execute_handler(request)
            return _normalize_execute_response(response)
        except Exception as exc:  # pragma: no cover - defensive wrapper
            return _execute_error(
                "OPERATION_RUNTIME_ERROR",
                str(exc),
                details={"handler": "execute", "exception_type": type(exc).__name__},
                catalog_version_hash=request.catalog_version_hash,
            )


def _validate_search_request_shape(request: Any) -> str | None:
    if not hasattr(request, "query"):
        return "Invalid search request: missing required field 'query'."
    if not isinstance(request.query, str):
        return "Invalid search request: 'query' must be a string."
    if not hasattr(request, "limit"):
        return "Invalid search request: missing required field 'limit'."
    limit = request.limit
    if not isinstance(limit, int) or isinstance(limit, bool):
        return "Invalid search request: 'limit' must be an integer."
    if limit < 1:
        return "Invalid search request: 'limit' must be >= 1."
    if hasattr(request, "tags") and request.tags is not None:
        tags = request.tags
        if not isinstance(tags, list):
            return "Invalid search request: 'tags' must be a list[str] when provided."
        if any(not isinstance(tag, str) for tag in tags):
            return "Invalid search request: each tag in 'tags' must be a string."
    return None


def _validate_execute_request_shape(request: Any) -> str | None:
    if not hasattr(request, "plan_code"):
        return "Invalid execute request: missing required field 'plan_code'."
    plan_code = request.plan_code
    if not isinstance(plan_code, str):
        return "Invalid execute request: 'plan_code' must be a string."
    if not plan_code.strip():
        return "Invalid execute request: 'plan_code' cannot be empty."
    if not hasattr(request, "context"):
        return "Invalid execute request: missing required field 'context'."
    context = request.context
    if not isinstance(context, dict):
        return "Invalid execute request: 'context' must be an object."
    if not hasattr(request, "catalog_version_hash"):
        return "Invalid execute request: missing required field 'catalog_version_hash'."
    catalog_version_hash = request.catalog_version_hash
    if catalog_version_hash is not None and not isinstance(catalog_version_hash, str):
        return "Invalid execute request: 'catalog_version_hash' must be a string when provided."
    return None


def _normalize_search_response(response: SearchResponse) -> SearchResponse:
    if response.error is not None:
        return SearchResponse(
            results=[],
            total=0,
            cursor=response.cursor,
            catalog_version_hash=response.catalog_version_hash,
            error=response.error,
        )

    normalized_results = [
        SearchResult(
            id=row.id,
            title=row.title,
            snippet=row.snippet,
            score=float(row.score),
            metadata=dict(row.metadata),
        )
        for row in response.results
    ]

    total = response.total
    if total < len(normalized_results):
        total = len(normalized_results)

    return SearchResponse(
        results=normalized_results,
        total=total,
        cursor=response.cursor,
        catalog_version_hash=response.catalog_version_hash,
        error=None,
    )


def _normalize_execute_response(response: ExecuteResponse) -> ExecuteResponse:
    if response.status == "ok":
        return ExecuteResponse(
            status="ok",
            result=response.result,
            error=None,
            trace=None if response.trace is None else dict(response.trace),
            catalog_version_hash=response.catalog_version_hash,
        )

    error = response.error
    if error is None:
        error = ErrorPayload(
            code="SCHEMA_VIOLATION_OUTPUT",
            message="Invalid execute response: failed status requires error payload.",
            retryable=False,
            details={"contract": "ExecuteResponse"},
        )

    normalized_details = _normalize_blocker_details(
        code=error.code,
        message=error.message,
        details=dict(error.details),
    )
    retryable = _normalize_retryable_semantics(
        code=error.code,
        details=normalized_details,
        retryable=error.retryable,
    )
    result_payload: Any = None
    if isinstance(response.result, dict) and str(response.result.get("mode", "")).strip().lower() == "preflight":
        result_payload = dict(response.result)
    return ExecuteResponse(
        status="failed",
        result=result_payload,
        error=ErrorPayload(
            code=error.code,
            message=error.message,
            retryable=retryable,
            details=normalized_details,
        ),
        trace=None if response.trace is None else dict(response.trace),
        catalog_version_hash=response.catalog_version_hash,
    )


def _catalog_entry_from_adapter(adapter: OperationAdapter) -> dict[str, Any]:
    spec = adapter.spec
    return {
        "id": spec.id,
        "tier": spec.tier,
        "title": spec.name,
        "description": spec.description,
        "tags": list(spec.tier_tags),
        "availability": spec.availability.value,
        "status": "deprecated" if spec.deprecated else "active",
        "deprecated": spec.deprecated,
        "replacement": spec.replaced_by,
        "schema": {
            "input": dict(spec.input_json_schema),
            "output": dict(spec.output_json_schema),
        },
    }


def _coerce_error_code(raw_code: Any) -> ErrorCode:
    if isinstance(raw_code, str) and raw_code in _ERROR_CODES:
        return cast(ErrorCode, raw_code)
    return "OPERATION_RUNTIME_ERROR"


def _policy_blocker_from_details(details: dict[str, Any], message: str) -> dict[str, Any]:
    if isinstance(details.get("boundary"), str):
        target = details.get("target")
        return {
            "type": "network_boundary_denied",
            "summary": "Network access was denied by policy.",
            "action": "Use a network-enabled profile or provide required local artifacts.",
            "policy_profile": details.get("policy_profile"),
            "boundary": details.get("boundary"),
            "target": str(target) if target is not None else None,
        }
    if isinstance(details.get("node_type"), str):
        return {
            "type": "restricted_python_construct",
            "summary": "Plan uses a Python construct blocked by policy.",
            "action": "Remove restricted constructs (import/global/nonlocal) from plan code.",
            "node_type": details.get("node_type"),
        }
    if isinstance(details.get("name"), str):
        return {
            "type": "restricted_builtin",
            "summary": "Plan references a builtin blocked by policy.",
            "action": "Use provided safe builtins and operation APIs instead.",
            "name": details.get("name"),
        }
    if isinstance(details.get("attribute"), str):
        return {
            "type": "restricted_attribute",
            "summary": "Plan references a dunder attribute blocked by policy.",
            "action": "Remove dunder attribute access from plan code.",
            "attribute": details.get("attribute"),
        }
    if isinstance(details.get("requested_profile"), str):
        return {
            "type": "policy_profile_denied",
            "summary": "Requested policy profile is not allowed.",
            "action": "Use one of the supported profiles: readonly_local, network_allowed.",
            "requested_profile": details.get("requested_profile"),
        }
    return {
        "type": "policy_denied",
        "summary": message,
        "action": "Adjust plan/context to satisfy runtime policy constraints.",
    }


def _dependency_blocker_from_details(details: dict[str, Any], message: str) -> dict[str, Any]:
    dependency = details.get("dependency")
    if not isinstance(dependency, str):
        dependency = details.get("extra")
    install_hint = details.get("install_hint")
    if not isinstance(install_hint, str) and isinstance(dependency, str) and dependency.strip():
        install_hint = f"pip install 'tess-vetter[{dependency}]'"
    blocker: dict[str, Any] = {
        "type": "optional_dependency_missing",
        "summary": message,
        "action": "Install required optional dependency and retry.",
        "dependency": dependency,
    }
    if isinstance(install_hint, str):
        blocker["install_hint"] = install_hint
    return blocker


def _normalize_blocker_details(
    *,
    code: ErrorCode,
    message: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(details)
    _ensure_blocker_arrays(normalized)
    if code == "POLICY_DENIED":
        if not normalized["policy_blockers"]:
            normalized["policy_blockers"] = [_policy_blocker_from_details(normalized, message)]
        _sort_blocker_arrays(normalized)
        return normalized

    if code == "DEPENDENCY_MISSING":
        if not normalized["dependency_blockers"] and not normalized["constructor_blockers"]:
            normalized["dependency_blockers"] = [_dependency_blocker_from_details(normalized, message)]
        _sort_blocker_arrays(normalized)
        return normalized

    if _is_preflight_payload(normalized):
        _sort_blocker_arrays(normalized)

    return normalized


def _ensure_blocker_arrays(details: dict[str, Any]) -> None:
    for key in ("policy_blockers", "dependency_blockers", "constructor_blockers"):
        value = details.get(key)
        if not isinstance(value, list):
            details[key] = []
            continue
        details[key] = [dict(item) for item in value if isinstance(item, dict)]


def _sort_blocker_arrays(details: dict[str, Any]) -> None:
    details["policy_blockers"] = sorted(
        details["policy_blockers"],
        key=lambda row: (
            str(row.get("operation_id")),
            str(row.get("type")),
            str(row.get("requirement")),
            int(row.get("call_site", {}).get("line", 0)),
            int(row.get("call_site", {}).get("column", 0)),
        ),
    )
    details["dependency_blockers"] = sorted(
        details["dependency_blockers"],
        key=lambda row: (
            str(row.get("operation_id")),
            str(row.get("type")),
            str(row.get("reason")),
            int(row.get("call_site", {}).get("line", 0)),
            int(row.get("call_site", {}).get("column", 0)),
        ),
    )
    details["constructor_blockers"] = sorted(
        details["constructor_blockers"],
        key=lambda row: (
            str(row.get("operation_id")),
            str(row.get("field")),
            str(row.get("type")),
            int(row.get("call_site", {}).get("line", 0)),
            int(row.get("call_site", {}).get("column", 0)),
        ),
    )


def _is_preflight_payload(details: dict[str, Any]) -> bool:
    return str(details.get("mode", "")).strip().lower() == "preflight"


def _normalize_retryable_semantics(
    *,
    code: ErrorCode,
    details: dict[str, Any],
    retryable: bool,
) -> bool:
    if retryable:
        return True
    if not _is_preflight_payload(details):
        return False
    if code in {"POLICY_DENIED", "DEPENDENCY_MISSING"}:
        return True
    return False


def _build_minimal_execute_plan_code(operation_id: str) -> str:
    return (
        "async def execute_plan(ops, context):\n"
        "    kwargs = context.get('operation_kwargs')\n"
        f"    op = ops['{operation_id}']\n"
        "    if isinstance(kwargs, dict):\n"
        "        return await op(**kwargs)\n"
        "    return await op()\n"
    )


def _callability_score_and_flags(
    *,
    entry: CatalogEntry,
    adapter: OperationAdapter | None,
) -> tuple[float, tuple[str, ...]]:
    score = 1.0
    blockers: list[str] = []
    hints: list[str] = []
    constructor_ready = False
    policy_ready = False

    if adapter is None:
        score -= 0.45
        blockers.append("adapter_missing")
        required_paths: tuple[str, ...] = ()
        requirements: dict[str, Any] = {}
    else:
        required_paths = required_input_paths_for_adapter(adapter)
        requirements = adapter.spec.safety_requirements.model_dump()
        if required_paths:
            blockers.append("constructor_inputs_required")
            score -= min(0.20, 0.03 * len(required_paths))
        else:
            constructor_ready = True
            hints.append("constructor_ready")

        if bool(requirements.get("needs_network")):
            blockers.append("policy_requires_network")
            score -= 0.12
        if bool(requirements.get("requires_human_review")):
            blockers.append("policy_requires_human_review")
            score -= 0.18
        if bool(requirements.get("needs_secrets")):
            blockers.append("policy_requires_secrets")
            score -= 0.18
        if bool(requirements.get("needs_filesystem")):
            blockers.append("policy_requires_filesystem")
            score -= 0.08
        if not any(
            bool(requirements.get(key))
            for key in (
                "needs_network",
                "requires_human_review",
                "needs_secrets",
                "needs_filesystem",
            )
        ):
            policy_ready = True
            hints.append("policy_ready")

    if entry.availability != "available":
        score -= 0.35
        blockers.append("availability_unavailable")
    if entry.status != "active":
        score -= 0.15
        blockers.append("status_inactive")
    if entry.deprecated:
        score -= 0.10
        blockers.append("deprecated")
    if entry.replacement:
        hints.append("replacement_available")

    if constructor_ready and policy_ready and not blockers:
        hints.insert(0, "direct_execute_ready")

    flags = [*hints, *blockers]

    if score < 0:
        score = 0.0
    return round(score, 2), tuple(flags)


def _search_metadata(
    *,
    entry: CatalogEntry,
    adapter: OperationAdapter | None,
    why_matched: tuple[str, ...],
) -> dict[str, Any]:
    callability_score, callability_reason_flags = _callability_score_and_flags(
        entry=entry,
        adapter=adapter,
    )
    required_paths = list(required_input_paths_for_adapter(adapter)) if adapter is not None else []
    request_required_paths = ["plan_code", "context", "catalog_version_hash"]
    metadata: dict[str, Any] = {
        "operation_id": entry.id,
        "operation_tier": entry.tier,
        "operation_tags": list(entry.tags),
        "callability_score": callability_score,
        "callability_reason_flags": list(callability_reason_flags),
        "operation_callability": {
            "tool": "execute",
            "request_type": "ExecuteRequest",
            "operation_field": "plan_code",
            "payload_field": "context",
            "sandbox_field": "catalog_version_hash",
            "hash_field": "catalog_version_hash",
            "sandbox_required": True,
            "operation": entry.id,
            "required_paths": list(required_paths),
            "request_required_paths": list(request_required_paths),
            "minimal_payload_example": {
                "plan_code": _build_minimal_execute_plan_code(entry.id),
                "context": {
                    "operation_kwargs": {
                        "example": "replace_with_valid_operation_kwargs"
                    }
                },
                "catalog_version_hash": "<from_search_response.catalog_version_hash>",
            },
            "callability_score": callability_score,
            "reason_flags": list(callability_reason_flags),
        },
        "why_matched": list(why_matched),
        "availability": entry.availability,
        "status": entry.status,
        "deprecated": entry.deprecated,
        "replacement": entry.replacement,
        "schema_fingerprint": entry.schema_fingerprint,
    }
    if adapter is not None:
        metadata["operation_version"] = adapter.spec.version
        metadata["operation_requirements"] = adapter.spec.safety_requirements.model_dump()
        metadata["operation_safety_class"] = adapter.spec.safety_class.value
        metadata["schema_snippet"] = {
            "input": dict(adapter.spec.input_json_schema),
            "output": dict(adapter.spec.output_json_schema),
        }
    else:
        metadata["operation_version"] = None
        metadata["operation_requirements"] = {}
        metadata["operation_safety_class"] = None
        metadata["schema_snippet"] = {"input": {}, "output": {}}
    return metadata


def _execute_via_runtime(
    request: ExecuteRequest,
    *,
    catalog: CatalogBuildResult,
    runtime_ops: dict[str, Any],
    preflight_operation_catalog: dict[str, dict[str, Any]],
) -> ExecuteResponse:
    context = dict(request.context)
    if request.catalog_version_hash is not None:
        context["catalog_version_hash"] = request.catalog_version_hash
    context["preflight_operation_catalog"] = {
        operation_id: dict(payload)
        for operation_id, payload in preflight_operation_catalog.items()
    }

    runtime_response = asyncio.run(
        runtime_execute(
            request.plan_code,
            runtime_ops,
            context,
            catalog_version_hash=catalog.catalog_version_hash,
        )
    )

    status = runtime_response.get("status")
    trace_payload = runtime_response.get("trace")
    trace = dict(trace_payload) if isinstance(trace_payload, dict) else None
    response_hash = runtime_response.get("catalog_version_hash")
    catalog_hash = response_hash if isinstance(response_hash, str) else catalog.catalog_version_hash

    if status == "ok":
        return ExecuteResponse(
            status="ok",
            result=runtime_response.get("result"),
            error=None,
            trace=trace,
            catalog_version_hash=catalog_hash,
        )

    raw_error = runtime_response.get("error")
    if isinstance(raw_error, dict):
        details = raw_error.get("details")
        code = _coerce_error_code(raw_error.get("code"))
        message = str(raw_error.get("message") or "Execution failed.")
        normalized_details = _normalize_blocker_details(
            code=code,
            message=message,
            details=dict(details) if isinstance(details, dict) else {},
        )
        raw_retryable = raw_error.get("retryable")
        retryable = _normalize_retryable_semantics(
            code=code,
            details=normalized_details,
            retryable=bool(raw_retryable) if isinstance(raw_retryable, bool) else False,
        )
        result_payload = runtime_response.get("result")
        return ExecuteResponse(
            status="failed",
            result=result_payload,
            error=ErrorPayload(
                code=code,
                message=message,
                retryable=retryable,
                details=normalized_details,
            ),
            trace=trace,
            catalog_version_hash=catalog_hash,
        )

    return ExecuteResponse(
        status="failed",
        result=None,
        error=ErrorPayload(
            code="OPERATION_RUNTIME_ERROR",
            message="Execution failed without structured error payload.",
            retryable=False,
            details={},
        ),
        trace=trace,
        catalog_version_hash=catalog_hash,
    )


def make_default_mcp_adapter() -> MCPAdapter:
    """Create a canonical MCP adapter wired to catalog search and runtime execute."""
    ops_library: OpsLibrary = make_default_ops_library()
    adapters = ops_library.list()
    catalog = build_catalog([_catalog_entry_from_adapter(adapter) for adapter in adapters])
    adapters_by_id = {adapter.id: adapter for adapter in adapters}
    runtime_ops = {adapter.id: adapter.fn for adapter in adapters}
    preflight_operation_catalog = _build_preflight_operation_catalog(adapters)

    def _search_handler(request: SearchRequest) -> SearchResponse:
        matches = search_catalog(
            catalog.entries,
            query=request.query,
            limit=request.limit,
            tags=request.tags or (),
        )
        results = [
            SearchResult(
                id=match.entry.id,
                title=match.entry.title,
                snippet=match.entry.description,
                score=float(match.score),
                metadata=_search_metadata(
                    entry=match.entry,
                    adapter=adapters_by_id.get(match.entry.id),
                    why_matched=match.why_matched,
                ),
            )
            for match in matches
        ]
        return SearchResponse(
            results=results,
            total=len(results),
            cursor=None,
            catalog_version_hash=catalog.catalog_version_hash,
            error=None,
        )

    def _execute_handler(request: ExecuteRequest) -> ExecuteResponse:
        return _execute_via_runtime(
            request,
            catalog=catalog,
            runtime_ops=runtime_ops,
            preflight_operation_catalog=preflight_operation_catalog,
        )

    return MCPAdapter(search_handler=_search_handler, execute_handler=_execute_handler)


def _build_preflight_operation_catalog(adapters: list[OperationAdapter]) -> dict[str, dict[str, Any]]:
    operation_catalog: dict[str, dict[str, Any]] = {}
    for adapter in adapters:
        schema = adapter.spec.input_json_schema if isinstance(adapter.spec.input_json_schema, dict) else {}
        properties = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
        required = schema.get("required", []) if isinstance(schema.get("required"), list) else []
        field_types: dict[str, list[str]] = {}
        for field_name, field_schema in properties.items():
            if not isinstance(field_name, str) or not isinstance(field_schema, dict):
                continue
            json_types = _extract_json_types(field_schema)
            if json_types:
                field_types[field_name] = list(json_types)
        operation_catalog[adapter.id] = {
            "availability": adapter.spec.availability.value,
            "required_fields": sorted(str(name) for name in required if isinstance(name, str)),
            "field_types": field_types,
            "safety_requirements": adapter.spec.safety_requirements.model_dump(),
        }
    return operation_catalog


def _extract_json_types(field_schema: dict[str, Any]) -> tuple[str, ...]:
    types: set[str] = set()
    direct_type = field_schema.get("type")
    if isinstance(direct_type, str):
        types.add(direct_type)
    elif isinstance(direct_type, list):
        for item in direct_type:
            if isinstance(item, str):
                types.add(item)

    for key in ("anyOf", "oneOf"):
        variants = field_schema.get(key)
        if not isinstance(variants, list):
            continue
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            variant_type = variant.get("type")
            if isinstance(variant_type, str):
                types.add(variant_type)
            elif isinstance(variant_type, list):
                for item in variant_type:
                    if isinstance(item, str):
                        types.add(item)
    return tuple(sorted(types))


__all__ = [
    "ErrorCode",
    "ErrorPayload",
    "ErrorPayloadDict",
    "ExecuteRequest",
    "ExecuteResponse",
    "ExecuteResponseDict",
    "MCPAdapter",
    "SearchRequest",
    "SearchResponse",
    "SearchResponseDict",
    "SearchResult",
    "SearchResultDict",
    "make_default_mcp_adapter",
]
