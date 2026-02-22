"""Typed MCP adapter surface for strict Code Mode contracts."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict, cast, get_args

from tess_vetter.code_mode.catalog import CatalogBuildResult, CatalogEntry, build_catalog
from tess_vetter.code_mode.ops_library import OperationAdapter, OpsLibrary, make_default_ops_library
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

    return ExecuteResponse(
        status="failed",
        result=None,
        error=error,
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


def _search_metadata(
    *,
    entry: CatalogEntry,
    adapter: OperationAdapter | None,
    why_matched: tuple[str, ...],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "operation_id": entry.id,
        "operation_tier": entry.tier,
        "operation_tags": list(entry.tags),
        "operation_callability": {
            "tool": "execute",
            "request_type": "ExecuteRequest",
            "operation_field": "plan_code",
            "payload_field": "context",
            "sandbox_field": "catalog_version_hash",
            "hash_field": "catalog_version_hash",
            "sandbox_required": True,
            "operation": entry.id,
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
) -> ExecuteResponse:
    context = dict(request.context)
    if request.catalog_version_hash is not None:
        context["catalog_version_hash"] = request.catalog_version_hash

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
        return ExecuteResponse(
            status="failed",
            result=None,
            error=ErrorPayload(
                code=_coerce_error_code(raw_error.get("code")),
                message=str(raw_error.get("message") or "Execution failed."),
                retryable=bool(raw_error.get("retryable", False)),
                details=dict(details) if isinstance(details, dict) else {},
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
        return _execute_via_runtime(request, catalog=catalog, runtime_ops=runtime_ops)

    return MCPAdapter(search_handler=_search_handler, execute_handler=_execute_handler)


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
