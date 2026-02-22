"""Typed MCP adapter surface with basic security guardrails.

This module defines a small host-facing adapter contract that can be reused by
MCP integrations without importing deeper app internals.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

ErrorCode = Literal[
    "network_denied",
    "sandbox_required",
    "hash_mismatch",
    "unsupported",
    "internal_error",
]


class ErrorPayloadDict(TypedDict, total=False):
    """Serialized error payload contract."""

    code: ErrorCode
    message: str
    retriable: bool
    details: dict[str, Any]


class SearchResultDict(TypedDict, total=False):
    """Serialized search result contract."""

    id: str
    title: str
    snippet: str
    score: float
    metadata: dict[str, Any]


class SearchResponseDict(TypedDict, total=False):
    """Serialized search response contract."""

    ok: bool
    results: list[SearchResultDict]
    error: ErrorPayloadDict


class ExecuteResponseDict(TypedDict, total=False):
    """Serialized execute response contract."""

    ok: bool
    output: dict[str, Any]
    error: ErrorPayloadDict


@dataclass(frozen=True)
class ErrorPayload:
    """Typed error payload returned by adapter calls."""

    code: ErrorCode
    message: str
    retriable: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> ErrorPayloadDict:
        payload: ErrorPayloadDict = {
            "code": self.code,
            "message": self.message,
            "retriable": self.retriable,
        }
        if self.details:
            payload["details"] = self.details
        return payload


@dataclass(frozen=True)
class SearchRequest:
    """Request contract for MCP search operations."""

    query: str
    limit: int = 20
    allow_network: bool = False


@dataclass(frozen=True)
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
            payload["metadata"] = self.metadata
        return payload


@dataclass(frozen=True)
class SearchResponse:
    """Response contract for MCP search operations."""

    ok: bool
    results: list[SearchResult] = field(default_factory=list)
    error: ErrorPayload | None = None

    def to_dict(self) -> SearchResponseDict:
        payload: SearchResponseDict = {"ok": self.ok, "results": [row.to_dict() for row in self.results]}
        if self.error is not None:
            payload["error"] = self.error.to_dict()
        return payload


@dataclass(frozen=True)
class ExecuteRequest:
    """Request contract for MCP execute operations."""

    operation: str
    payload: dict[str, Any] = field(default_factory=dict)
    sandboxed: bool = True
    expected_payload_sha256: str | None = None


@dataclass(frozen=True)
class ExecuteResponse:
    """Response contract for MCP execute operations."""

    ok: bool
    output: dict[str, Any] = field(default_factory=dict)
    error: ErrorPayload | None = None

    def to_dict(self) -> ExecuteResponseDict:
        payload: ExecuteResponseDict = {"ok": self.ok, "output": dict(self.output)}
        if self.error is not None:
            payload["error"] = self.error.to_dict()
        return payload


SearchHandler = Callable[[SearchRequest], SearchResponse]
ExecuteHandler = Callable[[ExecuteRequest], ExecuteResponse]


def _error_response(message: str, code: ErrorCode, details: dict[str, Any] | None = None) -> ExecuteResponse:
    return ExecuteResponse(
        ok=False,
        error=ErrorPayload(code=code, message=message, details=details or {}),
    )


class MCPAdapter:
    """Small additive adapter surface for host MCP integrations."""

    def __init__(
        self,
        *,
        search_handler: SearchHandler | None = None,
        execute_handler: ExecuteHandler | None = None,
    ) -> None:
        self._search_handler = search_handler
        self._execute_handler = execute_handler

    def search(self, request: SearchRequest) -> SearchResponse:
        """Run a search request through policy guardrails and host handler."""
        if not request.allow_network:
            return SearchResponse(
                ok=False,
                error=ErrorPayload(
                    code="network_denied",
                    message="Search requires network but allow_network=False.",
                ),
            )
        if self._search_handler is None:
            return SearchResponse(
                ok=False,
                error=ErrorPayload(code="unsupported", message="No search handler configured."),
            )
        try:
            return self._search_handler(request)
        except Exception as exc:  # pragma: no cover - defensive wrapper
            return SearchResponse(
                ok=False,
                error=ErrorPayload(code="internal_error", message=str(exc), retriable=True),
            )

    def execute(self, request: ExecuteRequest) -> ExecuteResponse:
        """Run an execute request through sandbox and integrity guardrails."""
        if not request.sandboxed:
            return _error_response("Execution denied: sandbox is required.", "sandbox_required")

        if request.expected_payload_sha256 is not None:
            actual_sha256 = hashlib.sha256(_canonical_json_bytes(request.payload)).hexdigest()
            if actual_sha256 != request.expected_payload_sha256:
                return _error_response(
                    "Execution denied: request payload hash mismatch.",
                    "hash_mismatch",
                    details={
                        "expected_payload_sha256": request.expected_payload_sha256,
                        "actual_payload_sha256": actual_sha256,
                    },
                )

        if self._execute_handler is None:
            return _error_response("No execute handler configured.", "unsupported")

        try:
            return self._execute_handler(request)
        except Exception as exc:  # pragma: no cover - defensive wrapper
            return _error_response(str(exc), "internal_error")


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    """Stable JSON encoding used for integrity checks."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


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
]
