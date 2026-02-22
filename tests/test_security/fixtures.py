from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from tess_vetter.code_mode.mcp_adapter import (
    ExecuteRequest,
    ExecuteResponse,
    MCPAdapter,
    SearchRequest,
    SearchResponse,
)


@dataclass(frozen=True)
class SecurityScenario:
    name: str
    request: SearchRequest | ExecuteRequest
    expected_error_code: str


def build_matrix_adapter() -> MCPAdapter:
    def _search_handler(_req: SearchRequest) -> SearchResponse:
        return SearchResponse(ok=True, results=[])

    def _execute_handler(_req: ExecuteRequest) -> ExecuteResponse:
        return ExecuteResponse(ok=True, output={"status": "ok"})

    return MCPAdapter(
        search_handler=_search_handler,
        execute_handler=_execute_handler,
    )


def canonical_payload_hash(payload: dict[str, Any]) -> str:
    import json

    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
