from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tess_vetter.code_mode.mcp_adapter import (
    ExecuteRequest,
    ExecuteResponse,
    MCPAdapter,
    SearchRequest,
    SearchResponse,
    SearchResult,
)


@dataclass(frozen=True)
class SecurityScenario:
    name: str
    request: Any
    expected_error_code: str


def build_matrix_adapter() -> MCPAdapter:
    def _search_handler(_req: SearchRequest) -> SearchResponse:
        return SearchResponse(
            results=[
                SearchResult(
                    id="ops.run",
                    title="Run",
                    snippet="Run plan",
                    score=1.0,
                    metadata={
                        "operation_id": "ops.run",
                        "operation_version": "1.2.3",
                        "operation_tier": "golden_path",
                        "operation_tags": ["core"],
                        "operation_requirements": {},
                        "operation_safety_class": "sandboxed",
                        "is_callable": True,
                    },
                )
            ],
            total=1,
            cursor=None,
            catalog_version_hash="catalog-v1",
            error=None,
        )

    def _execute_handler(req: ExecuteRequest) -> ExecuteResponse:
        return ExecuteResponse(
            status="ok",
            result={"status": "ok", "hash": req.catalog_version_hash},
            error=None,
            trace=None,
            catalog_version_hash=req.catalog_version_hash,
        )

    return MCPAdapter(
        search_handler=_search_handler,
        execute_handler=_execute_handler,
    )
