"""Operation-library composition and registry for code mode."""

from __future__ import annotations

import builtins

from tess_vetter.code_mode.adapters import (
    OperationAdapter,
    discover_api_export_adapters,
    manual_seed_adapters,
)


class OpsLibrary:
    """Registry for operation adapters with deterministic listings."""

    def __init__(self) -> None:
        self._ops: dict[str, OperationAdapter] = {}

    def register(self, adapter: OperationAdapter) -> None:
        if adapter.id in self._ops:
            raise ValueError(f"Operation '{adapter.id}' already registered")
        self._ops[adapter.id] = adapter

    def get(self, operation_id: str) -> OperationAdapter:
        if operation_id not in self._ops:
            raise KeyError(f"No operation registered for '{operation_id}'")
        return self._ops[operation_id]

    def list(self) -> builtins.list[OperationAdapter]:
        return sorted(self._ops.values(), key=lambda op: op.id)

    def list_ids(self) -> builtins.list[str]:
        return sorted(self._ops)

    def __contains__(self, operation_id: str) -> bool:
        return operation_id in self._ops

    def __len__(self) -> int:
        return len(self._ops)


def make_default_ops_library() -> OpsLibrary:
    """Construct the default operation library from seed and discovered adapters."""
    library = OpsLibrary()

    for adapter in manual_seed_adapters():
        library.register(adapter)

    for adapter in discover_api_export_adapters(existing_ids=set(library.list_ids())):
        library.register(adapter)

    return library


__all__ = [
    "OperationAdapter",
    "OpsLibrary",
    "make_default_ops_library",
]
