"""Auto-discovery helpers for adapting public API callables."""

from __future__ import annotations

import inspect

import tess_vetter.api as _api
from tess_vetter.code_mode.adapters.base import OperationAdapter
from tess_vetter.code_mode.operation_spec import OperationCitation, OperationSpec, SafetyClass
from tess_vetter.code_mode.registries.operation_ids import (
    build_operation_id,
    normalize_operation_name,
)
from tess_vetter.code_mode.registries.tiering import ApiSymbol, tier_for_api_symbol


def _iter_api_export_callables() -> list[tuple[ApiSymbol, object]]:
    """Resolve callable exports from ``tess_vetter.api`` export map deterministically."""
    export_map = _api._get_export_map()
    exports: list[tuple[ApiSymbol, object]] = []

    # Deterministic discovery order independent of dict insertion behavior.
    for export_name in sorted(export_map):
        module_name, _attr_name = export_map[export_name]
        symbol = ApiSymbol(module=module_name, name=export_name)
        try:
            value = getattr(_api, export_name)
        except (AttributeError, ImportError, ModuleNotFoundError):
            # Optional guarded exports are intentionally skipped when unavailable.
            continue

        if inspect.isclass(value):
            continue
        if not (inspect.isroutine(value) or callable(value)):
            continue

        exports.append((symbol, value))

    return exports


def _build_auto_adapter(operation_id: str, export_name: str, fn: object, *, module_name: str) -> OperationAdapter:
    doc = inspect.getdoc(fn)
    first_line = doc.splitlines()[0].strip() if doc else ""
    return OperationAdapter(
        spec=OperationSpec(
            id=operation_id,
            name=export_name.replace("_", " ").title(),
            description=first_line,
            tier_tags=("api-export", "auto-discovered"),
            safety_class=SafetyClass.SAFE,
            citations=(OperationCitation(label=f"{module_name}.{export_name}"),),
        ),
        fn=fn,
    )


def discover_api_export_adapters(existing_ids: set[str] | None = None) -> tuple[OperationAdapter, ...]:
    """Discover callable API exports and convert them into adapters deterministically."""
    used_ids = set(existing_ids or ())
    discovered: list[OperationAdapter] = []

    for symbol, fn in _iter_api_export_callables():
        operation_id = build_operation_id(
            tier=tier_for_api_symbol(symbol),
            name=normalize_operation_name(symbol.name),
        )
        if operation_id in used_ids:
            continue
        used_ids.add(operation_id)
        discovered.append(
            _build_auto_adapter(operation_id, symbol.name, fn, module_name=symbol.module)
        )

    return tuple(sorted(discovered, key=lambda adapter: adapter.id))


__all__ = ["discover_api_export_adapters"]
