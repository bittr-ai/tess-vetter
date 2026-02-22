"""Deterministic callable-surface inventory derived from discovery-aligned exports."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import tess_vetter.api as _api
from tess_vetter.api.primitives_catalog import list_primitives
from tess_vetter.code_mode.adapters.discovery import _iter_api_export_callables
from tess_vetter.code_mode.registries.operation_ids import (
    build_operation_id,
    normalize_operation_name,
)
from tess_vetter.code_mode.registries.tiering import ApiSymbol, tier_for_api_symbol

_TIER_ORDER: tuple[str, ...] = ("golden_path", "primitive", "internal")
_TIER_RANK: dict[str, int] = {tier: idx for idx, tier in enumerate(_TIER_ORDER)}


@dataclass(frozen=True, slots=True)
class SurfaceInventoryRow:
    """Single callable-surface row."""

    operation_id: str
    tier: str
    symbol: str
    module: str
    status: str
    replaced_by: str | None = None

def _sort_key(row: SurfaceInventoryRow) -> tuple[int, str, str, str, str, str]:
    return (
        _TIER_RANK.get(row.tier, len(_TIER_ORDER)),
        row.operation_id,
        row.symbol,
        row.module,
        row.status,
        row.replaced_by or "",
    )


def _iter_api_callable_rows() -> list[SurfaceInventoryRow]:
    callable_symbols = {symbol for symbol, _ in _iter_api_export_callables()}
    export_map = _api._get_export_map()
    rows: list[SurfaceInventoryRow] = []

    for export_name in sorted(export_map):
        module_name, _attr_name = export_map[export_name]
        symbol = ApiSymbol(module=module_name, name=export_name)
        tier = tier_for_api_symbol(symbol)
        operation_id = build_operation_id(
            tier=tier,
            name=normalize_operation_name(symbol.name),
        )
        try:
            value = getattr(_api, export_name)
        except (AttributeError, ImportError, ModuleNotFoundError):
            rows.append(
                SurfaceInventoryRow(
                    operation_id=operation_id,
                    tier=tier,
                    symbol=symbol.name,
                    module=symbol.module,
                    status="unavailable",
                    replaced_by=None,
                )
            )
            continue

        if symbol not in callable_symbols and (
            inspect.isclass(value) or not (inspect.isroutine(value) or callable(value))
        ):
            continue

        rows.append(
            SurfaceInventoryRow(
                operation_id=operation_id,
                tier=tier,
                symbol=symbol.name,
                module=symbol.module,
                status="available",
                replaced_by=None,
            )
        )
    return rows


def _iter_primitives_rows() -> list[SurfaceInventoryRow]:
    rows: list[SurfaceInventoryRow] = []
    for symbol, primitive_info in sorted(list_primitives(include_unimplemented=True).items()):
        operation_id = build_operation_id(
            tier="primitive",
            name=normalize_operation_name(symbol),
        )
        rows.append(
            SurfaceInventoryRow(
                operation_id=operation_id,
                tier="primitive",
                symbol=symbol,
                module="tess_vetter.compute",
                status=primitive_info.status,
                replaced_by=None,
            )
        )
    return rows


def build_surface_inventory() -> tuple[SurfaceInventoryRow, ...]:
    """Return deterministic callable-surface inventory rows."""
    rows = _iter_api_callable_rows()
    rows.extend(_iter_primitives_rows())
    return tuple(sorted(rows, key=_sort_key))


def surface_inventory_jsonable(
    rows: tuple[SurfaceInventoryRow, ...] | list[SurfaceInventoryRow] | None = None,
) -> list[dict[str, Any]]:
    """Return JSON-serializable inventory rows in deterministic order."""
    source_rows = build_surface_inventory() if rows is None else tuple(rows)
    sorted_rows = tuple(sorted(source_rows, key=_sort_key))
    return [
        {
            "operation_id": row.operation_id,
            "tier": row.tier,
            "symbol": row.symbol,
            "module": row.module,
            "status": row.status,
            "replaced_by": row.replaced_by,
        }
        for row in sorted_rows
    ]


__all__ = [
    "SurfaceInventoryRow",
    "build_surface_inventory",
    "surface_inventory_jsonable",
]
