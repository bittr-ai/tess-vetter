"""Deterministic callable-surface inventory derived from API exports and primitives."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Any

from tess_vetter import api as _api
from tess_vetter.api.primitives_catalog import list_primitives

_SLUG_RE = re.compile(r"[^a-z0-9_]+")
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


def _slugify(value: str) -> str:
    slug = _SLUG_RE.sub("_", value.strip().lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return slug or "unnamed"


def _operation_id(tier: str, symbol: str) -> str:
    return f"code_mode.{tier}.{_slugify(symbol)}"


def _tier_for_module(module: str) -> str:
    if module.startswith("tess_vetter.api.primitives") or module.startswith(
        "tess_vetter.api.sandbox_primitives"
    ):
        return "primitive"
    if module.startswith("tess_vetter.api"):
        return "golden_path"
    return "internal"


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
    rows: list[SurfaceInventoryRow] = []
    export_map = _api._get_export_map()

    for symbol in sorted(export_map):
        module, _ = export_map[symbol]
        try:
            value = getattr(_api, symbol)
        except Exception:
            continue
        if not inspect.isroutine(value):
            continue
        tier = _tier_for_module(module)
        rows.append(
            SurfaceInventoryRow(
                operation_id=_operation_id(tier, symbol),
                tier=tier,
                symbol=symbol,
                module=module,
                status="available",
                replaced_by=None,
            )
        )

    for alias, target in sorted(_api._ALIASES.items()):
        if alias not in getattr(_api, "__all__", ()) and alias not in export_map:
            continue
        try:
            alias_value = getattr(_api, alias)
        except Exception:
            continue
        if not inspect.isroutine(alias_value):
            continue

        target_module, _ = export_map.get(target, ("tess_vetter.api", target))
        tier = _tier_for_module(target_module)
        rows.append(
            SurfaceInventoryRow(
                operation_id=_operation_id(tier, alias),
                tier=tier,
                symbol=alias,
                module="tess_vetter.api",
                status="deprecated",
                replaced_by=_operation_id(tier, target),
            )
        )

    return rows


def _iter_primitives_rows() -> list[SurfaceInventoryRow]:
    rows: list[SurfaceInventoryRow] = []
    for symbol, primitive_info in sorted(list_primitives(include_unimplemented=True).items()):
        rows.append(
            SurfaceInventoryRow(
                operation_id=_operation_id("primitive", symbol),
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
