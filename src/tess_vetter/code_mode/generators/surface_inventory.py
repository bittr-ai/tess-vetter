"""Deterministic callable-surface inventory derived from discovery-aligned exports."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, cast

import tess_vetter.api as _api
from tess_vetter.api.primitives_catalog import list_primitives
from tess_vetter.code_mode.policy import (
    EXPORT_POLICY_ACTIONABLE,
    classify_api_export_policy,
)
from tess_vetter.code_mode.registries.operation_ids import (
    build_operation_id,
    normalize_operation_name,
)
from tess_vetter.code_mode.registries.tiering import ApiSymbol, tier_for_api_symbol

_TIER_ORDER: tuple[str, ...] = ("golden_path", "primitive", "internal")
_TIER_RANK: dict[str, int] = {tier: idx for idx, tier in enumerate(_TIER_ORDER)}
_LEGACY_DYNAMIC_EXPORTS: frozenset[str] = frozenset(
    {
        # Typing/enum alias artifacts intentionally exposed in public API.
        "ConsistencyClass",
        "ControlType",
        "ExportFormat",
        "LocalizationImages",
        "MatchClass",
        "PRFBackend",
        # Legacy variadic entrypoint retained for compatibility.
        "generate_control",
        # Plotting/reporting variadic helpers retained as legacy dynamic exports.
        "plot_alias_diagnostics",
        "plot_aperture_curve",
        "plot_asymmetry",
        "plot_centroid_shift",
        "plot_data_gaps",
        "plot_depth_stability",
        "plot_difference_image",
        "plot_duration_consistency",
        "plot_ephemeris_reliability",
        "plot_exofop_card",
        "plot_full_lightcurve",
        "plot_ghost_features",
        "plot_model_comparison",
        "plot_modshift",
        "plot_nearby_ebs",
        "plot_odd_even",
        "plot_phase_folded",
        "plot_secondary_eclipse",
        "plot_sector_consistency",
        "plot_sensitivity_sweep",
        "plot_sweet",
        "plot_transit_fit",
        "plot_v_shape",
        "save_vetting_report",
    }
)


def _api_export_map() -> dict[str, tuple[str, str]]:
    export_map_factory = getattr(_api, "_get_export_map", None)
    if not callable(export_map_factory):
        return {}
    export_map = export_map_factory()
    if not isinstance(export_map, dict):
        return {}
    return cast(dict[str, tuple[str, str]], export_map)


@dataclass(frozen=True, slots=True)
class SurfaceInventoryRow:
    """Single callable-surface row."""

    operation_id: str
    tier: str
    symbol: str
    module: str
    status: str
    export_policy: str
    replaced_by: str | None = None

def _sort_key(row: SurfaceInventoryRow) -> tuple[int, str, str, str, str, str, str]:
    return (
        _TIER_RANK.get(row.tier, len(_TIER_ORDER)),
        row.operation_id,
        row.symbol,
        row.module,
        row.status,
        row.export_policy,
        row.replaced_by or "",
    )


def _iter_api_callable_rows() -> list[SurfaceInventoryRow]:
    export_map = _api_export_map()
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
            export_policy = classify_api_export_policy(
                export_name=export_name,
                module_name=module_name,
                value=None,
            )
            rows.append(
                SurfaceInventoryRow(
                    operation_id=operation_id,
                    tier=tier,
                    symbol=symbol.name,
                    module=symbol.module,
                    status="unavailable" if export_policy == EXPORT_POLICY_ACTIONABLE else "legacy_dynamic",
                    export_policy=export_policy,
                    replaced_by=None,
                )
            )
            continue

        export_policy = classify_api_export_policy(
            export_name=export_name,
            module_name=module_name,
            value=value,
        )

        rows.append(
            SurfaceInventoryRow(
                operation_id=operation_id,
                tier=tier,
                symbol=symbol.name,
                module=symbol.module,
                status="available" if export_policy == EXPORT_POLICY_ACTIONABLE else "legacy_dynamic",
                export_policy=export_policy,
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
                export_policy=EXPORT_POLICY_ACTIONABLE,
                replaced_by=None,
            )
        )
    return rows


def _is_variadic_callable(value: object) -> bool:
    if inspect.isclass(value):
        return False
    if not (inspect.isroutine(value) or callable(value)):
        return False
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            return True
    return False


def dynamic_export_metrics() -> dict[str, Any]:
    """Return deterministic residual dynamic-export counts and symbol lists."""
    actionable: list[str] = []
    legacy: list[str] = []

    for export_name in sorted(_api_export_map()):
        try:
            value = getattr(_api, export_name)
        except (AttributeError, ImportError, ModuleNotFoundError):
            continue
        if not _is_variadic_callable(value):
            continue
        if export_name in _LEGACY_DYNAMIC_EXPORTS:
            legacy.append(export_name)
            continue
        actionable.append(export_name)

    return {
        "dynamic_export_counts": {
            "actionable_dynamic": len(actionable),
            "legacy_dynamic": len(legacy),
            "total_dynamic": len(actionable) + len(legacy),
        },
        "actionable_dynamic_symbols": tuple(actionable),
        "legacy_dynamic_symbols": tuple(legacy),
    }


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
            "export_policy": row.export_policy,
            "replaced_by": row.replaced_by,
        }
        for row in sorted_rows
    ]


__all__ = [
    "SurfaceInventoryRow",
    "build_surface_inventory",
    "dynamic_export_metrics",
    "surface_inventory_jsonable",
]
