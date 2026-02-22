"""Auto-discovery helpers for adapting public API callables."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import tess_vetter.api as _api
from tess_vetter.api.contracts import (
    opaque_object_schema,
)
from tess_vetter.code_mode.adapters.base import AdapterUnavailableError, OperationAdapter
from tess_vetter.code_mode.operation_spec import (
    OperationAvailability,
    OperationCitation,
    OperationSpec,
    SafetyClass,
    SafetyRequirements,
)
from tess_vetter.code_mode.policy import (
    EXPORT_POLICY_ACTIONABLE,
    classify_api_export_policy,
    is_actionable_api_export,
)
from tess_vetter.code_mode.registries.operation_ids import (
    build_operation_id,
    normalize_operation_name,
)
from tess_vetter.code_mode.registries.tiering import ApiSymbol, tier_for_api_symbol
from tess_vetter.code_mode.retry.wrappers import wrap_with_transient_retry

_NETWORK_PARAM_HINTS: frozenset[str] = frozenset(
    {
        "network",
        "allow_network",
        "use_network",
        "download",
        "allow_download",
    }
)


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

        if not is_actionable_api_export(
            export_name=export_name,
            module_name=module_name,
            value=value,
        ):
            continue

        exports.append((symbol, value))

    return exports


def _build_auto_adapter(
    operation_id: str,
    export_name: str,
    fn: object,
    *,
    module_name: str,
    needs_network: bool = False,
    availability: OperationAvailability = OperationAvailability.AVAILABLE,
) -> OperationAdapter:
    doc = inspect.getdoc(fn)
    first_line = doc.splitlines()[0].strip() if doc else ""
    wrapped_fn = wrap_with_transient_retry(fn) if needs_network and availability == OperationAvailability.AVAILABLE else fn
    return OperationAdapter(
        spec=OperationSpec(
            id=operation_id,
            name=export_name.replace("_", " ").title(),
            description=first_line,
            tier_tags=("api-export", "auto-discovered"),
            safety_class=SafetyClass.SAFE,
            safety_requirements=SafetyRequirements(needs_network=needs_network),
            input_json_schema=opaque_object_schema(),
            output_json_schema=opaque_object_schema(),
            availability=availability,
            citations=(OperationCitation(label=f"{module_name}.{export_name}"),),
        ),
        fn=wrapped_fn,
    )


def _is_network_bound_callable(fn: object) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False

    return any(param_name.lower() in _NETWORK_PARAM_HINTS for param_name in signature.parameters)


def _build_unavailable_adapter_fn(
    *,
    operation_id: str,
    export_name: str,
    module_name: str,
    cause: Exception,
) -> Callable[..., Any]:
    message = (
        f"Operation '{operation_id}' is unavailable because "
        f"'{module_name}.{export_name}' could not be imported: {cause}"
    )

    def _unavailable(*_args: Any, **_kwargs: Any) -> Any:
        raise AdapterUnavailableError(message) from cause

    return _unavailable


def discover_api_export_adapters(existing_ids: set[str] | None = None) -> tuple[OperationAdapter, ...]:
    """Discover callable API exports and convert them into adapters deterministically."""
    used_ids = set(existing_ids or ())
    discovered: list[OperationAdapter] = []

    for export_name, (module_name, _attr_name) in sorted(_api._get_export_map().items()):
        symbol = ApiSymbol(module=module_name, name=export_name)
        operation_id = build_operation_id(
            tier=tier_for_api_symbol(symbol),
            name=normalize_operation_name(symbol.name),
        )
        if operation_id in used_ids:
            continue

        try:
            value = getattr(_api, export_name)
        except (AttributeError, ImportError, ModuleNotFoundError) as exc:
            export_policy = classify_api_export_policy(
                export_name=export_name,
                module_name=module_name,
                value=None,
            )
            if export_policy != EXPORT_POLICY_ACTIONABLE:
                continue
            used_ids.add(operation_id)
            discovered.append(
                _build_auto_adapter(
                    operation_id,
                    symbol.name,
                    _build_unavailable_adapter_fn(
                        operation_id=operation_id,
                        export_name=symbol.name,
                        module_name=symbol.module,
                        cause=exc,
                    ),
                    module_name=symbol.module,
                    availability=OperationAvailability.UNAVAILABLE,
                )
            )
            continue

        if not is_actionable_api_export(
            export_name=export_name,
            module_name=module_name,
            value=value,
        ):
            continue

        used_ids.add(operation_id)
        discovered.append(
            _build_auto_adapter(
                operation_id,
                symbol.name,
                value,
                module_name=symbol.module,
                needs_network=_is_network_bound_callable(value),
            )
        )

    return tuple(sorted(discovered, key=lambda adapter: adapter.id))


__all__ = ["discover_api_export_adapters"]
