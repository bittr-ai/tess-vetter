"""Operation-library composition and registry for code mode."""

from __future__ import annotations

import builtins
from collections.abc import Callable
from typing import cast

import tess_vetter.api as public_api
from tess_vetter.code_mode.adapters import (
    OperationAdapter,
    default_adapter_registration_plan,
)
from tess_vetter.code_mode.catalog import (
    DEFAULT_REQUIRED_PATHS_CAP,
    extract_required_input_paths,
)
from tess_vetter.code_mode.operation_spec import OperationCitation, OperationSpec, SafetyClass
from tess_vetter.code_mode.policy import is_actionable_api_export
from tess_vetter.code_mode.registries.operation_ids import (
    build_operation_id,
    normalize_operation_name,
)
from tess_vetter.code_mode.registries.tiering import ApiSymbol, tier_for_api_symbol


def _api_export_map() -> dict[str, tuple[str, str]]:
    export_map_factory = getattr(public_api, "_get_export_map", None)
    if not callable(export_map_factory):
        return {}
    export_map = export_map_factory()
    if not isinstance(export_map, dict):
        return {}
    return cast(dict[str, tuple[str, str]], export_map)


def _guarded_export_names() -> set[str]:
    guarded: set[str] = set()
    for attr_name in ("_MLX_GUARDED_EXPORTS", "_MATPLOTLIB_GUARDED_EXPORTS"):
        names = getattr(public_api, attr_name, ())
        if isinstance(names, (set, frozenset, list, tuple)):
            guarded.update(str(name) for name in names)
    return guarded


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


def required_input_paths_for_adapter(
    adapter: OperationAdapter,
    *,
    max_paths: int = DEFAULT_REQUIRED_PATHS_CAP,
) -> tuple[str, ...]:
    """Return deterministic required input paths derived from adapter input schema."""
    return extract_required_input_paths(adapter.spec.input_json_schema, max_paths=max_paths)


def _build_unavailable_guarded_stub(*, export_name: str) -> Callable[..., object]:
    def _raise_unavailable(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise ImportError(f"Operation '{export_name}' is unavailable in this environment")

    return _raise_unavailable


def _iter_unavailable_guarded_export_adapters() -> builtins.list[OperationAdapter]:
    guarded_exports = _guarded_export_names()
    export_map = _api_export_map()
    unavailable: builtins.list[OperationAdapter] = []

    for export_name in sorted(export_map):
        if export_name not in guarded_exports:
            continue

        module_name, _attr_name = export_map[export_name]
        symbol = ApiSymbol(module=module_name, name=export_name)
        operation_id = build_operation_id(
            tier=tier_for_api_symbol(symbol),
            name=normalize_operation_name(symbol.name),
        )

        try:
            getattr(public_api, export_name)
        except (AttributeError, ImportError, ModuleNotFoundError):
            if not is_actionable_api_export(
                export_name=export_name,
                module_name=module_name,
                value=None,
            ):
                continue
            unavailable.append(
                OperationAdapter(
                    spec=OperationSpec(
                        id=operation_id,
                        name=export_name.replace("_", " ").title(),
                        description=f"Unavailable guarded export: {module_name}.{export_name}",
                        tier_tags=("api-export", "auto-discovered", "unavailable"),
                        safety_class=SafetyClass.GUARDED,
                        citations=(OperationCitation(label=f"{module_name}.{export_name}"),),
                    ),
                    fn=_build_unavailable_guarded_stub(export_name=export_name),
                )
            )

    return unavailable


def make_default_ops_library() -> OpsLibrary:
    """Construct the default operation library from the adapter registration plan."""
    library = OpsLibrary()
    for adapter in default_adapter_registration_plan():
        library.register(adapter)
    for unavailable_adapter in _iter_unavailable_guarded_export_adapters():
        if unavailable_adapter.id not in library:
            library.register(unavailable_adapter)
    return library


__all__ = [
    "OperationAdapter",
    "OpsLibrary",
    "make_default_ops_library",
    "required_input_paths_for_adapter",
]
