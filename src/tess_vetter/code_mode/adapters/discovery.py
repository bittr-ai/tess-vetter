"""Auto-discovery helpers for adapting public API callables."""

from __future__ import annotations

import inspect
import re
from types import ModuleType

import tess_vetter.api as _api
from tess_vetter.api import primitives as _api_primitives
from tess_vetter.code_mode.adapters.base import OperationAdapter
from tess_vetter.code_mode.operation_spec import OperationCitation, OperationSpec, SafetyClass

_SNAKE_BOUNDARY_RE = re.compile(r"(?<!^)(?=[A-Z])")
_NON_IDENT_RE = re.compile(r"[^a-z0-9_]+")


def _normalize_export_name(name: str) -> str:
    normalized = _SNAKE_BOUNDARY_RE.sub("_", name).lower()
    normalized = normalized.replace("-", "_")
    normalized = _NON_IDENT_RE.sub("_", normalized).strip("_")
    if not normalized:
        normalized = "export"
    if not normalized[0].isalpha():
        normalized = f"fn_{normalized}"
    return normalized


def _iter_module_callables(module: ModuleType) -> list[tuple[str, object]]:
    export_names = tuple(getattr(module, "__all__", ()))
    callables: list[tuple[str, object]] = []
    for name in sorted(set(export_names)):
        try:
            value = getattr(module, name)
        except (AttributeError, ImportError):
            continue
        if callable(value):
            callables.append((name, value))
    return callables


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
    discovery_plan: tuple[tuple[str, ModuleType], ...] = (
        ("api", _api),
        ("primitive", _api_primitives),
    )

    for tier, module in discovery_plan:
        for export_name, fn in _iter_module_callables(module):
            operation_id = f"code_mode.{tier}.{_normalize_export_name(export_name)}"
            if operation_id in used_ids:
                continue
            used_ids.add(operation_id)
            discovered.append(
                _build_auto_adapter(operation_id, export_name, fn, module_name=module.__name__)
            )

    return tuple(sorted(discovered, key=lambda adapter: adapter.id))


__all__ = ["discover_api_export_adapters"]
