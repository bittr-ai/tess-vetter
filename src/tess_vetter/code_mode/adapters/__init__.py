"""Operation adapter building blocks and registration helpers."""

from __future__ import annotations

from collections.abc import Iterable

from tess_vetter.code_mode.adapters.base import OperationAdapter
from tess_vetter.code_mode.adapters.check_wrappers import (
    CheckWrapperDefinition,
    CheckWrapperInput,
    TypedCheckResultBase,
    check_wrapper_definitions,
    check_wrapper_functions,
    make_check_wrapper,
)
from tess_vetter.code_mode.adapters.discovery import discover_api_export_adapters
from tess_vetter.code_mode.adapters.manual import legacy_manual_seed_ids, manual_seed_adapters


def _first_wins_dedup(adapters: Iterable[OperationAdapter]) -> tuple[OperationAdapter, ...]:
    """Deduplicate adapters by id, preserving first-seen deterministic order."""
    seen_ids: set[str] = set()
    deduped: list[OperationAdapter] = []
    for adapter in adapters:
        if adapter.id in seen_ids:
            continue
        seen_ids.add(adapter.id)
        deduped.append(adapter)
    return tuple(deduped)


def default_adapter_registration_plan() -> tuple[OperationAdapter, ...]:
    """Compose manual + discovered adapters with deterministic first-wins collisions.

    Manual seed ids register first for backward compatibility; discovered canonical
    ids are appended and retained whenever they do not collide.
    """
    manual = manual_seed_adapters()
    discovered = discover_api_export_adapters()
    return _first_wins_dedup((*manual, *discovered))

__all__ = [
    "OperationAdapter",
    "CheckWrapperDefinition",
    "CheckWrapperInput",
    "TypedCheckResultBase",
    "check_wrapper_definitions",
    "check_wrapper_functions",
    "make_check_wrapper",
    "default_adapter_registration_plan",
    "discover_api_export_adapters",
    "legacy_manual_seed_ids",
    "manual_seed_adapters",
]
