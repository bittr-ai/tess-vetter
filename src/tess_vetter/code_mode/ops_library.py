"""Lightweight operation adapters over existing public API callables."""

from __future__ import annotations

import builtins
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tess_vetter import api as _api
from tess_vetter.api import primitives as _api_primitives
from tess_vetter.code_mode.operation_spec import (
    OperationCitation,
    OperationExample,
    OperationSpec,
    SafetyClass,
    SafetyRequirements,
)


@dataclass(frozen=True)
class OperationAdapter:
    """Runtime adapter binding operation metadata to a callable."""

    spec: OperationSpec
    fn: Callable[..., Any]

    @property
    def id(self) -> str:
        return self.spec.id

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


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


def _seed_adapters() -> tuple[OperationAdapter, ...]:
    return (
        OperationAdapter(
            spec=OperationSpec(
                id="code_mode.golden.vet_candidate",
                name="Vet Candidate",
                description="Run the golden-path vetting pipeline.",
                tier_tags=("golden-path", "vetting"),
                safety_class=SafetyClass.GUARDED,
                safety_requirements=SafetyRequirements(needs_network=True),
                input_json_schema={"type": "object"},
                output_json_schema={"type": "object"},
                examples=(
                    OperationExample(
                        summary="Default vetting run",
                        input={"network": False, "preset": "default"},
                        output={"results": []},
                    ),
                ),
                citations=(
                    OperationCitation(label="tess_vetter.api.vet_candidate"),
                ),
            ),
            fn=_api.vet_candidate,
        ),
        OperationAdapter(
            spec=OperationSpec(
                id="code_mode.golden.run_periodogram",
                name="Run Periodogram",
                description="Run golden-path periodogram search.",
                tier_tags=("golden-path", "detection"),
                safety_class=SafetyClass.SAFE,
                input_json_schema={"type": "object"},
                output_json_schema={"type": "object"},
                examples=(
                    OperationExample(
                        summary="Fast TLS/auto search",
                        input={"preset": "fast", "method": "auto"},
                        output={"peaks": []},
                    ),
                ),
                citations=(
                    OperationCitation(label="tess_vetter.api.run_periodogram"),
                ),
            ),
            fn=_api.run_periodogram,
        ),
        OperationAdapter(
            spec=OperationSpec(
                id="code_mode.primitive.fold",
                name="Fold",
                description="Primitive seed for phase-folding.",
                tier_tags=("primitive-seed", "lightcurve"),
                safety_class=SafetyClass.SAFE,
                input_json_schema={"type": "object"},
                output_json_schema={"type": "object"},
                citations=(
                    OperationCitation(label="tess_vetter.api.primitives.fold"),
                ),
            ),
            fn=_api_primitives.fold,
        ),
        OperationAdapter(
            spec=OperationSpec(
                id="code_mode.primitive.median_detrend",
                name="Median Detrend",
                description="Primitive seed for robust detrending.",
                tier_tags=("primitive-seed", "lightcurve"),
                safety_class=SafetyClass.SAFE,
                input_json_schema={"type": "object"},
                output_json_schema={"type": "object"},
                citations=(
                    OperationCitation(label="tess_vetter.api.primitives.median_detrend"),
                ),
            ),
            fn=_api_primitives.median_detrend,
        ),
    )


def make_default_ops_library() -> OpsLibrary:
    """Construct the default operation library using current public API callables."""
    library = OpsLibrary()
    for adapter in _seed_adapters():
        library.register(adapter)
    return library


__all__ = [
    "OperationAdapter",
    "OpsLibrary",
    "make_default_ops_library",
]
