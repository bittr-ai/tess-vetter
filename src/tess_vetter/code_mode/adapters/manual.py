"""Manually curated operation adapters for stable seed coverage."""

from __future__ import annotations

import tess_vetter.api as _api
from tess_vetter.api import primitives as _api_primitives
from tess_vetter.code_mode.adapters.base import OperationAdapter
from tess_vetter.code_mode.adapters.check_wrappers import check_wrapper_functions
from tess_vetter.code_mode.operation_spec import (
    OperationCitation,
    OperationExample,
    OperationSpec,
    SafetyClass,
    SafetyRequirements,
)
from tess_vetter.code_mode.retry.wrappers import wrap_with_transient_retry

_LEGACY_MANUAL_SEED_IDS: tuple[str, ...] = (
    "code_mode.golden.vet_candidate",
    "code_mode.golden.run_periodogram",
    "code_mode.primitive.fold",
    "code_mode.primitive.median_detrend",
)


def legacy_manual_seed_ids() -> tuple[str, ...]:
    """Return legacy/manual seed operation ids in deterministic registration order."""
    return _LEGACY_MANUAL_SEED_IDS


def manual_seed_adapters() -> tuple[OperationAdapter, ...]:
    """Return stable seed adapters that are always present by design."""
    legacy = (
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
            fn=wrap_with_transient_retry(_api.vet_candidate),
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
    wrappers = tuple(
        OperationAdapter(
            spec=OperationSpec(
                id=definition.operation_id,
                name=definition.name,
                description=definition.description,
                tier_tags=("manual", "typed-check-wrapper", definition.check_id.lower()),
                safety_class=SafetyClass.GUARDED if definition.needs_network else SafetyClass.SAFE,
                safety_requirements=SafetyRequirements(needs_network=definition.needs_network),
                input_json_schema=definition.input_model.model_json_schema(mode="validation"),
                output_json_schema=definition.output_model.model_json_schema(mode="serialization"),
                citations=(
                    OperationCitation(label=f"tess_vetter.api.run_check[{definition.check_id}]"),
                ),
            ),
            fn=wrapper,
        )
        for definition, wrapper in check_wrapper_functions()
    )
    return (*legacy, *wrappers)


__all__ = ["legacy_manual_seed_ids", "manual_seed_adapters"]
