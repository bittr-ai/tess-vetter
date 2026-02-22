"""Manually curated operation adapters for stable seed coverage."""

from __future__ import annotations

import tess_vetter.api as _api
from tess_vetter.api import primitives as _api_primitives
from tess_vetter.code_mode.adapters.base import OperationAdapter
from tess_vetter.code_mode.operation_spec import (
    OperationCitation,
    OperationExample,
    OperationSpec,
    SafetyClass,
    SafetyRequirements,
)


def manual_seed_adapters() -> tuple[OperationAdapter, ...]:
    """Return stable seed adapters that are always present by design."""
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


__all__ = ["manual_seed_adapters"]
