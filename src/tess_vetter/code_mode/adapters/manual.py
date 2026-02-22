"""Manually curated operation adapters for stable seed coverage."""

from __future__ import annotations

import tess_vetter.api as _api
from tess_vetter.api import primitives as _api_primitives
from tess_vetter.api.constructor_contracts import (
    COMPOSE_CANDIDATE_INPUT_SCHEMA,
    COMPOSE_CANDIDATE_OUTPUT_SCHEMA,
    COMPOSE_LIGHTCURVE_INPUT_SCHEMA,
    COMPOSE_LIGHTCURVE_OUTPUT_SCHEMA,
    COMPOSE_STELLAR_INPUT_SCHEMA,
    COMPOSE_STELLAR_OUTPUT_SCHEMA,
    COMPOSE_TPF_INPUT_SCHEMA,
    COMPOSE_TPF_OUTPUT_SCHEMA,
    compose_candidate,
    compose_lightcurve,
    compose_stellar,
    compose_tpf,
)
from tess_vetter.api.contracts import (
    model_input_schema,
    model_output_schema,
    opaque_object_schema,
)
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


def _constructor_adapters() -> tuple[OperationAdapter, ...]:
    return (
        OperationAdapter(
            spec=OperationSpec(
                id="code_mode.primitive.compose_candidate",
                name="Compose Candidate",
                description="Compose a validated Candidate payload from ephemeris and depth fields.",
                tier_tags=("manual", "composer", "typed-constructor", "candidate"),
                safety_class=SafetyClass.SAFE,
                input_json_schema=COMPOSE_CANDIDATE_INPUT_SCHEMA,
                output_json_schema=COMPOSE_CANDIDATE_OUTPUT_SCHEMA,
                examples=(
                    OperationExample(
                        summary="Compose candidate with ephemeris and depth in ppm",
                        input={
                            "ephemeris": {
                                "period_days": 3.245,
                                "t0_btjd": 2012.345,
                                "duration_hours": 2.1,
                            },
                            "depth_ppm": 540.0,
                        },
                        output={
                            "candidate": {
                                "ephemeris": {
                                    "period_days": 3.245,
                                    "t0_btjd": 2012.345,
                                    "duration_hours": 2.1,
                                },
                                "depth_ppm": 540.0,
                                "depth_fraction": None,
                            }
                        },
                    ),
                ),
                citations=(OperationCitation(label="tess_vetter.api.Candidate"),),
            ),
            fn=compose_candidate,
        ),
        OperationAdapter(
            spec=OperationSpec(
                id="code_mode.primitive.compose_lightcurve",
                name="Compose Lightcurve",
                description="Compose a validated LightCurve payload for check wrappers.",
                tier_tags=("manual", "composer", "typed-constructor", "lightcurve"),
                safety_class=SafetyClass.SAFE,
                input_json_schema=COMPOSE_LIGHTCURVE_INPUT_SCHEMA,
                output_json_schema=COMPOSE_LIGHTCURVE_OUTPUT_SCHEMA,
                examples=(
                    OperationExample(
                        summary="Compose minimal light curve payload",
                        input={
                            "time": [1000.0, 1000.02, 1000.04],
                            "flux": [1.0, 0.9995, 1.0003],
                            "flux_err": [0.0005, 0.0005, 0.0005],
                        },
                        output={
                            "lc": {
                                "time": [1000.0, 1000.02, 1000.04],
                                "flux": [1.0, 0.9995, 1.0003],
                                "flux_err": [0.0005, 0.0005, 0.0005],
                                "quality": None,
                                "valid_mask": None,
                            }
                        },
                    ),
                ),
                citations=(OperationCitation(label="tess_vetter.api.LightCurve"),),
            ),
            fn=compose_lightcurve,
        ),
        OperationAdapter(
            spec=OperationSpec(
                id="code_mode.primitive.compose_stellar",
                name="Compose Stellar",
                description="Compose a validated StellarParams payload for duration checks.",
                tier_tags=("manual", "composer", "typed-constructor", "stellar"),
                safety_class=SafetyClass.SAFE,
                input_json_schema=COMPOSE_STELLAR_INPUT_SCHEMA,
                output_json_schema=COMPOSE_STELLAR_OUTPUT_SCHEMA,
                examples=(
                    OperationExample(
                        summary="Compose stellar payload with mass and radius",
                        input={"radius": 0.87, "mass": 0.91, "teff": 5200.0, "logg": 4.55},
                        output={
                            "stellar": {
                                "teff": 5200.0,
                                "logg": 4.55,
                                "radius": 0.87,
                                "mass": 0.91,
                                "tmag": None,
                                "contamination": None,
                                "luminosity": None,
                                "metallicity": None,
                            }
                        },
                    ),
                ),
                citations=(OperationCitation(label="tess_vetter.api.StellarParams"),),
            ),
            fn=compose_stellar,
        ),
        OperationAdapter(
            spec=OperationSpec(
                id="code_mode.primitive.compose_tpf",
                name="Compose Tpf",
                description="Compose a validated TPF payload for pixel-level checks.",
                tier_tags=("manual", "composer", "typed-constructor", "tpf"),
                safety_class=SafetyClass.SAFE,
                input_json_schema=COMPOSE_TPF_INPUT_SCHEMA,
                output_json_schema=COMPOSE_TPF_OUTPUT_SCHEMA,
                examples=(
                    OperationExample(
                        summary="Compose minimal TPF payload",
                        input={
                            "time": [1000.0, 1000.02],
                            "flux": [
                                [[120.0, 119.5], [118.9, 121.0]],
                                [[119.8, 119.3], [118.7, 120.8]],
                            ],
                        },
                        output={
                            "tpf": {
                                "time": [1000.0, 1000.02],
                                "flux": [
                                    [[120.0, 119.5], [118.9, 121.0]],
                                    [[119.8, 119.3], [118.7, 120.8]],
                                ],
                                "flux_err": None,
                                "aperture_mask": None,
                                "quality": None,
                            }
                        },
                    ),
                ),
                citations=(OperationCitation(label="tess_vetter.api.TPFStamp"),),
            ),
            fn=compose_tpf,
        ),
    )


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
                input_json_schema=opaque_object_schema(),
                output_json_schema=opaque_object_schema(),
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
                input_json_schema=opaque_object_schema(),
                output_json_schema=opaque_object_schema(),
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
                input_json_schema=opaque_object_schema(),
                output_json_schema=opaque_object_schema(),
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
                input_json_schema=opaque_object_schema(),
                output_json_schema=opaque_object_schema(),
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
                input_json_schema=model_input_schema(definition.input_model),
                output_json_schema=model_output_schema(definition.output_model),
                citations=(
                    OperationCitation(label=f"tess_vetter.api.run_check[{definition.check_id}]"),
                ),
            ),
            fn=wrapper,
        )
        for definition, wrapper in check_wrapper_functions()
    )
    return (*legacy, *_constructor_adapters(), *wrappers)


__all__ = ["legacy_manual_seed_ids", "manual_seed_adapters"]
