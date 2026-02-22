"""Typed wrappers for V01-V15 style vetting checks.

This module provides a strict adapter layer that:
- accepts structured Pydantic input payloads,
- executes legacy checks via public API entrypoints,
- returns typed Pydantic output models (never raw dicts), and
- exposes deterministic wrapper definitions for adapter registration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

import tess_vetter.api as _api

MetricScalar = float | int | str | bool | None
CheckStatus = Literal["ok", "skipped", "error"]


class EphemerisInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    period_days: float
    t0_btjd: float
    duration_hours: float


class CandidateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ephemeris: EphemerisInput
    depth_ppm: float | None = None
    depth_fraction: float | None = None


class LightCurveInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time: list[float]
    flux: list[float]
    flux_err: list[float] | None = None
    quality: list[int] | None = None
    valid_mask: list[bool] | None = None


class TPFInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time: list[float]
    flux: list[list[list[float]]]
    flux_err: list[list[list[float]]] | None = None
    aperture_mask: list[list[bool]] | None = None
    quality: list[int] | None = None


class CheckWrapperInput(BaseModel):
    """Minimal stable input contract for check wrappers."""

    model_config = ConfigDict(extra="forbid")

    lc: LightCurveInput
    candidate: CandidateInput
    stellar: dict[str, Any] | None = None
    tpf: TPFInput | None = None
    network: bool = False
    ra_deg: float | None = None
    dec_deg: float | None = None
    tic_id: int | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    preset: str = "default"


class V03CheckWrapperInput(CheckWrapperInput):
    stellar: dict[str, Any]


class V06CheckWrapperInput(CheckWrapperInput):
    network: bool = True
    ra_deg: float
    dec_deg: float


class V07CheckWrapperInput(CheckWrapperInput):
    network: bool = True
    tic_id: int


class PixelCheckWrapperInput(CheckWrapperInput):
    tpf: TPFInput


class CheckMetricsBase(BaseModel):
    """Known optional fields + passthrough extras for unknown metrics."""

    model_config = ConfigDict(extra="forbid")

    extras: dict[str, MetricScalar] = Field(
        default_factory=dict,
        description="Additional scalar metrics emitted by legacy checks.",
    )


class V01Metrics(CheckMetricsBase):
    odd_depth: float | None = None
    even_depth: float | None = None
    odd_even_diff_sigma: float | None = None


class V02Metrics(CheckMetricsBase):
    secondary_depth: float | None = None
    secondary_snr: float | None = None
    phase_secondary: float | None = None


class V03Metrics(CheckMetricsBase):
    duration_ratio: float | None = None
    expected_duration_hours: float | None = None
    observed_duration_hours: float | None = None


class V04Metrics(CheckMetricsBase):
    depth_scatter_ppm: float | None = None
    n_transits: int | None = None


class V05Metrics(CheckMetricsBase):
    v_shape_score: float | None = None
    flat_bottom_fraction: float | None = None


class V06Metrics(CheckMetricsBase):
    n_matches: int | None = None
    min_sep_arcsec: float | None = None


class V07Metrics(CheckMetricsBase):
    has_match: bool | None = None
    toi: float | None = None


class V08Metrics(CheckMetricsBase):
    centroid_shift_pixels: float | None = None
    centroid_shift_arcsec: float | None = None
    significance_sigma: float | None = None


class V09Metrics(CheckMetricsBase):
    max_depth_ppm: float | None = None
    target_depth_ppm: float | None = None
    concentration_ratio: float | None = None


class V10Metrics(CheckMetricsBase):
    stability_metric: float | None = None
    depth_variance_ppm2: float | None = None
    recommended_aperture_pixels: float | None = None


class V11Metrics(CheckMetricsBase):
    primary_signal: float | None = None
    secondary_signal: float | None = None
    secondary_primary_ratio: float | None = None


class V12Metrics(CheckMetricsBase):
    snr_half_period: float | None = None
    snr_at_period: float | None = None
    snr_double_period: float | None = None


class V13Metrics(CheckMetricsBase):
    gap_fraction: float | None = None
    n_gaps_near_transit: int | None = None


class V15Metrics(CheckMetricsBase):
    asymmetry_score: float | None = None
    asymmetry_sigma: float | None = None


class TypedCheckResultBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str
    check_name: str
    status: CheckStatus
    confidence: float | None = None
    flags: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    provenance: dict[str, MetricScalar] = Field(default_factory=dict)
    raw: dict[str, Any] | None = None


class V01Result(TypedCheckResultBase):
    check_id: Literal["V01"]
    metrics: V01Metrics


class V02Result(TypedCheckResultBase):
    check_id: Literal["V02"]
    metrics: V02Metrics


class V03Result(TypedCheckResultBase):
    check_id: Literal["V03"]
    metrics: V03Metrics


class V04Result(TypedCheckResultBase):
    check_id: Literal["V04"]
    metrics: V04Metrics


class V05Result(TypedCheckResultBase):
    check_id: Literal["V05"]
    metrics: V05Metrics


class V06Result(TypedCheckResultBase):
    check_id: Literal["V06"]
    metrics: V06Metrics


class V07Result(TypedCheckResultBase):
    check_id: Literal["V07"]
    metrics: V07Metrics


class V08Result(TypedCheckResultBase):
    check_id: Literal["V08"]
    metrics: V08Metrics


class V09Result(TypedCheckResultBase):
    check_id: Literal["V09"]
    metrics: V09Metrics


class V10Result(TypedCheckResultBase):
    check_id: Literal["V10"]
    metrics: V10Metrics


class V11Result(TypedCheckResultBase):
    check_id: Literal["V11"]
    metrics: V11Metrics


class V12Result(TypedCheckResultBase):
    check_id: Literal["V12"]
    metrics: V12Metrics


class V13Result(TypedCheckResultBase):
    check_id: Literal["V13"]
    metrics: V13Metrics


class V15Result(TypedCheckResultBase):
    check_id: Literal["V15"]
    metrics: V15Metrics


@dataclass(frozen=True)
class CheckWrapperDefinition:
    check_id: str
    name: str
    operation_id: str
    description: str
    input_model: type[CheckWrapperInput]
    metrics_model: type[CheckMetricsBase]
    output_model: type[TypedCheckResultBase]
    known_metric_fields: tuple[str, ...]
    needs_network: bool = False


def _to_api_candidate(candidate: CandidateInput) -> _api.Candidate:
    return _api.Candidate(
        ephemeris=_api.Ephemeris(
            period_days=candidate.ephemeris.period_days,
            t0_btjd=candidate.ephemeris.t0_btjd,
            duration_hours=candidate.ephemeris.duration_hours,
        ),
        depth_ppm=candidate.depth_ppm,
        depth_fraction=candidate.depth_fraction,
    )


def _to_api_lightcurve(lc: LightCurveInput) -> _api.LightCurve:
    return _api.LightCurve(
        time=lc.time,
        flux=lc.flux,
        flux_err=lc.flux_err,
        quality=lc.quality,
        valid_mask=lc.valid_mask,
    )


def _to_api_tpf(tpf: TPFInput | None) -> _api.TPFStamp | None:
    if tpf is None:
        return None
    return _api.TPFStamp(
        time=tpf.time,
        flux=tpf.flux,
        flux_err=tpf.flux_err,
        aperture_mask=tpf.aperture_mask,
        quality=tpf.quality,
    )


def _to_api_stellar(stellar: dict[str, Any] | None) -> _api.StellarParams | None:
    if stellar is None:
        return None
    return _api.StellarParams(**stellar)


def _typed_metrics(
    result: _api.CheckResult,
    *,
    metrics_model: type[CheckMetricsBase],
    known_fields: tuple[str, ...],
) -> CheckMetricsBase:
    raw_metrics = dict(result.metrics)
    known = {field: raw_metrics[field] for field in known_fields if field in raw_metrics}
    extras = {k: v for k, v in raw_metrics.items() if k not in known_fields}
    return metrics_model(**known, extras=extras)


def _build_result(
    result: _api.CheckResult,
    *,
    definition: CheckWrapperDefinition,
) -> TypedCheckResultBase:
    return definition.output_model(
        check_id=definition.check_id,
        check_name=result.name,
        status=result.status,
        confidence=result.confidence,
        metrics=_typed_metrics(
            result,
            metrics_model=definition.metrics_model,
            known_fields=definition.known_metric_fields,
        ),
        flags=list(result.flags),
        notes=list(result.notes),
        provenance=dict(result.provenance),
        raw=result.raw,
    )


def make_check_wrapper(definition: CheckWrapperDefinition) -> Callable[..., TypedCheckResultBase]:
    """Create a strict callable wrapper for one check definition."""

    def _wrapper(**kwargs: Any) -> TypedCheckResultBase:
        payload = definition.input_model.model_validate(kwargs)
        result = _api.run_check(
            lc=_to_api_lightcurve(payload.lc),
            candidate=_to_api_candidate(payload.candidate),
            check_id=definition.check_id,
            stellar=_to_api_stellar(payload.stellar),
            tpf=_to_api_tpf(payload.tpf),
            network=payload.network,
            ra_deg=payload.ra_deg,
            dec_deg=payload.dec_deg,
            tic_id=payload.tic_id,
            context=dict(payload.context),
            preset=payload.preset,
        )
        return _build_result(result, definition=definition)

    _wrapper.__name__ = f"check_wrapper_{definition.check_id.lower()}"
    _wrapper.__doc__ = f"Typed wrapper for {definition.check_id} ({definition.name})."
    return _wrapper


_CHECK_WRAPPER_DEFINITIONS: tuple[CheckWrapperDefinition, ...] = (
    CheckWrapperDefinition(
        check_id="V01",
        name="Odd-Even Depth",
        operation_id="code_mode.internal.check_v01_odd_even_depth",
        description="Typed wrapper for V01 odd/even depth check.",
        input_model=CheckWrapperInput,
        metrics_model=V01Metrics,
        output_model=V01Result,
        known_metric_fields=("odd_depth", "even_depth", "odd_even_diff_sigma"),
    ),
    CheckWrapperDefinition(
        check_id="V02",
        name="Secondary Eclipse",
        operation_id="code_mode.internal.check_v02_secondary_eclipse",
        description="Typed wrapper for V02 secondary-eclipse check.",
        input_model=CheckWrapperInput,
        metrics_model=V02Metrics,
        output_model=V02Result,
        known_metric_fields=("secondary_depth", "secondary_snr", "phase_secondary"),
    ),
    CheckWrapperDefinition(
        check_id="V03",
        name="Duration Consistency",
        operation_id="code_mode.internal.check_v03_duration_consistency",
        description="Typed wrapper for V03 duration-consistency check.",
        input_model=V03CheckWrapperInput,
        metrics_model=V03Metrics,
        output_model=V03Result,
        known_metric_fields=("duration_ratio", "expected_duration_hours", "observed_duration_hours"),
    ),
    CheckWrapperDefinition(
        check_id="V04",
        name="Depth Stability",
        operation_id="code_mode.internal.check_v04_depth_stability",
        description="Typed wrapper for V04 depth-stability check.",
        input_model=CheckWrapperInput,
        metrics_model=V04Metrics,
        output_model=V04Result,
        known_metric_fields=("depth_scatter_ppm", "n_transits"),
    ),
    CheckWrapperDefinition(
        check_id="V05",
        name="V-Shape",
        operation_id="code_mode.internal.check_v05_v_shape",
        description="Typed wrapper for V05 V-shape check.",
        input_model=CheckWrapperInput,
        metrics_model=V05Metrics,
        output_model=V05Result,
        known_metric_fields=("v_shape_score", "flat_bottom_fraction"),
    ),
    CheckWrapperDefinition(
        check_id="V06",
        name="Nearby EB Search",
        operation_id="code_mode.internal.check_v06_nearby_eb_search",
        description="Typed wrapper for V06 nearby-EB catalog check.",
        input_model=V06CheckWrapperInput,
        metrics_model=V06Metrics,
        output_model=V06Result,
        known_metric_fields=("n_matches", "min_sep_arcsec"),
        needs_network=True,
    ),
    CheckWrapperDefinition(
        check_id="V07",
        name="ExoFOP TOI Lookup",
        operation_id="code_mode.internal.check_v07_exofop_toi_lookup",
        description="Typed wrapper for V07 ExoFOP lookup check.",
        input_model=V07CheckWrapperInput,
        metrics_model=V07Metrics,
        output_model=V07Result,
        known_metric_fields=("has_match", "toi"),
        needs_network=True,
    ),
    CheckWrapperDefinition(
        check_id="V08",
        name="Centroid Shift",
        operation_id="code_mode.internal.check_v08_centroid_shift",
        description="Typed wrapper for V08 centroid-shift check.",
        input_model=PixelCheckWrapperInput,
        metrics_model=V08Metrics,
        output_model=V08Result,
        known_metric_fields=("centroid_shift_pixels", "centroid_shift_arcsec", "significance_sigma"),
    ),
    CheckWrapperDefinition(
        check_id="V09",
        name="Difference Image",
        operation_id="code_mode.internal.check_v09_difference_image",
        description="Typed wrapper for V09 difference-image localization check.",
        input_model=PixelCheckWrapperInput,
        metrics_model=V09Metrics,
        output_model=V09Result,
        known_metric_fields=("max_depth_ppm", "target_depth_ppm", "concentration_ratio"),
    ),
    CheckWrapperDefinition(
        check_id="V10",
        name="Aperture Dependence",
        operation_id="code_mode.internal.check_v10_aperture_dependence",
        description="Typed wrapper for V10 aperture-dependence check.",
        input_model=PixelCheckWrapperInput,
        metrics_model=V10Metrics,
        output_model=V10Result,
        known_metric_fields=("stability_metric", "depth_variance_ppm2", "recommended_aperture_pixels"),
    ),
    CheckWrapperDefinition(
        check_id="V11",
        name="ModShift",
        operation_id="code_mode.internal.check_v11_modshift",
        description="Typed wrapper for V11 ModShift check.",
        input_model=CheckWrapperInput,
        metrics_model=V11Metrics,
        output_model=V11Result,
        known_metric_fields=("primary_signal", "secondary_signal", "secondary_primary_ratio"),
    ),
    CheckWrapperDefinition(
        check_id="V12",
        name="SWEET",
        operation_id="code_mode.internal.check_v12_sweet",
        description="Typed wrapper for V12 SWEET check.",
        input_model=CheckWrapperInput,
        metrics_model=V12Metrics,
        output_model=V12Result,
        known_metric_fields=("snr_half_period", "snr_at_period", "snr_double_period"),
    ),
    CheckWrapperDefinition(
        check_id="V13",
        name="Data Gaps",
        operation_id="code_mode.internal.check_v13_data_gaps",
        description="Typed wrapper for V13 data-gap check.",
        input_model=CheckWrapperInput,
        metrics_model=V13Metrics,
        output_model=V13Result,
        known_metric_fields=("gap_fraction", "n_gaps_near_transit"),
    ),
    CheckWrapperDefinition(
        check_id="V15",
        name="Transit Asymmetry",
        operation_id="code_mode.internal.check_v15_transit_asymmetry",
        description="Typed wrapper for V15 transit-asymmetry check.",
        input_model=CheckWrapperInput,
        metrics_model=V15Metrics,
        output_model=V15Result,
        known_metric_fields=("asymmetry_score", "asymmetry_sigma"),
    ),
)


def check_wrapper_definitions() -> tuple[CheckWrapperDefinition, ...]:
    """Return deterministic check wrapper definitions in registration order."""
    return _CHECK_WRAPPER_DEFINITIONS


def check_wrapper_functions() -> tuple[tuple[CheckWrapperDefinition, Callable[..., TypedCheckResultBase]], ...]:
    """Return `(definition, callable)` pairs for adapter registration."""
    return tuple((definition, make_check_wrapper(definition)) for definition in _CHECK_WRAPPER_DEFINITIONS)


__all__ = [
    "CheckWrapperDefinition",
    "CheckWrapperInput",
    "TypedCheckResultBase",
    "check_wrapper_definitions",
    "check_wrapper_functions",
    "make_check_wrapper",
]
