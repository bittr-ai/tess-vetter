"""Typed constructor contracts for composing core API payload artifacts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

import tess_vetter.api as _api


class _EphemerisPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    period_days: float
    t0_btjd: float
    duration_hours: float


class _CandidatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ephemeris: _EphemerisPayload
    depth_ppm: float | None = None
    depth_fraction: float | None = None


class _ComposeCandidateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ephemeris: _EphemerisPayload
    depth_ppm: float | None = None
    depth_fraction: float | None = None


class _ComposeCandidateOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate: _CandidatePayload


class _LightCurvePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time: list[float]
    flux: list[float]
    flux_err: list[float] | None = None
    quality: list[int] | None = None
    valid_mask: list[bool] | None = None


class _ComposeLightCurveInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time: list[float]
    flux: list[float]
    flux_err: list[float] | None = None
    quality: list[int] | None = None
    valid_mask: list[bool] | None = None


class _ComposeLightCurveOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lc: _LightCurvePayload


class _StellarPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    teff: float | None = None
    logg: float | None = None
    radius: float | None = None
    mass: float | None = None
    tmag: float | None = None
    contamination: float | None = None
    luminosity: float | None = None
    metallicity: float | None = None


class _ComposeStellarOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stellar: _StellarPayload


class _TPFPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time: list[float]
    flux: list[list[list[float]]]
    flux_err: list[list[list[float]]] | None = None
    aperture_mask: list[list[bool]] | None = None
    quality: list[int] | None = None


class _ComposeTPFInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time: list[float]
    flux: list[list[list[float]]]
    flux_err: list[list[list[float]]] | None = None
    aperture_mask: list[list[bool]] | None = None
    quality: list[int] | None = None


class _ComposeTPFOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tpf: _TPFPayload


COMPOSE_CANDIDATE_INPUT_SCHEMA = _ComposeCandidateInput.model_json_schema(mode="validation")
COMPOSE_CANDIDATE_OUTPUT_SCHEMA = _ComposeCandidateOutput.model_json_schema(mode="serialization")
COMPOSE_LIGHTCURVE_INPUT_SCHEMA = _ComposeLightCurveInput.model_json_schema(mode="validation")
COMPOSE_LIGHTCURVE_OUTPUT_SCHEMA = _ComposeLightCurveOutput.model_json_schema(mode="serialization")
COMPOSE_STELLAR_INPUT_SCHEMA = _StellarPayload.model_json_schema(mode="validation")
COMPOSE_STELLAR_OUTPUT_SCHEMA = _ComposeStellarOutput.model_json_schema(mode="serialization")
COMPOSE_TPF_INPUT_SCHEMA = _ComposeTPFInput.model_json_schema(mode="validation")
COMPOSE_TPF_OUTPUT_SCHEMA = _ComposeTPFOutput.model_json_schema(mode="serialization")


def compose_candidate(**kwargs: Any) -> dict[str, Any]:
    payload = _ComposeCandidateInput.model_validate(kwargs)
    candidate = _api.Candidate(
        ephemeris=_api.Ephemeris(
            period_days=payload.ephemeris.period_days,
            t0_btjd=payload.ephemeris.t0_btjd,
            duration_hours=payload.ephemeris.duration_hours,
        ),
        depth_ppm=payload.depth_ppm,
        depth_fraction=payload.depth_fraction,
    )
    return _ComposeCandidateOutput(
        candidate=_CandidatePayload(
            ephemeris=_EphemerisPayload(
                period_days=candidate.ephemeris.period_days,
                t0_btjd=candidate.ephemeris.t0_btjd,
                duration_hours=candidate.ephemeris.duration_hours,
            ),
            depth_ppm=candidate.depth_ppm,
            depth_fraction=candidate.depth_fraction,
        )
    ).model_dump(mode="json")


def compose_lightcurve(**kwargs: Any) -> dict[str, Any]:
    payload = _ComposeLightCurveInput.model_validate(kwargs)
    lc = _api.LightCurve(
        time=payload.time,
        flux=payload.flux,
        flux_err=payload.flux_err,
        quality=payload.quality,
        valid_mask=payload.valid_mask,
    )
    lc.to_internal()
    return _ComposeLightCurveOutput(
        lc=_LightCurvePayload(
            time=[float(v) for v in payload.time],
            flux=[float(v) for v in payload.flux],
            flux_err=None if payload.flux_err is None else [float(v) for v in payload.flux_err],
            quality=None if payload.quality is None else [int(v) for v in payload.quality],
            valid_mask=None if payload.valid_mask is None else [bool(v) for v in payload.valid_mask],
        )
    ).model_dump(mode="json")


def compose_stellar(**kwargs: Any) -> dict[str, Any]:
    stellar = _api.StellarParams(**kwargs)
    return _ComposeStellarOutput(
        stellar=_StellarPayload(**stellar.model_dump(mode="python"))
    ).model_dump(mode="json")


def compose_tpf(**kwargs: Any) -> dict[str, Any]:
    payload = _ComposeTPFInput.model_validate(kwargs)
    _api.TPFStamp(
        time=payload.time,
        flux=payload.flux,
        flux_err=payload.flux_err,
        aperture_mask=payload.aperture_mask,
        quality=payload.quality,
    )
    return _ComposeTPFOutput(
        tpf=_TPFPayload(
            time=[float(v) for v in payload.time],
            flux=payload.flux,
            flux_err=payload.flux_err,
            aperture_mask=payload.aperture_mask,
            quality=None if payload.quality is None else [int(v) for v in payload.quality],
        )
    ).model_dump(mode="json")


__all__ = [
    "COMPOSE_CANDIDATE_INPUT_SCHEMA",
    "COMPOSE_CANDIDATE_OUTPUT_SCHEMA",
    "COMPOSE_LIGHTCURVE_INPUT_SCHEMA",
    "COMPOSE_LIGHTCURVE_OUTPUT_SCHEMA",
    "COMPOSE_STELLAR_INPUT_SCHEMA",
    "COMPOSE_STELLAR_OUTPUT_SCHEMA",
    "COMPOSE_TPF_INPUT_SCHEMA",
    "COMPOSE_TPF_OUTPUT_SCHEMA",
    "compose_candidate",
    "compose_lightcurve",
    "compose_stellar",
    "compose_tpf",
]
