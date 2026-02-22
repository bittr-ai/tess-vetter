"""Negative control generators for the public API.

Includes an additive typed boundary wrapper for ``generate_control`` while
preserving legacy behavior.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from tess_vetter.api.contracts import (
    callable_input_schema_from_signature,
    model_input_schema,
    model_output_schema,
)
from tess_vetter.validation.negative_controls import (  # noqa: F401
    ControlType,
    generate_control,
    generate_flux_invert,
    generate_null_inject,
    generate_phase_scramble,
    generate_time_scramble,
)


class GenerateControlTypedRequest(BaseModel):
    """Typed request payload for negative-control generation."""

    model_config = ConfigDict(extra="forbid")

    control_type: ControlType
    time: list[float]
    flux: list[float]
    flux_err: list[float]
    seed: int = 42
    period: float | None = None
    block_size: int | None = None
    n_bins: int | None = None


class GenerateControlTypedResponse(BaseModel):
    """Typed response payload for negative-control generation."""

    model_config = ConfigDict(extra="forbid")

    time: list[float]
    flux: list[float]
    flux_err: list[float]


def generate_control_typed(
    *,
    control_type: ControlType,
    time: list[float],
    flux: list[float],
    flux_err: list[float],
    seed: int = 42,
    period: float | None = None,
    block_size: int | None = None,
    n_bins: int | None = None,
) -> GenerateControlTypedResponse:
    """Typed boundary alternative for ``generate_control``."""
    payload = GenerateControlTypedRequest(
        control_type=control_type,
        time=time,
        flux=flux,
        flux_err=flux_err,
        seed=seed,
        period=period,
        block_size=block_size,
        n_bins=n_bins,
    )

    kwargs: dict[str, Any] = {}
    if payload.period is not None:
        kwargs["period"] = payload.period
    if payload.block_size is not None:
        kwargs["block_size"] = payload.block_size
    if payload.n_bins is not None:
        kwargs["n_bins"] = payload.n_bins

    out_time, out_flux, out_flux_err = generate_control(
        payload.control_type,
        np.asarray(payload.time, dtype=np.float64),
        np.asarray(payload.flux, dtype=np.float64),
        np.asarray(payload.flux_err, dtype=np.float64),
        seed=payload.seed,
        **kwargs,
    )
    return GenerateControlTypedResponse(
        time=np.asarray(out_time, dtype=np.float64).tolist(),
        flux=np.asarray(out_flux, dtype=np.float64).tolist(),
        flux_err=np.asarray(out_flux_err, dtype=np.float64).tolist(),
    )


GENERATE_CONTROL_TYPED_INPUT_SCHEMA = model_input_schema(GenerateControlTypedRequest)
GENERATE_CONTROL_TYPED_OUTPUT_SCHEMA = model_output_schema(GenerateControlTypedResponse)
GENERATE_CONTROL_TYPED_CALL_SCHEMA = callable_input_schema_from_signature(generate_control_typed)


__all__ = [
    "ControlType",
    "GENERATE_CONTROL_TYPED_CALL_SCHEMA",
    "GENERATE_CONTROL_TYPED_INPUT_SCHEMA",
    "GENERATE_CONTROL_TYPED_OUTPUT_SCHEMA",
    "GenerateControlTypedRequest",
    "GenerateControlTypedResponse",
    "generate_control",
    "generate_control_typed",
    "generate_flux_invert",
    "generate_null_inject",
    "generate_phase_scramble",
    "generate_time_scramble",
]
