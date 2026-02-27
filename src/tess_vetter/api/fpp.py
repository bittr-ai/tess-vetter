"""FPP (TRICERATOPS+) API with explicit runtime knobs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict

import numpy as np
from numpy.typing import NDArray

from tess_vetter.api.contracts import callable_input_schema_from_signature, opaque_object_schema
from tess_vetter.api.references import (
    GIACALONE_2021,
    TRICERATOPS_PLUS,
    TRICERATOPS_PLUS_MULTIBAND,
    cite,
    cites,
)
from tess_vetter.platform.io import PersistentCache
from tess_vetter.validation.triceratops_fpp import calculate_fpp_handler


class FppProgressPayload(TypedDict, total=False):
    """Progress payload shape for replicate execution callbacks."""

    stage: str
    replicate: int
    status: str
    message: str
    elapsed_seconds: float
    details: NotRequired[dict[str, object]]


ExternalLCFilter = Literal["g", "r", "i", "z", "J", "H", "K"]

# Module-level references for programmatic access (generated from central registry)
REFERENCES = [
    ref.to_dict() for ref in [GIACALONE_2021, TRICERATOPS_PLUS, TRICERATOPS_PLUS_MULTIBAND]
]

DEFAULT_MC_DRAWS = 50_000
DEFAULT_WINDOW_DURATION_MULT = 2.0
DEFAULT_POINT_REDUCTION: Literal["downsample", "bin", "none"] = "downsample"
DEFAULT_TARGET_POINTS = 1500
DEFAULT_BIN_STAT: Literal["mean", "median"] = "mean"
DEFAULT_BIN_ERR: Literal["propagate", "robust"] = "propagate"
DEFAULT_MIN_FLUX_ERR = 5e-5
DEFAULT_USE_EMPIRICAL_NOISE_FLOOR = True


@dataclass(frozen=True)
class ExternalLightCurve:
    """Ground-based light curve for TRICERATOPS+ multi-band FPP."""

    time_from_midtransit_days: NDArray[np.floating[Any]]
    flux: NDArray[np.floating[Any]]
    flux_err: NDArray[np.floating[Any]]
    filter: ExternalLCFilter


@dataclass(frozen=True)
class ContrastCurve:
    """High-resolution imaging contrast curve for companion exclusion."""

    separation_arcsec: NDArray[np.floating[Any]]
    delta_mag: NDArray[np.floating[Any]]
    filter: str


def _normalize_drop_scenario(value: str | list[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [item for item in value if isinstance(item, str)]


@cites(
    cite(GIACALONE_2021, "TRICERATOPS transit false positive probability framework"),
    cite(TRICERATOPS_PLUS, "TRICERATOPS+ multi-color / external light curves"),
    cite(TRICERATOPS_PLUS_MULTIBAND, "TRICERATOPS+ multi-band photometry validation"),
)
def calculate_fpp(
    *,
    cache: PersistentCache,
    tic_id: int,
    period: float,
    t0: float,
    depth_ppm: float,
    duration_hours: float | None = None,
    sectors: list[int] | None = None,
    stellar_radius: float | None = None,
    stellar_mass: float | None = None,
    tmag: float | None = None,
    timeout_seconds: float | None = None,
    mc_draws: int | None = DEFAULT_MC_DRAWS,
    window_duration_mult: float | None = DEFAULT_WINDOW_DURATION_MULT,
    point_reduction: Literal["downsample", "bin", "none"] = DEFAULT_POINT_REDUCTION,
    target_points: int | None = DEFAULT_TARGET_POINTS,
    bin_stat: Literal["mean", "median"] = DEFAULT_BIN_STAT,
    bin_err: Literal["propagate", "robust"] = DEFAULT_BIN_ERR,
    max_points: int | None = None,
    min_flux_err: float = DEFAULT_MIN_FLUX_ERR,
    use_empirical_noise_floor: bool = DEFAULT_USE_EMPIRICAL_NOISE_FLOOR,
    drop_scenario: str | list[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
    external_lightcurves: list[ExternalLightCurve] | None = None,
    contrast_curve: ContrastCurve | None = None,
    replicates: int | None = None,
    seed: int | None = None,
    allow_network: bool = True,
    progress_hook: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Calculate FPP using TRICERATOPS+ with explicit runtime knobs."""
    extra = dict(overrides or {})
    mc_draws = extra.get("mc_draws", mc_draws)
    window_duration_mult = extra.get("window_duration_mult", window_duration_mult)
    point_reduction = extra.get("point_reduction", point_reduction)
    target_points = extra.get("target_points", target_points)
    bin_stat = extra.get("bin_stat", bin_stat)
    bin_err = extra.get("bin_err", bin_err)
    max_points = extra.get("max_points", max_points)
    min_flux_err = extra.get("min_flux_err", min_flux_err)
    use_empirical_noise_floor = extra.get("use_empirical_noise_floor", use_empirical_noise_floor)
    drop_scenario = extra.get("drop_scenario", drop_scenario)

    return calculate_fpp_handler(
        cache=cache,
        tic_id=tic_id,
        period=period,
        t0=t0,
        depth_ppm=depth_ppm,
        duration_hours=duration_hours,
        sectors=sectors,
        stellar_radius=stellar_radius,
        stellar_mass=stellar_mass,
        tmag=tmag,
        timeout_seconds=timeout_seconds,
        mc_draws=mc_draws,
        window_duration_mult=window_duration_mult,
        point_reduction=point_reduction,
        target_points=target_points,
        bin_stat=bin_stat,
        bin_err=bin_err,
        max_points=max_points,
        min_flux_err=min_flux_err,
        use_empirical_noise_floor=use_empirical_noise_floor,
        drop_scenario=_normalize_drop_scenario(drop_scenario),
        replicates=replicates,
        seed=seed,
        external_lightcurves=external_lightcurves,
        contrast_curve=contrast_curve,
        allow_network=allow_network,
        progress_hook=progress_hook,
    )


CALCULATE_FPP_CALL_SCHEMA = callable_input_schema_from_signature(calculate_fpp)
CALCULATE_FPP_OUTPUT_SCHEMA = opaque_object_schema()
