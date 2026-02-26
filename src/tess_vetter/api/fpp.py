"""FPP (TRICERATOPS+) API with explicit presets.

This module exposes two intended usage modes:
- `standard`: closer to TRICERATOPS defaults (high-fidelity, slow)
- `fast`: bounded runtime defaults suitable for interactive usage

Now supports TRICERATOPS+ multi-band FPP via external_lightcurves parameter.

References:
    Giacalone, S., et al. 2021, AJ, 161, 24
    Barrientos et al. 2025, arxiv:2508.02782 (TRICERATOPS+ multi-color)
"""

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


class FppPresetOverrides(TypedDict, total=False):
    """Supported per-call preset override keys."""

    mc_draws: int | None
    window_duration_mult: float | None
    max_points: int | None
    min_flux_err: float
    use_empirical_noise_floor: bool
    drop_scenario: str | list[str]


class FppProgressPayload(TypedDict, total=False):
    """Progress payload shape for replicate execution callbacks."""

    stage: str
    replicate: int
    status: str
    message: str
    elapsed_seconds: float
    details: NotRequired[dict[str, object]]


class FppPresetMetadata(TypedDict):
    """Preset metadata used by CLI/runtime guidance."""

    intent: str
    defaults: FppPresetOverrides
    guidance_defaults: dict[str, int | float | None]


# Valid photometric filter designations for external light curves
ExternalLCFilter = Literal["g", "r", "i", "z", "J", "H", "K"]

# Module-level references for programmatic access (generated from central registry)
REFERENCES = [
    ref.to_dict() for ref in [GIACALONE_2021, TRICERATOPS_PLUS, TRICERATOPS_PLUS_MULTIBAND]
]


@dataclass(frozen=True)
class ExternalLightCurve:
    """Ground-based light curve for TRICERATOPS+ multi-band FPP.

    TRICERATOPS+ can incorporate ground-based photometry in multiple bands
    to improve FPP estimates by constraining achromatic vs chromatic scenarios.

    References:
        Barrientos et al. 2025, arxiv:2508.02782 (TRICERATOPS+ multi-color)
    """

    time_from_midtransit_days: NDArray[np.floating[Any]]
    """Time array relative to transit midpoint, in days."""

    flux: NDArray[np.floating[Any]]
    """Normalized flux values."""

    flux_err: NDArray[np.floating[Any]]
    """Flux uncertainties."""

    filter: ExternalLCFilter
    """Photometric filter: g, r, i, z, J, H, or K."""


@dataclass(frozen=True)
class ContrastCurve:
    """High-resolution imaging contrast curve for companion exclusion.

    Used to constrain the probability of unresolved companions at
    various angular separations.
    """

    separation_arcsec: NDArray[np.floating[Any]]
    """Angular separation from target in arcseconds."""

    delta_mag: NDArray[np.floating[Any]]
    """Contrast (magnitude difference) achieved at each separation."""

    filter: str
    """Filter/band of the imaging observation."""


@dataclass(frozen=True)
class TriceratopsFppPreset:
    """Controls tradeoffs between runtime, stability, and fidelity."""

    name: Literal["fast", "standard", "tutorial"]
    mc_draws: int | None
    window_duration_mult: float | None
    max_points: int | None
    min_flux_err: float
    use_empirical_noise_floor: bool


FAST_PRESET = TriceratopsFppPreset(
    name="fast",
    mc_draws=50_000,
    window_duration_mult=2.0,
    max_points=1500,
    min_flux_err=5e-5,
    use_empirical_noise_floor=True,
)

STANDARD_PRESET = TriceratopsFppPreset(
    name="standard",
    mc_draws=1_000_000,
    window_duration_mult=None,  # no windowing
    max_points=None,  # no downsampling
    min_flux_err=0.0,  # prefer TRICERATOPS-native uncertainty treatment
    use_empirical_noise_floor=False,
)

TUTORIAL_PRESET_OVERRIDES: FppPresetOverrides = {
    "mc_draws": 200_000,
    "max_points": 3000,
    "window_duration_mult": 2.0,
    "min_flux_err": 5e-5,
    "use_empirical_noise_floor": True,
}


def _preset_defaults(preset: TriceratopsFppPreset) -> FppPresetOverrides:
    return {
        "mc_draws": preset.mc_draws,
        "window_duration_mult": preset.window_duration_mult,
        "max_points": preset.max_points,
        "min_flux_err": preset.min_flux_err,
        "use_empirical_noise_floor": preset.use_empirical_noise_floor,
    }


def get_fpp_preset_metadata() -> dict[str, FppPresetMetadata]:
    """Single source for preset intent and default knobs."""
    fast_defaults = _preset_defaults(FAST_PRESET)
    standard_defaults = _preset_defaults(STANDARD_PRESET)
    tutorial_defaults: FppPresetOverrides = {**standard_defaults, **TUTORIAL_PRESET_OVERRIDES}
    return {
        "fast": {
            "intent": "Bounded-runtime interactive compute.",
            "defaults": fast_defaults,
            "guidance_defaults": {"replicates": 1, "timeout_seconds": None},
        },
        "standard": {
            "intent": "Higher-fidelity compute with longer runtime budgets.",
            "defaults": standard_defaults,
            "guidance_defaults": {"replicates": 3, "timeout_seconds": 900.0},
        },
        "tutorial": {
            "intent": "Stability fallback when standard outputs are degenerate.",
            "defaults": tutorial_defaults,
            "guidance_defaults": {"replicates": 1, "timeout_seconds": None},
        },
    }


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
    preset: Literal["fast", "standard", "tutorial"] = "fast",
    overrides: FppPresetOverrides | None = None,
    external_lightcurves: list[ExternalLightCurve] | None = None,
    contrast_curve: ContrastCurve | None = None,
    replicates: int | None = None,
    seed: int | None = None,
    allow_network: bool = True,
    progress_hook: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Calculate FPP using TRICERATOPS+ with an explicit preset.

    `standard` is intended for offline/non-interactive analysis and may take minutes.
    `fast` is intended for interactive workflows and may have higher variance.

    Args:
        cache: Persistent cache containing light curve data.
        tic_id: TESS Input Catalog identifier.
        period: Orbital period in days.
        t0: Transit epoch in BTJD.
        depth_ppm: Transit depth in parts per million.
        duration_hours: Transit duration in hours (estimated if None).
        sectors: Specific sectors to analyze (all cached if None).
        stellar_radius: Stellar radius in solar radii.
        stellar_mass: Stellar mass in solar masses.
        tmag: TESS magnitude (for saturation check).
        timeout_seconds: Overall timeout budget.
        preset: "fast", "standard", or "tutorial" preset selection.
        overrides: Override specific preset parameters.
        external_lightcurves: Ground-based light curves for multi-band FPP
            (TRICERATOPS+ feature). Up to 4 external LCs supported.
        contrast_curve: High-resolution imaging contrast curve used to constrain unresolved companions.
        replicates: If provided and >1, run multiple independent TRICERATOPS realizations
            (with incremented seeds) and report aggregate statistics.
        seed: Base RNG seed used for replicate runs (replicate i uses seed+i).
        allow_network: Whether network-dependent TRICERATOPS initialization/prefetch is allowed.
        progress_hook: Optional callback invoked during replicate execution.

    Returns:
        Dictionary with FPP results or error information. When ``replicates`` > 1 and
        multiple runs succeed, the returned dict includes:

        - ``fpp_summary`` / ``nfpp_summary``: ``{"median","p16","p84","values"}``
        - ``n_success`` / ``n_fail`` and ``base_seed``

        Headline ``fpp``/``nfpp`` are set to the median of the successful replicates.
    """
    base = FAST_PRESET if preset == "fast" else STANDARD_PRESET
    preset_overrides = TUTORIAL_PRESET_OVERRIDES if preset == "tutorial" else {}
    extra = {**preset_overrides, **(overrides or {})}
    missing = object()

    def _optional_int(value: object, fallback: int | None) -> int | None:
        if value is missing:
            return fallback
        if value is None:
            return None
        if isinstance(value, bool):
            return fallback
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float) and np.isfinite(value) and value.is_integer():
            return int(value)
        return fallback

    def _optional_float(value: object, fallback: float | None) -> float | None:
        if value is missing:
            return fallback
        if value is None:
            return None
        if isinstance(value, bool):
            return fallback
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            return float(value)
        return fallback

    def _required_float(value: object, fallback: float) -> float:
        out = _optional_float(value, fallback)
        return fallback if out is None else out

    def _bool(value: object, fallback: bool) -> bool:
        if isinstance(value, bool):
            return value
        return fallback

    def _drop_scenario(value: object) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return value
        return None

    mc_draws = _optional_int(extra.get("mc_draws", missing), base.mc_draws)
    window_duration_mult = _optional_float(
        extra.get("window_duration_mult", missing), base.window_duration_mult
    )
    max_points = _optional_int(extra.get("max_points", missing), base.max_points)
    min_flux_err = _required_float(extra.get("min_flux_err", missing), base.min_flux_err)
    use_empirical_noise_floor = _bool(
        extra.get("use_empirical_noise_floor"), base.use_empirical_noise_floor
    )
    drop_scenario = _drop_scenario(extra.get("drop_scenario"))

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
        max_points=max_points,
        min_flux_err=min_flux_err,
        use_empirical_noise_floor=use_empirical_noise_floor,
        drop_scenario=drop_scenario,
        replicates=replicates,
        seed=seed,
        external_lightcurves=external_lightcurves,
        contrast_curve=contrast_curve,
        allow_network=allow_network,
        progress_hook=progress_hook,
    )


CALCULATE_FPP_CALL_SCHEMA = callable_input_schema_from_signature(calculate_fpp)
CALCULATE_FPP_OUTPUT_SCHEMA = opaque_object_schema()
