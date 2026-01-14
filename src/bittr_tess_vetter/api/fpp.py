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

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from bittr_tess_vetter.api.references import (
    GIACALONE_2021,
    TRICERATOPS_PLUS,
    TRICERATOPS_PLUS_MULTIBAND,
    cite,
    cites,
)
from bittr_tess_vetter.validation.triceratops_fpp import calculate_fpp_handler

# PersistentCache type comes from the host application.
PersistentCache = Any

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

    name: Literal["fast", "standard"]
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
    preset: Literal["fast", "standard"] = "fast",
    overrides: dict[str, Any] | None = None,
    external_lightcurves: list[ExternalLightCurve] | None = None,
    contrast_curve: ContrastCurve | None = None,
) -> dict[str, Any]:
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
        preset: "fast" or "standard" preset selection.
        overrides: Override specific preset parameters.
        external_lightcurves: Ground-based light curves for multi-band FPP
            (TRICERATOPS+ feature). Up to 4 external LCs supported.
        contrast_curve: High-resolution imaging contrast curve (not yet implemented).

    Returns:
        Dictionary with FPP results or error information.
    """
    base = FAST_PRESET if preset == "fast" else STANDARD_PRESET
    extra = overrides or {}
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
        mc_draws=int(extra.get("mc_draws", base.mc_draws))
        if base.mc_draws is not None
        else extra.get("mc_draws"),
        window_duration_mult=extra.get("window_duration_mult", base.window_duration_mult),
        max_points=extra.get("max_points", base.max_points),
        min_flux_err=float(extra.get("min_flux_err", base.min_flux_err)),
        use_empirical_noise_floor=bool(
            extra.get("use_empirical_noise_floor", base.use_empirical_noise_floor)
        ),
        external_lightcurves=external_lightcurves,
        contrast_curve=contrast_curve,
    )
