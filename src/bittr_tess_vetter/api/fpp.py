"""FPP (TRICERATOPS) API with explicit presets.

This module exposes two intended usage modes:
- `standard`: closer to TRICERATOPS defaults (high-fidelity, slow)
- `fast`: bounded runtime defaults suitable for interactive usage
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from bittr_tess_vetter.validation.triceratops_fpp import calculate_fpp_handler

if False:  # TYPE_CHECKING without import cost in runtime environments
    from bittr_tess_vetter.io import PersistentCache  # pragma: no cover


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


def calculate_fpp(
    *,
    cache: "PersistentCache",
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
) -> dict[str, Any]:
    """Calculate FPP using TRICERATOPS with an explicit preset.

    `standard` is intended for offline/non-interactive analysis and may take minutes.
    `fast` is intended for interactive workflows and may have higher variance.
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
        mc_draws=int(extra.get("mc_draws", base.mc_draws)) if base.mc_draws is not None else extra.get("mc_draws"),
        window_duration_mult=extra.get("window_duration_mult", base.window_duration_mult),
        max_points=extra.get("max_points", base.max_points),
        min_flux_err=float(extra.get("min_flux_err", base.min_flux_err)),
        use_empirical_noise_floor=bool(extra.get("use_empirical_noise_floor", base.use_empirical_noise_floor)),
    )

