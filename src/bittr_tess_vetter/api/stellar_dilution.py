"""Stellar dilution + implied-companion-size physics (public API).

Re-exports metrics-only computations from `bittr_tess_vetter.validation.stellar_dilution`.

Naming conventions:
- `target_flux_fraction` is in (0, 1] and means: F_target / F_total
- `depth_correction_factor` is in [1, +inf) and means: F_total / F_target
"""

from __future__ import annotations

from bittr_tess_vetter.validation.stellar_dilution import (  # noqa: F401
    PLANET_MAX_RADIUS_RJUP,
    R_EARTH_TO_RSUN,
    R_JUP_TO_RSUN,
    RSUN_TO_REARTH,
    RSUN_TO_RJUP,
    STELLAR_MIN_RADIUS_RSUN,
    DilutionScenario,
    HostHypothesis,
    PhysicsFlags,
    build_host_hypotheses_from_profile,
    compute_depth_correction_factor_from_flux_fraction,
    compute_dilution_scenarios,
    compute_flux_fraction_from_mag_list,
    compute_implied_radius,
    compute_target_flux_fraction_from_neighbor_mags,
    evaluate_physics_flags,
)

__all__ = [
    # Models
    "HostHypothesis",
    "DilutionScenario",
    "PhysicsFlags",
    # Photometry helpers
    "compute_target_flux_fraction_from_neighbor_mags",
    "compute_flux_fraction_from_mag_list",
    "compute_depth_correction_factor_from_flux_fraction",
    # Transit/dilution physics
    "compute_implied_radius",
    "compute_dilution_scenarios",
    "evaluate_physics_flags",
    "build_host_hypotheses_from_profile",
    # Constants
    "R_EARTH_TO_RSUN",
    "R_JUP_TO_RSUN",
    "PLANET_MAX_RADIUS_RJUP",
    "STELLAR_MIN_RADIUS_RSUN",
    "RSUN_TO_REARTH",
    "RSUN_TO_RJUP",
]
