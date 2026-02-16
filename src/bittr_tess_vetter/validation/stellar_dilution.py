"""Stellar dilution + implied-companion-size physics (metrics-only).

This module centralizes the "dilution math" used for quick astrophysical
plausibility checks:
- Convert magnitudes -> flux fractions
- Convert observed depth -> true depth under dilution
- Convert true depth -> implied companion radius

Important naming conventions (to avoid confusion):
- `target_flux_fraction` is in (0, 1] and means: F_target / F_total
- `depth_correction_factor` is in [1, +inf) and means: F_total / F_target
  (i.e., multiply observed depth by this to get the true depth on the host)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# =============================================================================
# Physical Constants
# =============================================================================

# Radius conversion factors (relative to solar radius)
R_EARTH_TO_RSUN = 0.009167  # Earth radius in solar radii
R_JUP_TO_RSUN = 0.10045  # Jupiter radius in solar radii

# Thresholds for physics flags
PLANET_MAX_RADIUS_RJUP = 2.0  # Above this = likely not a planet
STELLAR_MIN_RADIUS_RSUN = 0.2  # Above this = clearly stellar

# Conversion factors for convenience
RSUN_TO_REARTH = 1.0 / R_EARTH_TO_RSUN  # ~109.1
RSUN_TO_RJUP = 1.0 / R_JUP_TO_RSUN  # ~9.96


# =============================================================================
# Models
# =============================================================================


@dataclass(frozen=True)
class HostHypothesis:
    """A candidate host star for the transit signal.

    `estimated_flux_fraction` is the fraction of *total* flux attributable to
    this source (F_host / F_total).
    """

    source_id: int
    name: str
    separation_arcsec: float
    g_mag: float | None = None
    estimated_flux_fraction: float = 1.0
    radius_rsun: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "name": self.name,
            "separation_arcsec": self.separation_arcsec,
            "g_mag": self.g_mag,
            "estimated_flux_fraction": self.estimated_flux_fraction,
            "radius_rsun": self.radius_rsun,
        }


@dataclass(frozen=True)
class DilutionScenario:
    """Depth correction + implied companion size for a given host hypothesis."""

    host: HostHypothesis
    observed_depth_ppm: float
    depth_correction_factor: float
    true_depth_ppm: float
    implied_companion_radius_rearth: float | None = None
    implied_companion_radius_rjup: float | None = None
    implied_companion_radius_rsun: float | None = None
    planet_radius_inconsistent: bool = False
    stellar_companion_likely: bool = False
    physically_impossible: bool = False
    scenario_plausibility: str = "unevaluated"

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host.to_dict(),
            "observed_depth_ppm": float(self.observed_depth_ppm),
            "depth_correction_factor": float(self.depth_correction_factor),
            "true_depth_ppm": float(self.true_depth_ppm),
            "implied_companion_radius_rearth": self.implied_companion_radius_rearth,
            "implied_companion_radius_rjup": self.implied_companion_radius_rjup,
            "implied_companion_radius_rsun": self.implied_companion_radius_rsun,
            "planet_radius_inconsistent": bool(self.planet_radius_inconsistent),
            "stellar_companion_likely": bool(self.stellar_companion_likely),
            "physically_impossible": bool(self.physically_impossible),
            "scenario_plausibility": str(self.scenario_plausibility),
        }


@dataclass(frozen=True)
class PhysicsFlags:
    """High-level plausibility summary derived from dilution scenarios."""

    planet_radius_inconsistent: bool
    requires_resolved_followup: bool
    rationale: str
    n_plausible_scenarios: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "planet_radius_inconsistent": bool(self.planet_radius_inconsistent),
            "requires_resolved_followup": bool(self.requires_resolved_followup),
            "rationale": str(self.rationale),
            "n_plausible_scenarios": int(self.n_plausible_scenarios),
        }


# =============================================================================
# Photometry helpers
# =============================================================================


def compute_target_flux_fraction_from_neighbor_mags(
    *,
    target_mag: float | None,
    neighbor_mags: list[float],
) -> float | None:
    """Compute target flux fraction (F_target/F_total) given neighbor magnitudes.

    This matches the convention historically used by
    `bittr_tess_vetter.platform.catalogs.crossmatch.compute_dilution_factor`.
    """
    if target_mag is None or not neighbor_mags:
        return None

    target_flux = 1.0
    neighbor_flux_total = sum(10 ** ((target_mag - mag) / 2.5) for mag in neighbor_mags)
    total_flux = target_flux + neighbor_flux_total
    if total_flux <= 0:
        return None
    return float(target_flux / total_flux)


def compute_flux_fraction_from_mag_list(target_mag: float, all_mags: list[float]) -> float:
    """Compute flux fraction (F_target/F_total) from a list that includes the target.

    Returns 1.0 when `all_mags` is empty or degenerate.
    """
    if not all_mags:
        return 1.0
    ref_mag = min(all_mags)
    total_flux = sum(10.0 ** ((ref_mag - m) / 2.5) for m in all_mags)
    if total_flux <= 0:
        return 1.0
    target_flux = 10.0 ** ((ref_mag - target_mag) / 2.5)
    return float(target_flux / total_flux)


def compute_depth_correction_factor_from_flux_fraction(target_flux_fraction: float) -> float:
    """Convert target flux fraction to depth correction factor (F_total/F_target)."""
    if target_flux_fraction <= 0.0:
        return 100.0
    return float(1.0 / target_flux_fraction)


# =============================================================================
# Transit/dilution physics
# =============================================================================


def compute_implied_radius(
    true_depth_frac: float,
    host_radius_rsun: float,
) -> tuple[float, float, float]:
    """Compute implied companion radius from true transit depth.

    Uses delta ≈ (R_comp / R_star)^2 for a central transit.
    """
    if true_depth_frac <= 0.0:
        return (0.0, 0.0, 0.0)

    radius_ratio = math.sqrt(true_depth_frac)
    radius_rsun = radius_ratio * host_radius_rsun
    radius_rearth = radius_rsun * RSUN_TO_REARTH
    radius_rjup = radius_rsun * RSUN_TO_RJUP
    return (float(radius_rearth), float(radius_rjup), float(radius_rsun))


def compute_dilution_scenarios(
    *,
    observed_depth_ppm: float,
    primary: HostHypothesis,
    companions: list[HostHypothesis],
) -> list[DilutionScenario]:
    """Compute dilution scenarios for each possible host."""
    scenarios: list[DilutionScenario] = []
    observed_depth_frac = float(observed_depth_ppm) / 1_000_000.0

    for host in [primary] + list(companions):
        depth_correction_factor = compute_depth_correction_factor_from_flux_fraction(
            float(host.estimated_flux_fraction)
        )
        true_depth_frac = observed_depth_frac * depth_correction_factor
        true_depth_ppm = true_depth_frac * 1_000_000.0

        r_earth: float | None = None
        r_jup: float | None = None
        r_sun: float | None = None
        planet_inconsistent = False
        stellar_likely = False
        physically_impossible = bool(true_depth_ppm > 1_000_000.0)
        scenario_plausibility = "unevaluated"

        if host.radius_rsun is not None and host.radius_rsun > 0:
            r_earth, r_jup, r_sun = compute_implied_radius(true_depth_frac, float(host.radius_rsun))
            planet_inconsistent = bool(r_jup > PLANET_MAX_RADIUS_RJUP)
            stellar_likely = bool(r_sun > STELLAR_MIN_RADIUS_RSUN)
            if physically_impossible:
                scenario_plausibility = "implausible_depth"
            elif planet_inconsistent or stellar_likely:
                scenario_plausibility = "implausible_radius"
            else:
                scenario_plausibility = "plausible"
        elif physically_impossible:
            scenario_plausibility = "implausible_depth"

        scenarios.append(
            DilutionScenario(
                host=host,
                observed_depth_ppm=float(observed_depth_ppm),
                depth_correction_factor=float(depth_correction_factor),
                true_depth_ppm=float(true_depth_ppm),
                implied_companion_radius_rearth=r_earth,
                implied_companion_radius_rjup=r_jup,
                implied_companion_radius_rsun=r_sun,
                planet_radius_inconsistent=planet_inconsistent,
                stellar_companion_likely=stellar_likely,
                physically_impossible=physically_impossible,
                scenario_plausibility=scenario_plausibility,
            )
        )

    return scenarios


def evaluate_physics_flags(scenarios: list[DilutionScenario], host_ambiguous: bool) -> PhysicsFlags:
    """Summarize plausibility/guardrails from dilution scenarios."""
    if not scenarios:
        return PhysicsFlags(
            planet_radius_inconsistent=False,
            requires_resolved_followup=bool(host_ambiguous),
            rationale=(
                "No dilution scenarios available"
                if not host_ambiguous
                else "Host ambiguous; requires resolved imaging"
            ),
            n_plausible_scenarios=0,
        )

    primary = scenarios[0]
    # Preserve historical contract: this flag tracks radius-based inconsistency,
    # not the separate depth>100% physical-impossibility condition.
    primary_inconsistent = bool(primary.planet_radius_inconsistent or primary.stellar_companion_likely)
    primary_stellar = bool(primary.stellar_companion_likely)
    n_plausible_scenarios = int(sum(1 for s in scenarios if s.scenario_plausibility == "plausible"))

    any_planet_consistent = bool(n_plausible_scenarios > 0)

    rationale_parts: list[str] = []
    if host_ambiguous:
        rationale_parts.append(
            "Host ambiguous within 1 TESS pixel; cannot determine true transit source"
        )

    if primary.physically_impossible:
        rationale_parts.append("Primary host scenario implies true depth > 100% (physically impossible)")
    elif primary_inconsistent:
        r_jup = primary.implied_companion_radius_rjup
        r_str = f"{r_jup:.2f}" if r_jup is not None else "unknown"
        rationale_parts.append(
            f"Primary host scenario implies R = {r_str} R_Jup (> 2 R_Jup planetary limit)"
        )

    if primary_stellar:
        r_sun = primary.implied_companion_radius_rsun
        r_str = f"{r_sun:.3f}" if r_sun is not None else "unknown"
        rationale_parts.append(
            f"Primary host scenario implies R = {r_str} R_Sun (stellar-sized companion)"
        )

    if any_planet_consistent and (primary_inconsistent or host_ambiguous):
        rationale_parts.append("At least one dilution scenario is consistent with a planet")

    if not rationale_parts:
        rationale_parts.append("Transit depth consistent with planetary interpretation")

    return PhysicsFlags(
        planet_radius_inconsistent=primary_inconsistent,
        requires_resolved_followup=bool(host_ambiguous),
        rationale="; ".join(rationale_parts),
        n_plausible_scenarios=n_plausible_scenarios,
    )


def build_host_hypotheses_from_profile(
    *,
    tic_id: int,
    primary_g_mag: float | None,
    primary_radius_rsun: float | None,
    close_bright_companions: list[tuple[int, float, float, float | None, float | None]],
) -> tuple[HostHypothesis, list[HostHypothesis]]:
    """Build host hypotheses from (primary + companion) metadata.

    `close_bright_companions` is a list of:
      (source_id, sep_arcsec, g_mag, delta_mag, radius_rsun)
    """
    all_mags: list[float] = []
    if primary_g_mag is not None:
        all_mags.append(float(primary_g_mag))
    for _, _, comp_g_mag, _, _ in close_bright_companions:
        if comp_g_mag is not None:
            all_mags.append(float(comp_g_mag))

    if primary_g_mag is not None and all_mags:
        primary_flux_frac = compute_flux_fraction_from_mag_list(float(primary_g_mag), all_mags)
    else:
        primary_flux_frac = 1.0

    primary = HostHypothesis(
        source_id=int(tic_id),
        name=f"TIC {int(tic_id)}",
        separation_arcsec=0.0,
        g_mag=float(primary_g_mag) if primary_g_mag is not None else None,
        estimated_flux_fraction=float(primary_flux_frac),
        radius_rsun=float(primary_radius_rsun) if primary_radius_rsun is not None else None,
    )

    companions: list[HostHypothesis] = []
    for source_id, sep_arcsec, comp_g_mag, _, radius_rsun in close_bright_companions:
        if comp_g_mag is not None and all_mags:
            flux_frac = compute_flux_fraction_from_mag_list(float(comp_g_mag), all_mags)
        else:
            flux_frac = 0.0
        companions.append(
            HostHypothesis(
                source_id=int(source_id),
                name=f"Gaia DR3 {int(source_id)}",
                separation_arcsec=float(sep_arcsec),
                g_mag=float(comp_g_mag) if comp_g_mag is not None else None,
                estimated_flux_fraction=float(flux_frac),
                radius_rsun=float(radius_rsun) if radius_rsun is not None else None,
            )
        )

    return primary, companions


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
