from __future__ import annotations

from bittr_tess_vetter.features.aggregates.host import build_host_plausibility_summary
from bittr_tess_vetter.validation.stellar_dilution import (
    HostHypothesis,
    compute_dilution_scenarios,
    evaluate_physics_flags,
)


def test_dilution_scenarios_flag_physically_impossible_for_faint_neighbor() -> None:
    primary = HostHypothesis(
        source_id=1,
        name="primary",
        separation_arcsec=0.0,
        estimated_flux_fraction=0.99,
        radius_rsun=1.0,
    )
    # Very faint neighbor: huge depth correction factor.
    neighbor = HostHypothesis(
        source_id=2,
        name="neighbor",
        separation_arcsec=10.0,
        estimated_flux_fraction=0.01,
        radius_rsun=1.0,
    )

    scenarios = compute_dilution_scenarios(observed_depth_ppm=500.0, primary=primary, companions=[neighbor])

    # Neighbor scenario should be flagged as stellar companion likely (implied radius too big).
    n = scenarios[1]
    assert n.depth_correction_factor > 10.0
    assert n.true_depth_ppm > 5000.0
    assert n.stellar_companion_likely or n.planet_radius_inconsistent

    flags = evaluate_physics_flags(scenarios, host_ambiguous=True)
    assert flags.requires_resolved_followup is True


def test_host_plausibility_summary_picks_best_feasible_host() -> None:
    host_plausibility = {
        "requires_resolved_followup": True,
        "rationale": "test",
        "physically_impossible_source_ids": ["2"],
        "scenarios": [
            {"source_id": "1", "depth_correction_factor": 1.0, "flux_fraction": 0.9, "true_depth_ppm": 500.0},
            {"source_id": "2", "depth_correction_factor": 100.0, "flux_fraction": 0.01, "true_depth_ppm": 50000.0},
        ],
    }
    s = build_host_plausibility_summary(host_plausibility)
    assert s["host_requires_resolved_followup"] is True
    assert s["host_physically_impossible_count"] == 1
    assert s["host_feasible_best_source_id"] == "1"


def test_host_radius_thresholds_are_strict_inequalities() -> None:
    # At exactly the stellar companion threshold (0.2 R_sun), stellar_companion_likely should be False.
    # This implies depth = (0.2)^2 = 0.04 => 40,000 ppm for a 1 R_sun host.
    depth_ppm_stellar_edge = (0.2**2) * 1_000_000.0
    primary = HostHypothesis(
        source_id=1,
        name="primary",
        separation_arcsec=0.0,
        estimated_flux_fraction=1.0,
        radius_rsun=1.0,
    )
    scenarios = compute_dilution_scenarios(
        observed_depth_ppm=depth_ppm_stellar_edge,
        primary=primary,
        companions=[],
    )
    s0 = scenarios[0]
    assert s0.implied_companion_radius_rjup is not None
    assert abs(s0.implied_companion_radius_rsun - 0.2) < 0.01
    assert s0.planet_radius_inconsistent is False
    assert s0.stellar_companion_likely is False

    # At exactly the planetary limit (2 R_Jup), planet_radius_inconsistent should still be False.
    # R_Jup -> R_sun conversion in code: R_JUP_TO_RSUN = 0.10045
    # So 2 R_Jup = 0.2009 R_sun, implying depth = (0.2009)^2 ~ 40360 ppm for a 1 R_sun host.
    depth_ppm_planet_edge = (0.2009**2) * 1_000_000.0
    scenarios2 = compute_dilution_scenarios(
        observed_depth_ppm=depth_ppm_planet_edge,
        primary=primary,
        companions=[],
    )
    s1 = scenarios2[0]
    assert s1.planet_radius_inconsistent is False

    # Slightly above the planetary limit should flip the planet inconsistency flag.
    depth_ppm_above = (0.2019**2) * 1_000_000.0
    scenarios3 = compute_dilution_scenarios(
        observed_depth_ppm=depth_ppm_above,
        primary=primary,
        companions=[],
    )
    s2 = scenarios3[0]
    assert s2.planet_radius_inconsistent is True
