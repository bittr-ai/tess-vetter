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

