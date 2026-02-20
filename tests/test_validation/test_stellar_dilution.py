from __future__ import annotations

import pytest

from tess_vetter.api.catalogs import (
    compute_dilution_factor as compute_target_flux_fraction_crossmatch,
)
from tess_vetter.validation.stellar_dilution import (
    HostHypothesis,
    compute_depth_correction_factor_from_flux_fraction,
    compute_dilution_scenarios,
    evaluate_physics_flags,
    compute_target_flux_fraction_from_neighbor_mags,
)


def test_flux_fraction_convention_matches_crossmatch() -> None:
    # target_mag=10, neighbor_mag=10 => equal brightness => flux fraction = 1/(1+1) = 0.5
    f1 = compute_target_flux_fraction_from_neighbor_mags(target_mag=10.0, neighbor_mags=[10.0])
    f2 = compute_target_flux_fraction_crossmatch(10.0, [10.0])
    assert f1 is not None
    assert f2 is not None
    assert f1 == pytest.approx(0.5)
    assert f2 == pytest.approx(0.5)
    assert f1 == pytest.approx(f2)


def test_depth_correction_factor_is_inverse_of_flux_fraction() -> None:
    # If only half the flux is from the host, the true depth is twice the observed.
    flux_fraction = 0.5
    corr = compute_depth_correction_factor_from_flux_fraction(flux_fraction)
    assert corr == pytest.approx(2.0)


def test_dilution_scenario_uses_depth_correction_factor() -> None:
    primary = HostHypothesis(
        source_id=1,
        name="TIC 1",
        separation_arcsec=0.0,
        g_mag=10.0,
        estimated_flux_fraction=0.5,
        radius_rsun=1.0,
    )
    scenarios = compute_dilution_scenarios(observed_depth_ppm=100.0, primary=primary, companions=[])
    assert len(scenarios) == 1
    s = scenarios[0]
    assert s.depth_correction_factor == pytest.approx(2.0)
    assert s.true_depth_ppm == pytest.approx(200.0)


def test_scenario_plausibility_unevaluated_when_radius_missing() -> None:
    primary = HostHypothesis(
        source_id=1,
        name="TIC 1",
        separation_arcsec=0.0,
        estimated_flux_fraction=1.0,
        radius_rsun=None,
    )
    scenarios = compute_dilution_scenarios(observed_depth_ppm=800.0, primary=primary, companions=[])
    assert scenarios[0].scenario_plausibility == "unevaluated"
    assert scenarios[0].physically_impossible is False


def test_scenario_plausibility_implausible_depth_when_true_depth_exceeds_100_percent() -> None:
    primary = HostHypothesis(
        source_id=1,
        name="TIC 1",
        separation_arcsec=0.0,
        estimated_flux_fraction=0.1,
        radius_rsun=None,
    )
    scenarios = compute_dilution_scenarios(observed_depth_ppm=200_000.0, primary=primary, companions=[])
    scenario = scenarios[0]
    assert scenario.true_depth_ppm > 1_000_000.0
    assert scenario.physically_impossible is True
    assert scenario.scenario_plausibility == "implausible_depth"


def test_evaluate_physics_flags_reports_n_plausible_scenarios() -> None:
    primary = HostHypothesis(
        source_id=1,
        name="primary",
        separation_arcsec=0.0,
        estimated_flux_fraction=0.9,
        radius_rsun=1.0,
    )
    companion = HostHypothesis(
        source_id=2,
        name="neighbor",
        separation_arcsec=2.0,
        estimated_flux_fraction=0.001,
        radius_rsun=1.0,
    )
    scenarios = compute_dilution_scenarios(observed_depth_ppm=1000.0, primary=primary, companions=[companion])
    flags = evaluate_physics_flags(scenarios, host_ambiguous=True)
    assert flags.n_plausible_scenarios == 1


def test_evaluate_physics_flags_keeps_radius_flag_false_for_depth_only_impossible() -> None:
    primary = HostHypothesis(
        source_id=1,
        name="primary",
        separation_arcsec=0.0,
        estimated_flux_fraction=0.1,
        radius_rsun=None,
    )
    scenarios = compute_dilution_scenarios(observed_depth_ppm=200_000.0, primary=primary, companions=[])
    flags = evaluate_physics_flags(scenarios, host_ambiguous=False)
    assert scenarios[0].scenario_plausibility == "implausible_depth"
    assert flags.planet_radius_inconsistent is False
