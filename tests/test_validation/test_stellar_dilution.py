from __future__ import annotations

import pytest

from bittr_tess_vetter.api.catalogs import (
    compute_dilution_factor as compute_target_flux_fraction_crossmatch,
)
from bittr_tess_vetter.validation.stellar_dilution import (
    HostHypothesis,
    compute_depth_correction_factor_from_flux_fraction,
    compute_dilution_scenarios,
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

