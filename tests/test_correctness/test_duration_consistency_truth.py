from __future__ import annotations

from bittr_tess_vetter.domain.target import StellarParameters
from bittr_tess_vetter.validation.lc_checks import check_duration_consistency


def test_duration_consistency_density_correction_monotonicity() -> None:
    # Same observed duration, different stellar densities should shift the ratio
    # in the expected direction via T_dur ∝ rho_star^(-1/3).
    period_days = 10.0
    period_years = period_days / 365.25
    expected_duration_solar = 13.0 * (period_years ** (1.0 / 3.0))

    # Use solar expectation as the "observed" value for clarity.
    duration_hours = expected_duration_solar

    r_none = check_duration_consistency(period=period_days, duration_hours=duration_hours, stellar=None)
    assert r_none.id == "V03"
    assert r_none.details["density_corrected"] is False

    solar = StellarParameters(mass=1.0, radius=1.0)
    m_dwarf = StellarParameters(mass=0.5, radius=0.5)  # rho=4x solar -> shorter expected duration
    giant = StellarParameters(mass=1.0, radius=3.0)  # rho~0.037x solar -> longer expected duration

    r_solar = check_duration_consistency(
        period=period_days, duration_hours=duration_hours, stellar=solar
    )
    r_md = check_duration_consistency(period=period_days, duration_hours=duration_hours, stellar=m_dwarf)
    r_giant = check_duration_consistency(
        period=period_days, duration_hours=duration_hours, stellar=giant
    )

    assert r_solar.details["density_corrected"] is True
    assert r_solar.details["stellar_density_solar"] == 1.0

    ratio_solar = float(r_solar.details["duration_ratio"])
    ratio_md = float(r_md.details["duration_ratio"])
    ratio_giant = float(r_giant.details["duration_ratio"])

    # For a denser star (M dwarf), expected duration is shorter -> ratio increases.
    assert ratio_md > ratio_solar
    # For a low-density star (giant), expected duration is longer -> ratio decreases.
    assert ratio_giant < ratio_solar

