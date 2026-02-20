from __future__ import annotations

import pytest
from pydantic import ValidationError

from tess_vetter.api.target import StellarParameters, Target


@pytest.fixture
def stellar_params_complete() -> StellarParameters:
    return StellarParameters(
        teff=5778.0,  # Solar
        logg=4.44,
        radius=1.0,
        mass=1.0,
        tmag=10.5,
        contamination=0.01,
        luminosity=1.0,
        metallicity=0.0,
    )


@pytest.fixture
def stellar_params_minimal() -> StellarParameters:
    return StellarParameters(radius=1.0, mass=1.0)


@pytest.fixture
def stellar_params_empty() -> StellarParameters:
    return StellarParameters()


@pytest.fixture
def target_complete(stellar_params_complete: StellarParameters) -> Target:
    return Target(
        tic_id=141914082,
        stellar=stellar_params_complete,
        ra=101.28715,
        dec=-16.71612,
        pmra=-546.01,
        pmdec=-1223.07,
        distance_pc=3.97,
        gaia_dr3_id=5167687064948389248,
        twomass_id="06450894-1642581",
        known_planet_count=7,
        toi_id="TOI-700",
    )


class TestStellarParametersValidation:
    def test_negative_teff_raises_error(self) -> None:
        with pytest.raises(ValidationError):
            StellarParameters(teff=-100.0)

    def test_negative_radius_raises_error(self) -> None:
        with pytest.raises(ValidationError):
            StellarParameters(radius=-1.0)

    def test_negative_mass_raises_error(self) -> None:
        with pytest.raises(ValidationError):
            StellarParameters(mass=-0.5)

    def test_negative_luminosity_raises_error(self) -> None:
        with pytest.raises(ValidationError):
            StellarParameters(luminosity=-1.0)

    def test_contamination_above_one_is_allowed(self) -> None:
        params = StellarParameters(contamination=1.5)
        assert params.contamination == 1.5

    def test_contamination_below_zero_raises_error(self) -> None:
        with pytest.raises(ValidationError):
            StellarParameters(contamination=-0.1)

    def test_valid_contamination_boundary_values(self) -> None:
        params_zero = StellarParameters(contamination=0.0)
        params_one = StellarParameters(contamination=1.0)

        assert params_zero.contamination == 0.0
        assert params_one.contamination == 1.0

    def test_zero_values_are_allowed(self) -> None:
        params = StellarParameters(
            teff=0.0,
            radius=0.0,
            mass=0.0,
            luminosity=0.0,
        )
        assert params.teff == 0.0
        assert params.radius == 0.0
        assert params.mass == 0.0
        assert params.luminosity == 0.0

    def test_logg_can_be_negative(self) -> None:
        params = StellarParameters(logg=-0.5)
        assert params.logg == -0.5

    def test_metallicity_can_be_negative(self) -> None:
        params = StellarParameters(metallicity=-2.0)
        assert params.metallicity == -2.0

    def test_tmag_can_be_any_value(self) -> None:
        params_bright = StellarParameters(tmag=-1.0)
        params_faint = StellarParameters(tmag=20.0)

        assert params_bright.tmag == -1.0
        assert params_faint.tmag == 20.0


class TestStellarParametersOptionalFields:
    def test_all_fields_optional(self) -> None:
        params = StellarParameters()

        assert params.teff is None
        assert params.logg is None
        assert params.radius is None
        assert params.mass is None
        assert params.tmag is None
        assert params.contamination is None
        assert params.luminosity is None
        assert params.metallicity is None

    def test_partial_fields(self) -> None:
        params = StellarParameters(teff=5500.0, radius=0.9)

        assert params.teff == 5500.0
        assert params.radius == 0.9
        assert params.mass is None

    def test_all_fields_set(self, stellar_params_complete: StellarParameters) -> None:
        assert stellar_params_complete.teff == 5778.0
        assert stellar_params_complete.logg == 4.44
        assert stellar_params_complete.radius == 1.0
        assert stellar_params_complete.mass == 1.0
        assert stellar_params_complete.tmag == 10.5
        assert stellar_params_complete.contamination == 0.01
        assert stellar_params_complete.luminosity == 1.0
        assert stellar_params_complete.metallicity == 0.0


class TestStellarParametersHasMinimumParams:
    def test_has_minimum_with_both(self, stellar_params_minimal: StellarParameters) -> None:
        assert stellar_params_minimal.has_minimum_params() is True

    def test_has_minimum_empty(self, stellar_params_empty: StellarParameters) -> None:
        assert stellar_params_empty.has_minimum_params() is False

    def test_has_minimum_missing_mass(self) -> None:
        params = StellarParameters(radius=1.0)
        assert params.has_minimum_params() is False

    def test_has_minimum_missing_radius(self) -> None:
        params = StellarParameters(mass=1.0)
        assert params.has_minimum_params() is False


class TestStellarParametersStellarDensity:
    def test_density_solar(self) -> None:
        params = StellarParameters(mass=1.0, radius=1.0)
        assert params.stellar_density_solar() == pytest.approx(1.0)

    def test_density_larger_radius(self) -> None:
        params = StellarParameters(mass=1.0, radius=2.0)
        assert params.stellar_density_solar() == pytest.approx(0.125)

    def test_density_smaller_radius(self) -> None:
        params = StellarParameters(mass=1.0, radius=0.5)
        assert params.stellar_density_solar() == pytest.approx(8.0)

    def test_density_missing_mass(self) -> None:
        params = StellarParameters(radius=1.0)
        assert params.stellar_density_solar() is None

    def test_density_missing_radius(self) -> None:
        params = StellarParameters(mass=1.0)
        assert params.stellar_density_solar() is None

    def test_density_zero_radius(self) -> None:
        params = StellarParameters(mass=1.0, radius=0.0)
        assert params.stellar_density_solar() is None

    def test_density_both_missing(self, stellar_params_empty: StellarParameters) -> None:
        assert stellar_params_empty.stellar_density_solar() is None


class TestTargetOptionalFields:
    def test_minimal_target(self) -> None:
        target = Target(tic_id=123456789)

        assert target.tic_id == 123456789
        assert target.stellar == StellarParameters()
        assert target.ra is None
        assert target.dec is None
        assert target.gaia_dr3_id is None
        assert target.known_planet_count == 0

    def test_complete_target(self, target_complete: Target) -> None:
        assert target_complete.tic_id == 141914082
        assert target_complete.ra == 101.28715
        assert target_complete.dec == -16.71612
        assert target_complete.gaia_dr3_id == 5167687064948389248
        assert target_complete.known_planet_count == 7
        assert target_complete.toi_id == "TOI-700"

    def test_has_position_both_present(self, target_complete: Target) -> None:
        assert target_complete.has_position() is True

    def test_has_position_missing(self) -> None:
        target = Target(tic_id=123)
        assert target.has_position() is False

    def test_has_position_only_ra(self) -> None:
        target = Target(tic_id=123, ra=100.0)
        assert target.has_position() is False

    def test_has_position_only_dec(self) -> None:
        target = Target(tic_id=123, dec=-20.0)
        assert target.has_position() is False


class TestTargetFromTicResponse:
    def test_from_tic_response_complete(self) -> None:
        tic_data = {
            "Teff": 5778.0,
            "logg": 4.44,
            "rad": 1.0,
            "mass": 1.0,
            "Tmag": 10.5,
            "contratio": 0.01,
            "lum": 1.0,
            "MH": 0.0,
            "ra": 101.28715,
            "dec": -16.71612,
            "pmRA": -546.01,
            "pmDEC": -1223.07,
            "d": 3.97,
            "GAIA": 5167687064948389248,
            "TWOMASS": "06450894-1642581",
        }

        target = Target.from_tic_response(141914082, tic_data)

        assert target.tic_id == 141914082
        assert target.stellar.teff == 5778.0
        assert target.stellar.radius == 1.0
        assert target.stellar.mass == 1.0
        assert target.ra == 101.28715
        assert target.dec == -16.71612
        assert target.distance_pc == 3.97
        assert target.gaia_dr3_id == 5167687064948389248

    def test_from_tic_response_partial(self) -> None:
        tic_data = {
            "Teff": 5500.0,
            "rad": 0.9,
            "ra": 100.0,
            "dec": -20.0,
        }

        target = Target.from_tic_response(123456, tic_data)

        assert target.tic_id == 123456
        assert target.stellar.teff == 5500.0
        assert target.stellar.radius == 0.9
        assert target.stellar.mass is None
        assert target.ra == 100.0
        assert target.dec == -20.0
        assert target.gaia_dr3_id is None

    def test_from_tic_response_empty(self) -> None:
        target = Target.from_tic_response(999999, {})

        assert target.tic_id == 999999
        assert target.stellar == StellarParameters()
        assert target.ra is None
        assert target.dec is None
