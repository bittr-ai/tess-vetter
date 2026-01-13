"""Tests for Gaia DR3 client with offline fixtures.

All tests use fixtures to avoid network calls. The client is tested for:
- Model parsing and validation
- Cone search neighbor sorting (by separation, then brightness)
- RUWE flagging (>1.4 = elevated)
- Close binary detection (PATHOS-39 case)
- Provenance tracking
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from bittr_tess_vetter.platform.catalogs.gaia_client import (
    RUWE_ELEVATED_THRESHOLD,
    GaiaAstrophysicalParams,
    GaiaClient,
    GaiaNeighbor,
    GaiaQueryResult,
    GaiaSourceRecord,
    _compute_separation_arcsec,
    _parse_astrophysical_params,
    _parse_gaia_source,
)
from bittr_tess_vetter.platform.catalogs.models import SourceRecord

# Fixture directory
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "gaia"


def load_fixture(name: str) -> dict[str, Any]:
    """Load a JSON fixture file."""
    path = FIXTURE_DIR / name
    with open(path) as f:
        return json.load(f)


# =============================================================================
# Model Tests
# =============================================================================


class TestGaiaSourceRecord:
    """Tests for GaiaSourceRecord model."""

    def test_parse_pi_mensae_fixture(self) -> None:
        """Should parse Pi Mensae source record from fixture."""
        data = load_fixture("gaia_source_261136679.json")
        source = _parse_gaia_source(data)

        assert source.source_id == 5290699119420871680
        assert source.ra == pytest.approx(84.29122447)
        assert source.dec == pytest.approx(-80.46911858)
        assert source.parallax == pytest.approx(54.7053)
        assert source.pmra == pytest.approx(311.276)
        assert source.phot_g_mean_mag == pytest.approx(5.0876)
        assert source.ruwe == pytest.approx(1.017)
        assert source.duplicated_source is False
        assert source.non_single_star == 0

    def test_parse_pathos39_fixture(self) -> None:
        """Should parse PATHOS-39 source record from fixture."""
        data = load_fixture("gaia_source_238235254.json")
        source = _parse_gaia_source(data)

        assert source.source_id == 5486768923498851584
        assert source.ruwe == pytest.approx(1.82)
        assert source.astrometric_excess_noise == pytest.approx(0.892)

    def test_ruwe_elevated_property_low(self) -> None:
        """RUWE below threshold should not be elevated."""
        source = GaiaSourceRecord(
            source_id=1234567890,
            ra=0.0,
            dec=0.0,
            ruwe=1.0,
        )
        assert source.ruwe_elevated is False

    def test_ruwe_elevated_property_high(self) -> None:
        """RUWE above threshold should be elevated."""
        source = GaiaSourceRecord(
            source_id=1234567890,
            ra=0.0,
            dec=0.0,
            ruwe=1.82,
        )
        assert source.ruwe_elevated is True

    def test_ruwe_elevated_at_threshold(self) -> None:
        """RUWE at exactly 1.4 should not be elevated (> not >=)."""
        source = GaiaSourceRecord(
            source_id=1234567890,
            ra=0.0,
            dec=0.0,
            ruwe=RUWE_ELEVATED_THRESHOLD,
        )
        assert source.ruwe_elevated is False

    def test_ruwe_elevated_none(self) -> None:
        """RUWE of None should not be elevated."""
        source = GaiaSourceRecord(
            source_id=1234567890,
            ra=0.0,
            dec=0.0,
            ruwe=None,
        )
        assert source.ruwe_elevated is False

    def test_distance_pc_property(self) -> None:
        """Should compute distance from parallax."""
        source = GaiaSourceRecord(
            source_id=1234567890,
            ra=0.0,
            dec=0.0,
            parallax=10.0,  # 10 mas = 100 pc
        )
        assert source.distance_pc == pytest.approx(100.0)

    def test_distance_pc_none_for_zero_parallax(self) -> None:
        """Distance should be None for zero or negative parallax."""
        source = GaiaSourceRecord(
            source_id=1234567890,
            ra=0.0,
            dec=0.0,
            parallax=0.0,
        )
        assert source.distance_pc is None

    def test_distance_pc_none_for_missing_parallax(self) -> None:
        """Distance should be None for missing parallax."""
        source = GaiaSourceRecord(
            source_id=1234567890,
            ra=0.0,
            dec=0.0,
            parallax=None,
        )
        assert source.distance_pc is None

    def test_is_frozen_immutable(self) -> None:
        """GaiaSourceRecord should be immutable."""
        source = GaiaSourceRecord(
            source_id=1234567890,
            ra=0.0,
            dec=0.0,
        )
        with pytest.raises(ValidationError):
            source.ra = 1.0  # type: ignore[misc]

    def test_serializes_to_dict(self) -> None:
        """Should serialize to dictionary."""
        source = GaiaSourceRecord(
            source_id=1234567890,
            ra=84.29,
            dec=-80.47,
            phot_g_mean_mag=5.08,
            ruwe=1.02,
        )
        data = source.model_dump()
        assert data["source_id"] == 1234567890
        assert data["ra"] == 84.29
        assert data["ruwe"] == 1.02


class TestGaiaAstrophysicalParams:
    """Tests for GaiaAstrophysicalParams model."""

    def test_parse_pi_mensae_fixture(self) -> None:
        """Should parse Pi Mensae astrophysical params from fixture."""
        data = load_fixture("gaia_astrophysical_261136679.json")
        params = _parse_astrophysical_params(data)

        assert params.source_id == 5290699119420871680
        assert params.teff_gspphot == pytest.approx(5950.0)
        assert params.logg_gspphot == pytest.approx(4.35)
        assert params.radius_gspphot == pytest.approx(1.10)
        assert params.mass_flame == pytest.approx(1.07)
        assert params.lum_flame == pytest.approx(1.52)

    def test_teff_uncertainty_property(self) -> None:
        """Should compute Teff uncertainty from confidence limits."""
        params = GaiaAstrophysicalParams(
            source_id=1234567890,
            teff_gspphot=5950.0,
            teff_gspphot_lower=5900.0,
            teff_gspphot_upper=6000.0,
        )
        assert params.teff_uncertainty == pytest.approx(50.0)

    def test_teff_uncertainty_none_for_missing_limits(self) -> None:
        """Teff uncertainty should be None if limits are missing."""
        params = GaiaAstrophysicalParams(
            source_id=1234567890,
            teff_gspphot=5950.0,
            teff_gspphot_lower=None,
            teff_gspphot_upper=6000.0,
        )
        assert params.teff_uncertainty is None

    def test_radius_uncertainty_property(self) -> None:
        """Should compute radius uncertainty from confidence limits."""
        params = GaiaAstrophysicalParams(
            source_id=1234567890,
            radius_gspphot=1.10,
            radius_gspphot_lower=1.08,
            radius_gspphot_upper=1.12,
        )
        assert params.radius_uncertainty == pytest.approx(0.02)

    def test_is_frozen_immutable(self) -> None:
        """GaiaAstrophysicalParams should be immutable."""
        params = GaiaAstrophysicalParams(source_id=1234567890)
        with pytest.raises(ValidationError):
            params.teff_gspphot = 6000.0  # type: ignore[misc]


class TestGaiaNeighbor:
    """Tests for GaiaNeighbor model."""

    def test_create_neighbor(self) -> None:
        """Should create a neighbor with all fields."""
        neighbor = GaiaNeighbor(
            source_id=1234567890,
            ra=84.30,
            dec=-80.48,
            separation_arcsec=10.5,
            phot_g_mean_mag=12.5,
            delta_mag=7.4,
            ruwe=1.02,
        )
        assert neighbor.source_id == 1234567890
        assert neighbor.separation_arcsec == 10.5
        assert neighbor.delta_mag == 7.4

    def test_ruwe_elevated_property(self) -> None:
        """Neighbor RUWE flagging should work."""
        elevated = GaiaNeighbor(
            source_id=1,
            ra=0.0,
            dec=0.0,
            separation_arcsec=1.0,
            ruwe=1.5,
        )
        normal = GaiaNeighbor(
            source_id=2,
            ra=0.0,
            dec=0.0,
            separation_arcsec=1.0,
            ruwe=1.2,
        )
        assert elevated.ruwe_elevated is True
        assert normal.ruwe_elevated is False


class TestGaiaQueryResult:
    """Tests for GaiaQueryResult model."""

    def test_create_result_with_all_fields(self) -> None:
        """Should create a complete query result."""
        source = GaiaSourceRecord(
            source_id=1234567890,
            ra=84.29,
            dec=-80.47,
        )
        astrophysical = GaiaAstrophysicalParams(
            source_id=1234567890,
            teff_gspphot=5950.0,
        )
        neighbors = [
            GaiaNeighbor(
                source_id=9876543210,
                ra=84.30,
                dec=-80.48,
                separation_arcsec=10.5,
            )
        ]
        source_record = SourceRecord(name="gaia_dr3", version="dr3")

        result = GaiaQueryResult(
            source=source,
            astrophysical=astrophysical,
            neighbors=neighbors,
            source_record=source_record,
        )

        assert result.source is not None
        assert result.source.source_id == 1234567890
        assert result.astrophysical is not None
        assert result.astrophysical.teff_gspphot == 5950.0
        assert len(result.neighbors) == 1
        assert result.source_record.name == "gaia_dr3"

    def test_create_empty_result(self) -> None:
        """Should create an empty result (no source found)."""
        source_record = SourceRecord(name="gaia_dr3")

        result = GaiaQueryResult(
            source=None,
            astrophysical=None,
            neighbors=[],
            source_record=source_record,
        )

        assert result.source is None
        assert result.astrophysical is None
        assert result.neighbors == []


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestComputeSeparation:
    """Tests for _compute_separation_arcsec function."""

    def test_zero_separation(self) -> None:
        """Same position should give zero separation."""
        sep = _compute_separation_arcsec(84.29, -80.47, 84.29, -80.47)
        assert sep == pytest.approx(0.0, abs=0.001)

    def test_known_separation(self) -> None:
        """Test with a known separation."""
        # 1 degree in dec = 3600 arcsec
        sep = _compute_separation_arcsec(0.0, 0.0, 0.0, 1.0)
        assert sep == pytest.approx(3600.0, abs=1.0)

    def test_small_separation(self) -> None:
        """Test small separation typical of close binaries."""
        # Approximately 10 arcsec
        sep = _compute_separation_arcsec(0.0, 0.0, 0.0, 10.0 / 3600.0)
        assert sep == pytest.approx(10.0, abs=0.1)


# =============================================================================
# Cone Search and Neighbor Sorting Tests
# =============================================================================


class TestConeSearchNeighborSorting:
    """Tests for cone search neighbor sorting logic."""

    def test_pi_mensae_cone_neighbor_sorting(self) -> None:
        """Pi Mensae cone should have neighbors sorted by separation, then brightness."""
        data = load_fixture("gaia_cone_261136679_60arcsec.json")
        sources = data["sources"]

        # Simulate creating neighbors and sorting
        primary_mag = sources[0]["phot_g_mean_mag"]
        neighbors: list[GaiaNeighbor] = []

        for s in sources[1:]:  # Skip primary
            delta_mag = s["phot_g_mean_mag"] - primary_mag if s.get("phot_g_mean_mag") else None
            neighbors.append(
                GaiaNeighbor(
                    source_id=s["source_id"],
                    ra=s["ra"],
                    dec=s["dec"],
                    separation_arcsec=s["separation_arcsec"],
                    phot_g_mean_mag=s.get("phot_g_mean_mag"),
                    delta_mag=delta_mag,
                    ruwe=s.get("ruwe"),
                )
            )

        # Sort by separation first, then by brightness
        neighbors.sort(
            key=lambda n: (
                n.separation_arcsec,
                n.phot_g_mean_mag if n.phot_g_mean_mag else 999.0,
            )
        )

        # Verify sorting order
        assert len(neighbors) == 2
        assert neighbors[0].separation_arcsec < neighbors[1].separation_arcsec
        # Both are faint, so delta_mag should be large
        assert neighbors[0].delta_mag is not None
        assert neighbors[0].delta_mag > 10.0  # Much fainter than Pi Mensae

    def test_pathos39_close_binary_detection(self) -> None:
        """PATHOS-39 cone should show the close companion within 21 arcsec."""
        data = load_fixture("gaia_cone_238235254_60arcsec.json")
        sources = data["sources"]

        primary_mag = sources[0]["phot_g_mean_mag"]
        neighbors: list[GaiaNeighbor] = []

        for s in sources[1:]:
            delta_mag = s["phot_g_mean_mag"] - primary_mag if s.get("phot_g_mean_mag") else None
            neighbors.append(
                GaiaNeighbor(
                    source_id=s["source_id"],
                    ra=s["ra"],
                    dec=s["dec"],
                    separation_arcsec=s["separation_arcsec"],
                    phot_g_mean_mag=s.get("phot_g_mean_mag"),
                    delta_mag=delta_mag,
                    ruwe=s.get("ruwe"),
                )
            )

        # Sort
        neighbors.sort(
            key=lambda n: (
                n.separation_arcsec,
                n.phot_g_mean_mag if n.phot_g_mean_mag else 999.0,
            )
        )

        # Verify close companion detection
        assert len(neighbors) == 3

        # First neighbor should be the close companion
        close_companion = neighbors[0]
        assert close_companion.separation_arcsec < 21.0  # Within 1 TESS pixel
        assert close_companion.delta_mag is not None
        assert close_companion.delta_mag < 1.0  # Nearly equal brightness (0.83 mag)

        # The close companion has elevated RUWE
        assert close_companion.ruwe is not None
        assert close_companion.ruwe > RUWE_ELEVATED_THRESHOLD


class TestCloseBinaryGuardrails:
    """Tests for close binary / host ambiguity detection."""

    def test_pi_mensae_no_close_bright_companions(self) -> None:
        """Pi Mensae should have no close bright companions."""
        data = load_fixture("gaia_cone_261136679_60arcsec.json")
        sources = data["sources"]
        primary_mag = sources[0]["phot_g_mean_mag"]

        # Check for companions within 21 arcsec with delta_mag < 3
        close_bright = [
            s
            for s in sources[1:]
            if s["separation_arcsec"] < 21.0
            and s.get("phot_g_mean_mag") is not None
            and s["phot_g_mean_mag"] - primary_mag < 3.0
        ]

        assert len(close_bright) == 0

    def test_pathos39_has_close_bright_companion(self) -> None:
        """PATHOS-39 should have a close bright companion (host ambiguity)."""
        data = load_fixture("gaia_cone_238235254_60arcsec.json")
        sources = data["sources"]
        primary_mag = sources[0]["phot_g_mean_mag"]

        # Check for companions within 21 arcsec with delta_mag < 3
        close_bright = [
            s
            for s in sources[1:]
            if s["separation_arcsec"] < 21.0
            and s.get("phot_g_mean_mag") is not None
            and s["phot_g_mean_mag"] - primary_mag < 3.0
        ]

        assert len(close_bright) == 1
        companion = close_bright[0]
        assert companion["separation_arcsec"] < 10.0  # Very close
        assert companion["phot_g_mean_mag"] - primary_mag < 1.0  # Nearly equal brightness


# =============================================================================
# RUWE Flagging Tests
# =============================================================================


class TestRuweFlagging:
    """Tests for RUWE-based quality flagging."""

    def test_pi_mensae_normal_ruwe(self) -> None:
        """Pi Mensae should have normal RUWE (single star)."""
        data = load_fixture("gaia_source_261136679.json")
        source = _parse_gaia_source(data)

        assert source.ruwe is not None
        assert source.ruwe < RUWE_ELEVATED_THRESHOLD
        assert source.ruwe_elevated is False

    def test_pathos39_elevated_ruwe(self) -> None:
        """PATHOS-39 should have elevated RUWE (binary)."""
        data = load_fixture("gaia_source_238235254.json")
        source = _parse_gaia_source(data)

        assert source.ruwe is not None
        assert source.ruwe > RUWE_ELEVATED_THRESHOLD
        assert source.ruwe_elevated is True


# =============================================================================
# Offline Fixture Tests (No Network)
# =============================================================================


class TestGaiaClientOffline:
    """Tests that verify the client can be tested with fixtures only."""

    def test_fixture_files_exist(self) -> None:
        """All required fixture files should exist."""
        fixtures = [
            "gaia_source_261136679.json",
            "gaia_astrophysical_261136679.json",
            "gaia_cone_261136679_60arcsec.json",
            "gaia_source_238235254.json",
            "gaia_cone_238235254_60arcsec.json",
        ]
        for fixture in fixtures:
            path = FIXTURE_DIR / fixture
            assert path.exists(), f"Fixture {fixture} not found"

    def test_all_fixtures_parse_without_error(self) -> None:
        """All fixtures should parse as valid JSON."""
        for fixture_path in FIXTURE_DIR.glob("*.json"):
            data = json.loads(fixture_path.read_text())
            assert isinstance(data, dict)

    def test_source_fixtures_have_required_fields(self) -> None:
        """Source fixtures should have required fields."""
        for name in ["gaia_source_261136679.json", "gaia_source_238235254.json"]:
            data = load_fixture(name)
            assert "source_id" in data
            assert "ra" in data
            assert "dec" in data

    def test_cone_fixtures_have_sources_list(self) -> None:
        """Cone search fixtures should have sources list."""
        for name in [
            "gaia_cone_261136679_60arcsec.json",
            "gaia_cone_238235254_60arcsec.json",
        ]:
            data = load_fixture(name)
            assert "sources" in data
            assert isinstance(data["sources"], list)
            assert len(data["sources"]) > 0


# =============================================================================
# Provenance Tests
# =============================================================================


class TestGaiaProvenance:
    """Tests for provenance tracking in Gaia queries."""

    def test_query_result_has_source_record(self) -> None:
        """Query result should include provenance source record."""
        source_record = SourceRecord(
            name="gaia_dr3",
            version="dr3",
            query="source_id = 1234567890, cone = 60 arcsec",
        )

        result = GaiaQueryResult(
            source=None,
            astrophysical=None,
            neighbors=[],
            source_record=source_record,
        )

        assert result.source_record.name == "gaia_dr3"
        assert result.source_record.version == "dr3"
        assert "source_id" in result.source_record.query  # type: ignore[operator]

    def test_source_record_serializes(self) -> None:
        """Source record should serialize for storage."""
        source_record = SourceRecord(
            name="gaia_dr3",
            version="dr3",
        )
        data = source_record.model_dump()
        assert data["name"] == "gaia_dr3"
        assert data["version"] == "dr3"


# =============================================================================
# Duplicated Source and NSS Flag Tests
# =============================================================================


class TestDuplicatedSourceAndNSSFlags:
    """Tests for duplicated_source and non_single_star flags."""

    def test_pi_mensae_not_duplicated(self) -> None:
        """Pi Mensae should not be flagged as duplicated."""
        data = load_fixture("gaia_source_261136679.json")
        source = _parse_gaia_source(data)

        assert source.duplicated_source is False

    def test_pi_mensae_nss_zero(self) -> None:
        """Pi Mensae should have NSS = 0 (single star)."""
        data = load_fixture("gaia_source_261136679.json")
        source = _parse_gaia_source(data)

        assert source.non_single_star == 0

    def test_pathos39_not_duplicated(self) -> None:
        """PATHOS-39 primary should not be duplicated (it has a companion, not duplicated entry)."""
        data = load_fixture("gaia_source_238235254.json")
        source = _parse_gaia_source(data)

        # Duplicated_source is about catalog artifacts, not physical binaries
        assert source.duplicated_source is False
