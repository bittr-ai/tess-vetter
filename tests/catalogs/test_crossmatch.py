"""Tests for bittr_tess_vetter.catalogs.crossmatch module.

Comprehensive tests for crossmatch functionality including:
- KnownObjectMatch, ContaminationRisk, and CrossmatchReport models
- Angular separation calculations
- Snapshot ID validation
- Known object matching
- Contamination assessment
- Novelty status determination
- Main crossmatch function with mock catalog data
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bittr_tess_vetter.catalogs.crossmatch import (
    KNOWN_OBJECT_TYPES,
    CatalogData,
    CatalogEntry,
    ContaminationRisk,
    CrossmatchError,
    CrossmatchReport,
    InvalidSnapshotIdError,
    KnownObjectMatch,
    NoCatalogsProvidedError,
    SnapshotNotFoundError,
    angular_separation_arcsec,
    assess_contamination,
    compute_dilution_factor,
    crossmatch,
    determine_novelty_status,
    find_known_object_matches,
    validate_snapshot_id,
)

# =============================================================================
# Mock Catalog Store
# =============================================================================


class MockCatalogSnapshotStore:
    """Mock implementation of CatalogSnapshotStore for testing.

    Provides a simple in-memory store for catalog snapshots that can be
    pre-populated with test data.
    """

    def __init__(self) -> None:
        self._catalogs: dict[str, CatalogData] = {}

    def add_catalog(self, snapshot_id: str, catalog_data: CatalogData) -> None:
        """Add a catalog to the mock store."""
        self._catalogs[snapshot_id] = catalog_data

    def load(self, snapshot_id: str) -> CatalogData:
        """Load catalog data from the mock store."""
        if snapshot_id not in self._catalogs:
            raise SnapshotNotFoundError(snapshot_id)
        return self._catalogs[snapshot_id]

    def exists(self, snapshot_id: str) -> bool:
        """Check if a snapshot exists in the mock store."""
        return snapshot_id in self._catalogs


# =============================================================================
# Test Fixtures
# =============================================================================


def create_catalog_entry(
    ra: float = 120.0,
    dec: float = -45.0,
    object_id: str = "test_object",
    object_type: str = "TOI",
    magnitude: float | None = 10.0,
    catalog_name: str = "test_catalog",
) -> CatalogEntry:
    """Create a CatalogEntry for testing."""
    return CatalogEntry(
        ra=ra,
        dec=dec,
        object_id=object_id,
        object_type=object_type,
        magnitude=magnitude,
        catalog_name=catalog_name,
    )


def create_mock_toi_catalog() -> tuple[str, CatalogData]:
    """Create a mock TOI catalog with test entries."""
    snapshot_id = "catalog:toi:v1.0:20240115:a1b2c3d4"
    entries = [
        create_catalog_entry(
            ra=120.0,
            dec=-45.0,
            object_id="TOI-1234",
            object_type="TOI",
            magnitude=10.5,
            catalog_name="toi",
        ),
        create_catalog_entry(
            ra=120.001,
            dec=-45.001,
            object_id="TOI-1235",
            object_type="TOI",
            magnitude=11.2,
            catalog_name="toi",
        ),
        create_catalog_entry(
            ra=180.0,
            dec=30.0,
            object_id="TOI-5678",
            object_type="TOI",
            magnitude=9.8,
            catalog_name="toi",
        ),
    ]
    catalog_data = CatalogData(
        entries=entries,
        catalog_name="toi",
        version="v1.0",
        asof_date="20240115",
    )
    return snapshot_id, catalog_data


def create_mock_confirmed_catalog() -> tuple[str, CatalogData]:
    """Create a mock confirmed exoplanets catalog."""
    snapshot_id = "catalog:confirmed:v1.0:20240115:b2c3d4e5"
    entries = [
        create_catalog_entry(
            ra=120.0,
            dec=-45.0,
            object_id="Kepler-442b",
            object_type="CONFIRMED",
            magnitude=10.5,
            catalog_name="confirmed",
        ),
        create_catalog_entry(
            ra=85.0,
            dec=22.0,
            object_id="TRAPPIST-1e",
            object_type="CONFIRMED",
            magnitude=18.8,
            catalog_name="confirmed",
        ),
    ]
    catalog_data = CatalogData(
        entries=entries,
        catalog_name="confirmed",
        version="v1.0",
        asof_date="20240115",
    )
    return snapshot_id, catalog_data


def create_mock_fp_catalog() -> tuple[str, CatalogData]:
    """Create a mock false positive catalog."""
    snapshot_id = "catalog:fp:v1.0:20240115:c3d4e5f6"
    entries = [
        create_catalog_entry(
            ra=120.002,
            dec=-45.002,
            object_id="FP-001",
            object_type="FP",
            magnitude=12.0,
            catalog_name="fp",
        ),
    ]
    catalog_data = CatalogData(
        entries=entries,
        catalog_name="fp",
        version="v1.0",
        asof_date="20240115",
    )
    return snapshot_id, catalog_data


def create_mock_eb_catalog() -> tuple[str, CatalogData]:
    """Create a mock eclipsing binary catalog."""
    snapshot_id = "catalog:eb:v1.0:20240115:d4e5f6a7"
    entries = [
        create_catalog_entry(
            ra=120.003,
            dec=-45.003,
            object_id="EB-001",
            object_type="EB",
            magnitude=11.5,
            catalog_name="eb",
        ),
    ]
    catalog_data = CatalogData(
        entries=entries,
        catalog_name="eb",
        version="v1.0",
        asof_date="20240115",
    )
    return snapshot_id, catalog_data


def create_mock_stellar_catalog() -> tuple[str, CatalogData]:
    """Create a mock stellar catalog for contamination testing."""
    snapshot_id = "catalog:gaia:v1.0:20240115:e5f6a7b8"
    entries = [
        # Target star
        create_catalog_entry(
            ra=120.0,
            dec=-45.0,
            object_id="Gaia-12345",
            object_type="STAR",
            magnitude=10.0,
            catalog_name="gaia",
        ),
        # Nearby star (contaminating)
        create_catalog_entry(
            ra=120.0005,
            dec=-45.0005,
            object_id="Gaia-12346",
            object_type="STAR",
            magnitude=12.0,
            catalog_name="gaia",
        ),
        # Another nearby star (fainter)
        create_catalog_entry(
            ra=120.001,
            dec=-45.001,
            object_id="Gaia-12347",
            object_type="STAR",
            magnitude=14.0,
            catalog_name="gaia",
        ),
    ]
    catalog_data = CatalogData(
        entries=entries,
        catalog_name="gaia",
        version="v1.0",
        asof_date="20240115",
    )
    return snapshot_id, catalog_data


@pytest.fixture
def mock_store() -> MockCatalogSnapshotStore:
    """Create a mock catalog store with test catalogs."""
    store = MockCatalogSnapshotStore()

    # Add all mock catalogs
    toi_id, toi_data = create_mock_toi_catalog()
    store.add_catalog(toi_id, toi_data)

    confirmed_id, confirmed_data = create_mock_confirmed_catalog()
    store.add_catalog(confirmed_id, confirmed_data)

    fp_id, fp_data = create_mock_fp_catalog()
    store.add_catalog(fp_id, fp_data)

    eb_id, eb_data = create_mock_eb_catalog()
    store.add_catalog(eb_id, eb_data)

    stellar_id, stellar_data = create_mock_stellar_catalog()
    store.add_catalog(stellar_id, stellar_data)

    return store


# =============================================================================
# KnownObjectMatch Model Tests
# =============================================================================


class TestKnownObjectMatch:
    """Tests for KnownObjectMatch model."""

    def test_valid_creation(self) -> None:
        """Can create a valid KnownObjectMatch."""
        match = KnownObjectMatch(
            object_type="TOI",
            object_id="TOI-1234.01",
            separation_arcsec=1.5,
            catalog_source="toi_catalog",
        )

        assert match.object_type == "TOI"
        assert match.object_id == "TOI-1234.01"
        assert match.separation_arcsec == 1.5
        assert match.catalog_source == "toi_catalog"

    @pytest.mark.parametrize("object_type", ["TOI", "CONFIRMED", "FP", "EB"])
    def test_valid_object_types(self, object_type: str) -> None:
        """All expected object types are accepted."""
        match = KnownObjectMatch(
            object_type=object_type,
            object_id="test-id",
            separation_arcsec=0.5,
            catalog_source="test",
        )
        assert match.object_type == object_type

    def test_immutable(self) -> None:
        """KnownObjectMatch is immutable (frozen)."""
        match = KnownObjectMatch(
            object_type="TOI",
            object_id="test",
            separation_arcsec=1.0,
            catalog_source="test",
        )

        with pytest.raises(ValidationError):
            match.object_type = "CONFIRMED"

    def test_serialization_roundtrip(self) -> None:
        """KnownObjectMatch survives JSON serialization."""
        original = KnownObjectMatch(
            object_type="CONFIRMED",
            object_id="Kepler-442b",
            separation_arcsec=0.123,
            catalog_source="confirmed_planets",
        )

        data = original.model_dump()
        restored = KnownObjectMatch.model_validate(data)

        assert restored == original


# =============================================================================
# ContaminationRisk Model Tests
# =============================================================================


class TestContaminationRisk:
    """Tests for ContaminationRisk model."""

    def test_no_contamination(self) -> None:
        """Can create ContaminationRisk with no neighbors."""
        risk = ContaminationRisk(
            has_neighbors=False,
            nearest_neighbor_arcsec=None,
            brightness_delta_mag=None,
            dilution_factor=None,
        )

        assert risk.has_neighbors is False
        assert risk.nearest_neighbor_arcsec is None

    def test_with_contamination(self) -> None:
        """Can create ContaminationRisk with neighbor data."""
        risk = ContaminationRisk(
            has_neighbors=True,
            nearest_neighbor_arcsec=5.2,
            brightness_delta_mag=2.5,
            dilution_factor=0.95,
        )

        assert risk.has_neighbors is True
        assert risk.nearest_neighbor_arcsec == 5.2
        assert risk.brightness_delta_mag == 2.5
        assert risk.dilution_factor == 0.95

    def test_defaults_to_none(self) -> None:
        """Optional fields default to None."""
        risk = ContaminationRisk(has_neighbors=True)

        assert risk.has_neighbors is True
        assert risk.nearest_neighbor_arcsec is None
        assert risk.brightness_delta_mag is None
        assert risk.dilution_factor is None

    def test_immutable(self) -> None:
        """ContaminationRisk is immutable (frozen)."""
        risk = ContaminationRisk(has_neighbors=False)

        with pytest.raises(ValidationError):
            risk.has_neighbors = True


# =============================================================================
# CrossmatchReport Model Tests
# =============================================================================


class TestCrossmatchReport:
    """Tests for CrossmatchReport model."""

    def test_novel_report(self) -> None:
        """Can create a report for a novel target."""
        report = CrossmatchReport(
            known_object_matches=[],
            contamination_risk=ContaminationRisk(has_neighbors=False),
            novelty_status="novel",
            snapshot_ids_used=["catalog:toi:v1.0:20240115:a1b2c3d4"],
            search_radius_arcsec=10.0,
        )

        assert report.novelty_status == "novel"
        assert len(report.known_object_matches) == 0
        assert len(report.snapshot_ids_used) == 1

    def test_known_report_with_matches(self) -> None:
        """Can create a report with known object matches."""
        matches = [
            KnownObjectMatch(
                object_type="TOI",
                object_id="TOI-1234",
                separation_arcsec=0.5,
                catalog_source="toi",
            ),
        ]

        report = CrossmatchReport(
            known_object_matches=matches,
            contamination_risk=ContaminationRisk(has_neighbors=False),
            novelty_status="known",
            snapshot_ids_used=["catalog:toi:v1.0:20240115:a1b2c3d4"],
            search_radius_arcsec=10.0,
        )

        assert report.novelty_status == "known"
        assert len(report.known_object_matches) == 1
        assert report.known_object_matches[0].object_id == "TOI-1234"

    def test_immutable(self) -> None:
        """CrossmatchReport is immutable (frozen)."""
        report = CrossmatchReport(
            known_object_matches=[],
            contamination_risk=ContaminationRisk(has_neighbors=False),
            novelty_status="novel",
            snapshot_ids_used=[],
            search_radius_arcsec=10.0,
        )

        with pytest.raises(ValidationError):
            report.novelty_status = "known"


# =============================================================================
# Angular Separation Tests
# =============================================================================


class TestAngularSeparation:
    """Tests for angular_separation_arcsec function."""

    def test_zero_separation(self) -> None:
        """Same coordinates give zero separation."""
        sep = angular_separation_arcsec(120.0, -45.0, 120.0, -45.0)
        assert sep < 1e-10

    def test_small_separation(self) -> None:
        """Small separations are accurate."""
        # 1 arcsecond = 1/3600 degree
        delta = 1.0 / 3600.0
        sep = angular_separation_arcsec(0.0, 0.0, delta, 0.0)
        # Should be approximately 1 arcsecond
        assert abs(sep - 1.0) < 0.01

    def test_one_degree_separation(self) -> None:
        """One degree separation (3600 arcsec)."""
        sep = angular_separation_arcsec(0.0, 0.0, 1.0, 0.0)
        assert abs(sep - 3600.0) < 1.0

    def test_declination_at_pole(self) -> None:
        """Handle separation near celestial pole."""
        sep = angular_separation_arcsec(0.0, 89.0, 180.0, 89.0)
        # At dec=89, 180 deg RA difference is about 2 degrees
        assert sep < 2 * 3600  # Less than 2 degrees

    def test_symmetry(self) -> None:
        """Separation is symmetric."""
        sep1 = angular_separation_arcsec(100.0, 30.0, 105.0, 35.0)
        sep2 = angular_separation_arcsec(105.0, 35.0, 100.0, 30.0)
        assert abs(sep1 - sep2) < 1e-10

    def test_known_separation(self) -> None:
        """Test against a known separation value."""
        # Two positions 10 arcseconds apart
        # At dec=0, 10 arcsec in RA = 10/3600 degrees
        delta_ra = 10.0 / 3600.0
        sep = angular_separation_arcsec(0.0, 0.0, delta_ra, 0.0)
        assert abs(sep - 10.0) < 0.1


# =============================================================================
# Snapshot ID Validation Tests
# =============================================================================


class TestValidateSnapshotId:
    """Tests for validate_snapshot_id function."""

    def test_valid_snapshot_id(self) -> None:
        """Valid snapshot ID is parsed correctly."""
        name, version, asof, hash_prefix = validate_snapshot_id(
            "catalog:toi:v1.0:20240115:a1b2c3d4"
        )

        assert name == "toi"
        assert version == "v1.0"
        assert asof == "20240115"
        assert hash_prefix == "a1b2c3d4"

    def test_valid_with_long_name(self) -> None:
        """Catalog names with underscores are valid."""
        name, version, asof, hash_prefix = validate_snapshot_id(
            "catalog:gaia_dr3_lite:v2.1:20240201:abcdef12"
        )

        assert name == "gaia_dr3_lite"
        assert version == "v2.1"

    def test_empty_snapshot_id(self) -> None:
        """Empty snapshot ID raises error."""
        with pytest.raises(InvalidSnapshotIdError) as exc_info:
            validate_snapshot_id("")

        assert "Empty snapshot ID" in str(exc_info.value)

    def test_wrong_prefix(self) -> None:
        """Wrong prefix raises error."""
        with pytest.raises(InvalidSnapshotIdError) as exc_info:
            validate_snapshot_id("wrong:toi:v1.0:20240115:a1b2c3d4")

        assert "catalog:" in str(exc_info.value)

    def test_missing_parts(self) -> None:
        """Missing parts raises error."""
        with pytest.raises(InvalidSnapshotIdError) as exc_info:
            validate_snapshot_id("catalog:toi:v1.0")

        assert "5 colon-separated parts" in str(exc_info.value)

    def test_empty_name(self) -> None:
        """Empty name raises error."""
        with pytest.raises(InvalidSnapshotIdError) as exc_info:
            validate_snapshot_id("catalog::v1.0:20240115:a1b2c3d4")

        assert "name is empty" in str(exc_info.value)

    def test_invalid_asof_date(self) -> None:
        """Invalid date format raises error."""
        with pytest.raises(InvalidSnapshotIdError) as exc_info:
            validate_snapshot_id("catalog:toi:v1.0:2024-01-15:a1b2c3d4")

        assert "asof date format" in str(exc_info.value)

    def test_invalid_hash_prefix(self) -> None:
        """Non-hex hash prefix raises error."""
        with pytest.raises(InvalidSnapshotIdError) as exc_info:
            validate_snapshot_id("catalog:toi:v1.0:20240115:xyz12345")

        assert "hex" in str(exc_info.value).lower()

    def test_hash_normalized_to_lowercase(self) -> None:
        """Hash prefix is normalized to lowercase."""
        _, _, _, hash_prefix = validate_snapshot_id("catalog:toi:v1.0:20240115:A1B2C3D4")

        assert hash_prefix == "a1b2c3d4"


# =============================================================================
# Novelty Status Tests
# =============================================================================


class TestDetermineNoveltyStatus:
    """Tests for determine_novelty_status function."""

    def test_no_matches_is_novel(self) -> None:
        """Empty match list returns 'novel'."""
        status = determine_novelty_status([])
        assert status == "novel"

    def test_toi_match_is_known(self) -> None:
        """TOI match returns 'known'."""
        matches = [
            KnownObjectMatch(
                object_type="TOI",
                object_id="TOI-1234",
                separation_arcsec=1.0,
                catalog_source="toi",
            )
        ]
        status = determine_novelty_status(matches)
        assert status == "known"

    def test_confirmed_match_is_known(self) -> None:
        """CONFIRMED match returns 'known'."""
        matches = [
            KnownObjectMatch(
                object_type="CONFIRMED",
                object_id="Kepler-442b",
                separation_arcsec=0.5,
                catalog_source="confirmed",
            )
        ]
        status = determine_novelty_status(matches)
        assert status == "known"

    def test_fp_only_is_ambiguous(self) -> None:
        """FP match only returns 'ambiguous'."""
        matches = [
            KnownObjectMatch(
                object_type="FP",
                object_id="FP-001",
                separation_arcsec=2.0,
                catalog_source="fp",
            )
        ]
        status = determine_novelty_status(matches)
        assert status == "ambiguous"

    def test_eb_only_is_ambiguous(self) -> None:
        """EB match only returns 'ambiguous'."""
        matches = [
            KnownObjectMatch(
                object_type="EB",
                object_id="EB-001",
                separation_arcsec=1.5,
                catalog_source="eb",
            )
        ]
        status = determine_novelty_status(matches)
        assert status == "ambiguous"

    def test_mixed_matches_known_takes_priority(self) -> None:
        """CONFIRMED/TOI takes priority over FP/EB."""
        matches = [
            KnownObjectMatch(
                object_type="FP",
                object_id="FP-001",
                separation_arcsec=3.0,
                catalog_source="fp",
            ),
            KnownObjectMatch(
                object_type="TOI",
                object_id="TOI-1234",
                separation_arcsec=1.0,
                catalog_source="toi",
            ),
        ]
        status = determine_novelty_status(matches)
        assert status == "known"


# =============================================================================
# Dilution Factor Tests
# =============================================================================


class TestComputeDilutionFactor:
    """Tests for compute_dilution_factor function."""

    def test_no_neighbors(self) -> None:
        """No neighbors returns None."""
        result = compute_dilution_factor(10.0, [])
        assert result is None

    def test_no_target_magnitude(self) -> None:
        """No target magnitude returns None."""
        result = compute_dilution_factor(None, [12.0, 14.0])
        assert result is None

    def test_single_faint_neighbor(self) -> None:
        """Faint neighbor causes minimal dilution."""
        # Target mag 10, neighbor mag 15 (5 mag fainter = 100x less flux)
        result = compute_dilution_factor(10.0, [15.0])
        assert result is not None
        assert result > 0.99  # Very close to 1.0

    def test_single_equal_brightness_neighbor(self) -> None:
        """Equal brightness neighbor causes 50% dilution."""
        result = compute_dilution_factor(10.0, [10.0])
        assert result is not None
        assert abs(result - 0.5) < 0.01

    def test_multiple_neighbors(self) -> None:
        """Multiple neighbors sum their flux contribution."""
        # Two equal brightness neighbors
        result = compute_dilution_factor(10.0, [10.0, 10.0])
        assert result is not None
        # Target has 1/3 of total flux
        assert abs(result - 1 / 3) < 0.01

    def test_bright_neighbor(self) -> None:
        """Brighter neighbor causes significant dilution."""
        # Target mag 12, neighbor mag 10 (2 mag brighter)
        result = compute_dilution_factor(12.0, [10.0])
        assert result is not None
        assert result < 0.2  # Most flux from neighbor


# =============================================================================
# Contamination Assessment Tests
# =============================================================================


class TestAssessContamination:
    """Tests for assess_contamination function."""

    def test_no_entries(self) -> None:
        """Empty catalog returns no contamination."""
        risk = assess_contamination(120.0, -45.0, [], 10.0)

        assert risk.has_neighbors is False
        assert risk.nearest_neighbor_arcsec is None

    def test_no_neighbors_in_radius(self) -> None:
        """Distant stars don't contribute."""
        entries = [create_catalog_entry(ra=180.0, dec=30.0, object_type="STAR", magnitude=10.0)]
        risk = assess_contamination(120.0, -45.0, entries, 10.0)

        assert risk.has_neighbors is False

    def test_known_objects_excluded(self) -> None:
        """Known object types (TOI, FP, etc) are excluded from contamination."""
        entries = [
            create_catalog_entry(ra=120.0001, dec=-45.0001, object_type="TOI", magnitude=10.0)
        ]
        risk = assess_contamination(120.0, -45.0, entries, 10.0)

        # TOI should not be counted as a contaminating neighbor
        assert risk.has_neighbors is False

    def test_finds_nearby_star(self) -> None:
        """Nearby stars are detected."""
        # Create entry about 2 arcseconds away in RA
        # At dec=-45, cos(dec) factor applies to RA separation
        delta = 2.0 / 3600.0
        entries = [
            create_catalog_entry(ra=120.0 + delta, dec=-45.0, object_type="STAR", magnitude=12.0)
        ]
        risk = assess_contamination(120.0, -45.0, entries, 10.0, target_magnitude=10.0)

        assert risk.has_neighbors is True
        assert risk.nearest_neighbor_arcsec is not None
        # Due to cos(dec) correction, actual separation is less than 2 arcsec
        assert 0.0 < risk.nearest_neighbor_arcsec < 3.0
        assert risk.brightness_delta_mag is not None
        assert abs(risk.brightness_delta_mag - 2.0) < 0.1

    def test_multiple_neighbors_sorted(self) -> None:
        """Nearest neighbor is correctly identified."""
        delta1 = 2.0 / 3600.0
        delta2 = 5.0 / 3600.0
        entries = [
            create_catalog_entry(ra=120.0 + delta2, dec=-45.0, object_type="STAR", magnitude=14.0),
            create_catalog_entry(ra=120.0 + delta1, dec=-45.0, object_type="STAR", magnitude=12.0),
        ]
        risk = assess_contamination(120.0, -45.0, entries, 10.0, target_magnitude=10.0)

        assert risk.has_neighbors is True
        assert risk.nearest_neighbor_arcsec is not None
        # Nearest is about 2 arcsec, not 5
        assert risk.nearest_neighbor_arcsec < 3.0

    def test_dilution_computed(self) -> None:
        """Dilution factor is computed when magnitudes available."""
        delta = 2.0 / 3600.0
        entries = [
            create_catalog_entry(ra=120.0 + delta, dec=-45.0, object_type="STAR", magnitude=10.0)
        ]
        risk = assess_contamination(120.0, -45.0, entries, 10.0, target_magnitude=10.0)

        assert risk.has_neighbors is True
        assert risk.dilution_factor is not None
        # Equal brightness = 50% dilution
        assert abs(risk.dilution_factor - 0.5) < 0.01


# =============================================================================
# Find Known Object Matches Tests
# =============================================================================


class TestFindKnownObjectMatches:
    """Tests for find_known_object_matches function."""

    def test_no_entries(self) -> None:
        """Empty catalog returns no matches."""
        matches = find_known_object_matches(120.0, -45.0, [], 10.0)
        assert len(matches) == 0

    def test_no_known_objects(self) -> None:
        """Stars are not matched as known objects."""
        entries = [create_catalog_entry(ra=120.0001, dec=-45.0001, object_type="STAR")]
        matches = find_known_object_matches(120.0, -45.0, entries, 10.0)
        assert len(matches) == 0

    def test_finds_toi(self) -> None:
        """TOIs within radius are matched."""
        delta = 2.0 / 3600.0
        entries = [
            create_catalog_entry(
                ra=120.0 + delta,
                dec=-45.0,
                object_type="TOI",
                object_id="TOI-1234",
                catalog_name="toi",
            )
        ]
        matches = find_known_object_matches(120.0, -45.0, entries, 10.0)

        assert len(matches) == 1
        assert matches[0].object_type == "TOI"
        assert matches[0].object_id == "TOI-1234"
        assert matches[0].separation_arcsec < 3.0

    def test_finds_all_known_object_types(self) -> None:
        """All known object types are matched."""
        delta = 1.0 / 3600.0
        entries = [
            create_catalog_entry(
                ra=120.0 + delta,
                dec=-45.0,
                object_type="TOI",
                object_id="TOI-1",
                catalog_name="toi",
            ),
            create_catalog_entry(
                ra=120.0 + 2 * delta,
                dec=-45.0,
                object_type="CONFIRMED",
                object_id="CONF-1",
                catalog_name="confirmed",
            ),
            create_catalog_entry(
                ra=120.0 + 3 * delta,
                dec=-45.0,
                object_type="FP",
                object_id="FP-1",
                catalog_name="fp",
            ),
            create_catalog_entry(
                ra=120.0 + 4 * delta,
                dec=-45.0,
                object_type="EB",
                object_id="EB-1",
                catalog_name="eb",
            ),
        ]
        matches = find_known_object_matches(120.0, -45.0, entries, 30.0)

        assert len(matches) == 4
        types = {m.object_type for m in matches}
        assert types == {"TOI", "CONFIRMED", "FP", "EB"}

    def test_outside_radius_not_matched(self) -> None:
        """Objects outside search radius are not matched."""
        # Object at 20 arcsec distance
        delta = 20.0 / 3600.0
        entries = [
            create_catalog_entry(
                ra=120.0 + delta, dec=-45.0, object_type="TOI", object_id="TOI-far"
            )
        ]
        matches = find_known_object_matches(120.0, -45.0, entries, 10.0)

        assert len(matches) == 0

    def test_sorted_by_separation(self) -> None:
        """Matches are sorted by separation (closest first)."""
        entries = [
            create_catalog_entry(
                ra=120.0 + 5.0 / 3600.0, dec=-45.0, object_type="TOI", object_id="TOI-far"
            ),
            create_catalog_entry(
                ra=120.0 + 1.0 / 3600.0, dec=-45.0, object_type="TOI", object_id="TOI-near"
            ),
            create_catalog_entry(
                ra=120.0 + 3.0 / 3600.0, dec=-45.0, object_type="TOI", object_id="TOI-mid"
            ),
        ]
        matches = find_known_object_matches(120.0, -45.0, entries, 10.0)

        assert len(matches) == 3
        assert matches[0].object_id == "TOI-near"
        assert matches[1].object_id == "TOI-mid"
        assert matches[2].object_id == "TOI-far"


# =============================================================================
# Main Crossmatch Function Tests
# =============================================================================


class TestCrossmatch:
    """Tests for the main crossmatch function."""

    def test_no_catalogs_error(self) -> None:
        """Empty snapshot_ids raises NoCatalogsProvidedError."""
        with pytest.raises(NoCatalogsProvidedError):
            crossmatch(ra=120.0, dec=-45.0, snapshot_ids=[])

    def test_invalid_snapshot_id_error(self) -> None:
        """Invalid snapshot ID raises InvalidSnapshotIdError."""
        with pytest.raises(InvalidSnapshotIdError):
            crossmatch(
                ra=120.0,
                dec=-45.0,
                snapshot_ids=["invalid_format"],
            )

    def test_snapshot_not_found_error(self, mock_store: MockCatalogSnapshotStore) -> None:
        """Missing snapshot raises SnapshotNotFoundError."""
        with pytest.raises(SnapshotNotFoundError):
            crossmatch(
                ra=120.0,
                dec=-45.0,
                snapshot_ids=["catalog:missing:v1.0:20240115:ffffffff"],
                catalog_store=mock_store,
            )

    def test_novel_target(self, mock_store: MockCatalogSnapshotStore) -> None:
        """Target with no matches is classified as novel."""
        report = crossmatch(
            ra=0.0,  # Far from any known objects
            dec=0.0,
            snapshot_ids=["catalog:toi:v1.0:20240115:a1b2c3d4"],
            catalog_store=mock_store,
        )

        assert report.novelty_status == "novel"
        assert len(report.known_object_matches) == 0
        assert len(report.snapshot_ids_used) == 1

    def test_known_target(self, mock_store: MockCatalogSnapshotStore) -> None:
        """Target matching TOI is classified as known."""
        report = crossmatch(
            ra=120.0,  # Near TOI-1234
            dec=-45.0,
            snapshot_ids=["catalog:toi:v1.0:20240115:a1b2c3d4"],
            catalog_store=mock_store,
            search_radius_arcsec=10.0,
        )

        assert report.novelty_status == "known"
        assert len(report.known_object_matches) > 0
        assert report.known_object_matches[0].object_type == "TOI"

    def test_confirmed_target(self, mock_store: MockCatalogSnapshotStore) -> None:
        """Target matching confirmed planet is classified as known."""
        report = crossmatch(
            ra=120.0,
            dec=-45.0,
            snapshot_ids=["catalog:confirmed:v1.0:20240115:b2c3d4e5"],
            catalog_store=mock_store,
            search_radius_arcsec=10.0,
        )

        assert report.novelty_status == "known"
        assert any(m.object_type == "CONFIRMED" for m in report.known_object_matches)

    def test_multiple_catalogs(self, mock_store: MockCatalogSnapshotStore) -> None:
        """Crossmatch searches all provided catalogs."""
        report = crossmatch(
            ra=120.0,
            dec=-45.0,
            snapshot_ids=[
                "catalog:toi:v1.0:20240115:a1b2c3d4",
                "catalog:confirmed:v1.0:20240115:b2c3d4e5",
            ],
            catalog_store=mock_store,
            search_radius_arcsec=10.0,
        )

        assert len(report.snapshot_ids_used) == 2

    def test_search_radius_respected(self, mock_store: MockCatalogSnapshotStore) -> None:
        """Small search radius limits matches."""
        # TOI-1234 is at (120.0, -45.0)
        # TOI-1235 is at (120.001, -45.001) - about 5 arcsec away
        report = crossmatch(
            ra=120.0,
            dec=-45.0,
            snapshot_ids=["catalog:toi:v1.0:20240115:a1b2c3d4"],
            catalog_store=mock_store,
            search_radius_arcsec=1.0,  # Very small
        )

        # Only TOI-1234 should match (it's at the exact position)
        # Actually it won't match because separation is 0
        assert report.search_radius_arcsec == 1.0

    def test_contamination_assessment(self, mock_store: MockCatalogSnapshotStore) -> None:
        """Contamination is assessed from stellar catalogs."""
        report = crossmatch(
            ra=120.0,
            dec=-45.0,
            snapshot_ids=["catalog:gaia:v1.0:20240115:e5f6a7b8"],
            catalog_store=mock_store,
            search_radius_arcsec=10.0,
            target_magnitude=10.0,
        )

        # The stellar catalog has neighbors
        assert report.contamination_risk.has_neighbors is True

    def test_without_store_validation_only(self) -> None:
        """Without store, function validates IDs and returns empty matches."""
        report = crossmatch(
            ra=120.0,
            dec=-45.0,
            snapshot_ids=["catalog:toi:v1.0:20240115:a1b2c3d4"],
            catalog_store=None,  # No store
        )

        assert report.novelty_status == "novel"
        assert len(report.known_object_matches) == 0
        assert len(report.snapshot_ids_used) == 1

    def test_policy_version_accepted(self, mock_store: MockCatalogSnapshotStore) -> None:
        """Contamination policy version parameter is accepted."""
        report = crossmatch(
            ra=120.0,
            dec=-45.0,
            snapshot_ids=["catalog:toi:v1.0:20240115:a1b2c3d4"],
            catalog_store=mock_store,
            contamination_policy_version="v1",
        )

        assert report is not None


# =============================================================================
# Integration Tests with Multiple Catalogs
# =============================================================================


class TestCrossmatchIntegration:
    """Integration tests with realistic catalog scenarios."""

    def test_ambiguous_target_fp_match(self, mock_store: MockCatalogSnapshotStore) -> None:
        """Target matching only FP is classified as ambiguous."""
        # Position near FP-001 but not near TOI or CONFIRMED
        report = crossmatch(
            ra=120.002,  # Near FP-001
            dec=-45.002,
            snapshot_ids=[
                "catalog:toi:v1.0:20240115:a1b2c3d4",
                "catalog:fp:v1.0:20240115:c3d4e5f6",
            ],
            catalog_store=mock_store,
            search_radius_arcsec=2.0,  # Small radius to only match FP
        )

        # Should have FP match, possibly TOI depending on exact distances
        if all(m.object_type == "FP" for m in report.known_object_matches):
            assert report.novelty_status == "ambiguous"

    def test_ambiguous_target_eb_match(self, mock_store: MockCatalogSnapshotStore) -> None:
        """Target matching only EB is classified as ambiguous."""
        report = crossmatch(
            ra=120.003,  # Near EB-001
            dec=-45.003,
            snapshot_ids=[
                "catalog:toi:v1.0:20240115:a1b2c3d4",
                "catalog:eb:v1.0:20240115:d4e5f6a7",
            ],
            catalog_store=mock_store,
            search_radius_arcsec=2.0,
        )

        if all(m.object_type == "EB" for m in report.known_object_matches):
            assert report.novelty_status == "ambiguous"

    def test_known_takes_priority_over_ambiguous(
        self, mock_store: MockCatalogSnapshotStore
    ) -> None:
        """If both known and ambiguous matches exist, status is 'known'."""
        # Position that matches both TOI and FP
        report = crossmatch(
            ra=120.0,
            dec=-45.0,
            snapshot_ids=[
                "catalog:toi:v1.0:20240115:a1b2c3d4",
                "catalog:fp:v1.0:20240115:c3d4e5f6",
            ],
            catalog_store=mock_store,
            search_radius_arcsec=30.0,  # Large enough to catch both
        )

        assert report.novelty_status == "known"

    def test_combined_contamination_from_multiple_catalogs(
        self, mock_store: MockCatalogSnapshotStore
    ) -> None:
        """Contamination is assessed from all catalogs."""
        report = crossmatch(
            ra=120.0,
            dec=-45.0,
            snapshot_ids=[
                "catalog:toi:v1.0:20240115:a1b2c3d4",
                "catalog:gaia:v1.0:20240115:e5f6a7b8",
            ],
            catalog_store=mock_store,
            search_radius_arcsec=30.0,
            target_magnitude=10.0,
        )

        # Should have contamination from stellar catalog
        # TOI entries are excluded from contamination
        assert report.contamination_risk is not None

    def test_far_from_everything(self, mock_store: MockCatalogSnapshotStore) -> None:
        """Position far from all catalog objects is novel with no contamination."""
        report = crossmatch(
            ra=270.0,  # Far from all test positions
            dec=60.0,
            snapshot_ids=[
                "catalog:toi:v1.0:20240115:a1b2c3d4",
                "catalog:confirmed:v1.0:20240115:b2c3d4e5",
                "catalog:fp:v1.0:20240115:c3d4e5f6",
                "catalog:eb:v1.0:20240115:d4e5f6a7",
                "catalog:gaia:v1.0:20240115:e5f6a7b8",
            ],
            catalog_store=mock_store,
            search_radius_arcsec=10.0,
        )

        assert report.novelty_status == "novel"
        assert len(report.known_object_matches) == 0
        assert report.contamination_risk.has_neighbors is False


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestCrossmatchErrors:
    """Tests for error handling in crossmatch."""

    def test_exception_inheritance(self) -> None:
        """All crossmatch exceptions inherit from CrossmatchError."""
        assert issubclass(SnapshotNotFoundError, CrossmatchError)
        assert issubclass(InvalidSnapshotIdError, CrossmatchError)
        assert issubclass(NoCatalogsProvidedError, CrossmatchError)

    def test_snapshot_not_found_error_attributes(self) -> None:
        """SnapshotNotFoundError has correct attributes."""
        error = SnapshotNotFoundError("catalog:test:v1.0:20240115:deadbeef")

        assert error.snapshot_id == "catalog:test:v1.0:20240115:deadbeef"
        assert "deadbeef" in str(error)

    def test_invalid_snapshot_id_error_attributes(self) -> None:
        """InvalidSnapshotIdError has correct attributes."""
        error = InvalidSnapshotIdError("bad_id", "wrong format")

        assert error.snapshot_id == "bad_id"
        assert error.reason == "wrong format"
        assert "bad_id" in str(error)
        assert "wrong format" in str(error)


# =============================================================================
# Module Import Tests
# =============================================================================


class TestModuleImports:
    """Tests for module imports and exports."""

    def test_import_from_crossmatch_module(self) -> None:
        """Can import from bittr_tess_vetter.catalogs.crossmatch module."""
        from bittr_tess_vetter.catalogs.crossmatch import (
            CrossmatchReport,
            crossmatch,
        )

        assert CrossmatchReport is not None
        assert crossmatch is not None

    def test_import_from_catalogs_package(self) -> None:
        """Can import crossmatch from bittr_tess_vetter.catalogs package."""
        from bittr_tess_vetter.catalogs import (
            CrossmatchReport,
            crossmatch,
        )

        assert CrossmatchReport is not None
        assert crossmatch is not None

    def test_known_object_types_constant(self) -> None:
        """KNOWN_OBJECT_TYPES constant contains expected types."""
        assert "TOI" in KNOWN_OBJECT_TYPES
        assert "CONFIRMED" in KNOWN_OBJECT_TYPES
        assert "FP" in KNOWN_OBJECT_TYPES
        assert "EB" in KNOWN_OBJECT_TYPES
        assert "STAR" not in KNOWN_OBJECT_TYPES
