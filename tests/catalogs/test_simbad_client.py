"""Tests for SIMBAD TAP client with offline fixtures.

All tests use fixtures to avoid network calls. The client is tested for:
- Model parsing and validation
- Spectral type classification (early-type, giant detection)
- Object type classification (star, binary, variable)
- Identifier parsing (HD, HIP, TIC, Gaia DR3)
- Provenance tracking
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from bittr_tess_vetter.platform.catalogs.simbad_client import (
    SIMBAD_TAP_ENDPOINT,
    SimbadClient,
    SimbadIdentifiers,
    SimbadObjectType,
    SimbadQueryResult,
    SimbadSpectralInfo,
    classify_object_type,
    parse_spectral_type,
)
from bittr_reason_core.provenance import SourceRecord


# Fixture directory
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "simbad"


def load_fixture(name: str) -> dict:
    """Load a JSON fixture file."""
    path = FIXTURE_DIR / name
    with open(path) as f:
        return json.load(f)


# =============================================================================
# Spectral Type Parsing Tests
# =============================================================================


class TestParseSpectralType:
    """Tests for spectral type parsing and classification."""

    def test_parse_g0v_main_sequence(self) -> None:
        """G0V should be main sequence, not early type, not giant."""
        result = parse_spectral_type("G0V")

        assert result.spectral_type == "G0V"
        assert result.luminosity_class == "V"
        assert result.is_early_type is False
        assert result.is_giant is False

    def test_parse_a5iii_early_giant(self) -> None:
        """A5III should be early type AND giant."""
        result = parse_spectral_type("A5III")

        assert result.spectral_type == "A5III"
        assert result.luminosity_class == "III"
        assert result.is_early_type is True
        assert result.is_giant is True

    def test_parse_b2v_early_dwarf(self) -> None:
        """B2V should be early type but not giant."""
        result = parse_spectral_type("B2V")

        assert result.spectral_type == "B2V"
        assert result.luminosity_class == "V"
        assert result.is_early_type is True
        assert result.is_giant is False

    def test_parse_k2iii_late_giant(self) -> None:
        """K2III should be giant but not early type."""
        result = parse_spectral_type("K2III")

        assert result.spectral_type == "K2III"
        assert result.luminosity_class == "III"
        assert result.is_early_type is False
        assert result.is_giant is True

    def test_parse_o9v_hottest_early_type(self) -> None:
        """O9V should be early type."""
        result = parse_spectral_type("O9V")

        assert result.is_early_type is True
        assert result.is_giant is False

    def test_parse_m5iii_cool_giant(self) -> None:
        """M5III should be giant but not early type."""
        result = parse_spectral_type("M5III")

        assert result.is_early_type is False
        assert result.is_giant is True

    def test_parse_f5iv_subgiant(self) -> None:
        """F5IV (subgiant) should not be considered giant."""
        result = parse_spectral_type("F5IV")

        assert result.luminosity_class == "IV"
        assert result.is_giant is False

    def test_parse_k0ii_bright_giant(self) -> None:
        """K0II (bright giant) should be considered giant."""
        result = parse_spectral_type("K0II")

        assert result.luminosity_class == "II"
        assert result.is_giant is True

    def test_parse_g2ia_supergiant(self) -> None:
        """G2Ia (supergiant) should be considered giant."""
        result = parse_spectral_type("G2Ia")

        assert result.luminosity_class == "Ia"
        assert result.is_giant is True

    def test_parse_none_returns_defaults(self) -> None:
        """None spectral type should return safe defaults."""
        result = parse_spectral_type(None)

        assert result.spectral_type is None
        assert result.luminosity_class is None
        assert result.is_early_type is False
        assert result.is_giant is False

    def test_parse_empty_string_returns_defaults(self) -> None:
        """Empty string should return safe defaults."""
        result = parse_spectral_type("")

        assert result.spectral_type is None
        assert result.is_early_type is False
        assert result.is_giant is False

    def test_parse_whitespace_only_returns_defaults(self) -> None:
        """Whitespace-only string should return safe defaults."""
        result = parse_spectral_type("   ")

        assert result.spectral_type is None

    def test_parse_complex_spectral_type(self) -> None:
        """Complex spectral types like 'K2/3V' should be handled."""
        result = parse_spectral_type("K2/3V")

        assert result.spectral_type == "K2/3V"
        assert result.luminosity_class == "V"
        assert result.is_early_type is False
        assert result.is_giant is False


# =============================================================================
# Object Type Classification Tests
# =============================================================================


class TestClassifyObjectType:
    """Tests for object type classification."""

    def test_star_type(self) -> None:
        """Single star '*' should be classified as star."""
        result = classify_object_type("*")

        assert result.main_type == "*"
        assert result.is_star is True
        assert result.is_binary is False
        assert result.is_variable is False

    def test_eclipsing_binary(self) -> None:
        """EB* should be classified as binary."""
        result = classify_object_type("EB*")

        assert result.is_star is True
        assert result.is_binary is True

    def test_spectroscopic_binary(self) -> None:
        """SB* should be classified as binary."""
        result = classify_object_type("SB*")

        assert result.is_binary is True

    def test_double_star(self) -> None:
        """** (double star) should be classified as binary."""
        result = classify_object_type("**")

        assert result.is_star is True
        assert result.is_binary is True

    def test_variable_star(self) -> None:
        """V* should be classified as variable."""
        result = classify_object_type("V*")

        assert result.is_star is True
        assert result.is_variable is True

    def test_rr_lyrae(self) -> None:
        """RR* should be classified as variable."""
        result = classify_object_type("RR*")

        assert result.is_variable is True

    def test_high_proper_motion_star(self) -> None:
        """PM* should be classified as star."""
        result = classify_object_type("PM*")

        assert result.is_star is True
        assert result.is_binary is False

    def test_unknown_type_returns_unknown(self) -> None:
        """None type should default to 'Unknown'."""
        result = classify_object_type(None)

        assert result.main_type == "Unknown"
        assert result.is_star is False

    def test_other_types_contribute(self) -> None:
        """Other types should contribute to classification."""
        result = classify_object_type("*", ["V*", "BY*"])

        assert result.is_star is True
        assert result.is_variable is True
        assert result.other_types == ["V*", "BY*"]


# =============================================================================
# SimbadIdentifiers Model Tests
# =============================================================================


class TestSimbadIdentifiers:
    """Tests for SimbadIdentifiers model."""

    def test_create_with_all_fields(self) -> None:
        """Should create identifiers with all catalog IDs."""
        idents = SimbadIdentifiers(
            main_id="* pi. Men",
            hd="HD 39091",
            hip="HIP 26394",
            gaia_dr3="Gaia DR3 5290699119420871680",
            tic="TIC 261136679",
            all_ids=["* pi. Men", "HD 39091", "HIP 26394"],
        )

        assert idents.main_id == "* pi. Men"
        assert idents.hd == "HD 39091"
        assert idents.hip == "HIP 26394"
        assert idents.gaia_dr3 == "Gaia DR3 5290699119420871680"
        assert idents.tic == "TIC 261136679"
        assert len(idents.all_ids) == 3

    def test_create_minimal(self) -> None:
        """Should create identifiers with only main_id."""
        idents = SimbadIdentifiers(main_id="Unknown Star")

        assert idents.main_id == "Unknown Star"
        assert idents.hd is None
        assert idents.all_ids == []

    def test_is_frozen(self) -> None:
        """SimbadIdentifiers should be immutable."""
        idents = SimbadIdentifiers(main_id="* pi. Men")

        with pytest.raises(ValidationError):
            idents.main_id = "Changed"  # type: ignore[misc]


# =============================================================================
# SimbadObjectType Model Tests
# =============================================================================


class TestSimbadObjectType:
    """Tests for SimbadObjectType model."""

    def test_create_star(self) -> None:
        """Should create a star object type."""
        obj = SimbadObjectType(
            main_type="*",
            is_star=True,
            is_binary=False,
            is_variable=False,
        )

        assert obj.main_type == "*"
        assert obj.is_star is True

    def test_create_binary(self) -> None:
        """Should create a binary object type."""
        obj = SimbadObjectType(
            main_type="**",
            is_star=True,
            is_binary=True,
            is_variable=False,
        )

        assert obj.is_binary is True

    def test_serializes_to_dict(self) -> None:
        """Should serialize properly."""
        obj = SimbadObjectType(
            main_type="V*",
            is_star=True,
            is_binary=False,
            is_variable=True,
        )

        data = obj.model_dump()
        assert data["main_type"] == "V*"
        assert data["is_variable"] is True


# =============================================================================
# SimbadSpectralInfo Model Tests
# =============================================================================


class TestSimbadSpectralInfo:
    """Tests for SimbadSpectralInfo model."""

    def test_create_main_sequence(self) -> None:
        """Should create main sequence spectral info."""
        spec = SimbadSpectralInfo(
            spectral_type="G0V",
            luminosity_class="V",
            is_early_type=False,
            is_giant=False,
        )

        assert spec.spectral_type == "G0V"
        assert spec.luminosity_class == "V"
        assert spec.is_early_type is False
        assert spec.is_giant is False

    def test_create_early_giant(self) -> None:
        """Should create early-type giant spectral info."""
        spec = SimbadSpectralInfo(
            spectral_type="A5III",
            luminosity_class="III",
            is_early_type=True,
            is_giant=True,
        )

        assert spec.is_early_type is True
        assert spec.is_giant is True


# =============================================================================
# SimbadQueryResult Model Tests
# =============================================================================


class TestSimbadQueryResult:
    """Tests for SimbadQueryResult model."""

    def test_create_complete_result(self) -> None:
        """Should create a complete query result."""
        result = SimbadQueryResult(
            identifiers=SimbadIdentifiers(
                main_id="* pi. Men",
                hd="HD 39091",
            ),
            object_type=SimbadObjectType(
                main_type="*",
                is_star=True,
                is_binary=False,
                is_variable=False,
            ),
            spectral=SimbadSpectralInfo(
                spectral_type="G0V",
                luminosity_class="V",
                is_early_type=False,
                is_giant=False,
            ),
            ra=84.29122447,
            dec=-80.46911858,
            source_record=SourceRecord(name="simbad", version="TAP"),
        )

        assert result.identifiers.main_id == "* pi. Men"
        assert result.object_type.is_star is True
        assert result.spectral is not None
        assert result.spectral.spectral_type == "G0V"
        assert result.ra == pytest.approx(84.29, rel=1e-3)
        assert result.source_record.name == "simbad"

    def test_create_without_spectral(self) -> None:
        """Should create result without spectral info."""
        result = SimbadQueryResult(
            identifiers=SimbadIdentifiers(main_id="Unknown"),
            object_type=SimbadObjectType(
                main_type="*",
                is_star=True,
                is_binary=False,
                is_variable=False,
            ),
            spectral=None,
            ra=None,
            dec=None,
            source_record=SourceRecord(name="simbad"),
        )

        assert result.spectral is None


# =============================================================================
# Fixture-Based Tests
# =============================================================================


class TestPiMensaeFixture:
    """Tests using Pi Mensae fixture data."""

    def test_fixture_exists(self) -> None:
        """Pi Mensae fixture should exist."""
        path = FIXTURE_DIR / "simbad_pi_mensae.json"
        assert path.exists()

    def test_parse_fixture_basic(self) -> None:
        """Should parse Pi Mensae fixture basic data."""
        data = load_fixture("simbad_pi_mensae.json")

        assert data["basic"]["main_id"] == "* pi. Men"
        assert data["basic"]["otype"] == "*"
        assert data["basic"]["ra"] == pytest.approx(84.29, rel=1e-3)

    def test_pi_mensae_is_single_star(self) -> None:
        """Pi Mensae should be classified as single star."""
        data = load_fixture("simbad_pi_mensae.json")

        obj_type = classify_object_type(data["basic"]["otype"])

        assert obj_type.is_star is True
        assert obj_type.is_binary is False
        assert obj_type.is_variable is False

    def test_pi_mensae_spectral_type(self) -> None:
        """Pi Mensae G0V should not be early type or giant."""
        data = load_fixture("simbad_pi_mensae.json")

        spectral = parse_spectral_type(data["spectral_type"]["sptype"])

        assert spectral.spectral_type == "G0V"
        assert spectral.is_early_type is False
        assert spectral.is_giant is False
        assert spectral.luminosity_class == "V"

    def test_pi_mensae_has_expected_identifiers(self) -> None:
        """Pi Mensae should have HD, HIP, TIC, Gaia identifiers."""
        data = load_fixture("simbad_pi_mensae.json")
        ids = data["identifiers"]

        assert any("HD 39091" in id_str for id_str in ids)
        assert any("HIP 26394" in id_str for id_str in ids)
        assert any("TIC 261136679" in id_str for id_str in ids)
        assert any("Gaia DR3" in id_str for id_str in ids)


class TestHD66006Fixture:
    """Tests using HD 66006 / PATHOS-39 fixture data."""

    def test_fixture_exists(self) -> None:
        """HD 66006 fixture should exist."""
        path = FIXTURE_DIR / "simbad_hd66006.json"
        assert path.exists()

    def test_parse_fixture_basic(self) -> None:
        """Should parse HD 66006 fixture basic data."""
        data = load_fixture("simbad_hd66006.json")

        assert data["basic"]["main_id"] == "HD 66006"
        assert data["basic"]["otype"] == "**"

    def test_hd66006_is_binary(self) -> None:
        """HD 66006 should be classified as binary (visual double)."""
        data = load_fixture("simbad_hd66006.json")

        obj_type = classify_object_type(data["basic"]["otype"])

        assert obj_type.is_star is True
        assert obj_type.is_binary is True

    def test_hd66006_spectral_type(self) -> None:
        """HD 66006 G5V should not be early type or giant."""
        data = load_fixture("simbad_hd66006.json")

        spectral = parse_spectral_type(data["spectral_type"]["sptype"])

        assert spectral.spectral_type == "G5V"
        assert spectral.is_early_type is False
        assert spectral.is_giant is False


# =============================================================================
# Offline/No-Network Tests
# =============================================================================


class TestSimbadClientOffline:
    """Tests that verify the client can be tested with fixtures only."""

    def test_fixture_files_exist(self) -> None:
        """All required fixture files should exist."""
        fixtures = [
            "simbad_pi_mensae.json",
            "simbad_hd66006.json",
        ]
        for fixture in fixtures:
            path = FIXTURE_DIR / fixture
            assert path.exists(), f"Fixture {fixture} not found"

    def test_all_fixtures_parse_without_error(self) -> None:
        """All fixtures should parse as valid JSON."""
        for fixture_path in FIXTURE_DIR.glob("*.json"):
            data = json.loads(fixture_path.read_text())
            assert isinstance(data, dict)

    def test_fixtures_have_required_fields(self) -> None:
        """Fixtures should have required fields."""
        for fixture in ["simbad_pi_mensae.json", "simbad_hd66006.json"]:
            data = load_fixture(fixture)
            assert "basic" in data
            assert "main_id" in data["basic"]
            assert "otype" in data["basic"]
            assert "identifiers" in data


# =============================================================================
# Provenance Tests
# =============================================================================


class TestSimbadProvenance:
    """Tests for provenance tracking in SIMBAD queries."""

    def test_query_result_has_source_record(self) -> None:
        """Query result should include provenance source record."""
        source_record = SourceRecord(
            name="simbad",
            version="TAP",
            query="identifier = TIC 261136679",
        )

        result = SimbadQueryResult(
            identifiers=SimbadIdentifiers(main_id="* pi. Men"),
            object_type=SimbadObjectType(
                main_type="*",
                is_star=True,
                is_binary=False,
                is_variable=False,
            ),
            spectral=None,
            ra=None,
            dec=None,
            source_record=source_record,
        )

        assert result.source_record.name == "simbad"
        assert result.source_record.version == "TAP"
        assert "identifier" in result.source_record.query  # type: ignore[operator]

    def test_source_record_serializes(self) -> None:
        """Source record should serialize for storage."""
        source_record = SourceRecord(
            name="simbad",
            version="TAP",
        )
        data = source_record.model_dump()
        assert data["name"] == "simbad"
        assert data["version"] == "TAP"


# =============================================================================
# Early Type and Giant Detection Tests
# =============================================================================


class TestEarlyTypeDetection:
    """Tests for early-type star detection."""

    @pytest.mark.parametrize(
        "sptype,expected",
        [
            ("O5V", True),
            ("O9.5Ia", True),
            ("B0V", True),
            ("B9III", True),
            ("A0V", True),
            ("A9IV", True),
            ("F0V", False),
            ("G0V", False),
            ("K5III", False),
            ("M2V", False),
        ],
    )
    def test_early_type_classification(self, sptype: str, expected: bool) -> None:
        """Should correctly identify O, B, A type stars."""
        result = parse_spectral_type(sptype)
        assert result.is_early_type is expected


class TestGiantDetection:
    """Tests for giant star detection."""

    @pytest.mark.parametrize(
        "sptype,expected",
        [
            ("G0V", False),
            ("K5V", False),
            ("F5IV", False),  # Subgiant is NOT giant
            ("G2III", True),
            ("K0II", True),
            ("M5III", True),
            ("G2Ib", True),
            ("K0Ia", True),
            ("F5III", True),
        ],
    )
    def test_giant_classification(self, sptype: str, expected: bool) -> None:
        """Should correctly identify giant luminosity classes (III, II, I)."""
        result = parse_spectral_type(sptype)
        assert result.is_giant is expected
