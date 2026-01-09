"""Tests for bittr_tess_vetter.catalogs.snapshot_id module.

Comprehensive tests for snapshot ID generation, parsing, and validation.
Snapshot IDs are critical for catalog versioning and reproducibility.
"""

from __future__ import annotations

import hashlib

import pytest

from bittr_tess_vetter.catalogs.snapshot_id import (
    SHA256_PREFIX_LENGTH,
    SNAPSHOT_PREFIX,
    SNAPSHOT_SEPARATOR,
    SnapshotComponents,
    generate_snapshot_id,
    parse_snapshot_id,
    validate_snapshot_id,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def valid_components() -> dict[str, str]:
    """Valid snapshot components for testing."""
    return {
        "name": "tic",
        "version": "v8.2",
        "asof_date": "20240115",
        "sha256_prefix": "a1b2c3d4",
    }


@pytest.fixture
def sample_data() -> bytes:
    """Sample catalog data for hashing tests."""
    return b"TIC catalog data with stellar parameters and identifiers"


@pytest.fixture
def valid_snapshot_id() -> str:
    """A valid snapshot ID string for testing."""
    return "catalog:tic:v8.2:20240115:a1b2c3d4"


# =============================================================================
# SnapshotComponents Tests
# =============================================================================


class TestSnapshotComponents:
    """Tests for SnapshotComponents dataclass."""

    def test_valid_construction(self, valid_components: dict[str, str]) -> None:
        """Valid components create a SnapshotComponents instance."""
        components = SnapshotComponents(**valid_components)
        assert components.name == "tic"
        assert components.version == "v8.2"
        assert components.asof_date == "20240115"
        assert components.sha256_prefix == "a1b2c3d4"

    def test_frozen_after_construction(self, valid_components: dict[str, str]) -> None:
        """SnapshotComponents are immutable after construction."""
        components = SnapshotComponents(**valid_components)
        with pytest.raises(AttributeError):
            components.name = "modified"  # type: ignore[misc]

    def test_to_id_format(self, valid_components: dict[str, str]) -> None:
        """to_id() produces correct snapshot ID format."""
        components = SnapshotComponents(**valid_components)
        result = components.to_id()
        assert result == "catalog:tic:v8.2:20240115:a1b2c3d4"

    def test_to_id_roundtrip(self, valid_components: dict[str, str]) -> None:
        """to_id() output can be parsed back to same components."""
        components = SnapshotComponents(**valid_components)
        id_str = components.to_id()
        parsed = parse_snapshot_id(id_str)
        assert parsed == components

    # Name validation tests
    def test_valid_names(self) -> None:
        """Various valid catalog names are accepted."""
        valid_names = [
            "tic",
            "kepler",
            "gaia",
            "gaia-dr3",
            "tess_spoc",
            "catalog123",
            "a",
            "ab",
            "a1b2c3",
        ]
        for name in valid_names:
            components = SnapshotComponents(
                name=name,
                version="v1",
                asof_date="20240101",
                sha256_prefix="abcd1234",
            )
            assert components.name == name

    def test_invalid_name_uppercase(self) -> None:
        """Uppercase names are rejected."""
        with pytest.raises(ValueError, match="Invalid catalog name"):
            SnapshotComponents(
                name="TIC",
                version="v1",
                asof_date="20240101",
                sha256_prefix="abcd1234",
            )

    def test_invalid_name_starts_with_digit(self) -> None:
        """Names starting with digits are rejected."""
        with pytest.raises(ValueError, match="Invalid catalog name"):
            SnapshotComponents(
                name="123catalog",
                version="v1",
                asof_date="20240101",
                sha256_prefix="abcd1234",
            )

    def test_invalid_name_starts_with_underscore(self) -> None:
        """Names starting with underscore are rejected."""
        with pytest.raises(ValueError, match="Invalid catalog name"):
            SnapshotComponents(
                name="_catalog",
                version="v1",
                asof_date="20240101",
                sha256_prefix="abcd1234",
            )

    def test_invalid_name_empty(self) -> None:
        """Empty names are rejected."""
        with pytest.raises(ValueError, match="Invalid catalog name"):
            SnapshotComponents(
                name="",
                version="v1",
                asof_date="20240101",
                sha256_prefix="abcd1234",
            )

    def test_invalid_name_special_chars(self) -> None:
        """Names with special characters (except underscore/hyphen) are rejected."""
        invalid_names = ["cat.log", "cat@log", "cat log", "cat/log"]
        for name in invalid_names:
            with pytest.raises(ValueError, match="Invalid catalog name"):
                SnapshotComponents(
                    name=name,
                    version="v1",
                    asof_date="20240101",
                    sha256_prefix="abcd1234",
                )

    # Version validation tests
    def test_valid_versions(self) -> None:
        """Various valid version formats are accepted."""
        valid_versions = ["v1", "v8.2", "v1.0.0", "v12.34.56", "v0"]
        for version in valid_versions:
            components = SnapshotComponents(
                name="test",
                version=version,
                asof_date="20240101",
                sha256_prefix="abcd1234",
            )
            assert components.version == version

    def test_invalid_version_no_v_prefix(self) -> None:
        """Versions without 'v' prefix are rejected."""
        with pytest.raises(ValueError, match="Invalid version"):
            SnapshotComponents(
                name="test",
                version="8.2",
                asof_date="20240101",
                sha256_prefix="abcd1234",
            )

    def test_invalid_version_uppercase_v(self) -> None:
        """Versions with uppercase 'V' are rejected."""
        with pytest.raises(ValueError, match="Invalid version"):
            SnapshotComponents(
                name="test",
                version="V8.2",
                asof_date="20240101",
                sha256_prefix="abcd1234",
            )

    def test_invalid_version_text(self) -> None:
        """Versions with text (e.g., 'v8.2-beta') are rejected."""
        with pytest.raises(ValueError, match="Invalid version"):
            SnapshotComponents(
                name="test",
                version="v8.2-beta",
                asof_date="20240101",
                sha256_prefix="abcd1234",
            )

    def test_invalid_version_empty(self) -> None:
        """Empty versions are rejected."""
        with pytest.raises(ValueError, match="Invalid version"):
            SnapshotComponents(
                name="test",
                version="",
                asof_date="20240101",
                sha256_prefix="abcd1234",
            )

    # As-of date validation tests
    def test_valid_asof_dates(self) -> None:
        """Various valid dates are accepted."""
        valid_dates = ["20240101", "20231231", "19990615", "20501225"]
        for asof_date in valid_dates:
            components = SnapshotComponents(
                name="test",
                version="v1",
                asof_date=asof_date,
                sha256_prefix="abcd1234",
            )
            assert components.asof_date == asof_date

    def test_invalid_asof_date_format(self) -> None:
        """Incorrectly formatted dates are rejected."""
        invalid_dates = [
            "2024-01-15",  # dashes
            "01/15/2024",  # slashes
            "15012024",  # wrong order (fails on year validation)
            "202401",  # missing day
            "2024011",  # too short
            "202401150",  # too long
        ]
        for asof_date in invalid_dates:
            # Match either format error or year/month/day validation error
            with pytest.raises(ValueError, match="Invalid"):
                SnapshotComponents(
                    name="test",
                    version="v1",
                    asof_date=asof_date,
                    sha256_prefix="abcd1234",
                )

    def test_invalid_asof_date_month(self) -> None:
        """Invalid months are rejected."""
        with pytest.raises(ValueError, match="Invalid month"):
            SnapshotComponents(
                name="test",
                version="v1",
                asof_date="20241301",  # month 13
                sha256_prefix="abcd1234",
            )

    def test_invalid_asof_date_day(self) -> None:
        """Invalid days are rejected."""
        with pytest.raises(ValueError, match="Invalid day"):
            SnapshotComponents(
                name="test",
                version="v1",
                asof_date="20240132",  # day 32
                sha256_prefix="abcd1234",
            )

    def test_invalid_asof_date_day_for_month(self) -> None:
        """Days invalid for specific month are rejected."""
        with pytest.raises(ValueError, match="Invalid day"):
            SnapshotComponents(
                name="test",
                version="v1",
                asof_date="20240230",  # Feb 30
                sha256_prefix="abcd1234",
            )

    # SHA256 prefix validation tests
    def test_valid_sha256_prefixes(self) -> None:
        """Various valid SHA256 prefixes are accepted."""
        valid_prefixes = ["a1b2c3d4", "00000000", "ffffffff", "12345678"]
        for prefix in valid_prefixes:
            components = SnapshotComponents(
                name="test",
                version="v1",
                asof_date="20240101",
                sha256_prefix=prefix,
            )
            assert components.sha256_prefix == prefix

    def test_invalid_sha256_prefix_too_short(self) -> None:
        """SHA256 prefixes shorter than 8 chars are rejected."""
        with pytest.raises(ValueError, match="Invalid SHA-256 prefix"):
            SnapshotComponents(
                name="test",
                version="v1",
                asof_date="20240101",
                sha256_prefix="a1b2c3",  # 6 chars
            )

    def test_invalid_sha256_prefix_too_long(self) -> None:
        """SHA256 prefixes longer than 8 chars are rejected."""
        with pytest.raises(ValueError, match="Invalid SHA-256 prefix"):
            SnapshotComponents(
                name="test",
                version="v1",
                asof_date="20240101",
                sha256_prefix="a1b2c3d4e5",  # 10 chars
            )

    def test_invalid_sha256_prefix_uppercase(self) -> None:
        """Uppercase SHA256 prefixes are rejected."""
        with pytest.raises(ValueError, match="Invalid SHA-256 prefix"):
            SnapshotComponents(
                name="test",
                version="v1",
                asof_date="20240101",
                sha256_prefix="A1B2C3D4",
            )

    def test_invalid_sha256_prefix_non_hex(self) -> None:
        """Non-hex SHA256 prefixes are rejected."""
        with pytest.raises(ValueError, match="Invalid SHA-256 prefix"):
            SnapshotComponents(
                name="test",
                version="v1",
                asof_date="20240101",
                sha256_prefix="ghijklmn",  # not hex
            )


# =============================================================================
# generate_snapshot_id Tests
# =============================================================================


class TestGenerateSnapshotId:
    """Tests for generate_snapshot_id function."""

    def test_generates_correct_format(self, sample_data: bytes) -> None:
        """Generated ID has correct format."""
        result = generate_snapshot_id("tic", "v8.2", "20240115", sample_data)

        # Check format: catalog:<name>:<version>:<asof>:<hash>
        assert result.startswith("catalog:")
        parts = result.split(":")
        assert len(parts) == 5
        assert parts[0] == "catalog"
        assert parts[1] == "tic"
        assert parts[2] == "v8.2"
        assert parts[3] == "20240115"
        assert len(parts[4]) == 8

    def test_sha256_prefix_from_data(self, sample_data: bytes) -> None:
        """SHA256 prefix is computed from data."""
        result = generate_snapshot_id("tic", "v8.2", "20240115", sample_data)

        # Compute expected hash
        expected_hash = hashlib.sha256(sample_data).hexdigest()[:8]

        parts = result.split(":")
        assert parts[4] == expected_hash

    def test_different_data_different_hash(self) -> None:
        """Different data produces different SHA256 prefix."""
        data1 = b"catalog data version 1"
        data2 = b"catalog data version 2"

        id1 = generate_snapshot_id("test", "v1", "20240101", data1)
        id2 = generate_snapshot_id("test", "v1", "20240101", data2)

        # Extract hash prefixes
        hash1 = id1.split(":")[-1]
        hash2 = id2.split(":")[-1]

        assert hash1 != hash2

    def test_same_data_same_hash(self, sample_data: bytes) -> None:
        """Same data always produces same SHA256 prefix."""
        id1 = generate_snapshot_id("test", "v1", "20240101", sample_data)
        id2 = generate_snapshot_id("test", "v1", "20240101", sample_data)

        assert id1 == id2

    def test_validates_name(self) -> None:
        """Invalid names are rejected."""
        with pytest.raises(ValueError, match="Invalid catalog name"):
            generate_snapshot_id("INVALID", "v1", "20240101", b"data")

    def test_validates_version(self) -> None:
        """Invalid versions are rejected."""
        with pytest.raises(ValueError, match="Invalid version"):
            generate_snapshot_id("test", "1.0", "20240101", b"data")

    def test_validates_asof_date(self) -> None:
        """Invalid dates are rejected."""
        with pytest.raises(ValueError, match="Invalid as-of date"):
            generate_snapshot_id("test", "v1", "2024-01-01", b"data")

    def test_rejects_non_bytes_data(self) -> None:
        """Non-bytes data is rejected."""
        with pytest.raises(TypeError, match="data must be bytes"):
            generate_snapshot_id("test", "v1", "20240101", "string data")  # type: ignore[arg-type]

    def test_rejects_empty_data(self) -> None:
        """Empty data is rejected."""
        with pytest.raises(ValueError, match="data must not be empty"):
            generate_snapshot_id("test", "v1", "20240101", b"")

    def test_result_is_parseable(self, sample_data: bytes) -> None:
        """Generated ID can be parsed back."""
        result = generate_snapshot_id("tic", "v8.2", "20240115", sample_data)
        components = parse_snapshot_id(result)

        assert components.name == "tic"
        assert components.version == "v8.2"
        assert components.asof_date == "20240115"

    def test_result_is_valid(self, sample_data: bytes) -> None:
        """Generated ID passes validation."""
        result = generate_snapshot_id("tic", "v8.2", "20240115", sample_data)
        assert validate_snapshot_id(result) is True


# =============================================================================
# parse_snapshot_id Tests
# =============================================================================


class TestParseSnapshotId:
    """Tests for parse_snapshot_id function."""

    def test_parses_valid_id(self, valid_snapshot_id: str) -> None:
        """Valid snapshot ID is parsed correctly."""
        components = parse_snapshot_id(valid_snapshot_id)

        assert components.name == "tic"
        assert components.version == "v8.2"
        assert components.asof_date == "20240115"
        assert components.sha256_prefix == "a1b2c3d4"

    def test_parses_various_valid_ids(self) -> None:
        """Various valid snapshot IDs are parsed correctly."""
        test_cases = [
            ("catalog:kepler:v1:20200101:00001111", "kepler", "v1", "20200101", "00001111"),
            ("catalog:gaia-dr3:v3.0:20220630:ffffffff", "gaia-dr3", "v3.0", "20220630", "ffffffff"),
            (
                "catalog:tess_spoc:v2.1.5:20231215:abcdef12",
                "tess_spoc",
                "v2.1.5",
                "20231215",
                "abcdef12",
            ),
        ]
        for id_str, name, version, asof, hash_prefix in test_cases:
            components = parse_snapshot_id(id_str)
            assert components.name == name
            assert components.version == version
            assert components.asof_date == asof
            assert components.sha256_prefix == hash_prefix

    def test_rejects_wrong_prefix(self) -> None:
        """IDs with wrong prefix are rejected."""
        with pytest.raises(ValueError, match="Invalid snapshot ID prefix"):
            parse_snapshot_id("snapshot:tic:v8.2:20240115:a1b2c3d4")

    def test_rejects_too_few_parts(self) -> None:
        """IDs with too few parts are rejected."""
        with pytest.raises(ValueError, match="expected 5 colon-separated parts"):
            parse_snapshot_id("catalog:tic:v8.2:20240115")

    def test_rejects_too_many_parts(self) -> None:
        """IDs with too many parts are rejected."""
        with pytest.raises(ValueError, match="expected 5 colon-separated parts"):
            parse_snapshot_id("catalog:tic:v8.2:20240115:a1b2c3d4:extra")

    def test_rejects_empty_string(self) -> None:
        """Empty string is rejected."""
        with pytest.raises(ValueError, match="expected 5 colon-separated parts"):
            parse_snapshot_id("")

    def test_rejects_non_string(self) -> None:
        """Non-string input is rejected."""
        with pytest.raises(TypeError, match="id_str must be a string"):
            parse_snapshot_id(12345)  # type: ignore[arg-type]

    def test_rejects_invalid_name_in_id(self) -> None:
        """IDs with invalid names are rejected."""
        with pytest.raises(ValueError, match="Invalid catalog name"):
            parse_snapshot_id("catalog:UPPERCASE:v1:20240101:a1b2c3d4")

    def test_rejects_invalid_version_in_id(self) -> None:
        """IDs with invalid versions are rejected."""
        with pytest.raises(ValueError, match="Invalid version"):
            parse_snapshot_id("catalog:tic:8.2:20240101:a1b2c3d4")

    def test_rejects_invalid_date_in_id(self) -> None:
        """IDs with invalid dates are rejected."""
        with pytest.raises(ValueError, match="Invalid as-of date"):
            parse_snapshot_id("catalog:tic:v8.2:2024-01-01:a1b2c3d4")

    def test_rejects_invalid_hash_in_id(self) -> None:
        """IDs with invalid hash prefixes are rejected."""
        with pytest.raises(ValueError, match="Invalid SHA-256 prefix"):
            parse_snapshot_id("catalog:tic:v8.2:20240101:UPPERCASE")


# =============================================================================
# validate_snapshot_id Tests
# =============================================================================


class TestValidateSnapshotId:
    """Tests for validate_snapshot_id function."""

    def test_valid_id_returns_true(self, valid_snapshot_id: str) -> None:
        """Valid snapshot ID returns True."""
        assert validate_snapshot_id(valid_snapshot_id) is True

    def test_various_valid_ids(self) -> None:
        """Various valid IDs all return True."""
        valid_ids = [
            "catalog:tic:v8.2:20240115:a1b2c3d4",
            "catalog:kepler:v1:20200101:00001111",
            "catalog:gaia-dr3:v3.0:20220630:ffffffff",
            "catalog:tess_spoc:v2.1.5:20231215:abcdef12",
        ]
        for id_str in valid_ids:
            assert validate_snapshot_id(id_str) is True

    def test_invalid_prefix_returns_false(self) -> None:
        """Invalid prefix returns False (not exception)."""
        assert validate_snapshot_id("snapshot:tic:v8.2:20240115:a1b2c3d4") is False

    def test_wrong_part_count_returns_false(self) -> None:
        """Wrong number of parts returns False."""
        assert validate_snapshot_id("catalog:tic:v8.2:20240115") is False
        assert validate_snapshot_id("catalog:tic:v8.2:20240115:a1b2c3d4:extra") is False

    def test_invalid_name_returns_false(self) -> None:
        """Invalid name returns False."""
        assert validate_snapshot_id("catalog:UPPERCASE:v1:20240101:a1b2c3d4") is False
        assert validate_snapshot_id("catalog:123start:v1:20240101:a1b2c3d4") is False

    def test_invalid_version_returns_false(self) -> None:
        """Invalid version returns False."""
        assert validate_snapshot_id("catalog:tic:8.2:20240101:a1b2c3d4") is False
        assert validate_snapshot_id("catalog:tic:V8.2:20240101:a1b2c3d4") is False

    def test_invalid_date_returns_false(self) -> None:
        """Invalid date returns False."""
        assert validate_snapshot_id("catalog:tic:v8.2:2024-01-01:a1b2c3d4") is False
        assert validate_snapshot_id("catalog:tic:v8.2:20241301:a1b2c3d4") is False

    def test_invalid_hash_returns_false(self) -> None:
        """Invalid hash prefix returns False."""
        assert validate_snapshot_id("catalog:tic:v8.2:20240101:a1b2c3") is False  # too short
        assert validate_snapshot_id("catalog:tic:v8.2:20240101:a1b2c3d4e5") is False  # too long
        assert validate_snapshot_id("catalog:tic:v8.2:20240101:ABCD1234") is False  # uppercase

    def test_empty_string_returns_false(self) -> None:
        """Empty string returns False."""
        assert validate_snapshot_id("") is False

    def test_non_string_returns_false(self) -> None:
        """Non-string input returns False."""
        assert validate_snapshot_id(None) is False  # type: ignore[arg-type]
        assert validate_snapshot_id(12345) is False  # type: ignore[arg-type]
        assert validate_snapshot_id(["catalog", "tic"]) is False  # type: ignore[arg-type]


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_snapshot_prefix(self) -> None:
        """SNAPSHOT_PREFIX is 'catalog'."""
        assert SNAPSHOT_PREFIX == "catalog"

    def test_snapshot_separator(self) -> None:
        """SNAPSHOT_SEPARATOR is ':'."""
        assert SNAPSHOT_SEPARATOR == ":"

    def test_sha256_prefix_length(self) -> None:
        """SHA256_PREFIX_LENGTH is 8."""
        assert SHA256_PREFIX_LENGTH == 8


# =============================================================================
# Immutability Tests
# =============================================================================


class TestImmutability:
    """Tests verifying immutability guarantees."""

    def test_components_are_frozen(self, valid_components: dict[str, str]) -> None:
        """SnapshotComponents cannot be modified."""
        components = SnapshotComponents(**valid_components)

        with pytest.raises(AttributeError):
            components.name = "changed"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            components.version = "v2"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            components.asof_date = "20250101"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            components.sha256_prefix = "11111111"  # type: ignore[misc]

    def test_reindexing_requires_version_bump(self, sample_data: bytes) -> None:
        """Same name/date with different data requires version bump for uniqueness."""
        # If we reindex the same catalog on the same date with different data,
        # we get a different hash, making the IDs unique
        old_data = b"original catalog data v1"
        new_data = b"reindexed catalog data v2"

        old_id = generate_snapshot_id("tic", "v8.2", "20240115", old_data)
        new_id = generate_snapshot_id("tic", "v8.2", "20240115", new_data)

        # IDs differ because content differs
        assert old_id != new_id

        # To properly distinguish reindexing, bump version:
        proper_new_id = generate_snapshot_id("tic", "v8.3", "20240115", new_data)
        assert proper_new_id != old_id
        assert proper_new_id != new_id


# =============================================================================
# Equality and Hashing Tests
# =============================================================================


class TestEquality:
    """Tests for equality and hashing of SnapshotComponents."""

    def test_equal_components(self) -> None:
        """Components with same values are equal."""
        c1 = SnapshotComponents("tic", "v8.2", "20240115", "a1b2c3d4")
        c2 = SnapshotComponents("tic", "v8.2", "20240115", "a1b2c3d4")
        assert c1 == c2

    def test_different_name_not_equal(self) -> None:
        """Components with different names are not equal."""
        c1 = SnapshotComponents("tic", "v8.2", "20240115", "a1b2c3d4")
        c2 = SnapshotComponents("kepler", "v8.2", "20240115", "a1b2c3d4")
        assert c1 != c2

    def test_different_version_not_equal(self) -> None:
        """Components with different versions are not equal."""
        c1 = SnapshotComponents("tic", "v8.2", "20240115", "a1b2c3d4")
        c2 = SnapshotComponents("tic", "v8.3", "20240115", "a1b2c3d4")
        assert c1 != c2

    def test_different_date_not_equal(self) -> None:
        """Components with different dates are not equal."""
        c1 = SnapshotComponents("tic", "v8.2", "20240115", "a1b2c3d4")
        c2 = SnapshotComponents("tic", "v8.2", "20240116", "a1b2c3d4")
        assert c1 != c2

    def test_different_hash_not_equal(self) -> None:
        """Components with different hashes are not equal."""
        c1 = SnapshotComponents("tic", "v8.2", "20240115", "a1b2c3d4")
        c2 = SnapshotComponents("tic", "v8.2", "20240115", "d4c3b2a1")
        assert c1 != c2

    def test_hashable(self) -> None:
        """SnapshotComponents are hashable (for use in sets/dicts)."""
        c1 = SnapshotComponents("tic", "v8.2", "20240115", "a1b2c3d4")
        c2 = SnapshotComponents("tic", "v8.2", "20240115", "a1b2c3d4")
        c3 = SnapshotComponents("kepler", "v1", "20200101", "11112222")

        # Can use in set
        components_set = {c1, c2, c3}
        assert len(components_set) == 2  # c1 and c2 are equal

        # Can use as dict key
        components_dict = {c1: "first", c3: "third"}
        assert components_dict[c2] == "first"  # c2 equals c1


# =============================================================================
# Edge Cases and Boundary Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_minimum_valid_name(self) -> None:
        """Single character name is valid."""
        components = SnapshotComponents("a", "v1", "20240101", "12345678")
        assert components.name == "a"

    def test_long_name(self) -> None:
        """Long name is valid."""
        long_name = "a" + "b" * 100
        components = SnapshotComponents(long_name, "v1", "20240101", "12345678")
        assert components.name == long_name

    def test_minimum_version(self) -> None:
        """Single digit version is valid."""
        components = SnapshotComponents("test", "v0", "20240101", "12345678")
        assert components.version == "v0"

    def test_long_version(self) -> None:
        """Long version is valid."""
        components = SnapshotComponents("test", "v123.456.789", "20240101", "12345678")
        assert components.version == "v123.456.789"

    def test_earliest_valid_date(self) -> None:
        """Date at year boundary is valid."""
        components = SnapshotComponents("test", "v1", "19000101", "12345678")
        assert components.asof_date == "19000101"

    def test_latest_valid_date(self) -> None:
        """Date at year boundary is valid."""
        components = SnapshotComponents("test", "v1", "21001231", "12345678")
        assert components.asof_date == "21001231"

    def test_february_29_leap_year(self) -> None:
        """Feb 29 in leap year is valid."""
        components = SnapshotComponents("test", "v1", "20240229", "12345678")
        assert components.asof_date == "20240229"

    def test_all_zeros_hash(self) -> None:
        """All zeros hash is valid."""
        components = SnapshotComponents("test", "v1", "20240101", "00000000")
        assert components.sha256_prefix == "00000000"

    def test_all_f_hash(self) -> None:
        """All f hash is valid."""
        components = SnapshotComponents("test", "v1", "20240101", "ffffffff")
        assert components.sha256_prefix == "ffffffff"

    def test_large_data_hashing(self) -> None:
        """Large data can be hashed."""
        large_data = b"x" * (1024 * 1024)  # 1MB
        result = generate_snapshot_id("test", "v1", "20240101", large_data)
        assert validate_snapshot_id(result) is True

    def test_binary_data_hashing(self) -> None:
        """Binary data (with null bytes) can be hashed."""
        binary_data = bytes(range(256)) * 100
        result = generate_snapshot_id("test", "v1", "20240101", binary_data)
        assert validate_snapshot_id(result) is True


# =============================================================================
# Documentation Examples Tests
# =============================================================================


class TestDocumentationExamples:
    """Tests that verify documentation examples work correctly."""

    def test_module_docstring_example(self) -> None:
        """Example from module docstring works."""
        data = b"catalog content here"
        snapshot_id = generate_snapshot_id("tic", "v8.2", "20240115", data)

        # Verify format
        assert snapshot_id.startswith("catalog:tic:v8.2:20240115:")
        assert len(snapshot_id.split(":")[-1]) == 8

        # Parse it
        components = parse_snapshot_id(snapshot_id)
        assert components.name == "tic"

        # Validate it
        assert validate_snapshot_id(snapshot_id) is True

    def test_format_example(self) -> None:
        """Example format from docstring is valid."""
        example_id = "catalog:tic:v8.2:20240115:a1b2c3d4"
        assert validate_snapshot_id(example_id) is True

        components = parse_snapshot_id(example_id)
        assert components.name == "tic"
        assert components.version == "v8.2"
        assert components.asof_date == "20240115"
        assert components.sha256_prefix == "a1b2c3d4"

    def test_to_id_docstring_example(self) -> None:
        """Example from to_id() docstring works."""
        components = SnapshotComponents("tic", "v8.2", "20240115", "a1b2c3d4")
        result = components.to_id()
        assert result == "catalog:tic:v8.2:20240115:a1b2c3d4"
