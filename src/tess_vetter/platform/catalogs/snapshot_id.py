"""Snapshot ID generation, parsing, and validation for catalog versioning.

Snapshot IDs provide immutable, content-addressed identifiers for catalog data.
They encode the catalog name, version, as-of date, and a SHA-256 content prefix
to ensure reproducibility and detect changes.

Format: catalog:<name>:<version>:<asof_yyyymmdd>:<sha256prefix>
Example: catalog:tic:v8.2:20240115:a1b2c3d4

Key guarantees:
- Immutable: Snapshot IDs are immutable - any reindexing must bump version
- Content-addressed: The sha256 prefix ensures data integrity
- Parseable: Components can be extracted for routing and validation
- Human-readable: Format is designed for debugging and logging

Usage:
    >>> from tess_vetter.platform.catalogs.snapshot_id import (
    ...     generate_snapshot_id,
    ...     parse_snapshot_id,
    ...     validate_snapshot_id,
    ... )
    >>> data = b"catalog content here"
    >>> snapshot_id = generate_snapshot_id("tic", "v8.2", "20240115", data)
    >>> snapshot_id
    'catalog:tic:v8.2:20240115:a1b2c3d4'
    >>> components = parse_snapshot_id(snapshot_id)
    >>> components.name
    'tic'
    >>> validate_snapshot_id(snapshot_id)
    True
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

# Snapshot ID format components
SNAPSHOT_PREFIX = "catalog"
SNAPSHOT_SEPARATOR = ":"
SHA256_PREFIX_LENGTH = 8

# Validation patterns
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
VERSION_PATTERN = re.compile(r"^v\d+(\.\d+)*$")
ASOF_DATE_PATTERN = re.compile(r"^\d{8}$")
SHA256_PREFIX_PATTERN = re.compile(r"^[a-f0-9]{8}$")

# Full snapshot ID pattern for validation
SNAPSHOT_ID_PATTERN = re.compile(r"^catalog:[a-z][a-z0-9_-]*:v\d+(\.\d+)*:\d{8}:[a-f0-9]{8}$")


@dataclass(frozen=True)
class SnapshotComponents:
    """Parsed components of a snapshot ID.

    All fields are immutable after construction to ensure snapshot IDs
    remain stable references.

    Attributes:
        name: Catalog name (e.g., 'tic', 'kepler', 'gaia').
            Must start with lowercase letter, followed by lowercase
            letters, digits, underscores, or hyphens.
        version: Catalog version (e.g., 'v8.2', 'v1', 'v2.0.1').
            Must start with 'v' followed by version numbers.
        asof_date: As-of date in YYYYMMDD format (e.g., '20240115').
            Represents the date when the catalog data was frozen.
        sha256_prefix: First 8 characters of SHA-256 hash of catalog data.
            Provides content addressing for integrity verification.
    """

    name: str
    version: str
    asof_date: str  # YYYYMMDD format
    sha256_prefix: str  # 8 chars

    def __post_init__(self) -> None:
        """Validate all components after initialization."""
        if not NAME_PATTERN.match(self.name):
            raise ValueError(
                f"Invalid catalog name '{self.name}': must start with lowercase letter, "
                f"followed by lowercase letters, digits, underscores, or hyphens"
            )
        if not VERSION_PATTERN.match(self.version):
            raise ValueError(
                f"Invalid version '{self.version}': must be in format 'vX.Y.Z' "
                f"(e.g., 'v1', 'v8.2', 'v2.0.1')"
            )
        if not ASOF_DATE_PATTERN.match(self.asof_date):
            raise ValueError(f"Invalid as-of date '{self.asof_date}': must be in YYYYMMDD format")
        # Validate the date is plausible (basic check)
        _validate_asof_date(self.asof_date)

        if not SHA256_PREFIX_PATTERN.match(self.sha256_prefix):
            raise ValueError(
                f"Invalid SHA-256 prefix '{self.sha256_prefix}': "
                f"must be exactly 8 lowercase hex characters"
            )

    def to_id(self) -> str:
        """Convert components back to a snapshot ID string.

        Returns:
            The full snapshot ID string.

        Example:
            >>> components = SnapshotComponents("tic", "v8.2", "20240115", "a1b2c3d4")
            >>> components.to_id()
            'catalog:tic:v8.2:20240115:a1b2c3d4'
        """
        return SNAPSHOT_SEPARATOR.join(
            [
                SNAPSHOT_PREFIX,
                self.name,
                self.version,
                self.asof_date,
                self.sha256_prefix,
            ]
        )


def _validate_asof_date(asof_date: str) -> None:
    """Validate that the as-of date is a plausible calendar date.

    Args:
        asof_date: Date string in YYYYMMDD format.

    Raises:
        ValueError: If the date is not a valid calendar date.
    """
    year = int(asof_date[:4])
    month = int(asof_date[4:6])
    day = int(asof_date[6:8])

    # Basic sanity checks
    if year < 1900 or year > 2100:
        raise ValueError(f"Invalid year {year} in as-of date: must be between 1900 and 2100")
    if month < 1 or month > 12:
        raise ValueError(f"Invalid month {month} in as-of date: must be 1-12")
    if day < 1 or day > 31:
        raise ValueError(f"Invalid day {day} in as-of date: must be 1-31")

    # Check days per month (simplified - doesn't handle leap years precisely)
    days_in_month = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if day > days_in_month[month]:
        raise ValueError(f"Invalid day {day} for month {month} in as-of date")


def generate_snapshot_id(
    name: str,
    version: str,
    asof_date: str,
    data: bytes,
) -> str:
    """Generate a snapshot ID from catalog data.

    Creates an immutable, content-addressed identifier for a catalog snapshot.
    The SHA-256 prefix is computed from the provided data bytes, ensuring
    that any change to the catalog content will produce a different ID.

    Args:
        name: Catalog name (e.g., 'tic', 'kepler').
            Must start with lowercase letter, followed by lowercase
            letters, digits, underscores, or hyphens.
        version: Catalog version (e.g., 'v8.2', 'v1.0.0').
            Must start with 'v' followed by version numbers.
            Any reindexing of the catalog MUST bump the version.
        asof_date: As-of date in YYYYMMDD format (e.g., '20240115').
            Represents when the catalog data was frozen.
        data: Raw bytes of the catalog data to hash.
            The first 8 characters of the SHA-256 hash are used.

    Returns:
        Snapshot ID string in format 'catalog:<name>:<version>:<asof>:<hash>'.

    Raises:
        ValueError: If any component is invalid.
        TypeError: If data is not bytes.

    Example:
        >>> data = b"TIC catalog data..."
        >>> generate_snapshot_id("tic", "v8.2", "20240115", data)
        'catalog:tic:v8.2:20240115:a1b2c3d4'
    """
    if not isinstance(data, bytes):
        raise TypeError(f"data must be bytes, not {type(data).__name__}")
    if len(data) == 0:
        raise ValueError("data must not be empty")

    # Compute SHA-256 prefix
    sha256_hash = hashlib.sha256(data).hexdigest()
    sha256_prefix = sha256_hash[:SHA256_PREFIX_LENGTH]

    # Create and validate components
    components = SnapshotComponents(
        name=name,
        version=version,
        asof_date=asof_date,
        sha256_prefix=sha256_prefix,
    )

    return components.to_id()


def parse_snapshot_id(id_str: str) -> SnapshotComponents:
    """Parse a snapshot ID string into its components.

    Extracts and validates all components from a snapshot ID string.
    This is useful for routing queries, logging, and verification.

    Args:
        id_str: Snapshot ID string to parse.
            Must be in format 'catalog:<name>:<version>:<asof>:<hash>'.

    Returns:
        SnapshotComponents with all parsed and validated fields.

    Raises:
        ValueError: If the ID format is invalid or any component fails validation.

    Example:
        >>> components = parse_snapshot_id("catalog:tic:v8.2:20240115:a1b2c3d4")
        >>> components.name
        'tic'
        >>> components.version
        'v8.2'
        >>> components.asof_date
        '20240115'
        >>> components.sha256_prefix
        'a1b2c3d4'
    """
    if not isinstance(id_str, str):
        raise TypeError(f"id_str must be a string, not {type(id_str).__name__}")

    parts = id_str.split(SNAPSHOT_SEPARATOR)

    if len(parts) != 5:
        raise ValueError(
            f"Invalid snapshot ID format: expected 5 colon-separated parts, "
            f"got {len(parts)}. Format: catalog:<name>:<version>:<asof>:<hash>"
        )

    prefix, name, version, asof_date, sha256_prefix = parts

    if prefix != SNAPSHOT_PREFIX:
        raise ValueError(
            f"Invalid snapshot ID prefix: expected '{SNAPSHOT_PREFIX}', got '{prefix}'"
        )

    # SnapshotComponents constructor validates all fields
    return SnapshotComponents(
        name=name,
        version=version,
        asof_date=asof_date,
        sha256_prefix=sha256_prefix,
    )


def validate_snapshot_id(id_str: str) -> bool:
    """Validate snapshot ID format without raising exceptions.

    Performs full validation of the snapshot ID string, including:
    - Correct number of components
    - Valid prefix ('catalog')
    - Valid catalog name format
    - Valid version format
    - Valid as-of date format and plausibility
    - Valid SHA-256 prefix format

    Args:
        id_str: Snapshot ID string to validate.

    Returns:
        True if the ID is valid, False otherwise.

    Example:
        >>> validate_snapshot_id("catalog:tic:v8.2:20240115:a1b2c3d4")
        True
        >>> validate_snapshot_id("invalid-format")
        False
        >>> validate_snapshot_id("catalog:TIC:v8.2:20240115:a1b2c3d4")  # uppercase
        False
    """
    if not isinstance(id_str, str):
        return False

    try:
        parse_snapshot_id(id_str)
        return True
    except (ValueError, TypeError):
        return False
