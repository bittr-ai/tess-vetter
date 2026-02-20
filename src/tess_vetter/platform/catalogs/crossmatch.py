"""Crossmatch tool for astronomical catalog matching.

This module provides functionality to crossmatch astronomical positions against
known object catalogs (TOI, confirmed planets, false positives, eclipsing binaries).
It also assesses contamination risk from nearby sources.

The crossmatch function uses the CatalogSnapshotStore to load versioned catalog
snapshots, ensuring reproducible results tied to specific catalog versions.

Usage:
    >>> from tess_vetter.platform.catalogs.crossmatch import crossmatch
    >>> report = crossmatch(
    ...     ra=120.5,
    ...     dec=-45.3,
    ...     snapshot_ids=["catalog:toi:v1.0:20240115:abc123"],
    ...     search_radius_arcsec=10.0
    ... )
    >>> print(report.novelty_status)
    'novel'
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

# =============================================================================
# Data Models
# =============================================================================


class KnownObjectMatch(BaseModel):
    """A match to a known object in a catalog.

    Attributes:
        object_type: Type of the known object. One of:
            - "TOI": TESS Object of Interest
            - "CONFIRMED": Confirmed exoplanet
            - "FP": False positive
            - "EB": Eclipsing binary
        object_id: Unique identifier for the object in its catalog.
        separation_arcsec: Angular separation between query position and
            object position in arcseconds.
        catalog_source: Name/identifier of the source catalog.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    object_type: str  # "TOI", "CONFIRMED", "FP", "EB"
    object_id: str
    separation_arcsec: float
    catalog_source: str


class ContaminationRisk(BaseModel):
    """Assessment of contamination risk from nearby sources.

    Contamination occurs when light from nearby stars falls within the
    photometric aperture, potentially diluting or mimicking transit signals.

    Attributes:
        has_neighbors: Whether any neighbors exist within the search radius.
        nearest_neighbor_arcsec: Angular distance to the nearest neighbor
            in arcseconds, or None if no neighbors found.
        brightness_delta_mag: Magnitude difference between target and
            nearest contaminating source, or None if no neighbors.
            Positive values mean neighbor is fainter.
        dilution_factor: Estimated flux dilution factor from neighbors.
            1.0 means no dilution, <1.0 means some flux is from neighbors.
            None if no neighbors or cannot be computed.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    has_neighbors: bool
    nearest_neighbor_arcsec: float | None = None
    brightness_delta_mag: float | None = None
    dilution_factor: float | None = None


class CrossmatchReport(BaseModel):
    """Complete crossmatch analysis report.

    Contains all known object matches, contamination risk assessment,
    and novelty status determination.

    Attributes:
        known_object_matches: List of matches to known objects (TOI,
            confirmed planets, FPs, EBs) sorted by separation.
        contamination_risk: Assessment of contamination from neighbors.
        novelty_status: Classification of the target:
            - "novel": No matches to any known objects
            - "known": Matches a confirmed planet or TOI
            - "ambiguous": Matches an FP or EB, requires review
        snapshot_ids_used: List of catalog snapshot IDs that were searched.
        search_radius_arcsec: The search radius that was used in arcseconds.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    known_object_matches: list[KnownObjectMatch]
    contamination_risk: ContaminationRisk
    novelty_status: str  # "novel", "known", "ambiguous"
    snapshot_ids_used: list[str]
    search_radius_arcsec: float


# =============================================================================
# Catalog Entry Types
# =============================================================================


@dataclass(frozen=True)
class CatalogEntry:
    """A single entry in an astronomical catalog.

    Attributes:
        ra: Right ascension in degrees.
        dec: Declination in degrees.
        object_id: Unique identifier for this object.
        object_type: Type classification (TOI, CONFIRMED, FP, EB, STAR, etc).
        magnitude: Brightness in a standard bandpass, or None if unknown.
        catalog_name: Name of the catalog this entry belongs to.
    """

    ra: float
    dec: float
    object_id: str
    object_type: str
    magnitude: float | None = None
    catalog_name: str = ""


@dataclass
class CatalogData:
    """Data loaded from a catalog snapshot.

    Attributes:
        entries: List of catalog entries.
        catalog_name: Name/identifier of the catalog.
        version: Version string for this catalog snapshot.
        asof_date: Date the catalog was retrieved (YYYYMMDD format).
    """

    entries: list[CatalogEntry]
    catalog_name: str
    version: str
    asof_date: str


# =============================================================================
# Catalog Snapshot Store Protocol
# =============================================================================


@runtime_checkable
class CatalogSnapshotStore(Protocol):
    """Protocol for catalog snapshot storage and retrieval.

    Implementations must provide methods to load catalog data from
    versioned, checksummed snapshots. This ensures reproducibility
    by tying crossmatch results to specific catalog versions.

    Snapshot ID Format:
        catalog:<name>:<version>:<asof_yyyymmdd>:<sha256prefix>

    Example:
        catalog:toi:v1.0:20240115:a1b2c3d4
    """

    def load(self, snapshot_id: str) -> CatalogData:
        """Load catalog data from a snapshot.

        Args:
            snapshot_id: The snapshot identifier in format
                catalog:<name>:<version>:<asof>:<hash>

        Returns:
            CatalogData containing all entries from the snapshot.

        Raises:
            SnapshotNotFoundError: If the snapshot ID doesn't exist.
            SnapshotChecksumError: If the snapshot fails checksum validation.
        """
        ...

    def exists(self, snapshot_id: str) -> bool:
        """Check if a snapshot exists in the store.

        Args:
            snapshot_id: The snapshot identifier to check.

        Returns:
            True if the snapshot exists and is valid, False otherwise.
        """
        ...


# =============================================================================
# Exceptions
# =============================================================================


class CrossmatchError(Exception):
    """Base exception for crossmatch operations."""

    pass


class SnapshotNotFoundError(CrossmatchError):
    """Raised when a catalog snapshot cannot be found."""

    def __init__(self, snapshot_id: str) -> None:
        self.snapshot_id = snapshot_id
        super().__init__(f"Catalog snapshot not found: {snapshot_id}")


class InvalidSnapshotIdError(CrossmatchError):
    """Raised when a snapshot ID format is invalid."""

    def __init__(self, snapshot_id: str, reason: str) -> None:
        self.snapshot_id = snapshot_id
        self.reason = reason
        super().__init__(f"Invalid snapshot ID '{snapshot_id}': {reason}")


class NoCatalogsProvidedError(CrossmatchError):
    """Raised when crossmatch is called with no snapshot IDs."""

    def __init__(self) -> None:
        super().__init__("At least one snapshot ID must be provided")


# =============================================================================
# Utility Functions
# =============================================================================


def angular_separation_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Calculate angular separation between two celestial positions.

    Uses the haversine formula for accuracy at small separations.

    Args:
        ra1: Right ascension of first position in degrees.
        dec1: Declination of first position in degrees.
        ra2: Right ascension of second position in degrees.
        dec2: Declination of second position in degrees.

    Returns:
        Angular separation in arcseconds.
    """
    # Convert to radians
    ra1_rad = math.radians(ra1)
    dec1_rad = math.radians(dec1)
    ra2_rad = math.radians(ra2)
    dec2_rad = math.radians(dec2)

    # Haversine formula
    delta_ra = ra2_rad - ra1_rad
    delta_dec = dec2_rad - dec1_rad

    a = (
        math.sin(delta_dec / 2) ** 2
        + math.cos(dec1_rad) * math.cos(dec2_rad) * math.sin(delta_ra / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))

    # Convert radians to arcseconds
    # 1 radian = 206264.806 arcseconds
    return c * 206264.806


def validate_snapshot_id(snapshot_id: str) -> tuple[str, str, str, str]:
    """Validate and parse a snapshot ID.

    Args:
        snapshot_id: The snapshot ID to validate.

    Returns:
        Tuple of (name, version, asof_date, hash_prefix).

    Raises:
        InvalidSnapshotIdError: If the format is invalid.
    """
    if not snapshot_id:
        raise InvalidSnapshotIdError(snapshot_id, "Empty snapshot ID")

    parts = snapshot_id.split(":")
    if len(parts) != 5:
        raise InvalidSnapshotIdError(
            snapshot_id,
            f"Expected 5 colon-separated parts, got {len(parts)}",
        )

    prefix, name, version, asof, hash_prefix = parts

    if prefix != "catalog":
        raise InvalidSnapshotIdError(
            snapshot_id,
            f"Must start with 'catalog:', got '{prefix}:'",
        )

    if not name:
        raise InvalidSnapshotIdError(snapshot_id, "Catalog name is empty")

    if not version:
        raise InvalidSnapshotIdError(snapshot_id, "Version is empty")

    if not asof or len(asof) != 8 or not asof.isdigit():
        raise InvalidSnapshotIdError(
            snapshot_id,
            f"Invalid asof date format, expected YYYYMMDD: {asof}",
        )

    if not hash_prefix or not all(c in "0123456789abcdefABCDEF" for c in hash_prefix):
        raise InvalidSnapshotIdError(
            snapshot_id,
            f"Invalid hash prefix (must be hex): {hash_prefix}",
        )

    return name, version, asof, hash_prefix.lower()


def determine_novelty_status(matches: list[KnownObjectMatch]) -> str:
    """Determine novelty status based on known object matches.

    Args:
        matches: List of matches to known objects.

    Returns:
        One of:
        - "novel": No matches found
        - "known": Matches a confirmed planet or TOI
        - "ambiguous": Matches an FP or EB (requires review)
    """
    if not matches:
        return "novel"

    # Check match types in priority order
    for match in matches:
        if match.object_type in ("CONFIRMED", "TOI"):
            return "known"

    # If we have matches but none are confirmed/TOI, it's ambiguous
    # (could be FP or EB)
    return "ambiguous"


def compute_dilution_factor(target_mag: float | None, neighbor_mags: list[float]) -> float | None:
    """Compute flux dilution factor from neighboring sources.

    The dilution factor represents what fraction of the measured flux
    actually comes from the target star.

    Args:
        target_mag: Magnitude of the target star, or None if unknown.
        neighbor_mags: List of magnitudes for neighboring stars.

    Returns:
        Dilution factor (0 < factor <= 1), or None if cannot compute.
        1.0 means all flux is from target (no dilution).
    """
    if target_mag is None or not neighbor_mags:
        return None

    # Convert magnitudes to relative fluxes
    # flux_ratio = 10^((m1 - m2) / 2.5)
    target_flux = 1.0  # Normalize target flux to 1
    neighbor_flux_total = sum(10 ** ((target_mag - mag) / 2.5) for mag in neighbor_mags)

    total_flux = target_flux + neighbor_flux_total
    dilution_factor = target_flux / total_flux

    return dilution_factor


# =============================================================================
# Contamination Assessment
# =============================================================================


# Known object types that are relevant for crossmatch
KNOWN_OBJECT_TYPES = frozenset({"TOI", "CONFIRMED", "FP", "EB"})


def assess_contamination(
    ra: float,
    dec: float,
    catalog_entries: list[CatalogEntry],
    search_radius_arcsec: float,
    target_magnitude: float | None = None,
) -> ContaminationRisk:
    """Assess contamination risk from nearby sources.

    Args:
        ra: Target right ascension in degrees.
        dec: Target declination in degrees.
        catalog_entries: Entries from stellar catalogs to check.
        search_radius_arcsec: Maximum distance to search in arcseconds.
        target_magnitude: Target star magnitude, for dilution calculation.

    Returns:
        ContaminationRisk assessment.
    """
    # Find all neighbors within search radius (excluding known object types)
    neighbors: list[tuple[float, CatalogEntry]] = []

    for entry in catalog_entries:
        # Skip entries that are known objects (TOI, FP, etc.) - only consider stars
        if entry.object_type in KNOWN_OBJECT_TYPES:
            continue

        sep = angular_separation_arcsec(ra, dec, entry.ra, entry.dec)
        if sep <= search_radius_arcsec and sep > 0.1:  # Exclude very close matches (self)
            neighbors.append((sep, entry))

    if not neighbors:
        return ContaminationRisk(
            has_neighbors=False,
            nearest_neighbor_arcsec=None,
            brightness_delta_mag=None,
            dilution_factor=None,
        )

    # Sort by separation
    neighbors.sort(key=lambda x: x[0])

    nearest_sep, nearest_entry = neighbors[0]

    # Compute brightness delta if magnitudes available
    brightness_delta: float | None = None
    if target_magnitude is not None and nearest_entry.magnitude is not None:
        brightness_delta = nearest_entry.magnitude - target_magnitude

    # Compute dilution factor from all neighbors
    neighbor_mags = [entry.magnitude for _, entry in neighbors if entry.magnitude is not None]
    dilution = compute_dilution_factor(target_magnitude, neighbor_mags)

    return ContaminationRisk(
        has_neighbors=True,
        nearest_neighbor_arcsec=nearest_sep,
        brightness_delta_mag=brightness_delta,
        dilution_factor=dilution,
    )


# =============================================================================
# Known Object Matching
# =============================================================================


def find_known_object_matches(
    ra: float,
    dec: float,
    catalog_entries: list[CatalogEntry],
    search_radius_arcsec: float,
) -> list[KnownObjectMatch]:
    """Find matches to known objects (TOI, confirmed, FP, EB).

    Args:
        ra: Target right ascension in degrees.
        dec: Target declination in degrees.
        catalog_entries: Entries from catalogs to search.
        search_radius_arcsec: Maximum distance to search in arcseconds.

    Returns:
        List of KnownObjectMatch sorted by separation (closest first).
    """
    matches: list[KnownObjectMatch] = []

    for entry in catalog_entries:
        # Only match known object types
        if entry.object_type not in KNOWN_OBJECT_TYPES:
            continue

        sep = angular_separation_arcsec(ra, dec, entry.ra, entry.dec)
        if sep <= search_radius_arcsec:
            matches.append(
                KnownObjectMatch(
                    object_type=entry.object_type,
                    object_id=entry.object_id,
                    separation_arcsec=sep,
                    catalog_source=entry.catalog_name,
                )
            )

    # Sort by separation (closest first)
    matches.sort(key=lambda m: m.separation_arcsec)

    return matches


# =============================================================================
# Main Crossmatch Function
# =============================================================================


def crossmatch(
    ra: float,
    dec: float,
    snapshot_ids: list[str],
    search_radius_arcsec: float = 10.0,
    contamination_policy_version: str = "v1",
    catalog_store: CatalogSnapshotStore | None = None,
    target_magnitude: float | None = None,
) -> CrossmatchReport:
    """Crossmatch a position against catalog snapshots.

    Searches multiple catalog snapshots for known objects and assesses
    contamination risk from nearby sources.

    Args:
        ra: Right ascension in degrees (J2000).
        dec: Declination in degrees (J2000).
        snapshot_ids: List of catalog snapshot IDs to search. Must have
            format: catalog:<name>:<version>:<asof_yyyymmdd>:<sha256prefix>
        search_radius_arcsec: Maximum search radius in arcseconds.
            Default is 10.0 arcseconds.
        contamination_policy_version: Version of contamination assessment
            policy to use. Currently only "v1" is supported.
        catalog_store: Optional CatalogSnapshotStore implementation.
            If None, snapshots must be loaded externally.
        target_magnitude: Optional magnitude of target for dilution
            calculation.

    Returns:
        CrossmatchReport containing matches, contamination assessment,
        and novelty status.

    Raises:
        NoCatalogsProvidedError: If snapshot_ids is empty.
        InvalidSnapshotIdError: If any snapshot ID format is invalid.
        SnapshotNotFoundError: If a catalog snapshot cannot be loaded.

    Example:
        >>> from tess_vetter.platform.catalogs.crossmatch import crossmatch
        >>> report = crossmatch(
        ...     ra=120.5,
        ...     dec=-45.3,
        ...     snapshot_ids=[
        ...         "catalog:toi:v1.0:20240115:abc123",
        ...         "catalog:gaia_dr3:v1.0:20240115:def456"
        ...     ],
        ...     search_radius_arcsec=10.0
        ... )
        >>> print(f"Novelty: {report.novelty_status}")
        >>> print(f"Matches: {len(report.known_object_matches)}")
    """
    # Validate inputs
    if not snapshot_ids:
        raise NoCatalogsProvidedError()

    # Validate all snapshot IDs upfront
    for sid in snapshot_ids:
        validate_snapshot_id(sid)

    # Collect all catalog entries from all snapshots
    all_entries: list[CatalogEntry] = []
    used_snapshot_ids: list[str] = []

    if catalog_store is not None:
        for snapshot_id in snapshot_ids:
            if not catalog_store.exists(snapshot_id):
                raise SnapshotNotFoundError(snapshot_id)

            catalog_data = catalog_store.load(snapshot_id)
            all_entries.extend(catalog_data.entries)
            used_snapshot_ids.append(snapshot_id)
    else:
        # If no store provided, just validate IDs and return empty results
        # This allows for testing without a real store
        used_snapshot_ids = list(snapshot_ids)

    # Find known object matches
    matches = find_known_object_matches(ra, dec, all_entries, search_radius_arcsec)

    # Assess contamination risk
    contamination = assess_contamination(
        ra, dec, all_entries, search_radius_arcsec, target_magnitude
    )

    # Determine novelty status
    novelty = determine_novelty_status(matches)

    return CrossmatchReport(
        known_object_matches=matches,
        contamination_risk=contamination,
        novelty_status=novelty,
        snapshot_ids_used=used_snapshot_ids,
        search_radius_arcsec=search_radius_arcsec,
    )


__all__ = [
    # Data models
    "KnownObjectMatch",
    "ContaminationRisk",
    "CrossmatchReport",
    "CatalogEntry",
    "CatalogData",
    # Protocol
    "CatalogSnapshotStore",
    # Exceptions
    "CrossmatchError",
    "SnapshotNotFoundError",
    "InvalidSnapshotIdError",
    "NoCatalogsProvidedError",
    # Functions
    "crossmatch",
    "angular_separation_arcsec",
    "validate_snapshot_id",
    "determine_novelty_status",
    "compute_dilution_factor",
    "assess_contamination",
    "find_known_object_matches",
]
