"""Disk-backed catalog storage for versioned, checksummed catalog snapshots.

This module provides a filesystem-based storage layer for catalog data,
enabling reproducible analysis with versioned catalog snapshots that
support offline operation after initial installation.

Storage structure:
    <storage_root>/
        catalogs/
            <name>/
                <version>/
                    data.json       - catalog data
                    metadata.json   - installation metadata (source_url, checksum, etc.)

Key features:
- Content-addressed storage using SHA-256 checksums
- Versioned catalogs for reproducibility
- Offline operation after installation
- Checksum verification for data integrity

Usage:
    >>> from astro_arc.catalogs.store import CatalogSnapshotStore
    >>> store = CatalogSnapshotStore("/path/to/storage")
    >>> snapshot_id = store.install("my_catalog", "1.0.0", "https://example.com/catalog.json")
    >>> catalog = store.load(snapshot_id)
    >>> store.verify_checksum(snapshot_id)
    True
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class CatalogNotFoundError(Exception):
    """Raised when a catalog snapshot cannot be found in storage.

    Attributes:
        snapshot_id: The snapshot ID that was not found.
        reason: Additional context about why the catalog was not found.
    """

    def __init__(self, snapshot_id: str, reason: str | None = None) -> None:
        self.snapshot_id = snapshot_id
        self.reason = reason
        message = f"Catalog snapshot not found: {snapshot_id}"
        if reason:
            message += f" ({reason})"
        super().__init__(message)


class CatalogChecksumError(Exception):
    """Raised when catalog data fails checksum verification.

    Attributes:
        snapshot_id: The snapshot ID that failed verification.
        expected: The expected checksum.
        actual: The actual computed checksum.
    """

    def __init__(self, snapshot_id: str, expected: str, actual: str) -> None:
        self.snapshot_id = snapshot_id
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Checksum verification failed for {snapshot_id}: "
            f"expected {expected[:16]}..., got {actual[:16]}..."
        )


class CatalogInstallError(Exception):
    """Raised when catalog installation fails.

    Attributes:
        name: The catalog name.
        version: The catalog version.
        source_url: The source URL that failed.
        reason: The underlying error message.
    """

    def __init__(self, name: str, version: str, source_url: str, reason: str) -> None:
        self.name = name
        self.version = version
        self.source_url = source_url
        self.reason = reason
        super().__init__(f"Failed to install catalog {name}@{version} from {source_url}: {reason}")


@dataclass(frozen=True)
class CatalogData:
    """Immutable container for loaded catalog data.

    Represents a versioned catalog snapshot loaded from disk storage.
    The data is read-only after loading to ensure consistency.

    Attributes:
        name: The catalog name (e.g., "tic", "gaia_dr3").
        version: The catalog version string (e.g., "1.0.0", "2024.01").
        snapshot_id: The unique identifier for this snapshot.
        data: The catalog data as a dictionary. Structure depends on catalog type.
        checksum: SHA-256 checksum of the data file.
        source_url: The original URL from which the catalog was installed.
        installed_at: ISO8601 timestamp of when the catalog was installed.
        entry_count: Number of entries in the catalog (if applicable).
    """

    name: str
    version: str
    snapshot_id: str
    data: dict[str, Any]
    checksum: str
    source_url: str
    installed_at: str
    entry_count: int = 0

    def __post_init__(self) -> None:
        """Validate required fields after initialization."""
        if not self.name:
            raise ValueError("Catalog name cannot be empty")
        if not self.version:
            raise ValueError("Catalog version cannot be empty")
        if not self.snapshot_id:
            raise ValueError("Snapshot ID cannot be empty")

    def get_entries(self) -> list[dict[str, Any]]:
        """Return catalog entries if data contains an 'entries' list.

        Returns:
            List of catalog entry dictionaries, or empty list if not applicable.
        """
        entries = self.data.get("entries", [])
        if isinstance(entries, list):
            return entries
        return []


@dataclass
class CatalogMetadata:
    """Internal metadata for installed catalog snapshots.

    Stored alongside catalog data to track provenance and enable verification.
    """

    name: str
    version: str
    snapshot_id: str
    source_url: str
    checksum: str
    installed_at: str
    file_size_bytes: int = 0
    entry_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "snapshot_id": self.snapshot_id,
            "source_url": self.source_url,
            "checksum": self.checksum,
            "installed_at": self.installed_at,
            "file_size_bytes": self.file_size_bytes,
            "entry_count": self.entry_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CatalogMetadata:
        """Create from dictionary loaded from JSON."""
        return cls(
            name=data["name"],
            version=data["version"],
            snapshot_id=data["snapshot_id"],
            source_url=data["source_url"],
            checksum=data["checksum"],
            installed_at=data["installed_at"],
            file_size_bytes=data.get("file_size_bytes", 0),
            entry_count=data.get("entry_count", 0),
        )


class CatalogSnapshotStore:
    """Filesystem-based persistence layer for catalog snapshots.

    Provides versioned, checksummed storage for catalog data with support
    for offline operation after initial installation. Catalogs are stored
    in a structured directory hierarchy organized by name and version.

    Storage layout:
        <storage_root>/catalogs/<name>/<version>/data.json
        <storage_root>/catalogs/<name>/<version>/metadata.json

    Thread Safety:
        This class is NOT thread-safe. External synchronization is required
        if accessed from multiple threads.

    Attributes:
        storage_root: Base directory for catalog storage.

    Example:
        >>> store = CatalogSnapshotStore("/tmp/catalogs")
        >>> snapshot_id = store.install("tic", "v8.2", "https://example.com/tic.json")
        >>> catalog = store.load(snapshot_id)
        >>> print(catalog.name, catalog.version)
        tic v8.2
        >>> store.verify_checksum(snapshot_id)
        True
    """

    def __init__(self, storage_root: str | Path) -> None:
        """Initialize the catalog snapshot store.

        Args:
            storage_root: Base directory for catalog storage. Will be created
                if it doesn't exist during first install operation.
        """
        self.storage_root = Path(storage_root)
        self._catalogs_dir = self.storage_root / "catalogs"

    def install(self, name: str, version: str, source_url: str) -> str:
        """Download and install a catalog snapshot.

        Downloads catalog data from the source URL, computes its checksum,
        and stores it with versioned metadata. The snapshot can then be
        loaded offline without network access.

        Args:
            name: Catalog name (e.g., "tic", "gaia_dr3"). Must be non-empty
                and filesystem-safe (alphanumeric, underscore, hyphen).
            version: Version string (e.g., "1.0.0", "2024.01"). Must be non-empty
                and filesystem-safe.
            source_url: URL to download catalog data from. Must return valid JSON.
                Supports http://, https://, and file:// protocols.

        Returns:
            Snapshot ID in the format "<name>:<version>:<checksum_prefix>".
            The checksum prefix is the first 12 characters of the SHA-256 hash.

        Raises:
            CatalogInstallError: If download fails, URL is unreachable, or
                response is not valid JSON.
            ValueError: If name or version contains invalid characters.

        Example:
            >>> store = CatalogSnapshotStore("/tmp/catalogs")
            >>> snapshot_id = store.install("tic", "v8", "https://example.com/tic.json")
            >>> print(snapshot_id)
            'tic:v8:a1b2c3d4e5f6'
        """
        # Validate name and version
        self._validate_name(name)
        self._validate_version(version)

        # Create catalog directory
        catalog_dir = self._catalogs_dir / name / version
        catalog_dir.mkdir(parents=True, exist_ok=True)

        # Download catalog data
        try:
            data_bytes = self._download(source_url)
        except Exception as e:
            raise CatalogInstallError(name, version, source_url, str(e)) from e

        # Parse and validate JSON
        try:
            data = json.loads(data_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise CatalogInstallError(name, version, source_url, f"Invalid JSON: {e}") from e

        # Compute checksum
        checksum = hashlib.sha256(data_bytes).hexdigest()
        checksum_prefix = checksum[:12]
        snapshot_id = f"{name}:{version}:{checksum_prefix}"

        # Count entries if present
        entry_count = 0
        if isinstance(data, dict) and "entries" in data:
            entries = data.get("entries", [])
            if isinstance(entries, list):
                entry_count = len(entries)

        # Create metadata
        metadata = CatalogMetadata(
            name=name,
            version=version,
            snapshot_id=snapshot_id,
            source_url=source_url,
            checksum=checksum,
            installed_at=datetime.now(UTC).isoformat(),
            file_size_bytes=len(data_bytes),
            entry_count=entry_count,
        )

        # Write data file
        data_path = catalog_dir / "data.json"
        data_path.write_bytes(data_bytes)

        # Write metadata file
        metadata_path = catalog_dir / "metadata.json"
        metadata_json = json.dumps(metadata.to_dict(), indent=2)
        metadata_path.write_text(metadata_json, encoding="utf-8")

        return snapshot_id

    def install_from_data(
        self, name: str, version: str, data: dict[str, Any], source_url: str = ""
    ) -> str:
        """Install a catalog snapshot from in-memory data.

        Useful for testing or when catalog data is already available in memory.

        Args:
            name: Catalog name.
            version: Version string.
            data: Catalog data as a dictionary.
            source_url: Optional source URL for provenance tracking.

        Returns:
            Snapshot ID in the format "<name>:<version>:<checksum_prefix>".

        Raises:
            ValueError: If name or version contains invalid characters.
        """
        # Validate name and version
        self._validate_name(name)
        self._validate_version(version)

        # Create catalog directory
        catalog_dir = self._catalogs_dir / name / version
        catalog_dir.mkdir(parents=True, exist_ok=True)

        # Serialize data
        data_bytes = json.dumps(data, indent=2).encode("utf-8")

        # Compute checksum
        checksum = hashlib.sha256(data_bytes).hexdigest()
        checksum_prefix = checksum[:12]
        snapshot_id = f"{name}:{version}:{checksum_prefix}"

        # Count entries if present
        entry_count = 0
        if "entries" in data:
            entries = data.get("entries", [])
            if isinstance(entries, list):
                entry_count = len(entries)

        # Create metadata
        metadata = CatalogMetadata(
            name=name,
            version=version,
            snapshot_id=snapshot_id,
            source_url=source_url or "memory://",
            checksum=checksum,
            installed_at=datetime.now(UTC).isoformat(),
            file_size_bytes=len(data_bytes),
            entry_count=entry_count,
        )

        # Write data file
        data_path = catalog_dir / "data.json"
        data_path.write_bytes(data_bytes)

        # Write metadata file
        metadata_path = catalog_dir / "metadata.json"
        metadata_json = json.dumps(metadata.to_dict(), indent=2)
        metadata_path.write_text(metadata_json, encoding="utf-8")

        return snapshot_id

    def load(self, snapshot_id: str) -> CatalogData:
        """Load catalog data by snapshot ID.

        Loads the catalog data and metadata from disk. This operation is
        fully offline - no network access is required after installation.

        Args:
            snapshot_id: The snapshot ID returned by install(). Format is
                "<name>:<version>:<checksum_prefix>".

        Returns:
            CatalogData containing the loaded catalog.

        Raises:
            CatalogNotFoundError: If the snapshot ID is not found or
                has an invalid format.

        Example:
            >>> catalog = store.load("tic:v8:a1b2c3d4e5f6")
            >>> print(catalog.name, catalog.version)
            tic v8
            >>> print(len(catalog.get_entries()))
            1000
        """
        # Parse snapshot ID
        name, version, checksum_prefix = self._parse_snapshot_id(snapshot_id)

        # Find catalog directory
        catalog_dir = self._catalogs_dir / name / version
        if not catalog_dir.exists():
            raise CatalogNotFoundError(snapshot_id, f"No catalog found at {name}/{version}")

        # Load metadata
        metadata_path = catalog_dir / "metadata.json"
        if not metadata_path.exists():
            raise CatalogNotFoundError(snapshot_id, "Metadata file missing")

        try:
            metadata_json = metadata_path.read_text(encoding="utf-8")
            metadata = CatalogMetadata.from_dict(json.loads(metadata_json))
        except (json.JSONDecodeError, KeyError) as e:
            raise CatalogNotFoundError(snapshot_id, f"Invalid metadata: {e}") from e

        # Verify checksum prefix matches
        if not metadata.checksum.startswith(checksum_prefix):
            raise CatalogNotFoundError(
                snapshot_id,
                f"Checksum mismatch: expected prefix {checksum_prefix}, "
                f"got {metadata.checksum[:12]}",
            )

        # Load data
        data_path = catalog_dir / "data.json"
        if not data_path.exists():
            raise CatalogNotFoundError(snapshot_id, "Data file missing")

        try:
            data_bytes = data_path.read_bytes()
            data = json.loads(data_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise CatalogNotFoundError(snapshot_id, f"Invalid data: {e}") from e

        return CatalogData(
            name=metadata.name,
            version=metadata.version,
            snapshot_id=metadata.snapshot_id,
            data=data,
            checksum=metadata.checksum,
            source_url=metadata.source_url,
            installed_at=metadata.installed_at,
            entry_count=metadata.entry_count,
        )

    def list_installed(self) -> list[str]:
        """List all installed snapshot IDs.

        Scans the storage directory for all installed catalog versions
        and returns their snapshot IDs.

        Returns:
            List of snapshot IDs in the format "<name>:<version>:<checksum_prefix>".
            Returns empty list if no catalogs are installed.

        Example:
            >>> store.list_installed()
            ['tic:v8:a1b2c3d4e5f6', 'gaia:dr3:b2c3d4e5f678']
        """
        if not self._catalogs_dir.exists():
            return []

        snapshot_ids: list[str] = []

        for name_dir in self._catalogs_dir.iterdir():
            if not name_dir.is_dir():
                continue

            for version_dir in name_dir.iterdir():
                if not version_dir.is_dir():
                    continue

                metadata_path = version_dir / "metadata.json"
                if not metadata_path.exists():
                    continue

                try:
                    metadata_json = metadata_path.read_text(encoding="utf-8")
                    metadata = CatalogMetadata.from_dict(json.loads(metadata_json))
                    snapshot_ids.append(metadata.snapshot_id)
                except (json.JSONDecodeError, KeyError, OSError):
                    # Skip corrupted entries
                    continue

        return sorted(snapshot_ids)

    def verify_checksum(self, snapshot_id: str) -> bool:
        """Verify catalog data integrity.

        Recomputes the SHA-256 checksum of the stored data file and
        compares it against the stored checksum in metadata.

        Args:
            snapshot_id: The snapshot ID to verify.

        Returns:
            True if the checksum matches, False if verification fails.

        Raises:
            CatalogNotFoundError: If the snapshot ID is not found.

        Example:
            >>> store.verify_checksum("tic:v8:a1b2c3d4e5f6")
            True
        """
        # Parse snapshot ID
        name, version, _ = self._parse_snapshot_id(snapshot_id)

        # Find catalog directory
        catalog_dir = self._catalogs_dir / name / version
        if not catalog_dir.exists():
            raise CatalogNotFoundError(snapshot_id, f"No catalog found at {name}/{version}")

        # Load metadata
        metadata_path = catalog_dir / "metadata.json"
        if not metadata_path.exists():
            raise CatalogNotFoundError(snapshot_id, "Metadata file missing")

        try:
            metadata_json = metadata_path.read_text(encoding="utf-8")
            metadata = CatalogMetadata.from_dict(json.loads(metadata_json))
        except (json.JSONDecodeError, KeyError) as e:
            raise CatalogNotFoundError(snapshot_id, f"Invalid metadata: {e}") from e

        # Load and hash data
        data_path = catalog_dir / "data.json"
        if not data_path.exists():
            raise CatalogNotFoundError(snapshot_id, "Data file missing")

        data_bytes = data_path.read_bytes()
        actual_checksum = hashlib.sha256(data_bytes).hexdigest()

        return actual_checksum == metadata.checksum

    def exists(self, snapshot_id: str) -> bool:
        """Check if a snapshot exists in storage.

        Args:
            snapshot_id: The snapshot ID to check.

        Returns:
            True if the snapshot exists, False otherwise.
        """
        try:
            name, version, checksum_prefix = self._parse_snapshot_id(snapshot_id)
        except (ValueError, CatalogNotFoundError):
            return False

        catalog_dir = self._catalogs_dir / name / version
        if not catalog_dir.exists():
            return False

        metadata_path = catalog_dir / "metadata.json"
        if not metadata_path.exists():
            return False

        try:
            metadata_json = metadata_path.read_text(encoding="utf-8")
            metadata = CatalogMetadata.from_dict(json.loads(metadata_json))
            return metadata.checksum.startswith(checksum_prefix)
        except (json.JSONDecodeError, KeyError, OSError):
            return False

    def delete(self, snapshot_id: str) -> bool:
        """Delete a catalog snapshot from storage.

        Args:
            snapshot_id: The snapshot ID to delete.

        Returns:
            True if the snapshot was deleted, False if it didn't exist.

        Raises:
            CatalogNotFoundError: If the snapshot ID format is invalid.
        """
        name, version, _ = self._parse_snapshot_id(snapshot_id)

        catalog_dir = self._catalogs_dir / name / version
        if not catalog_dir.exists():
            return False

        # Remove all files in the catalog directory
        for file_path in catalog_dir.iterdir():
            file_path.unlink()

        # Remove the version directory
        catalog_dir.rmdir()

        # Remove name directory if empty
        name_dir = self._catalogs_dir / name
        with contextlib.suppress(OSError):
            name_dir.rmdir()

        return True

    def _parse_snapshot_id(self, snapshot_id: str) -> tuple[str, str, str]:
        """Parse a snapshot ID into components.

        Args:
            snapshot_id: The snapshot ID string.

        Returns:
            Tuple of (name, version, checksum_prefix).

        Raises:
            CatalogNotFoundError: If the format is invalid.
        """
        parts = snapshot_id.split(":")
        if len(parts) != 3:
            raise CatalogNotFoundError(
                snapshot_id,
                f"Invalid format: expected 'name:version:checksum', got {len(parts)} parts",
            )

        name, version, checksum_prefix = parts
        if not name or not version or not checksum_prefix:
            raise CatalogNotFoundError(snapshot_id, "Empty component in snapshot ID")

        return name, version, checksum_prefix

    def _validate_name(self, name: str) -> None:
        """Validate catalog name for filesystem safety.

        Args:
            name: The catalog name to validate.

        Raises:
            ValueError: If the name is invalid.
        """
        if not name:
            raise ValueError("Catalog name cannot be empty")
        if not all(c.isalnum() or c in "_-" for c in name):
            raise ValueError(
                f"Invalid catalog name '{name}': only alphanumeric, underscore, "
                "and hyphen characters are allowed"
            )
        if name.startswith(".") or name.startswith("-"):
            raise ValueError(f"Invalid catalog name '{name}': cannot start with '.' or '-'")

    def _validate_version(self, version: str) -> None:
        """Validate version string for filesystem safety.

        Args:
            version: The version string to validate.

        Raises:
            ValueError: If the version is invalid.
        """
        if not version:
            raise ValueError("Catalog version cannot be empty")
        if not all(c.isalnum() or c in "_-." for c in version):
            raise ValueError(
                f"Invalid catalog version '{version}': only alphanumeric, "
                "underscore, hyphen, and period characters are allowed"
            )
        if version.startswith(".") or version.startswith("-"):
            raise ValueError(f"Invalid catalog version '{version}': cannot start with '.' or '-'")

    def _download(self, url: str) -> bytes:
        """Download data from a URL.

        Args:
            url: The URL to download from.

        Returns:
            The downloaded data as bytes.

        Raises:
            Exception: If the download fails.
        """
        with urllib.request.urlopen(url, timeout=60) as response:
            result: bytes = response.read()
            return result


__all__ = [
    "CatalogData",
    "CatalogChecksumError",
    "CatalogInstallError",
    "CatalogNotFoundError",
    "CatalogSnapshotStore",
]
