"""Tests for tess_vetter.catalogs.store module.

Comprehensive tests for the CatalogSnapshotStore persistence layer including:
- Installation from URL and in-memory data
- Loading installed catalogs
- Checksum verification and data integrity
- Listing installed snapshots
- Error handling for invalid operations
- Filesystem structure validation
- Offline operation (no network after install)
"""

from __future__ import annotations

import hashlib
import http.server
import json
import socketserver
import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from tess_vetter.platform.catalogs.store import (
    CatalogChecksumError,
    CatalogData,
    CatalogInstallError,
    CatalogNotFoundError,
    CatalogSnapshotStore,
)

# =============================================================================
# Test Fixtures
# =============================================================================


def create_sample_catalog_data(
    num_entries: int = 10,
    catalog_name: str = "test_catalog",
) -> dict[str, Any]:
    """Create sample catalog data for testing."""
    entries = [
        {
            "id": f"obj_{i:04d}",
            "ra": 120.0 + i * 0.1,
            "dec": 45.0 + i * 0.05,
            "magnitude": 10.0 + i * 0.2,
        }
        for i in range(num_entries)
    ]
    return {
        "catalog_name": catalog_name,
        "version": "1.0.0",
        "description": f"Test catalog with {num_entries} entries",
        "entries": entries,
    }


@pytest.fixture
def temp_storage(tmp_path: Path) -> CatalogSnapshotStore:
    """Create a temporary CatalogSnapshotStore."""
    return CatalogSnapshotStore(tmp_path / "catalog_store")


@pytest.fixture
def sample_catalog_data() -> dict[str, Any]:
    """Create sample catalog data."""
    return create_sample_catalog_data()


@pytest.fixture
def installed_catalog(
    temp_storage: CatalogSnapshotStore, sample_catalog_data: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Install a sample catalog and return (snapshot_id, data)."""
    snapshot_id = temp_storage.install_from_data(
        name="test_catalog",
        version="1.0.0",
        data=sample_catalog_data,
        source_url="https://example.com/catalog.json",
    )
    return snapshot_id, sample_catalog_data


class SimpleHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler for test server."""

    catalog_data: dict[str, Any] = {}

    def do_GET(self) -> None:
        """Handle GET request."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response = json.dumps(self.catalog_data).encode("utf-8")
        self.wfile.write(response)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress logging."""
        pass


@pytest.fixture
def http_server(sample_catalog_data: dict[str, Any]) -> Generator[str, None, None]:
    """Start a local HTTP server serving sample catalog data."""
    SimpleHTTPHandler.catalog_data = sample_catalog_data

    with socketserver.TCPServer(("localhost", 0), SimpleHTTPHandler) as server:
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()

        yield f"http://localhost:{port}/catalog.json"

        server.shutdown()


# =============================================================================
# CatalogSnapshotStore Initialization Tests
# =============================================================================


class TestCatalogSnapshotStoreInit:
    """Tests for CatalogSnapshotStore initialization."""

    def test_init_with_string_path(self, tmp_path: Path) -> None:
        """Can initialize with string path."""
        store = CatalogSnapshotStore(str(tmp_path / "test_store"))
        assert store.storage_root == tmp_path / "test_store"

    def test_init_with_path_object(self, tmp_path: Path) -> None:
        """Can initialize with Path object."""
        store = CatalogSnapshotStore(tmp_path / "test_store")
        assert store.storage_root == tmp_path / "test_store"

    def test_storage_root_not_created_on_init(self, tmp_path: Path) -> None:
        """Storage directory is not created until first install."""
        store = CatalogSnapshotStore(tmp_path / "test_store")
        assert not store.storage_root.exists()


# =============================================================================
# Install from Data Tests
# =============================================================================


class TestInstallFromData:
    """Tests for install_from_data() method."""

    def test_install_returns_snapshot_id(
        self, temp_storage: CatalogSnapshotStore, sample_catalog_data: dict[str, Any]
    ) -> None:
        """Install returns a valid snapshot ID."""
        snapshot_id = temp_storage.install_from_data(
            name="test_catalog",
            version="1.0.0",
            data=sample_catalog_data,
        )
        assert snapshot_id.startswith("test_catalog:1.0.0:")
        # Checksum prefix should be 12 hex characters
        parts = snapshot_id.split(":")
        assert len(parts) == 3
        assert len(parts[2]) == 12
        assert all(c in "0123456789abcdef" for c in parts[2])

    def test_install_creates_directory_structure(
        self, temp_storage: CatalogSnapshotStore, sample_catalog_data: dict[str, Any]
    ) -> None:
        """Install creates the correct directory structure."""
        temp_storage.install_from_data(
            name="test_catalog",
            version="1.0.0",
            data=sample_catalog_data,
        )

        catalog_dir = temp_storage._catalogs_dir / "test_catalog" / "1.0.0"
        assert catalog_dir.exists()
        assert (catalog_dir / "data.json").exists()
        assert (catalog_dir / "metadata.json").exists()

    def test_install_creates_valid_data_file(
        self, temp_storage: CatalogSnapshotStore, sample_catalog_data: dict[str, Any]
    ) -> None:
        """Install creates a valid JSON data file."""
        temp_storage.install_from_data(
            name="test_catalog",
            version="1.0.0",
            data=sample_catalog_data,
        )

        data_path = temp_storage._catalogs_dir / "test_catalog" / "1.0.0" / "data.json"
        content = json.loads(data_path.read_text())
        assert content["catalog_name"] == "test_catalog"
        assert len(content["entries"]) == 10

    def test_install_creates_valid_metadata_file(
        self, temp_storage: CatalogSnapshotStore, sample_catalog_data: dict[str, Any]
    ) -> None:
        """Install creates a valid metadata file."""
        snapshot_id = temp_storage.install_from_data(
            name="test_catalog",
            version="1.0.0",
            data=sample_catalog_data,
            source_url="https://example.com/catalog.json",
        )

        metadata_path = temp_storage._catalogs_dir / "test_catalog" / "1.0.0" / "metadata.json"
        metadata = json.loads(metadata_path.read_text())

        assert metadata["name"] == "test_catalog"
        assert metadata["version"] == "1.0.0"
        assert metadata["snapshot_id"] == snapshot_id
        assert metadata["source_url"] == "https://example.com/catalog.json"
        assert len(metadata["checksum"]) == 64  # Full SHA-256
        assert metadata["entry_count"] == 10
        assert metadata["file_size_bytes"] > 0

    def test_install_computes_correct_checksum(
        self, temp_storage: CatalogSnapshotStore, sample_catalog_data: dict[str, Any]
    ) -> None:
        """Install computes the correct SHA-256 checksum."""
        snapshot_id = temp_storage.install_from_data(
            name="test_catalog",
            version="1.0.0",
            data=sample_catalog_data,
        )

        data_path = temp_storage._catalogs_dir / "test_catalog" / "1.0.0" / "data.json"
        data_bytes = data_path.read_bytes()
        expected_checksum = hashlib.sha256(data_bytes).hexdigest()

        metadata_path = temp_storage._catalogs_dir / "test_catalog" / "1.0.0" / "metadata.json"
        metadata = json.loads(metadata_path.read_text())

        assert metadata["checksum"] == expected_checksum
        assert snapshot_id.endswith(f":{expected_checksum[:12]}")

    def test_install_with_empty_data(self, temp_storage: CatalogSnapshotStore) -> None:
        """Install works with empty catalog data."""
        snapshot_id = temp_storage.install_from_data(
            name="empty_catalog",
            version="0.0.1",
            data={},
        )
        assert snapshot_id.startswith("empty_catalog:0.0.1:")

    def test_install_with_nested_data(self, temp_storage: CatalogSnapshotStore) -> None:
        """Install works with deeply nested data."""
        nested_data = {
            "level1": {
                "level2": {
                    "level3": [1, 2, 3],
                },
            },
        }
        snapshot_id = temp_storage.install_from_data(
            name="nested",
            version="1.0",
            data=nested_data,
        )

        catalog = temp_storage.load(snapshot_id)
        assert catalog.data == nested_data


# =============================================================================
# Install from URL Tests
# =============================================================================


class TestInstallFromUrl:
    """Tests for install() method with HTTP URLs."""

    def test_install_from_http(
        self,
        temp_storage: CatalogSnapshotStore,
        http_server: str,
        sample_catalog_data: dict[str, Any],
    ) -> None:
        """Install downloads and stores catalog from HTTP URL."""
        snapshot_id = temp_storage.install(
            name="remote_catalog",
            version="1.0.0",
            source_url=http_server,
        )

        assert snapshot_id.startswith("remote_catalog:1.0.0:")

        # Verify catalog can be loaded
        catalog = temp_storage.load(snapshot_id)
        assert catalog.name == "remote_catalog"
        assert catalog.version == "1.0.0"
        assert catalog.data["catalog_name"] == sample_catalog_data["catalog_name"]

    def test_install_from_file_url(
        self, temp_storage: CatalogSnapshotStore, tmp_path: Path
    ) -> None:
        """Install works with file:// URLs."""
        # Create a local JSON file
        catalog_data = create_sample_catalog_data(5)
        file_path = tmp_path / "local_catalog.json"
        file_path.write_text(json.dumps(catalog_data))

        snapshot_id = temp_storage.install(
            name="local_catalog",
            version="1.0.0",
            source_url=f"file://{file_path}",
        )

        catalog = temp_storage.load(snapshot_id)
        assert catalog.name == "local_catalog"
        assert len(catalog.get_entries()) == 5

    def test_install_invalid_url_raises(self, temp_storage: CatalogSnapshotStore) -> None:
        """Install raises CatalogInstallError for unreachable URL."""
        with pytest.raises(CatalogInstallError) as exc_info:
            temp_storage.install(
                name="bad_catalog",
                version="1.0.0",
                source_url="http://localhost:99999/nonexistent",
            )

        assert exc_info.value.name == "bad_catalog"
        assert exc_info.value.version == "1.0.0"
        assert "localhost:99999" in exc_info.value.source_url

    def test_install_invalid_json_raises(
        self, temp_storage: CatalogSnapshotStore, tmp_path: Path
    ) -> None:
        """Install raises CatalogInstallError for invalid JSON."""
        # Create a file with invalid JSON
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{{")

        with pytest.raises(CatalogInstallError) as exc_info:
            temp_storage.install(
                name="bad_json",
                version="1.0.0",
                source_url=f"file://{bad_file}",
            )

        assert "Invalid JSON" in exc_info.value.reason


# =============================================================================
# Load Tests
# =============================================================================


class TestLoad:
    """Tests for load() method."""

    def test_load_returns_catalog_data(
        self,
        temp_storage: CatalogSnapshotStore,
        installed_catalog: tuple[str, dict[str, Any]],
    ) -> None:
        """Load returns CatalogData with correct attributes."""
        snapshot_id, original_data = installed_catalog
        catalog = temp_storage.load(snapshot_id)

        assert isinstance(catalog, CatalogData)
        assert catalog.name == "test_catalog"
        assert catalog.version == "1.0.0"
        assert catalog.snapshot_id == snapshot_id
        assert catalog.data == original_data

    def test_load_preserves_all_data(
        self,
        temp_storage: CatalogSnapshotStore,
        installed_catalog: tuple[str, dict[str, Any]],
    ) -> None:
        """Load preserves all catalog data fields."""
        snapshot_id, original_data = installed_catalog
        catalog = temp_storage.load(snapshot_id)

        assert catalog.data["catalog_name"] == original_data["catalog_name"]
        assert catalog.data["entries"] == original_data["entries"]
        assert catalog.entry_count == len(original_data["entries"])

    def test_load_get_entries(
        self,
        temp_storage: CatalogSnapshotStore,
        installed_catalog: tuple[str, dict[str, Any]],
    ) -> None:
        """get_entries() returns the entries list."""
        snapshot_id, _ = installed_catalog
        catalog = temp_storage.load(snapshot_id)

        entries = catalog.get_entries()
        assert len(entries) == 10
        assert entries[0]["id"] == "obj_0000"
        assert entries[9]["id"] == "obj_0009"

    def test_load_nonexistent_raises(self, temp_storage: CatalogSnapshotStore) -> None:
        """Load raises CatalogNotFoundError for nonexistent snapshot."""
        with pytest.raises(CatalogNotFoundError) as exc_info:
            temp_storage.load("missing:1.0.0:abcd12345678")

        assert exc_info.value.snapshot_id == "missing:1.0.0:abcd12345678"
        assert "missing/1.0.0" in str(exc_info.value)

    def test_load_invalid_format_raises(self, temp_storage: CatalogSnapshotStore) -> None:
        """Load raises CatalogNotFoundError for invalid snapshot ID format."""
        # Missing parts
        with pytest.raises(CatalogNotFoundError) as exc_info:
            temp_storage.load("only_one_part")
        assert "Invalid format" in exc_info.value.reason

        # Too many parts
        with pytest.raises(CatalogNotFoundError):
            temp_storage.load("a:b:c:d")

    def test_load_checksum_mismatch_raises(
        self,
        temp_storage: CatalogSnapshotStore,
        installed_catalog: tuple[str, dict[str, Any]],
    ) -> None:
        """Load raises error when checksum prefix doesn't match."""
        snapshot_id, _ = installed_catalog
        parts = snapshot_id.split(":")
        wrong_id = f"{parts[0]}:{parts[1]}:wrongchecksum"

        with pytest.raises(CatalogNotFoundError) as exc_info:
            temp_storage.load(wrong_id)

        assert "Checksum mismatch" in exc_info.value.reason

    def test_load_offline_operation(
        self,
        temp_storage: CatalogSnapshotStore,
        installed_catalog: tuple[str, dict[str, Any]],
    ) -> None:
        """Load works without network access (offline operation)."""
        snapshot_id, original_data = installed_catalog

        # Create new store instance (simulates offline use)
        offline_store = CatalogSnapshotStore(temp_storage.storage_root)

        # Should load successfully without any network calls
        catalog = offline_store.load(snapshot_id)
        assert catalog.data == original_data


# =============================================================================
# List Installed Tests
# =============================================================================


class TestListInstalled:
    """Tests for list_installed() method."""

    def test_list_empty_store(self, temp_storage: CatalogSnapshotStore) -> None:
        """Returns empty list for empty store."""
        result = temp_storage.list_installed()
        assert result == []

    def test_list_single_catalog(
        self,
        temp_storage: CatalogSnapshotStore,
        installed_catalog: tuple[str, dict[str, Any]],
    ) -> None:
        """Returns single installed catalog."""
        snapshot_id, _ = installed_catalog
        result = temp_storage.list_installed()

        assert len(result) == 1
        assert snapshot_id in result

    def test_list_multiple_catalogs(self, temp_storage: CatalogSnapshotStore) -> None:
        """Returns all installed catalogs."""
        # Install multiple catalogs
        ids = []
        for i in range(3):
            snapshot_id = temp_storage.install_from_data(
                name=f"catalog_{i}",
                version="1.0.0",
                data=create_sample_catalog_data(i + 1),
            )
            ids.append(snapshot_id)

        result = temp_storage.list_installed()

        assert len(result) == 3
        for snapshot_id in ids:
            assert snapshot_id in result

    def test_list_multiple_versions(self, temp_storage: CatalogSnapshotStore) -> None:
        """Returns all versions of the same catalog."""
        ids = []
        for version in ["1.0.0", "1.1.0", "2.0.0"]:
            snapshot_id = temp_storage.install_from_data(
                name="versioned_catalog",
                version=version,
                data=create_sample_catalog_data(),
            )
            ids.append(snapshot_id)

        result = temp_storage.list_installed()

        assert len(result) == 3
        for snapshot_id in ids:
            assert snapshot_id in result

    def test_list_sorted_output(self, temp_storage: CatalogSnapshotStore) -> None:
        """Returns snapshot IDs in sorted order."""
        names = ["zebra", "alpha", "middle"]
        for name in names:
            temp_storage.install_from_data(
                name=name,
                version="1.0.0",
                data={},
            )

        result = temp_storage.list_installed()

        # Should be sorted
        assert result[0].startswith("alpha:")
        assert result[1].startswith("middle:")
        assert result[2].startswith("zebra:")


# =============================================================================
# Verify Checksum Tests
# =============================================================================


class TestVerifyChecksum:
    """Tests for verify_checksum() method."""

    def test_verify_valid_checksum(
        self,
        temp_storage: CatalogSnapshotStore,
        installed_catalog: tuple[str, dict[str, Any]],
    ) -> None:
        """Verify returns True for valid checksum."""
        snapshot_id, _ = installed_catalog
        assert temp_storage.verify_checksum(snapshot_id) is True

    def test_verify_corrupted_data(
        self,
        temp_storage: CatalogSnapshotStore,
        installed_catalog: tuple[str, dict[str, Any]],
    ) -> None:
        """Verify returns False for corrupted data."""
        snapshot_id, _ = installed_catalog

        # Corrupt the data file
        data_path = temp_storage._catalogs_dir / "test_catalog" / "1.0.0" / "data.json"
        data_path.write_text('{"corrupted": true}')

        assert temp_storage.verify_checksum(snapshot_id) is False

    def test_verify_nonexistent_raises(self, temp_storage: CatalogSnapshotStore) -> None:
        """Verify raises CatalogNotFoundError for nonexistent snapshot."""
        with pytest.raises(CatalogNotFoundError):
            temp_storage.verify_checksum("missing:1.0.0:abcd12345678")

    def test_verify_after_reinstall(self, temp_storage: CatalogSnapshotStore) -> None:
        """Verify works after reinstalling same catalog."""
        data = create_sample_catalog_data()

        snapshot_id1 = temp_storage.install_from_data(
            name="test",
            version="1.0",
            data=data,
        )
        assert temp_storage.verify_checksum(snapshot_id1)

        # Reinstall (should overwrite)
        snapshot_id2 = temp_storage.install_from_data(
            name="test",
            version="1.0",
            data=data,
        )

        # Both should be the same ID
        assert snapshot_id1 == snapshot_id2
        assert temp_storage.verify_checksum(snapshot_id2)


# =============================================================================
# Exists Tests
# =============================================================================


class TestExists:
    """Tests for exists() method."""

    def test_exists_returns_true_after_install(
        self,
        temp_storage: CatalogSnapshotStore,
        installed_catalog: tuple[str, dict[str, Any]],
    ) -> None:
        """Exists returns True for installed catalog."""
        snapshot_id, _ = installed_catalog
        assert temp_storage.exists(snapshot_id) is True

    def test_exists_returns_false_for_nonexistent(self, temp_storage: CatalogSnapshotStore) -> None:
        """Exists returns False for nonexistent catalog."""
        assert temp_storage.exists("missing:1.0.0:abcd12345678") is False

    def test_exists_returns_false_for_invalid_format(
        self, temp_storage: CatalogSnapshotStore
    ) -> None:
        """Exists returns False for invalid format."""
        assert temp_storage.exists("invalid") is False
        assert temp_storage.exists("") is False
        assert temp_storage.exists("a:b:c:d") is False


# =============================================================================
# Delete Tests
# =============================================================================


class TestDelete:
    """Tests for delete() method."""

    def test_delete_removes_catalog(
        self,
        temp_storage: CatalogSnapshotStore,
        installed_catalog: tuple[str, dict[str, Any]],
    ) -> None:
        """Delete removes catalog from storage."""
        snapshot_id, _ = installed_catalog

        assert temp_storage.exists(snapshot_id)
        result = temp_storage.delete(snapshot_id)

        assert result is True
        assert not temp_storage.exists(snapshot_id)

    def test_delete_nonexistent_returns_false(self, temp_storage: CatalogSnapshotStore) -> None:
        """Delete returns False for nonexistent catalog."""
        result = temp_storage.delete("missing:1.0.0:abcd12345678")
        assert result is False

    def test_delete_removes_files(
        self,
        temp_storage: CatalogSnapshotStore,
        installed_catalog: tuple[str, dict[str, Any]],
    ) -> None:
        """Delete removes all catalog files."""
        snapshot_id, _ = installed_catalog
        catalog_dir = temp_storage._catalogs_dir / "test_catalog" / "1.0.0"

        assert catalog_dir.exists()
        temp_storage.delete(snapshot_id)

        assert not catalog_dir.exists()

    def test_delete_preserves_other_versions(self, temp_storage: CatalogSnapshotStore) -> None:
        """Delete only removes the specified version."""
        id1 = temp_storage.install_from_data("test", "1.0", data={})
        id2 = temp_storage.install_from_data("test", "2.0", data={"v": 2})

        temp_storage.delete(id1)

        assert not temp_storage.exists(id1)
        assert temp_storage.exists(id2)


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidation:
    """Tests for input validation."""

    def test_invalid_name_empty(self, temp_storage: CatalogSnapshotStore) -> None:
        """Empty name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            temp_storage.install_from_data(name="", version="1.0", data={})
        assert "empty" in str(exc_info.value).lower()

    def test_invalid_name_special_chars(self, temp_storage: CatalogSnapshotStore) -> None:
        """Special characters in name raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            temp_storage.install_from_data(name="bad/name", version="1.0", data={})
        assert "Invalid catalog name" in str(exc_info.value)

    def test_invalid_name_starts_with_dot(self, temp_storage: CatalogSnapshotStore) -> None:
        """Name starting with dot raises ValueError."""
        with pytest.raises(ValueError):
            temp_storage.install_from_data(name=".hidden", version="1.0", data={})

    def test_invalid_version_empty(self, temp_storage: CatalogSnapshotStore) -> None:
        """Empty version raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            temp_storage.install_from_data(name="test", version="", data={})
        assert "empty" in str(exc_info.value).lower()

    def test_invalid_version_special_chars(self, temp_storage: CatalogSnapshotStore) -> None:
        """Special characters in version raise ValueError."""
        with pytest.raises(ValueError):
            temp_storage.install_from_data(name="test", version="1.0/bad", data={})

    def test_valid_name_with_underscore_hyphen(self, temp_storage: CatalogSnapshotStore) -> None:
        """Names with underscores and hyphens are valid."""
        snapshot_id = temp_storage.install_from_data(
            name="my_catalog-v2",
            version="1.0.0",
            data={},
        )
        assert snapshot_id.startswith("my_catalog-v2:")

    def test_valid_version_with_dots(self, temp_storage: CatalogSnapshotStore) -> None:
        """Versions with dots are valid."""
        snapshot_id = temp_storage.install_from_data(
            name="test",
            version="1.2.3-alpha.1",
            data={},
        )
        assert ":1.2.3-alpha.1:" in snapshot_id


# =============================================================================
# CatalogData Tests
# =============================================================================


class TestCatalogData:
    """Tests for CatalogData dataclass."""

    def test_catalog_data_is_frozen(self) -> None:
        """CatalogData is immutable (frozen)."""
        catalog = CatalogData(
            name="test",
            version="1.0",
            snapshot_id="test:1.0:abc123",
            data={"key": "value"},
            checksum="a" * 64,
            source_url="https://example.com",
            installed_at="2024-01-01T00:00:00Z",
            entry_count=0,
        )

        with pytest.raises(AttributeError):
            catalog.name = "modified"  # type: ignore

    def test_catalog_data_required_fields(self) -> None:
        """CatalogData validates required fields."""
        with pytest.raises(ValueError):
            CatalogData(
                name="",
                version="1.0",
                snapshot_id="test:1.0:abc123",
                data={},
                checksum="a" * 64,
                source_url="",
                installed_at="",
            )

    def test_get_entries_with_entries(self) -> None:
        """get_entries() returns entries when present."""
        entries = [{"id": 1}, {"id": 2}]
        catalog = CatalogData(
            name="test",
            version="1.0",
            snapshot_id="test:1.0:abc123",
            data={"entries": entries},
            checksum="a" * 64,
            source_url="",
            installed_at="",
            entry_count=2,
        )

        assert catalog.get_entries() == entries

    def test_get_entries_without_entries(self) -> None:
        """get_entries() returns empty list when no entries."""
        catalog = CatalogData(
            name="test",
            version="1.0",
            snapshot_id="test:1.0:abc123",
            data={"other": "data"},
            checksum="a" * 64,
            source_url="",
            installed_at="",
        )

        assert catalog.get_entries() == []


# =============================================================================
# Exception Tests
# =============================================================================


class TestExceptions:
    """Tests for exception classes."""

    def test_catalog_not_found_error_attributes(self) -> None:
        """CatalogNotFoundError has correct attributes."""
        error = CatalogNotFoundError("test:1.0:abc123", "reason message")
        assert error.snapshot_id == "test:1.0:abc123"
        assert error.reason == "reason message"
        assert "test:1.0:abc123" in str(error)
        assert "reason message" in str(error)

    def test_catalog_not_found_error_no_reason(self) -> None:
        """CatalogNotFoundError works without reason."""
        error = CatalogNotFoundError("test:1.0:abc123")
        assert error.reason is None
        assert "test:1.0:abc123" in str(error)

    def test_catalog_checksum_error_attributes(self) -> None:
        """CatalogChecksumError has correct attributes."""
        error = CatalogChecksumError(
            "test:1.0:abc123",
            expected="expected_hash",
            actual="actual_hash",
        )
        assert error.snapshot_id == "test:1.0:abc123"
        assert error.expected == "expected_hash"
        assert error.actual == "actual_hash"
        assert "test:1.0:abc123" in str(error)

    def test_catalog_install_error_attributes(self) -> None:
        """CatalogInstallError has correct attributes."""
        error = CatalogInstallError(
            name="test",
            version="1.0",
            source_url="https://example.com",
            reason="Connection refused",
        )
        assert error.name == "test"
        assert error.version == "1.0"
        assert error.source_url == "https://example.com"
        assert error.reason == "Connection refused"
        assert "test@1.0" in str(error)
        assert "example.com" in str(error)


# =============================================================================
# Filesystem Structure Tests
# =============================================================================


class TestFilesystemStructure:
    """Tests for filesystem storage structure."""

    def test_catalog_directory_structure(self, temp_storage: CatalogSnapshotStore) -> None:
        """Catalogs are stored in name/version directory structure."""
        temp_storage.install_from_data(
            name="my_catalog",
            version="2.1.0",
            data={"test": True},
        )

        expected_dir = temp_storage._catalogs_dir / "my_catalog" / "2.1.0"
        assert expected_dir.exists()
        assert (expected_dir / "data.json").exists()
        assert (expected_dir / "metadata.json").exists()

    def test_data_file_is_valid_utf8(
        self,
        temp_storage: CatalogSnapshotStore,
        installed_catalog: tuple[str, dict[str, Any]],
    ) -> None:
        """Data file is valid UTF-8 encoded JSON."""
        snapshot_id, _ = installed_catalog

        data_path = temp_storage._catalogs_dir / "test_catalog" / "1.0.0" / "data.json"
        content = data_path.read_text(encoding="utf-8")
        parsed = json.loads(content)

        assert "catalog_name" in parsed

    def test_metadata_file_is_valid_utf8(
        self,
        temp_storage: CatalogSnapshotStore,
        installed_catalog: tuple[str, dict[str, Any]],
    ) -> None:
        """Metadata file is valid UTF-8 encoded JSON."""
        snapshot_id, _ = installed_catalog

        metadata_path = temp_storage._catalogs_dir / "test_catalog" / "1.0.0" / "metadata.json"
        content = metadata_path.read_text(encoding="utf-8")
        parsed = json.loads(content)

        assert parsed["name"] == "test_catalog"
        assert parsed["version"] == "1.0.0"
        assert "installed_at" in parsed


# =============================================================================
# Round-trip Tests
# =============================================================================


class TestRoundTrip:
    """Tests for install/load round-trip behavior."""

    def test_simple_roundtrip(self, temp_storage: CatalogSnapshotStore) -> None:
        """Simple data survives install/load round-trip."""
        original_data = {"key": "value", "number": 42}

        snapshot_id = temp_storage.install_from_data(
            name="roundtrip",
            version="1.0",
            data=original_data,
        )

        catalog = temp_storage.load(snapshot_id)
        assert catalog.data == original_data

    def test_complex_roundtrip(self, temp_storage: CatalogSnapshotStore) -> None:
        """Complex nested data survives round-trip."""
        original_data = {
            "metadata": {
                "name": "Test Catalog",
                "created": "2024-01-01",
            },
            "entries": [
                {"id": 1, "values": [1.1, 2.2, 3.3]},
                {"id": 2, "values": [4.4, 5.5, 6.6]},
            ],
            "config": {
                "nested": {
                    "deep": {
                        "value": True,
                    },
                },
            },
        }

        snapshot_id = temp_storage.install_from_data(
            name="complex",
            version="1.0",
            data=original_data,
        )

        catalog = temp_storage.load(snapshot_id)
        assert catalog.data == original_data

    def test_unicode_roundtrip(self, temp_storage: CatalogSnapshotStore) -> None:
        """Unicode data survives round-trip."""
        original_data = {
            "name": "Catalog with Unicode",
            "description": "Contains special characters",
            "entries": [{"name": "Alpha Centauri"}],
        }

        snapshot_id = temp_storage.install_from_data(
            name="unicode",
            version="1.0",
            data=original_data,
        )

        catalog = temp_storage.load(snapshot_id)
        assert catalog.data == original_data


# =============================================================================
# Import Tests
# =============================================================================


class TestModuleImports:
    """Tests for module imports and exports."""

    def test_import_from_catalogs_package(self) -> None:
        """Can import from tess_vetter.platform.catalogs package."""
        from tess_vetter.platform.catalogs import (
            CatalogChecksumError,
            CatalogData,
            CatalogInstallError,
            CatalogNotFoundError,
            CatalogSnapshotStore,
        )

        assert CatalogSnapshotStore is not None
        assert CatalogData is not None
        assert CatalogNotFoundError is not None
        assert CatalogChecksumError is not None
        assert CatalogInstallError is not None

    def test_import_from_store_module(self) -> None:
        """Can import from tess_vetter.platform.catalogs.store module."""
        from tess_vetter.platform.catalogs.store import (
            CatalogChecksumError,
            CatalogData,
            CatalogInstallError,
            CatalogNotFoundError,
            CatalogSnapshotStore,
        )

        assert CatalogSnapshotStore is not None
        assert CatalogData is not None
        assert CatalogNotFoundError is not None
        assert CatalogChecksumError is not None
        assert CatalogInstallError is not None
