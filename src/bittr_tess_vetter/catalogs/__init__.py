"""Catalog utilities for astronomical data processing.

Provides:
- Snapshot ID generation, parsing, and validation for catalog versioning
- Spatial indexing utilities for astronomical catalogs
- Disk-backed storage for versioned, checksummed catalog snapshots
- Crossmatch tool for matching positions against known object catalogs
- NASA Exoplanet Archive integration for querying known planets
- Gaia DR3 TAP client for stellar characterization

Usage:
    >>> from bittr_tess_vetter.catalogs import CatalogSnapshotStore, CatalogData
    >>> store = CatalogSnapshotStore("/path/to/storage")
    >>> snapshot_id = store.install("tic", "v8.2", "https://example.com/tic.json")
    >>> catalog = store.load(snapshot_id)
    >>> store.verify_checksum(snapshot_id)
    True

Crossmatch example:
    >>> from bittr_tess_vetter.catalogs import crossmatch
    >>> report = crossmatch(
    ...     ra=120.5, dec=-45.3,
    ...     snapshot_ids=["catalog:toi:v1.0:20240115:abc12345"],
    ... )
    >>> print(report.novelty_status)

Exoplanet Archive example:
    >>> from bittr_tess_vetter.catalogs import get_known_planets
    >>> result = get_known_planets(tic_id=150428135)
    >>> print(f"Found {result.n_planets} planets for {result.toi_id}")
"""

from bittr_tess_vetter.catalogs.crossmatch import CatalogData as CrossmatchCatalogData
from bittr_tess_vetter.catalogs.crossmatch import (
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
)
from bittr_tess_vetter.catalogs.crossmatch import (
    CatalogSnapshotStore as CrossmatchCatalogSnapshotStore,
)
from bittr_tess_vetter.catalogs.crossmatch import (
    validate_snapshot_id as validate_crossmatch_snapshot_id,
)
from bittr_tess_vetter.catalogs.exoplanet_archive import (
    ExoplanetArchiveClient,
    ExoplanetArchiveError,
    KnownPlanet,
    KnownPlanetsResult,
    TAPQueryError,
    get_known_planets,
)
from bittr_tess_vetter.catalogs.exoplanet_archive import (
    get_client as get_exoplanet_archive_client,
)
from bittr_tess_vetter.catalogs.gaia_client import (
    GAIA_TAP_ENDPOINT,
    RUWE_ELEVATED_THRESHOLD,
    GaiaAstrophysicalParams,
    GaiaClient,
    GaiaNeighbor,
    GaiaQueryError,
    GaiaQueryResult,
    GaiaSourceRecord,
    GaiaTAPError,
    query_gaia_by_id_sync,
    query_gaia_by_position_sync,
)
from bittr_tess_vetter.catalogs.simbad_client import (
    SIMBAD_TAP_ENDPOINT,
    SimbadClient,
    SimbadIdentifiers,
    SimbadObjectType,
    SimbadQueryError,
    SimbadQueryResult,
    SimbadSpectralInfo,
    SimbadTAPError,
    classify_object_type,
    parse_spectral_type,
    query_simbad_by_id_sync,
    query_simbad_by_position_sync,
)
from bittr_tess_vetter.catalogs.snapshot_id import (
    SnapshotComponents,
    generate_snapshot_id,
    parse_snapshot_id,
    validate_snapshot_id,
)
from bittr_tess_vetter.catalogs.spatial import SpatialIndex
from bittr_tess_vetter.catalogs.store import (
    CatalogChecksumError,
    CatalogData,
    CatalogInstallError,
    CatalogNotFoundError,
    CatalogSnapshotStore,
)

__all__ = [
    # Snapshot ID utilities
    "SnapshotComponents",
    "generate_snapshot_id",
    "parse_snapshot_id",
    "validate_snapshot_id",
    # Spatial indexing
    "SpatialIndex",
    # Catalog storage
    "CatalogChecksumError",
    "CatalogData",
    "CatalogInstallError",
    "CatalogNotFoundError",
    "CatalogSnapshotStore",
    # Crossmatch
    "ContaminationRisk",
    "CrossmatchError",
    "CrossmatchReport",
    "InvalidSnapshotIdError",
    "KnownObjectMatch",
    "NoCatalogsProvidedError",
    "SnapshotNotFoundError",
    "angular_separation_arcsec",
    "assess_contamination",
    "compute_dilution_factor",
    "crossmatch",
    "determine_novelty_status",
    "find_known_object_matches",
    "CatalogEntry",
    "CrossmatchCatalogData",
    "CrossmatchCatalogSnapshotStore",
    "validate_crossmatch_snapshot_id",
    # Exoplanet Archive
    "ExoplanetArchiveClient",
    "ExoplanetArchiveError",
    "KnownPlanet",
    "KnownPlanetsResult",
    "TAPQueryError",
    "get_exoplanet_archive_client",
    "get_known_planets",
    # Gaia DR3
    "GAIA_TAP_ENDPOINT",
    "RUWE_ELEVATED_THRESHOLD",
    "GaiaAstrophysicalParams",
    "GaiaClient",
    "GaiaNeighbor",
    "GaiaQueryError",
    "GaiaQueryResult",
    "GaiaSourceRecord",
    "GaiaTAPError",
    "query_gaia_by_id_sync",
    "query_gaia_by_position_sync",
    # SIMBAD
    "SIMBAD_TAP_ENDPOINT",
    "SimbadClient",
    "SimbadIdentifiers",
    "SimbadObjectType",
    "SimbadQueryError",
    "SimbadQueryResult",
    "SimbadSpectralInfo",
    "SimbadTAPError",
    "classify_object_type",
    "parse_spectral_type",
    "query_simbad_by_id_sync",
    "query_simbad_by_position_sync",
]
