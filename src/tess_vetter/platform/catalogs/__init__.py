"""Catalog utilities for astronomical data processing (platform-facing).

Canonical home for networked catalog lookups and on-disk catalog snapshot
storage.
"""

from __future__ import annotations

from tess_vetter.platform.catalogs.crossmatch import CatalogData as CrossmatchCatalogData
from tess_vetter.platform.catalogs.crossmatch import (
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
from tess_vetter.platform.catalogs.crossmatch import (
    CatalogSnapshotStore as CrossmatchCatalogSnapshotStore,
)
from tess_vetter.platform.catalogs.crossmatch import (
    validate_snapshot_id as validate_crossmatch_snapshot_id,
)
from tess_vetter.platform.catalogs.exoplanet_archive import (
    ExoplanetArchiveClient,
    ExoplanetArchiveError,
    KnownPlanet,
    KnownPlanetMatchResult,
    KnownPlanetsResult,
    TAPQueryError,
    get_known_planets,
    match_known_planet_ephemeris,
)
from tess_vetter.platform.catalogs.exoplanet_archive import (
    get_client as get_exoplanet_archive_client,
)
from tess_vetter.platform.catalogs.gaia_client import (
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
from tess_vetter.platform.catalogs.simbad_client import (
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
from tess_vetter.platform.catalogs.snapshot_id import (
    SnapshotComponents,
    generate_snapshot_id,
    parse_snapshot_id,
    validate_snapshot_id,
)
from tess_vetter.platform.catalogs.spatial import SpatialIndex
from tess_vetter.platform.catalogs.store import (
    CatalogChecksumError,
    CatalogData,
    CatalogInstallError,
    CatalogNotFoundError,
    CatalogSnapshotStore,
)
from tess_vetter.platform.catalogs.toi_resolution import (
    LookupStatus,
    TICCoordinateLookupResult,
    ToiResolutionResult,
    lookup_tic_coordinates,
    resolve_toi_to_tic_ephemeris_depth,
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
    # TOI/TIC resolution helpers
    "LookupStatus",
    "ToiResolutionResult",
    "TICCoordinateLookupResult",
    "resolve_toi_to_tic_ephemeris_depth",
    "lookup_tic_coordinates",
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
    "KnownPlanetMatchResult",
    "KnownPlanetsResult",
    "TAPQueryError",
    "get_exoplanet_archive_client",
    "get_known_planets",
    "match_known_planet_ephemeris",
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
