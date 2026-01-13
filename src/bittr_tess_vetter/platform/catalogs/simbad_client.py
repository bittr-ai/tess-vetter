"""SIMBAD TAP service client for identifier resolution and object classification.

This module provides functionality to query SIMBAD to retrieve object identifiers,
object types (star, binary, variable), and spectral classifications. It supports
querying by any identifier (TIC, HD, HIP, Gaia DR3, etc.) or by position.

Usage:
    >>> from astro_arc.catalogs.simbad_client import SimbadClient
    >>> client = SimbadClient()
    >>> result = await client.query_by_id("TIC 261136679")  # Pi Mensae
    >>> print(f"Main ID: {result.identifiers.main_id}")
    >>> print(f"Spectral type: {result.spectral.spectral_type}")
    >>> print(f"Is binary: {result.object_type.is_binary}")

Technical Notes:
    - SIMBAD TAP endpoint: https://simbad.u-strasbg.fr/simbad/sim-tap
    - Queries basic table for object types and coordinates
    - Queries ident table for cross-match identifiers
    - Queries mesSpT table for spectral types
    - Early-type classification: O, B, A stars (Teff > ~7500 K)
    - Giant classification: luminosity class III, II, or I
"""

from __future__ import annotations

import logging
import re
import time
from datetime import UTC, datetime
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from bittr_tess_vetter.platform.catalogs.models import SourceRecord

logger = logging.getLogger(__name__)

# SIMBAD TAP endpoint
SIMBAD_TAP_ENDPOINT = "https://simbad.u-strasbg.fr/simbad/sim-tap"

# Spectral type patterns for classification
EARLY_TYPE_PATTERN = re.compile(r"^[OBA]", re.IGNORECASE)
GIANT_CLASS_PATTERN = re.compile(r"(III|II|Ib|Ia|I)($|[^IV])", re.IGNORECASE)
SPECTRAL_TYPE_PATTERN = re.compile(r"^([OBAFGKM])(\d)?\.?(\d)?")
LUMINOSITY_CLASS_PATTERN = re.compile(r"(Ia|Ib|II|III|IV|V)($|[^IVa-z])", re.IGNORECASE)

# Object type patterns for classification
BINARY_TYPE_PATTERNS = frozenset(
    {
        "EB*",  # Eclipsing binary
        "SB*",  # Spectroscopic binary
        "**",  # Double/multiple star
        "V*EB",  # Variable star, eclipsing binary
        "Sy*",  # Symbiotic star
        "Al*",  # Algol-type eclipsing binary
        "bL*",  # Beta Lyrae-type eclipsing binary
        "WU*",  # W UMa-type eclipsing binary
    }
)

VARIABLE_TYPE_PATTERNS = frozenset(
    {
        "V*",  # Variable star
        "Pu*",  # Pulsating variable
        "LP*",  # Long-period variable
        "Ir*",  # Irregular variable
        "Er*",  # Eruptive variable
        "Ce*",  # Cepheid
        "RR*",  # RR Lyrae
        "dS*",  # Delta Scuti
        "gD*",  # Gamma Doradus
        "Ro*",  # Rotationally variable
        "BY*",  # BY Dra variable
        "a2*",  # alpha2 CVn variable
    }
)

STAR_TYPE_PATTERNS = frozenset(
    {
        "*",  # Star
        "**",  # Double star
        "PM*",  # High proper motion star
        "HB*",  # Horizontal branch star
        "Be*",  # Be star
        "WR*",  # Wolf-Rayet star
        "s*r",  # S-type star
        "s*y",  # Young stellar object
        "C*",  # Carbon star
    }
)


class SimbadQueryError(Exception):
    """Base exception for SIMBAD query errors."""

    pass


class SimbadTAPError(SimbadQueryError):
    """Error during TAP query execution."""

    pass


class SimbadIdentifiers(BaseModel):
    """Cross-match identifiers from SIMBAD.

    Captures the main identifier and common catalog cross-matches.

    Attributes:
        main_id: SIMBAD main identifier (e.g., "* pi Men")
        hd: Henry Draper catalog ID (e.g., "HD 39091")
        hip: Hipparcos catalog ID (e.g., "HIP 26394")
        gaia_dr3: Gaia DR3 source ID string (e.g., "Gaia DR3 5290699119420871680")
        tic: TESS Input Catalog ID (e.g., "TIC 261136679")
        all_ids: Complete list of all identifiers
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    main_id: str = Field(description="SIMBAD main identifier")
    hd: str | None = Field(default=None, description="Henry Draper catalog ID")
    hip: str | None = Field(default=None, description="Hipparcos catalog ID")
    gaia_dr3: str | None = Field(default=None, description="Gaia DR3 source ID")
    tic: str | None = Field(default=None, description="TESS Input Catalog ID")
    all_ids: list[str] = Field(default_factory=list, description="All identifiers")


class SimbadObjectType(BaseModel):
    """SIMBAD object classification.

    Captures the object type from SIMBAD with derived boolean flags
    for common classification needs.

    Attributes:
        main_type: Primary object type (e.g., "Star", "EB*", "PM*")
        other_types: Additional object types
        is_star: Whether the object is classified as a star
        is_binary: Whether the object shows signs of binarity
        is_variable: Whether the object is a variable star
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    main_type: str = Field(description="Primary object type code")
    other_types: list[str] = Field(default_factory=list, description="Additional type codes")
    is_star: bool = Field(description="Whether object is classified as a star")
    is_binary: bool = Field(description="Whether object shows binarity")
    is_variable: bool = Field(description="Whether object is variable")


class SimbadSpectralInfo(BaseModel):
    """Spectral type and luminosity class from SIMBAD.

    Parses and classifies spectral types to identify early-type stars
    (O, B, A) and giants (luminosity class III, II, I).

    Attributes:
        spectral_type: Full spectral type string (e.g., "G0V", "A5III")
        luminosity_class: Extracted luminosity class (e.g., "V", "III")
        is_early_type: True for O, B, A type stars
        is_giant: True for luminosity class III, II, or I
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    spectral_type: str | None = Field(default=None, description="Full spectral type string")
    luminosity_class: str | None = Field(default=None, description="Luminosity class")
    is_early_type: bool = Field(description="True for O, B, A type stars")
    is_giant: bool = Field(description="True for luminosity class III, II, I")


class SimbadQueryResult(BaseModel):
    """Complete SIMBAD query result.

    Combines identifiers, object classification, spectral information,
    coordinates, and provenance tracking.

    Attributes:
        identifiers: Cross-match identifiers
        object_type: Object classification
        spectral: Spectral type information (may be None if unavailable)
        ra: Right ascension in degrees (ICRS)
        dec: Declination in degrees (ICRS)
        source_record: Provenance record for the query
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    identifiers: SimbadIdentifiers = Field(description="Cross-match identifiers")
    object_type: SimbadObjectType = Field(description="Object classification")
    spectral: SimbadSpectralInfo | None = Field(
        default=None, description="Spectral type information"
    )
    ra: float | None = Field(default=None, description="Right ascension (degrees, ICRS)")
    dec: float | None = Field(default=None, description="Declination (degrees, ICRS)")
    source_record: SourceRecord = Field(description="Provenance record")


def parse_spectral_type(sptype: str | None) -> SimbadSpectralInfo:
    """Parse a spectral type string into structured information.

    Args:
        sptype: Spectral type string (e.g., "G0V", "A5III", "K2/3V")

    Returns:
        SimbadSpectralInfo with parsed components
    """
    if not sptype or not sptype.strip():
        return SimbadSpectralInfo(
            spectral_type=None,
            luminosity_class=None,
            is_early_type=False,
            is_giant=False,
        )

    sptype = sptype.strip()

    # Check for early type (O, B, A)
    is_early_type = bool(EARLY_TYPE_PATTERN.match(sptype))

    # Extract luminosity class
    luminosity_class: str | None = None
    lum_match = LUMINOSITY_CLASS_PATTERN.search(sptype)
    if lum_match:
        luminosity_class = lum_match.group(1)

    # Check for giant (III, II, I but not IV or V)
    is_giant = bool(GIANT_CLASS_PATTERN.search(sptype))

    return SimbadSpectralInfo(
        spectral_type=sptype,
        luminosity_class=luminosity_class,
        is_early_type=is_early_type,
        is_giant=is_giant,
    )


def classify_object_type(otype: str | None, all_types: list[str] | None = None) -> SimbadObjectType:
    """Classify an object type into structured flags.

    Args:
        otype: Primary object type code
        all_types: All associated type codes

    Returns:
        SimbadObjectType with classification flags
    """
    if not otype:
        otype = "Unknown"

    other_types = all_types or []

    # Check all types for classification
    all_type_set = {otype} | set(other_types)

    # Is it a star?
    is_star = any(
        t in STAR_TYPE_PATTERNS or t.startswith("*") or "Star" in t or t.endswith("*")
        for t in all_type_set
    )

    # Is it a binary?
    is_binary = any(t in BINARY_TYPE_PATTERNS for t in all_type_set)

    # Is it variable?
    is_variable = any(t in VARIABLE_TYPE_PATTERNS or t.startswith("V*") for t in all_type_set)

    return SimbadObjectType(
        main_type=otype,
        other_types=list(other_types),
        is_star=is_star,
        is_binary=is_binary,
        is_variable=is_variable,
    )


def _normalize_whitespace(s: str) -> str:
    """Normalize whitespace in an identifier string.

    Collapses multiple spaces to single space and strips.

    Args:
        s: Input string

    Returns:
        Normalized string
    """
    return " ".join(s.split())


def _parse_identifiers(all_ids: list[str], main_id: str) -> SimbadIdentifiers:
    """Parse a list of identifiers into structured form.

    Args:
        all_ids: List of all identifiers
        main_id: SIMBAD main identifier

    Returns:
        SimbadIdentifiers with extracted catalog IDs
    """
    hd: str | None = None
    hip: str | None = None
    gaia_dr3: str | None = None
    tic: str | None = None

    for id_str in all_ids:
        id_upper = id_str.upper().strip()

        if id_upper.startswith("HD "):
            # Normalize "HD  39091" -> "HD 39091"
            hd = _normalize_whitespace(id_str)
        elif id_upper.startswith("HIP "):
            hip = _normalize_whitespace(id_str)
        elif "GAIA DR3" in id_upper or id_upper.startswith("GAIA DR3"):
            gaia_dr3 = _normalize_whitespace(id_str)
        elif id_upper.startswith("TIC "):
            tic = _normalize_whitespace(id_str)

    return SimbadIdentifiers(
        main_id=_normalize_whitespace(main_id),
        hd=hd,
        hip=hip,
        gaia_dr3=gaia_dr3,
        tic=tic,
        all_ids=[_normalize_whitespace(i) for i in all_ids],
    )


class SimbadClient:
    """Client for SIMBAD TAP queries.

    Provides methods to query SIMBAD for object identifiers, classifications,
    and spectral types. Supports querying by any identifier or by position.

    Attributes:
        tap_url: SIMBAD TAP endpoint URL
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        tap_url: str = SIMBAD_TAP_ENDPOINT,
        timeout: int = 60,
    ) -> None:
        """Initialize the SIMBAD client.

        Args:
            tap_url: SIMBAD TAP endpoint URL
            timeout: Request timeout in seconds
        """
        self.tap_url = tap_url
        self.timeout = timeout

    def _execute_tap_query(self, query: str) -> list[dict[str, Any]]:
        """Execute a synchronous TAP query.

        Args:
            query: ADQL query string

        Returns:
            List of result rows as dictionaries

        Raises:
            SimbadTAPError: If the query fails
        """
        import requests

        params = {
            "request": "doQuery",
            "lang": "adql",
            "format": "json",
            "query": query,
        }

        max_retries = 3
        last_exception: Exception | None = None

        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"{self.tap_url}/sync",
                    params=params,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()

                # SIMBAD TAP returns data in VOTable JSON format
                if isinstance(data, dict):
                    if "data" in data:
                        columns = data.get("metadata", [])
                        col_names = [c.get("name") for c in columns]
                        rows = []
                        for row_data in data["data"]:
                            rows.append(dict(zip(col_names, row_data, strict=False)))
                        return rows
                    return [data] if data else []
                elif isinstance(data, list):
                    return data
                return []

            except requests.exceptions.Timeout as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"SIMBAD TAP timeout, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                raise SimbadTAPError(
                    f"SIMBAD TAP query timed out after {max_retries} attempts"
                ) from e

            except requests.exceptions.RequestException as e:
                last_exception = e
                status_code = (
                    getattr(e.response, "status_code", None) if hasattr(e, "response") else None
                )
                if status_code is not None and 400 <= status_code < 500:
                    raise SimbadTAPError(f"SIMBAD TAP query failed: {e}") from e

                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"SIMBAD TAP query failed, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    time.sleep(wait_time)
                    continue
                raise SimbadTAPError(
                    f"SIMBAD TAP query failed after {max_retries} attempts: {e}"
                ) from e

            except ValueError as e:
                raise SimbadTAPError(f"Failed to parse SIMBAD TAP response: {e}") from e

        raise SimbadTAPError(
            f"SIMBAD TAP query failed after {max_retries} attempts"
        ) from last_exception

    async def query_by_id(self, identifier: str) -> SimbadQueryResult | None:
        """Query SIMBAD by any identifier.

        Accepts TIC IDs, HD numbers, HIP numbers, Gaia DR3 IDs, star names, etc.

        Args:
            identifier: Any valid SIMBAD identifier

        Returns:
            SimbadQueryResult or None if not found
        """
        now = datetime.now(UTC)
        escaped_id = identifier.replace("'", "''")

        # Query basic data
        basic_query = f"""
        SELECT TOP 1
            main_id, ra, dec, otype
        FROM basic
        WHERE main_id = '{escaped_id}'
           OR oid IN (SELECT oidref FROM ident WHERE id = '{escaped_id}')
        """

        try:
            basic_rows = self._execute_tap_query(basic_query)
        except SimbadTAPError:
            return None

        if not basic_rows:
            return None

        row = basic_rows[0]
        main_id = row.get("main_id", identifier)
        ra = row.get("ra")
        dec = row.get("dec")
        otype = row.get("otype")

        # Query identifiers
        escaped_main = main_id.replace("'", "''")
        ident_query = f"""
        SELECT id
        FROM ident
        WHERE oidref IN (SELECT oid FROM basic WHERE main_id = '{escaped_main}')
        """

        try:
            ident_rows = self._execute_tap_query(ident_query)
            all_ids = [r.get("id", "") for r in ident_rows if r.get("id")]
        except SimbadTAPError:
            all_ids = [main_id]

        # Query spectral type
        sp_query = f"""
        SELECT sptype
        FROM mesSpT
        WHERE oidref IN (SELECT oid FROM basic WHERE main_id = '{escaped_main}')
        ORDER BY bibcode DESC
        """

        spectral: SimbadSpectralInfo | None = None
        try:
            sp_rows = self._execute_tap_query(sp_query)
            if sp_rows:
                sptype = sp_rows[0].get("sptype")
                spectral = parse_spectral_type(sptype)
        except SimbadTAPError:
            spectral = None

        identifiers = _parse_identifiers(all_ids, main_id)
        object_type = classify_object_type(otype)

        return SimbadQueryResult(
            identifiers=identifiers,
            object_type=object_type,
            spectral=spectral,
            ra=ra,
            dec=dec,
            source_record=SourceRecord(
                name="simbad",
                version="TAP",
                retrieved_at=now,
                query=f"identifier = {identifier}",
            ),
        )

    async def query_by_position(
        self,
        ra: float,
        dec: float,
        radius_arcsec: float = 5.0,
    ) -> list[SimbadQueryResult]:
        """Query SIMBAD by sky position.

        Performs a cone search and returns all matching objects.

        Args:
            ra: Right ascension in degrees
            dec: Declination in degrees
            radius_arcsec: Search radius in arcseconds

        Returns:
            List of SimbadQueryResult for objects in the cone
        """
        now = datetime.now(UTC)
        radius_deg = radius_arcsec / 3600.0

        cone_query = f"""
        SELECT main_id, ra, dec, otype
        FROM basic
        WHERE CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
        ) = 1
        ORDER BY DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra}, {dec})) ASC
        """

        try:
            rows = self._execute_tap_query(cone_query)
        except SimbadTAPError:
            return []

        results: list[SimbadQueryResult] = []
        for row in rows:
            main_id = row.get("main_id", "Unknown")
            obj_ra = row.get("ra")
            obj_dec = row.get("dec")
            otype = row.get("otype")

            identifiers = SimbadIdentifiers(main_id=main_id, all_ids=[main_id])
            object_type = classify_object_type(otype)

            results.append(
                SimbadQueryResult(
                    identifiers=identifiers,
                    object_type=object_type,
                    spectral=None,
                    ra=obj_ra,
                    dec=obj_dec,
                    source_record=SourceRecord(
                        name="simbad",
                        version="TAP",
                        retrieved_at=now,
                        query=f"cone({ra}, {dec}, {radius_arcsec} arcsec)",
                    ),
                )
            )

        return results


# Module-level synchronous wrappers


def query_simbad_by_id_sync(
    identifier: str,
    tap_url: str = SIMBAD_TAP_ENDPOINT,
    timeout: int = 60,
) -> SimbadQueryResult | None:
    """Synchronous wrapper to query SIMBAD by identifier.

    Args:
        identifier: Any valid SIMBAD identifier
        tap_url: SIMBAD TAP endpoint URL
        timeout: Request timeout in seconds

    Returns:
        SimbadQueryResult or None if not found
    """
    import asyncio

    client = SimbadClient(tap_url=tap_url, timeout=timeout)
    return asyncio.run(client.query_by_id(identifier))


def query_simbad_by_position_sync(
    ra: float,
    dec: float,
    radius_arcsec: float = 5.0,
    tap_url: str = SIMBAD_TAP_ENDPOINT,
    timeout: int = 60,
) -> list[SimbadQueryResult]:
    """Synchronous wrapper to query SIMBAD by position.

    Args:
        ra: Right ascension in degrees
        dec: Declination in degrees
        radius_arcsec: Search radius in arcseconds
        tap_url: SIMBAD TAP endpoint URL
        timeout: Request timeout in seconds

    Returns:
        List of SimbadQueryResult for objects in the cone
    """
    import asyncio

    client = SimbadClient(tap_url=tap_url, timeout=timeout)
    return asyncio.run(client.query_by_position(ra, dec, radius_arcsec))


__all__ = [
    # Exceptions
    "SimbadQueryError",
    "SimbadTAPError",
    # Models
    "SimbadIdentifiers",
    "SimbadObjectType",
    "SimbadSpectralInfo",
    "SimbadQueryResult",
    # Parsing utilities
    "parse_spectral_type",
    "classify_object_type",
    # Client
    "SimbadClient",
    # Sync helpers
    "query_simbad_by_id_sync",
    "query_simbad_by_position_sync",
    # Constants
    "SIMBAD_TAP_ENDPOINT",
]
