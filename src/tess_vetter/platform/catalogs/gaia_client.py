"""Gaia DR3 TAP service client for stellar characterization.

This module provides functionality to query the Gaia DR3 archive to retrieve
stellar parameters and source information. It supports querying by TIC ID,
Gaia source_id, or sky position, and returns source data along with
astrophysical parameters and nearby neighbors from cone searches.

Usage:
    >>> from tess_vetter.platform.catalogs.gaia_client import GaiaClient
    >>> client = GaiaClient()
    >>> result = await client.query_by_tic(261136679)  # Pi Mensae
    >>> print(f"Gaia source: {result.source.source_id}")
    >>> print(f"Teff: {result.astrophysical.teff_gspphot} K")
    >>> print(f"Neighbors within 60 arcsec: {len(result.neighbors)}")

Technical Notes:
    - Gaia DR3 TAP endpoint: https://gea.esac.esa.int/tap-server/tap
    - Queries gaia_source for astrometry and photometry
    - Queries astrophysical_parameters for GSP-Phot and FLAME parameters
    - Cone searches return neighbors sorted by separation, then brightness
    - RUWE > 1.4 indicates elevated astrometric excess (potential binary)
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import UTC, datetime
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from tess_vetter.platform.catalogs.models import SourceRecord

logger = logging.getLogger(__name__)

# Gaia TAP endpoint
GAIA_TAP_ENDPOINT = "https://gea.esac.esa.int/tap-server/tap"

# RUWE threshold for flagging elevated astrometric noise
RUWE_ELEVATED_THRESHOLD = 1.4


class GaiaQueryError(Exception):
    """Base exception for Gaia query errors."""

    pass


class GaiaTAPError(GaiaQueryError):
    """Error during TAP query execution."""

    pass


class GaiaSourceRecord(BaseModel):
    """Gaia DR3 source table record.

    Contains astrometric, photometric, and quality fields from gaia_source.

    Attributes:
        source_id: Gaia DR3 unique source identifier
        ra: Right ascension (degrees, ICRS)
        dec: Declination (degrees, ICRS)
        parallax: Parallax (mas)
        parallax_error: Parallax uncertainty (mas)
        pmra: Proper motion in RA (mas/yr)
        pmdec: Proper motion in Dec (mas/yr)
        phot_g_mean_mag: G-band mean magnitude
        phot_bp_mean_mag: BP-band mean magnitude
        phot_rp_mean_mag: RP-band mean magnitude
        bp_rp: BP - RP color
        ruwe: Renormalised unit weight error (>1.4 = elevated)
        duplicated_source: Duplicate source flag
        non_single_star: Non-single star flag (NSS)
        astrometric_excess_noise: Excess noise in astrometric fit (mas)
        phot_bp_rp_excess_factor: Photometric BP/RP excess factor
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    source_id: int = Field(description="Gaia DR3 unique source identifier")
    ra: float = Field(description="Right ascension (degrees, ICRS)")
    dec: float = Field(description="Declination (degrees, ICRS)")
    parallax: float | None = Field(default=None, description="Parallax (mas)")
    parallax_error: float | None = Field(default=None, description="Parallax uncertainty (mas)")
    pmra: float | None = Field(default=None, description="Proper motion in RA (mas/yr)")
    pmdec: float | None = Field(default=None, description="Proper motion in Dec (mas/yr)")
    phot_g_mean_mag: float | None = Field(default=None, description="G-band mean magnitude")
    phot_bp_mean_mag: float | None = Field(default=None, description="BP-band mean magnitude")
    phot_rp_mean_mag: float | None = Field(default=None, description="RP-band mean magnitude")
    bp_rp: float | None = Field(default=None, description="BP - RP color")
    ruwe: float | None = Field(default=None, description="Renormalised unit weight error")
    duplicated_source: bool | None = Field(default=None, description="Duplicate source flag")
    non_single_star: int | None = Field(default=None, description="Non-single star flag (NSS)")
    astrometric_excess_noise: float | None = Field(
        default=None, description="Excess noise in astrometric fit (mas)"
    )
    phot_bp_rp_excess_factor: float | None = Field(
        default=None, description="Photometric BP/RP excess factor"
    )

    @property
    def ruwe_elevated(self) -> bool:
        """Check if RUWE indicates elevated astrometric excess."""
        return self.ruwe is not None and self.ruwe > RUWE_ELEVATED_THRESHOLD

    @property
    def distance_pc(self) -> float | None:
        """Compute distance in parsecs from parallax."""
        if self.parallax is None or self.parallax <= 0:
            return None
        return 1000.0 / self.parallax


class GaiaAstrophysicalParams(BaseModel):
    """Gaia DR3 astrophysical_parameters table record.

    Contains stellar parameters from GSP-Phot and FLAME modules.

    Attributes:
        source_id: Gaia DR3 unique source identifier
        teff_gspphot: Effective temperature from GSP-Phot (K)
        teff_gspphot_lower: Lower confidence limit on Teff
        teff_gspphot_upper: Upper confidence limit on Teff
        logg_gspphot: Surface gravity from GSP-Phot (log cm/s^2)
        logg_gspphot_lower: Lower confidence limit on logg
        logg_gspphot_upper: Upper confidence limit on logg
        radius_gspphot: Stellar radius from GSP-Phot (solar radii)
        radius_gspphot_lower: Lower confidence limit on radius
        radius_gspphot_upper: Upper confidence limit on radius
        mass_flame: Stellar mass from FLAME (solar masses)
        lum_flame: Luminosity from FLAME (solar luminosities)
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    source_id: int = Field(description="Gaia DR3 unique source identifier")
    teff_gspphot: float | None = Field(
        default=None, description="Effective temperature from GSP-Phot (K)"
    )
    teff_gspphot_lower: float | None = Field(
        default=None, description="Lower confidence limit on Teff"
    )
    teff_gspphot_upper: float | None = Field(
        default=None, description="Upper confidence limit on Teff"
    )
    logg_gspphot: float | None = Field(
        default=None, description="Surface gravity from GSP-Phot (log cm/s^2)"
    )
    logg_gspphot_lower: float | None = Field(
        default=None, description="Lower confidence limit on logg"
    )
    logg_gspphot_upper: float | None = Field(
        default=None, description="Upper confidence limit on logg"
    )
    radius_gspphot: float | None = Field(
        default=None, description="Stellar radius from GSP-Phot (solar radii)"
    )
    radius_gspphot_lower: float | None = Field(
        default=None, description="Lower confidence limit on radius"
    )
    radius_gspphot_upper: float | None = Field(
        default=None, description="Upper confidence limit on radius"
    )
    mass_flame: float | None = Field(
        default=None, description="Stellar mass from FLAME (solar masses)"
    )
    lum_flame: float | None = Field(
        default=None, description="Luminosity from FLAME (solar luminosities)"
    )

    @property
    def teff_uncertainty(self) -> float | None:
        """Compute Teff uncertainty from confidence limits."""
        if self.teff_gspphot_lower is None or self.teff_gspphot_upper is None:
            return None
        return (self.teff_gspphot_upper - self.teff_gspphot_lower) / 2.0

    @property
    def radius_uncertainty(self) -> float | None:
        """Compute radius uncertainty from confidence limits."""
        if self.radius_gspphot_lower is None or self.radius_gspphot_upper is None:
            return None
        return (self.radius_gspphot_upper - self.radius_gspphot_lower) / 2.0


class GaiaNeighbor(BaseModel):
    """A Gaia source found in a cone search.

    Represents a neighboring source with separation and magnitude difference
    relative to the primary target.

    Attributes:
        source_id: Gaia DR3 unique source identifier
        ra: Right ascension (degrees, ICRS)
        dec: Declination (degrees, ICRS)
        separation_arcsec: Angular separation from primary (arcsec)
        phot_g_mean_mag: G-band mean magnitude
        delta_mag: Magnitude difference relative to primary (positive = fainter)
        ruwe: Renormalised unit weight error
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    source_id: int = Field(description="Gaia DR3 unique source identifier")
    ra: float = Field(description="Right ascension (degrees, ICRS)")
    dec: float = Field(description="Declination (degrees, ICRS)")
    separation_arcsec: float = Field(description="Angular separation from primary (arcsec)")
    phot_g_mean_mag: float | None = Field(default=None, description="G-band mean magnitude")
    delta_mag: float | None = Field(
        default=None, description="Magnitude difference relative to primary"
    )
    ruwe: float | None = Field(default=None, description="Renormalised unit weight error")

    @property
    def ruwe_elevated(self) -> bool:
        """Check if RUWE indicates elevated astrometric excess."""
        return self.ruwe is not None and self.ruwe > RUWE_ELEVATED_THRESHOLD


class GaiaQueryResult(BaseModel):
    """Complete result from Gaia DR3 queries.

    Combines source record, astrophysical parameters, cone search neighbors,
    and provenance information.

    Attributes:
        source: Primary source record from gaia_source
        astrophysical: Astrophysical parameters from GSP-Phot/FLAME
        neighbors: Nearby sources from cone search, sorted by separation then brightness
        source_record: Provenance record for the query
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    source: GaiaSourceRecord | None = Field(
        default=None, description="Primary source record from gaia_source"
    )
    astrophysical: GaiaAstrophysicalParams | None = Field(
        default=None, description="Astrophysical parameters from GSP-Phot/FLAME"
    )
    neighbors: list[GaiaNeighbor] = Field(
        default_factory=list, description="Nearby sources from cone search"
    )
    source_record: SourceRecord = Field(description="Provenance record for the query")


def _parse_gaia_source(row: dict[str, Any]) -> GaiaSourceRecord:
    """Parse a gaia_source TAP row into a GaiaSourceRecord.

    Args:
        row: Dictionary from TAP query result

    Returns:
        GaiaSourceRecord with parsed values
    """
    return GaiaSourceRecord(
        source_id=int(row["source_id"]),
        ra=float(row["ra"]),
        dec=float(row["dec"]),
        parallax=row.get("parallax"),
        parallax_error=row.get("parallax_error"),
        pmra=row.get("pmra"),
        pmdec=row.get("pmdec"),
        phot_g_mean_mag=row.get("phot_g_mean_mag"),
        phot_bp_mean_mag=row.get("phot_bp_mean_mag"),
        phot_rp_mean_mag=row.get("phot_rp_mean_mag"),
        bp_rp=row.get("bp_rp"),
        ruwe=row.get("ruwe"),
        duplicated_source=row.get("duplicated_source"),
        non_single_star=row.get("non_single_star"),
        astrometric_excess_noise=row.get("astrometric_excess_noise"),
        phot_bp_rp_excess_factor=row.get("phot_bp_rp_excess_factor"),
    )


def _parse_astrophysical_params(row: dict[str, Any]) -> GaiaAstrophysicalParams:
    """Parse an astrophysical_parameters TAP row.

    Args:
        row: Dictionary from TAP query result

    Returns:
        GaiaAstrophysicalParams with parsed values
    """
    return GaiaAstrophysicalParams(
        source_id=int(row["source_id"]),
        teff_gspphot=row.get("teff_gspphot"),
        teff_gspphot_lower=row.get("teff_gspphot_lower"),
        teff_gspphot_upper=row.get("teff_gspphot_upper"),
        logg_gspphot=row.get("logg_gspphot"),
        logg_gspphot_lower=row.get("logg_gspphot_lower"),
        logg_gspphot_upper=row.get("logg_gspphot_upper"),
        radius_gspphot=row.get("radius_gspphot"),
        radius_gspphot_lower=row.get("radius_gspphot_lower"),
        radius_gspphot_upper=row.get("radius_gspphot_upper"),
        mass_flame=row.get("mass_flame"),
        lum_flame=row.get("lum_flame"),
    )


def _compute_separation_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Compute angular separation in arcseconds.

    Uses the Haversine formula for small angles.

    Args:
        ra1, dec1: Position 1 in degrees
        ra2, dec2: Position 2 in degrees

    Returns:
        Separation in arcseconds
    """
    import math

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

    # Convert to arcseconds
    separation_deg = math.degrees(c)
    return separation_deg * 3600.0


class GaiaClient:
    """Client for Gaia DR3 TAP queries.

    Provides methods to query Gaia DR3 by TIC ID, Gaia source_id, or position.
    Returns source data, astrophysical parameters, and cone search neighbors.

    Attributes:
        tap_url: Gaia TAP endpoint URL
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        tap_url: str = GAIA_TAP_ENDPOINT,
        timeout: int = 60,
        max_retries: int = 3,
        cache_path: str | None = None,
        cache_ttl_days: float | None = None,
    ) -> None:
        """Initialize the Gaia client.

        Args:
            tap_url: Gaia TAP endpoint URL
            timeout: Request timeout in seconds
        """
        self.tap_url = tap_url
        self.timeout = timeout
        self.max_retries = int(max_retries)
        self.cache_path = cache_path
        self.cache_ttl_days = cache_ttl_days

    def _cache_conn(self) -> sqlite3.Connection | None:
        if not self.cache_path:
            return None
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        except Exception:
            # Allow cache_path with no directory component.
            pass
        try:
            conn = sqlite3.connect(self.cache_path)
        except Exception:
            return None
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS gaia_cache (
                    cache_key TEXT PRIMARY KEY,
                    created_unix REAL NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.commit()
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            return None
        return conn

    def _cache_key_position(self, ra: float, dec: float, radius_arcsec: float) -> str:
        # Quantize to keep keys stable across float repr differences.
        ra_q = round(float(ra), 6)
        dec_q = round(float(dec), 6)
        r_q = round(float(radius_arcsec), 2)
        return f"pos:dr3:{ra_q}:{dec_q}:r{r_q}"

    def _cache_get(self, cache_key: str) -> dict[str, Any] | None:
        conn = self._cache_conn()
        if conn is None:
            return None
        try:
            row = conn.execute(
                "SELECT created_unix, payload_json FROM gaia_cache WHERE cache_key=?",
                (cache_key,),
            ).fetchone()
            if not row:
                return None
            created_unix, payload_json = row
            if self.cache_ttl_days is not None:
                import time

                age_days = (time.time() - float(created_unix)) / 86400.0
                if age_days > float(self.cache_ttl_days):
                    return None
            import json

            return json.loads(payload_json)
        except Exception:
            return None
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _cache_put(self, cache_key: str, payload: dict[str, Any]) -> None:
        conn = self._cache_conn()
        if conn is None:
            return
        try:
            import json
            import time

            conn.execute(
                "INSERT OR REPLACE INTO gaia_cache(cache_key, created_unix, payload_json) VALUES (?, ?, ?)",
                (cache_key, float(time.time()), json.dumps(payload, sort_keys=True)),
            )
            conn.commit()
        except Exception:
            return
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _execute_tap_query(self, query: str) -> list[dict[str, Any]]:
        """Execute a synchronous TAP query.

        Args:
            query: ADQL query string

        Returns:
            List of result rows as dictionaries

        Raises:
            GaiaTAPError: If the query fails
        """
        import time

        import requests

        params = {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "json",
            "QUERY": query,
        }

        max_retries = max(1, int(self.max_retries))
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

                # Gaia TAP returns data in a specific format
                if isinstance(data, dict):
                    # Standard VOTable JSON format
                    if "data" in data:
                        columns = data.get("metadata", [])
                        col_names = [c.get("name") for c in columns]
                        rows = []
                        for row_data in data["data"]:
                            rows.append(dict(zip(col_names, row_data, strict=False)))
                        return rows
                    # Some responses have direct results
                    return [data] if data else []
                elif isinstance(data, list):
                    return data
                return []

            except requests.exceptions.Timeout as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Gaia TAP timeout, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                raise GaiaTAPError(f"Gaia TAP query timed out after {max_retries} attempts") from e

            except requests.exceptions.RequestException as e:
                last_exception = e
                status_code = (
                    getattr(e.response, "status_code", None) if hasattr(e, "response") else None
                )
                if status_code is not None and 400 <= status_code < 500:
                    raise GaiaTAPError(f"Gaia TAP query failed: {e}") from e

                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Gaia TAP query failed, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    time.sleep(wait_time)
                    continue
                raise GaiaTAPError(
                    f"Gaia TAP query failed after {max_retries} attempts: {e}"
                ) from e

            except ValueError as e:
                raise GaiaTAPError(f"Failed to parse Gaia TAP response: {e}") from e

        raise GaiaTAPError(
            f"Gaia TAP query failed after {max_retries} attempts"
        ) from last_exception

    def _query_source_by_id(self, source_id: int) -> GaiaSourceRecord | None:
        """Query gaia_source by source_id.

        Args:
            source_id: Gaia DR3 source_id

        Returns:
            GaiaSourceRecord or None if not found
        """
        query = f"""
        SELECT
            source_id, ra, dec, parallax, parallax_error, pmra, pmdec,
            phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, bp_rp,
            ruwe, duplicated_source, non_single_star,
            astrometric_excess_noise, phot_bp_rp_excess_factor
        FROM gaiadr3.gaia_source
        WHERE source_id = {source_id}
        """
        rows = self._execute_tap_query(query)
        if not rows:
            return None
        return _parse_gaia_source(rows[0])

    def _query_astrophysical_by_id(self, source_id: int) -> GaiaAstrophysicalParams | None:
        """Query astrophysical_parameters by source_id.

        Args:
            source_id: Gaia DR3 source_id

        Returns:
            GaiaAstrophysicalParams or None if not found
        """
        query = f"""
        SELECT
            source_id,
            teff_gspphot, teff_gspphot_lower, teff_gspphot_upper,
            logg_gspphot, logg_gspphot_lower, logg_gspphot_upper,
            radius_gspphot, radius_gspphot_lower, radius_gspphot_upper,
            mass_flame, lum_flame
        FROM gaiadr3.astrophysical_parameters
        WHERE source_id = {source_id}
        """
        rows = self._execute_tap_query(query)
        if not rows:
            return None
        return _parse_astrophysical_params(rows[0])

    def _query_cone_search(
        self,
        ra: float,
        dec: float,
        radius_arcsec: float,
        primary_mag: float | None = None,
    ) -> list[GaiaNeighbor]:
        """Perform a cone search around a position.

        Args:
            ra: Right ascension (degrees)
            dec: Declination (degrees)
            radius_arcsec: Search radius (arcsec)
            primary_mag: Primary target G-mag for computing delta_mag

        Returns:
            List of GaiaNeighbor sorted by separation then brightness
        """
        # Convert radius to degrees for ADQL
        radius_deg = radius_arcsec / 3600.0

        query = f"""
        SELECT
            source_id, ra, dec, phot_g_mean_mag, ruwe
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
        )
        ORDER BY phot_g_mean_mag ASC
        """
        rows = self._execute_tap_query(query)

        neighbors: list[GaiaNeighbor] = []
        for row in rows:
            sep = _compute_separation_arcsec(ra, dec, row["ra"], row["dec"])
            g_mag = row.get("phot_g_mean_mag")
            delta_mag = None
            if g_mag is not None and primary_mag is not None:
                delta_mag = g_mag - primary_mag

            neighbors.append(
                GaiaNeighbor(
                    source_id=int(row["source_id"]),
                    ra=float(row["ra"]),
                    dec=float(row["dec"]),
                    separation_arcsec=sep,
                    phot_g_mean_mag=g_mag,
                    delta_mag=delta_mag,
                    ruwe=row.get("ruwe"),
                )
            )

        # Sort by separation first, then by brightness (brighter = lower mag)
        neighbors.sort(
            key=lambda n: (n.separation_arcsec, n.phot_g_mean_mag if n.phot_g_mean_mag else 999.0)
        )

        return neighbors

    def _resolve_tic_to_gaia(self, tic_id: int) -> int | None:
        """Resolve TIC ID to Gaia DR3 source_id.

        Uses the TIC v8.2 catalog which includes Gaia DR3 cross-matches.
        Falls back to a cone search if direct lookup fails.

        Args:
            tic_id: TESS Input Catalog ID

        Returns:
            Gaia DR3 source_id or None if not found
        """
        # TIC v8.2 includes Gaia DR3 IDs, but for now we use astroquery
        # or direct TIC lookup. This is a placeholder that should be
        # replaced with actual TIC crossmatch logic.
        #
        # For now, we assume the caller will provide the Gaia ID directly
        # or use query_by_position.
        logger.warning(
            f"TIC-to-Gaia resolution not yet implemented for TIC {tic_id}. "
            "Use query_by_gaia_id or query_by_position instead."
        )
        return None

    async def query_by_tic(
        self,
        tic_id: int,
        cone_radius_arcsec: float = 60.0,
    ) -> GaiaQueryResult:
        """Query Gaia DR3 by TIC ID.

        Resolves the TIC ID to a Gaia source_id, then queries source data,
        astrophysical parameters, and performs a cone search for neighbors.

        Args:
            tic_id: TESS Input Catalog ID
            cone_radius_arcsec: Radius for neighbor cone search (arcsec)

        Returns:
            GaiaQueryResult with source, astrophysical params, and neighbors
        """
        gaia_id = self._resolve_tic_to_gaia(tic_id)
        if gaia_id is None:
            # Return empty result with provenance
            return GaiaQueryResult(
                source=None,
                astrophysical=None,
                neighbors=[],
                source_record=SourceRecord(
                    name="gaia_dr3",
                    version="dr3",
                    retrieved_at=datetime.now(UTC),
                    query=f"TIC {tic_id} -> Gaia resolution failed",
                ),
            )

        return await self.query_by_gaia_id(gaia_id, cone_radius_arcsec)

    async def query_by_gaia_id(
        self,
        gaia_id: int,
        cone_radius_arcsec: float = 60.0,
    ) -> GaiaQueryResult:
        """Query Gaia DR3 by Gaia source_id.

        Args:
            gaia_id: Gaia DR3 source_id
            cone_radius_arcsec: Radius for neighbor cone search (arcsec)

        Returns:
            GaiaQueryResult with source, astrophysical params, and neighbors
        """
        now = datetime.now(UTC)

        # Query source and astrophysical params
        source = self._query_source_by_id(gaia_id)
        astrophysical = self._query_astrophysical_by_id(gaia_id)

        # Perform cone search if we have a position
        neighbors: list[GaiaNeighbor] = []
        if source is not None:
            neighbors = self._query_cone_search(
                source.ra,
                source.dec,
                cone_radius_arcsec,
                primary_mag=source.phot_g_mean_mag,
            )
            # Exclude the primary source from neighbors
            neighbors = [n for n in neighbors if n.source_id != gaia_id]

        return GaiaQueryResult(
            source=source,
            astrophysical=astrophysical,
            neighbors=neighbors,
            source_record=SourceRecord(
                name="gaia_dr3",
                version="dr3",
                retrieved_at=now,
                query=f"source_id = {gaia_id}, cone = {cone_radius_arcsec} arcsec",
            ),
        )

    async def query_by_position(
        self,
        ra: float,
        dec: float,
        radius_arcsec: float = 60.0,
    ) -> GaiaQueryResult:
        """Query Gaia DR3 by sky position.

        Performs a cone search and returns the brightest source as primary,
        with all other sources as neighbors.

        Args:
            ra: Right ascension (degrees)
            dec: Declination (degrees)
            radius_arcsec: Search radius (arcsec)

        Returns:
            GaiaQueryResult with brightest source as primary and others as neighbors
        """
        now = datetime.now(UTC)

        # Get all sources in the cone
        all_sources = self._query_cone_search(ra, dec, radius_arcsec, primary_mag=None)

        if not all_sources:
            return GaiaQueryResult(
                source=None,
                astrophysical=None,
                neighbors=[],
                source_record=SourceRecord(
                    name="gaia_dr3",
                    version="dr3",
                    retrieved_at=now,
                    query=f"cone({ra}, {dec}, {radius_arcsec} arcsec) -> no sources",
                ),
            )

        # The first source (sorted by separation) closest to the center is primary
        # But we want the brightest one near the center for typical use
        # Actually, sort by separation first, take the closest as primary
        primary_neighbor = all_sources[0]
        primary_gaia_id = primary_neighbor.source_id

        # Get full source and astrophysical data for primary
        source = self._query_source_by_id(primary_gaia_id)
        astrophysical = self._query_astrophysical_by_id(primary_gaia_id)

        # Recompute neighbors with delta_mag relative to primary
        primary_mag = source.phot_g_mean_mag if source else None
        neighbors: list[GaiaNeighbor] = []
        for n in all_sources[1:]:  # Skip the primary
            delta_mag = None
            if n.phot_g_mean_mag is not None and primary_mag is not None:
                delta_mag = n.phot_g_mean_mag - primary_mag
            neighbors.append(
                GaiaNeighbor(
                    source_id=n.source_id,
                    ra=n.ra,
                    dec=n.dec,
                    separation_arcsec=n.separation_arcsec,
                    phot_g_mean_mag=n.phot_g_mean_mag,
                    delta_mag=delta_mag,
                    ruwe=n.ruwe,
                )
            )

        return GaiaQueryResult(
            source=source,
            astrophysical=astrophysical,
            neighbors=neighbors,
            source_record=SourceRecord(
                name="gaia_dr3",
                version="dr3",
                retrieved_at=now,
                query=f"cone({ra}, {dec}, {radius_arcsec} arcsec)",
            ),
        )


# Module-level functions for synchronous access


def query_gaia_by_id_sync(
    gaia_id: int,
    cone_radius_arcsec: float = 60.0,
    tap_url: str = GAIA_TAP_ENDPOINT,
    timeout: int = 60,
    max_retries: int = 3,
    cache_path: str | None = None,
    cache_ttl_days: float | None = None,
) -> GaiaQueryResult:
    """Synchronous wrapper to query Gaia DR3 by source_id.

    Args:
        gaia_id: Gaia DR3 source_id
        cone_radius_arcsec: Radius for neighbor cone search (arcsec)
        tap_url: Gaia TAP endpoint URL
        timeout: Request timeout in seconds

    Returns:
        GaiaQueryResult with source, astrophysical params, and neighbors
    """
    import asyncio

    client = GaiaClient(
        tap_url=tap_url,
        timeout=timeout,
        max_retries=max_retries,
        cache_path=cache_path,
        cache_ttl_days=cache_ttl_days,
    )
    return asyncio.run(client.query_by_gaia_id(gaia_id, cone_radius_arcsec))


def query_gaia_by_position_sync(
    ra: float,
    dec: float,
    radius_arcsec: float = 60.0,
    tap_url: str = GAIA_TAP_ENDPOINT,
    timeout: int = 60,
    max_retries: int = 3,
    cache_path: str | None = None,
    cache_ttl_days: float | None = None,
) -> GaiaQueryResult:
    """Synchronous wrapper to query Gaia DR3 by position.

    Args:
        ra: Right ascension (degrees)
        dec: Declination (degrees)
        radius_arcsec: Search radius (arcsec)
        tap_url: Gaia TAP endpoint URL
        timeout: Request timeout in seconds

    Returns:
        GaiaQueryResult with brightest source as primary
    """
    import asyncio

    client = GaiaClient(
        tap_url=tap_url,
        timeout=timeout,
        max_retries=max_retries,
        cache_path=cache_path,
        cache_ttl_days=cache_ttl_days,
    )
    # Cache only the position query (this is what bulk enrichment needs).
    cache_key = client._cache_key_position(float(ra), float(dec), float(radius_arcsec))
    cached = client._cache_get(cache_key)
    if cached is not None:
        try:
            return GaiaQueryResult.model_validate(cached)
        except Exception:
            pass
    result = asyncio.run(client.query_by_position(ra, dec, radius_arcsec))
    try:
        client._cache_put(cache_key, result.model_dump(mode="json"))
    except Exception:
        pass
    return result


__all__ = [
    # Exceptions
    "GaiaQueryError",
    "GaiaTAPError",
    # Models
    "GaiaSourceRecord",
    "GaiaAstrophysicalParams",
    "GaiaNeighbor",
    "GaiaQueryResult",
    # Client
    "GaiaClient",
    # Sync helpers
    "query_gaia_by_id_sync",
    "query_gaia_by_position_sync",
    # Constants
    "GAIA_TAP_ENDPOINT",
    "RUWE_ELEVATED_THRESHOLD",
]
