"""NASA Exoplanet Archive TAP service client for querying known planets.

This module provides functionality to query the NASA Exoplanet Archive TAP service
to retrieve known exoplanet parameters for a given star. It supports querying by
TIC ID or target name, and returns ephemeris data in BTJD format suitable for use
with TESS light curve analysis tools.

Usage:
    >>> from tess_vetter.platform.catalogs.exoplanet_archive import ExoplanetArchiveClient
    >>> client = ExoplanetArchiveClient()
    >>> result = client.get_known_planets(tic_id=150428135)
    >>> print(result.n_planets)
    4
    >>> for planet in result.planets:
    ...     print(f"{planet.name}: P={planet.period:.3f}d")

Technical Notes:
    - NASA Exoplanet Archive TAP endpoint: https://exoplanetarchive.ipac.caltech.edu/TAP/sync
    - Queries the PS (Planetary Systems) table for confirmed planets
    - Queries the TOI table for TESS Objects of Interest
    - Converts t0 from BJD to BTJD (subtracts 2457000) for TESS compatibility
    - Results are cached in memory for the session
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import requests
from tess_vetter.platform.catalogs.time_conventions import normalize_epoch_to_btjd

logger = logging.getLogger(__name__)

# NASA Exoplanet Archive TAP endpoint
TAP_ENDPOINT = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


class ExoplanetArchiveError(Exception):
    """Base exception for Exoplanet Archive errors."""

    pass


class TAPQueryError(ExoplanetArchiveError):
    """Error during TAP query execution."""

    pass


@dataclass
class KnownPlanet:
    """Represents a known exoplanet with its orbital parameters.

    Attributes:
        name: Planet name (e.g., "TOI-700 d", "WASP-18 b")
        period: Orbital period in days
        period_err: Period uncertainty in days (None if not available)
        t0: Reference transit epoch in BTJD
        t0_err: t0 uncertainty in days (None if not available)
        duration_hours: Transit duration in hours (None if not available)
        depth_ppm: Transit depth in parts per million (None if not available)
        radius_earth: Planet radius in Earth radii (None if not available)
        status: Discovery/confirmation status ("CONFIRMED", "CANDIDATE", "TOI")
        disposition: TOI disposition (for candidates)
        reference: Discovery/publication reference
        discovery_facility: Facility that discovered the planet
    """

    name: str
    period: float
    period_err: float | None
    t0: float  # BTJD
    t0_err: float | None
    duration_hours: float | None
    depth_ppm: float | None
    radius_earth: float | None
    status: str
    disposition: str | None
    reference: str | None
    discovery_facility: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "period": self.period,
            "period_err": self.period_err,
            "t0": self.t0,
            "t0_err": self.t0_err,
            "duration_hours": self.duration_hours,
            "depth_ppm": self.depth_ppm,
            "radius_earth": self.radius_earth,
            "status": self.status,
            "disposition": self.disposition,
            "reference": self.reference,
            "discovery_facility": self.discovery_facility,
        }


@dataclass
class KnownPlanetsResult:
    """Result of a known planets query.

    Attributes:
        tic_id: TESS Input Catalog ID that was queried
        n_planets: Number of planets found
        planets: List of known planets
        toi_id: TOI designation if applicable (e.g., "TOI-700")
        source: Data source description
    """

    tic_id: int
    n_planets: int
    planets: list[KnownPlanet]
    toi_id: str | None
    source: str = "NASA Exoplanet Archive"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "tic_id": self.tic_id,
            "n_planets": self.n_planets,
            "planets": [p.to_dict() for p in self.planets],
            "toi_id": self.toi_id,
            "source": self.source,
        }


@dataclass
class KnownPlanetMatchResult:
    """Period-level planet matching result for a TIC host."""

    status: Literal[
        "confirmed_same_planet",
        "confirmed_same_star_different_period",
        "no_confirmed_match",
        "ambiguous_multi_match",
    ]
    tic_id: int
    candidate_period_days: float
    period_tolerance_days: float
    period_tolerance_fraction: float | None
    matched_planet: KnownPlanet | None
    matched_planets: list[KnownPlanet]
    confirmed_planets: list[KnownPlanet]
    best_period_offset_days: float | None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "tic_id": int(self.tic_id),
            "candidate_period_days": float(self.candidate_period_days),
            "period_tolerance_days": float(self.period_tolerance_days),
            "period_tolerance_fraction": (
                float(self.period_tolerance_fraction)
                if self.period_tolerance_fraction is not None
                else None
            ),
            "matched_planet": self.matched_planet.to_dict() if self.matched_planet is not None else None,
            "matched_planets": [p.to_dict() for p in self.matched_planets],
            "confirmed_planets": [p.to_dict() for p in self.confirmed_planets],
            "best_period_offset_days": (
                float(self.best_period_offset_days) if self.best_period_offset_days is not None else None
            ),
            "notes": [str(x) for x in self.notes],
        }


@dataclass
class ExoplanetArchiveClient:
    """Client for querying the NASA Exoplanet Archive TAP service.

    Attributes:
        timeout: Request timeout in seconds
        cache: In-memory cache for query results
    """

    timeout: int = 30
    cache: dict[int, KnownPlanetsResult] = field(default_factory=dict)

    def _execute_tap_query(self, query: str) -> list[dict[str, Any]]:
        """Execute a TAP query against the NASA Exoplanet Archive.

        Args:
            query: ADQL query string

        Returns:
            List of result rows as dictionaries

        Raises:
            TAPQueryError: If the query fails
        """
        params = {
            "query": query,
            "format": "json",
        }

        max_retries = 3
        last_exception: Exception | None = None

        for attempt in range(max_retries):
            try:
                response = requests.get(TAP_ENDPOINT, params=params, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()

                # The TAP response format has results in a specific structure
                # Handle both old and new API response formats
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    # Some TAP responses wrap results in a structure
                    if "data" in data:
                        return list(data["data"])
                    return [data]
                return []

            except requests.exceptions.Timeout as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(
                        f"TAP query timeout, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                raise TAPQueryError(f"TAP query timed out after {max_retries} attempts") from e

            except requests.exceptions.RequestException as e:
                last_exception = e
                # Retry on 5xx errors and connection errors, not on 4xx client errors
                status_code = (
                    getattr(e.response, "status_code", None) if hasattr(e, "response") else None
                )
                if status_code is not None and 400 <= status_code < 500:
                    # Client error (bad request) - don't retry
                    raise TAPQueryError(f"TAP query failed: {e}") from e

                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"TAP query failed, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    time.sleep(wait_time)
                    continue
                raise TAPQueryError(f"TAP query failed after {max_retries} attempts: {e}") from e

            except ValueError as e:
                raise TAPQueryError(f"Failed to parse TAP response: {e}") from e

        # Should not reach here, but just in case
        raise TAPQueryError(f"TAP query failed after {max_retries} attempts") from last_exception

    def _query_ps_table(self, tic_id: int) -> list[dict[str, Any]]:
        """Query the Planetary Systems (PS) table for confirmed planets.

        The PS table contains one row per planet with best available parameters.

        Args:
            tic_id: TESS Input Catalog ID

        Returns:
            List of planet records
        """
        # Query PS table for confirmed planets matching this TIC ID
        # NOTE: The tic_id column in PS table is a string with format 'TIC 123456'
        query = f"""
        SELECT
            pl_name,
            pl_orbper,
            pl_orbpererr1,
            pl_tranmid,
            pl_tranmiderr1,
            pl_trandur,
            pl_trandep,
            pl_rade,
            disc_facility,
            disc_refname,
            tic_id,
            hostname,
            default_flag
        FROM ps
        WHERE tic_id = 'TIC {tic_id}'
        AND default_flag = 1
        ORDER BY pl_orbper ASC
        """
        return self._execute_tap_query(query)

    def _query_ps_by_hostname(self, hostname: str) -> list[dict[str, Any]]:
        """Query the Planetary Systems table by hostname.

        Args:
            hostname: Host star name (e.g., "WASP-18", "TOI-700")

        Returns:
            List of planet records
        """
        # Escape single quotes in hostname
        safe_hostname = hostname.replace("'", "''")

        query = f"""
        SELECT
            pl_name,
            pl_orbper,
            pl_orbpererr1,
            pl_tranmid,
            pl_tranmiderr1,
            pl_trandur,
            pl_trandep,
            pl_rade,
            disc_facility,
            disc_refname,
            tic_id,
            hostname,
            default_flag
        FROM ps
        WHERE (hostname LIKE '{safe_hostname}%' OR pl_name LIKE '{safe_hostname}%')
        AND default_flag = 1
        ORDER BY pl_orbper ASC
        """
        return self._execute_tap_query(query)

    def _query_toi_table(self, tic_id: int) -> list[dict[str, Any]]:
        """Query the TOI (TESS Objects of Interest) table.

        The TOI table contains candidates that may not yet be confirmed.

        Args:
            tic_id: TESS Input Catalog ID

        Returns:
            List of TOI records
        """
        # NOTE: TOI table uses 'tid' (integer) for TIC ID, not 'tic_id'
        # and 'pl_trandurh' for transit duration (in hours)
        query = f"""
        SELECT
            toi,
            toipfx,
            pl_orbper,
            pl_orbpererr1,
            pl_tranmid,
            pl_tranmiderr1,
            pl_trandurh,
            pl_trandep,
            pl_rade,
            tid,
            tfopwg_disp
        FROM toi
        WHERE tid = {tic_id}
        ORDER BY toi ASC
        """
        return self._execute_tap_query(query)

    def _bjd_to_btjd(self, bjd: float | None) -> float | None:
        """Convert BJD to BTJD (TESS Barycentric Julian Date).

        BTJD = BJD - 2457000

        Args:
            bjd: Barycentric Julian Date

        Returns:
            BTJD value, or None if input is None
        """
        return normalize_epoch_to_btjd(bjd)

    def _parse_ps_record(self, row: dict[str, Any]) -> KnownPlanet | None:
        """Parse a PS table record into a KnownPlanet object.

        Args:
            row: Dictionary from TAP query result

        Returns:
            KnownPlanet object, or None if essential data is missing
        """
        # Essential fields must be present
        name = row.get("pl_name")
        period = row.get("pl_orbper")
        t0_bjd = row.get("pl_tranmid")

        if not name or period is None:
            return None

        # Convert t0 to BTJD when available; preserve row with NaN otherwise.
        t0_converted = self._bjd_to_btjd(t0_bjd)
        t0: float = float("nan") if t0_converted is None else t0_converted

        # Convert transit duration from hours if available
        duration_hours = row.get("pl_trandur")

        # Transit depth - PS table gives it in percentage, convert to ppm
        depth_pct = row.get("pl_trandep")
        depth_ppm = depth_pct * 10000 if depth_pct is not None else None

        return KnownPlanet(
            name=name,
            period=float(period),
            period_err=row.get("pl_orbpererr1"),
            t0=t0,
            t0_err=row.get("pl_tranmiderr1"),
            duration_hours=duration_hours,
            depth_ppm=depth_ppm,
            radius_earth=row.get("pl_rade"),
            status="CONFIRMED",
            disposition="CONFIRMED",
            reference=row.get("disc_refname"),
            discovery_facility=row.get("disc_facility"),
        )

    def _parse_toi_record(self, row: dict[str, Any]) -> KnownPlanet | None:
        """Parse a TOI table record into a KnownPlanet object.

        Args:
            row: Dictionary from TAP query result

        Returns:
            KnownPlanet object, or None if essential data is missing
        """
        toi = row.get("toi")
        period = row.get("pl_orbper")
        t0_bjd = row.get("pl_tranmid")

        if toi is None or period is None:
            return None

        # Format TOI name
        name = f"TOI-{toi}"

        # Convert t0 to BTJD
        t0_converted = self._bjd_to_btjd(t0_bjd)
        t0: float = float("nan") if t0_converted is None else t0_converted

        # Duration in hours (TOI table uses pl_trandurh)
        duration_hours = row.get("pl_trandurh")

        # Depth - TOI table gives it in ppm already
        depth_ppm = row.get("pl_trandep")

        # Determine status from disposition
        # CP = Confirmed Planet, KP = Known Planet (previously confirmed)
        # PC = Planetary Candidate, APC = Ambiguous Planetary Candidate
        # FP = False Positive, FA = False Alarm
        disposition = row.get("tfopwg_disp")
        if disposition in ("CP", "KP"):
            status = "CONFIRMED"
        elif disposition in ("PC", "APC"):
            status = "CANDIDATE"
        elif disposition in ("FP", "FA"):
            status = "FALSE_POSITIVE"
        else:
            status = "TOI"

        return KnownPlanet(
            name=name,
            period=float(period),
            period_err=row.get("pl_orbpererr1"),
            t0=t0,
            t0_err=row.get("pl_tranmiderr1"),
            duration_hours=duration_hours,
            depth_ppm=depth_ppm,
            radius_earth=row.get("pl_rade"),
            status=status,
            disposition=disposition,
            reference=None,  # TOI table doesn't have reference field
            discovery_facility="TESS",
        )

    def get_known_planets(
        self,
        tic_id: int | None = None,
        target: str | None = None,
        include_candidates: bool = True,
    ) -> KnownPlanetsResult:
        """Query known planets for a given target.

        Queries both the Planetary Systems (PS) table for confirmed planets
        and the TOI table for TESS Objects of Interest.

        Args:
            tic_id: TESS Input Catalog ID (preferred)
            target: Target name for hostname matching (used if tic_id not provided)
            include_candidates: Include unconfirmed TOIs (default True)

        Returns:
            KnownPlanetsResult with list of planets and metadata

        Raises:
            ValueError: If neither tic_id nor target is provided
            TAPQueryError: If the TAP query fails
        """
        if tic_id is None and target is None:
            raise ValueError("Either tic_id or target must be provided")

        # Check cache
        if tic_id is not None and tic_id in self.cache:
            cached = self.cache[tic_id]
            if include_candidates:
                return cached
            # Filter out candidates if not requested
            confirmed_planets = [p for p in cached.planets if p.status == "CONFIRMED"]
            return KnownPlanetsResult(
                tic_id=cached.tic_id,
                n_planets=len(confirmed_planets),
                planets=confirmed_planets,
                toi_id=cached.toi_id,
            )

        planets: list[KnownPlanet] = []
        toi_id: str | None = None
        effective_tic_id = tic_id

        try:
            # Query PS table first (confirmed planets)
            if tic_id is not None:
                ps_results = self._query_ps_table(tic_id)
            elif target is not None:
                ps_results = self._query_ps_by_hostname(target)
            else:
                ps_results = []

            for row in ps_results:
                planet = self._parse_ps_record(row)
                if planet is not None:
                    planets.append(planet)
                # Extract TIC ID from PS results if we queried by name
                # PS table tic_id format is 'TIC 123456' so we need to parse it
                if effective_tic_id is None and row.get("tic_id"):
                    tic_str = str(row["tic_id"])
                    if tic_str.startswith("TIC "):
                        effective_tic_id = int(tic_str[4:])
                    elif tic_str.isdigit():
                        effective_tic_id = int(tic_str)
                # Extract TOI ID from hostname if it starts with TOI-
                if toi_id is None and row.get("hostname"):
                    hostname = str(row["hostname"])
                    if hostname.startswith("TOI-"):
                        toi_id = hostname

            # Query TOI table for candidates (only if we have a TIC ID)
            if effective_tic_id is not None and include_candidates:
                toi_results = self._query_toi_table(effective_tic_id)

                for row in toi_results:
                    planet = self._parse_toi_record(row)
                    if planet is not None:
                        # Skip confirmed planets (disposition="CP") since they're
                        # already in the PS table. Also skip false positives.
                        # Only include candidates (PC, APC) or unclassified TOIs.
                        if planet.status not in ("CONFIRMED", "FALSE_POSITIVE"):
                            planets.append(planet)

                        # Extract TOI designation
                        if toi_id is None and row.get("toi"):
                            toi_val = row["toi"]
                            # TOI is usually a float like 700.01, extract the integer part
                            toi_int = int(float(toi_val))
                            toi_id = f"TOI-{toi_int}"

        except TAPQueryError:
            # Re-raise TAP errors
            raise
        except Exception as e:
            logger.warning(f"Error querying exoplanet archive: {e}")
            # Return empty result rather than failing
            pass

        # Sort planets by period
        planets.sort(key=lambda p: p.period)

        # Use provided tic_id or 0 if not available
        result_tic_id = effective_tic_id if effective_tic_id is not None else 0

        result = KnownPlanetsResult(
            tic_id=result_tic_id,
            n_planets=len(planets),
            planets=planets,
            toi_id=toi_id,
        )

        # Cache the result
        if result_tic_id > 0:
            self.cache[result_tic_id] = result

        return result

    def match_known_planet_ephemeris(
        self,
        *,
        tic_id: int,
        period_days: float,
        period_tolerance_days: float = 0.01,
        period_tolerance_fraction: float | None = 0.001,
    ) -> KnownPlanetMatchResult:
        """Match a candidate period against confirmed planets on the same TIC host.

        Matching is host+period based (not host-only), which avoids false equivalence
        on multi-planet systems.
        """
        candidate_period = float(period_days)
        abs_tol_days = float(max(period_tolerance_days, 0.0))
        frac_tol = (
            float(period_tolerance_fraction)
            if period_tolerance_fraction is not None and float(period_tolerance_fraction) > 0.0
            else None
        )

        kp = self.get_known_planets(tic_id=int(tic_id), include_candidates=False)
        confirmed = [p for p in kp.planets if str(p.status).upper() == "CONFIRMED"]
        if not confirmed:
            return KnownPlanetMatchResult(
                status="no_confirmed_match",
                tic_id=int(tic_id),
                candidate_period_days=candidate_period,
                period_tolerance_days=abs_tol_days,
                period_tolerance_fraction=frac_tol,
                matched_planet=None,
                matched_planets=[],
                confirmed_planets=[],
                best_period_offset_days=None,
                notes=["No confirmed planets found for TIC in Exoplanet Archive."],
            )

        scored: list[tuple[float, KnownPlanet]] = []
        matches: list[KnownPlanet] = []
        for planet in confirmed:
            try:
                p_period = float(planet.period)
            except Exception:
                continue
            diff = abs(candidate_period - p_period)
            tol = abs_tol_days
            if frac_tol is not None:
                tol = max(tol, abs(p_period) * frac_tol)
            scored.append((diff, planet))
            if diff <= tol:
                matches.append(planet)

        scored.sort(key=lambda item: item[0])
        matches_sorted = sorted(matches, key=lambda p: abs(candidate_period - float(p.period)))
        best_match = matches_sorted[0] if matches_sorted else None
        best_diff = scored[0][0] if scored else None

        if len(matches_sorted) == 1:
            status: Literal[
                "confirmed_same_planet",
                "confirmed_same_star_different_period",
                "no_confirmed_match",
                "ambiguous_multi_match",
            ] = "confirmed_same_planet"
            notes: list[str] = []
        elif len(matches_sorted) > 1:
            status = "ambiguous_multi_match"
            notes = ["Multiple confirmed planets matched candidate period within tolerance."]
        else:
            status = "confirmed_same_star_different_period"
            notes = ["Confirmed planets exist on host, but none matched candidate period."]

        return KnownPlanetMatchResult(
            status=status,
            tic_id=int(tic_id),
            candidate_period_days=candidate_period,
            period_tolerance_days=abs_tol_days,
            period_tolerance_fraction=frac_tol,
            matched_planet=best_match,
            matched_planets=matches_sorted,
            confirmed_planets=confirmed,
            best_period_offset_days=float(best_diff) if best_diff is not None else None,
            notes=notes,
        )

    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self.cache.clear()


# Module-level singleton for convenience
_client: ExoplanetArchiveClient | None = None


def get_client() -> ExoplanetArchiveClient:
    """Get the module-level ExoplanetArchiveClient singleton."""
    global _client
    if _client is None:
        _client = ExoplanetArchiveClient()
    return _client


def get_known_planets(
    tic_id: int | None = None,
    target: str | None = None,
    include_candidates: bool = True,
) -> KnownPlanetsResult:
    """Convenience function to query known planets using the singleton client.

    Args:
        tic_id: TESS Input Catalog ID
        target: Target name
        include_candidates: Include unconfirmed TOIs

    Returns:
        KnownPlanetsResult
    """
    return get_client().get_known_planets(
        tic_id=tic_id,
        target=target,
        include_candidates=include_candidates,
    )


def match_known_planet_ephemeris(
    *,
    tic_id: int,
    period_days: float,
    period_tolerance_days: float = 0.01,
    period_tolerance_fraction: float | None = 0.001,
) -> KnownPlanetMatchResult:
    """Convenience wrapper for period-level known-planet matching."""
    return get_client().match_known_planet_ephemeris(
        tic_id=int(tic_id),
        period_days=float(period_days),
        period_tolerance_days=float(period_tolerance_days),
        period_tolerance_fraction=(
            float(period_tolerance_fraction)
            if period_tolerance_fraction is not None
            else None
        ),
    )
