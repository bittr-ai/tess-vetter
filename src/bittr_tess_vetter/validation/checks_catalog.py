"""Catalog-based vetting checks for transit candidate validation (V06-V07).

This module implements pre-filter checks that query external catalogs to
quickly identify known false positives before expensive analysis:

- V06 (NearbyEBCheck): Query TESS-EB catalog for known EBs near target
- V07 (ExoFOPDispositionCheck): Query ExoFOP-TESS for existing dispositions

These checks are fast (<1s) and can reject obvious false positives before
running expensive TRICERATOPS FPP calculations (~30s).
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import requests

from bittr_tess_vetter.domain.detection import VetterCheckResult
from bittr_tess_vetter.validation.base import CheckConfig, VetterCheck, make_result

if TYPE_CHECKING:
    from bittr_tess_vetter.domain.detection import TransitCandidate
    from bittr_tess_vetter.domain.lightcurve import LightCurveData
    from bittr_tess_vetter.domain.target import StellarParameters

logger = logging.getLogger(__name__)

# API endpoints
VIZIER_TAP_URL = "https://vizier.cds.unistra.fr/viz-bin/votable"
EXOFOP_TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php"

# ExoFOP disposition codes that indicate false positive
FP_DISPOSITIONS = {"FP", "FA", "EB", "NEB", "BEB", "V", "IS", "O"}
# Codes that indicate confirmed/known planet (pass)
PLANET_DISPOSITIONS = {"CP", "KP", "PC", "APC"}

# Request timeout in seconds
REQUEST_TIMEOUT = 10.0


# =============================================================================
# Data Classes for Results
# =============================================================================


@dataclass
class NearbyEBResult:
    """Result of nearby EB catalog search."""

    passed: bool  # True = no EB found
    confidence: float  # 0.95 if clean, lower if uncertain
    n_ebs_found: int
    closest_eb_arcsec: float | None
    closest_eb_tic: int | None
    closest_eb_period: float | None  # EB period in days
    period_match: bool  # True if EB period ~ candidate period or 2x


@dataclass
class ExoFOPDispositionResult:
    """Result of ExoFOP disposition check."""

    passed: bool  # False if FP/EB disposition
    confidence: float  # 0.95 for known FP, 0.5 for unknown
    disposition: str | None  # Raw disposition string
    toi_number: float | None  # TOI number if exists
    comments: str | None  # TFOPWG comments
    last_updated: str | None  # ISO timestamp


# =============================================================================
# V06: Nearby EB Catalog Search
# =============================================================================


class NearbyEBCheck(VetterCheck):
    """V06: Search for known eclipsing binaries near target.

    Queries the TESS-EB catalog (VizieR J/ApJS/258/16) for known eclipsing
    binaries within the TESS aperture (~21 arcsec pixel scale).

    A known EB within the photometric aperture can contaminate the light curve
    and produce transit-like signals that mimic planetary transits.

    Pass Criteria:
    - No known EB found within search radius (default 42 arcsec = 2 TESS pixels)
    - If EB found, fails if period matches candidate within 10%

    Confidence:
    - 0.95 if no EB found and query succeeded
    - 0.0 if network error (returns PASS with low confidence)
    """

    id: ClassVar[str] = "V06"
    name: ClassVar[str] = "nearby_eb_search"

    def __init__(
        self,
        config: CheckConfig | None = None,
        search_radius_arcsec: float = 42.0,
        period_tolerance: float = 0.1,
    ) -> None:
        """Initialize NearbyEBCheck.

        Args:
            config: Optional check configuration
            search_radius_arcsec: Cone search radius (default 42 = 2 TESS pixels)
            period_tolerance: Fractional tolerance for period match (default 0.1 = 10%)
        """
        super().__init__(config)
        self.search_radius_arcsec = search_radius_arcsec
        self.period_tolerance = period_tolerance

    @classmethod
    def _default_config(cls) -> CheckConfig:
        """Default configuration for nearby EB search."""
        return CheckConfig(
            enabled=True,
            threshold=42.0,  # arcsec search radius
            additional={
                "period_tolerance": 0.1,
                "catalog": "J/ApJS/258/16/tess-ebs",
            },
        )

    def run(
        self,
        candidate: TransitCandidate,
        lightcurve: LightCurveData | None = None,
        stellar: StellarParameters | None = None,
        *,
        ra: float | None = None,
        dec: float | None = None,
        tic_id: int | None = None,
    ) -> VetterCheckResult:
        """Search for nearby EBs in TESS-EB catalog.

        Args:
            candidate: Transit candidate with period
            lightcurve: Not used for this check
            stellar: Not used for this check
            ra: Target RA in degrees (required)
            dec: Target Dec in degrees (required)
            tic_id: Optional TIC ID for logging

        Returns:
            VetterCheckResult with nearby EB search results
        """
        # Check required parameters
        if ra is None or dec is None:
            return make_result(
                self.id,
                passed=True,
                confidence=0.0,
                details={
                    "reason": "no_coordinates",
                    "note": "RA/Dec required for nearby EB search",
                },
            )

        # Query VizieR TESS-EB catalog
        try:
            ebs = self._query_tess_eb_catalog(ra, dec, self.search_radius_arcsec)
        except Exception as e:
            logger.warning(f"V06: Failed to query TESS-EB catalog: {e}")
            return make_result(
                self.id,
                passed=True,
                confidence=0.0,
                details={
                    "reason": "query_failed",
                    "error": str(e),
                    "note": "Network error querying TESS-EB catalog",
                },
            )

        # No EBs found - pass with high confidence
        if not ebs:
            return make_result(
                self.id,
                passed=True,
                confidence=0.95,
                details={
                    "n_ebs_found": 0,
                    "search_radius_arcsec": self.search_radius_arcsec,
                    "ra": ra,
                    "dec": dec,
                    "note": "No known EBs found within search radius",
                },
            )

        # EBs found - check for period match
        candidate_period = candidate.period
        closest_eb = min(ebs, key=lambda x: x.get("separation_arcsec", float("inf")))

        # Check if any EB period matches candidate period (or 2x period)
        period_match = False
        matching_eb = None
        for eb in ebs:
            eb_period = eb.get("period")
            if eb_period is not None and candidate_period > 0:
                # Check if periods match within tolerance
                ratio = eb_period / candidate_period
                if abs(ratio - 1.0) < self.period_tolerance:
                    period_match = True
                    matching_eb = eb
                    break
                # Also check 2x period (EB at half the candidate period)
                if abs(ratio - 2.0) < self.period_tolerance:
                    period_match = True
                    matching_eb = eb
                    break
                # Check half period (candidate at 2x EB period)
                if abs(ratio - 0.5) < self.period_tolerance:
                    period_match = True
                    matching_eb = eb
                    break

        # Build result
        passed = not period_match  # Fail if period matches
        confidence = 0.90 if not period_match else 0.95

        details: dict[str, Any] = {
            "n_ebs_found": len(ebs),
            "search_radius_arcsec": self.search_radius_arcsec,
            "closest_eb_arcsec": closest_eb.get("separation_arcsec"),
            "closest_eb_tic": closest_eb.get("tic_id"),
            "closest_eb_period": closest_eb.get("period"),
            "period_match": period_match,
            "candidate_period": candidate_period,
            "ra": ra,
            "dec": dec,
        }

        if period_match and matching_eb:
            details["matching_eb_tic"] = matching_eb.get("tic_id")
            details["matching_eb_period"] = matching_eb.get("period")
            details["note"] = (
                f"Known EB found with matching period "
                f'(P={matching_eb.get("period"):.4f}d, sep={matching_eb.get("separation_arcsec"):.1f}")'
            )
        else:
            details["note"] = f"Found {len(ebs)} EB(s) but no period match with candidate"

        return make_result(
            self.id,
            passed=passed,
            confidence=confidence,
            details=details,
        )

    def _query_tess_eb_catalog(
        self, ra: float, dec: float, radius_arcsec: float
    ) -> list[dict[str, Any]]:
        """Query TESS-EB catalog via VizieR.

        Args:
            ra: RA in degrees
            dec: Dec in degrees
            radius_arcsec: Search radius in arcsec

        Returns:
            List of EB dictionaries with tic_id, period, separation_arcsec
        """
        # Build VizieR TAP query URL
        # Using simple cone search format
        params = {
            "-source": "J/ApJS/258/16/tess-ebs",
            "-c": f"{ra},{dec}",
            "-c.rs": str(radius_arcsec),
            "-out": "TIC,Per,_r",  # TIC ID, Period, angular distance
            "-out.max": "100",
        }

        response = requests.get(
            VIZIER_TAP_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        # Parse VOTable response
        ebs = self._parse_votable_response(response.text)
        return ebs

    def _parse_votable_response(self, votable_text: str) -> list[dict[str, Any]]:
        """Parse VizieR VOTable response.

        Args:
            votable_text: VOTable XML text

        Returns:
            List of EB dictionaries
        """
        ebs: list[dict[str, Any]] = []

        # Simple XML parsing for TABLEDATA
        # Look for <TR><TD>...</TD></TR> patterns
        import re

        # Find all table rows
        row_pattern = re.compile(r"<TR>(.*?)</TR>", re.DOTALL)
        td_pattern = re.compile(r"<TD[^>]*>(.*?)</TD>", re.DOTALL)

        for row_match in row_pattern.finditer(votable_text):
            row_text = row_match.group(1)
            cells = td_pattern.findall(row_text)

            if len(cells) >= 3:
                try:
                    tic_id_str = cells[0].strip()
                    period_str = cells[1].strip()
                    sep_str = cells[2].strip()

                    eb: dict[str, Any] = {}

                    if tic_id_str:
                        eb["tic_id"] = int(float(tic_id_str))
                    if period_str:
                        eb["period"] = float(period_str)
                    if sep_str:
                        eb["separation_arcsec"] = float(sep_str)

                    if eb:
                        ebs.append(eb)
                except (ValueError, IndexError):
                    continue

        return ebs


# =============================================================================
# V07: ExoFOP Disposition Check
# =============================================================================


class ExoFOPDispositionCheck(VetterCheck):
    """V07: Check ExoFOP-TESS for existing dispositions.

    Queries ExoFOP-TESS API to check if the target already has a
    TFOPWG disposition flagging it as a false positive.

    Pass Criteria:
    - No FP disposition in ExoFOP (FP, FA, EB, NEB, BEB, V, IS, O)
    - Pass if no ExoFOP entry exists (not yet vetted)
    - Pass if disposition is planet candidate or confirmed (CP, KP, PC, APC)

    Confidence:
    - 0.95 for known FP (confident fail)
    - 0.90 for confirmed planet (confident pass)
    - 0.50 for no disposition/not found
    """

    id: ClassVar[str] = "V07"
    name: ClassVar[str] = "exofop_disposition"

    @classmethod
    def _default_config(cls) -> CheckConfig:
        """Default configuration for ExoFOP check."""
        return CheckConfig(
            enabled=True,
            threshold=None,
            additional={
                "fp_dispositions": list(FP_DISPOSITIONS),
                "planet_dispositions": list(PLANET_DISPOSITIONS),
            },
        )

    def run(
        self,
        candidate: TransitCandidate,
        lightcurve: LightCurveData | None = None,
        stellar: StellarParameters | None = None,
        *,
        tic_id: int | None = None,
        toi: float | None = None,
    ) -> VetterCheckResult:
        """Check ExoFOP for existing disposition.

        Args:
            candidate: Transit candidate (period used for TOI matching)
            lightcurve: Not used for this check
            stellar: Not used for this check
            tic_id: TIC ID to query (required)
            toi: Optional specific TOI number (e.g., 123.01)

        Returns:
            VetterCheckResult with ExoFOP disposition results
        """
        if tic_id is None:
            return make_result(
                self.id,
                passed=True,
                confidence=0.0,
                details={
                    "reason": "no_tic_id",
                    "note": "TIC ID required for ExoFOP disposition check",
                },
            )

        # Query ExoFOP
        try:
            toi_entries = self._query_exofop(tic_id)
        except Exception as e:
            logger.warning(f"V07: Failed to query ExoFOP: {e}")
            return make_result(
                self.id,
                passed=True,
                confidence=0.0,
                details={
                    "reason": "query_failed",
                    "error": str(e),
                    "tic_id": tic_id,
                    "note": "Network error querying ExoFOP",
                },
            )

        # No TOI entries found - pass with moderate confidence
        if not toi_entries:
            return make_result(
                self.id,
                passed=True,
                confidence=0.50,
                details={
                    "tic_id": tic_id,
                    "n_toi_entries": 0,
                    "disposition": None,
                    "note": "No TOI entry found in ExoFOP",
                },
            )

        # Find matching TOI entry (by TOI number or closest period)
        matching_entry = self._find_matching_entry(toi_entries, candidate.period, toi)

        if matching_entry is None:
            return make_result(
                self.id,
                passed=True,
                confidence=0.50,
                details={
                    "tic_id": tic_id,
                    "n_toi_entries": len(toi_entries),
                    "disposition": None,
                    "note": "No matching TOI entry for this candidate period",
                },
            )

        # Check disposition
        disposition = matching_entry.get("disposition", "").strip().upper()
        toi_number = matching_entry.get("toi")
        comments = matching_entry.get("comments")

        # Determine pass/fail based on disposition
        is_fp = disposition in FP_DISPOSITIONS
        is_planet = disposition in PLANET_DISPOSITIONS

        if is_fp:
            passed = False
            confidence = 0.95
            note = f"ExoFOP disposition is {disposition} (false positive)"
        elif is_planet:
            passed = True
            confidence = 0.90
            note = f"ExoFOP disposition is {disposition} (confirmed/candidate planet)"
        else:
            passed = True
            confidence = 0.50
            note = f"ExoFOP disposition is '{disposition}' (not classified as FP)"

        return make_result(
            self.id,
            passed=passed,
            confidence=confidence,
            details={
                "tic_id": tic_id,
                "toi_number": toi_number,
                "disposition": disposition if disposition else None,
                "comments": comments,
                "is_known_fp": is_fp,
                "is_known_planet": is_planet,
                "n_toi_entries": len(toi_entries),
                "note": note,
            },
        )

    def _query_exofop(self, tic_id: int) -> list[dict[str, Any]]:
        """Query ExoFOP-TESS for TOI entries.

        Args:
            tic_id: TIC ID to query

        Returns:
            List of TOI entry dictionaries
        """
        # ExoFOP TOI table query
        params = {
            "sort": "toi",
            "output": "pipe",
        }

        response = requests.get(
            EXOFOP_TOI_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        # Parse pipe-delimited response
        entries = self._parse_exofop_response(response.text, tic_id)
        return entries

    def _parse_exofop_response(
        self, response_text: str, target_tic_id: int
    ) -> list[dict[str, Any]]:
        """Parse ExoFOP pipe-delimited response.

        Args:
            response_text: Pipe-delimited text
            target_tic_id: TIC ID to filter for

        Returns:
            List of TOI entries matching target TIC ID
        """
        entries: list[dict[str, Any]] = []
        lines = response_text.strip().split("\n")

        if not lines:
            return entries

        # First line is header
        header = lines[0].split("|")
        header_map = {col.strip().lower(): i for i, col in enumerate(header)}

        # Find column indices
        tic_idx = header_map.get("tic id", header_map.get("tic", -1))
        toi_idx = header_map.get("toi", -1)
        disp_idx = header_map.get("tfopwg disposition", header_map.get("disposition", -1))
        period_idx = header_map.get("period (days)", header_map.get("period", -1))
        comments_idx = header_map.get("comments", -1)

        for line in lines[1:]:
            cols = line.split("|")
            if len(cols) <= max(tic_idx, toi_idx, disp_idx):
                continue

            try:
                tic_str = cols[tic_idx].strip() if tic_idx >= 0 else ""
                if not tic_str:
                    continue
                tic_id = int(tic_str)

                if tic_id != target_tic_id:
                    continue

                entry: dict[str, Any] = {"tic_id": tic_id}

                if toi_idx >= 0 and toi_idx < len(cols):
                    toi_str = cols[toi_idx].strip()
                    if toi_str:
                        entry["toi"] = float(toi_str)

                if disp_idx >= 0 and disp_idx < len(cols):
                    entry["disposition"] = cols[disp_idx].strip()

                if period_idx >= 0 and period_idx < len(cols):
                    period_str = cols[period_idx].strip()
                    if period_str:
                        with contextlib.suppress(ValueError):
                            entry["period"] = float(period_str)

                if comments_idx >= 0 and comments_idx < len(cols):
                    entry["comments"] = cols[comments_idx].strip()

                entries.append(entry)

            except (ValueError, IndexError):
                continue

        return entries

    def _find_matching_entry(
        self,
        entries: list[dict[str, Any]],
        candidate_period: float,
        toi: float | None,
    ) -> dict[str, Any] | None:
        """Find TOI entry matching candidate.

        Args:
            entries: List of TOI entries
            candidate_period: Candidate orbital period in days
            toi: Optional specific TOI number

        Returns:
            Matching entry or None
        """
        # If specific TOI provided, find exact match
        if toi is not None:
            for entry in entries:
                if entry.get("toi") == toi:
                    return entry

        # Otherwise find closest period match
        best_match = None
        best_diff = float("inf")

        for entry in entries:
            entry_period = entry.get("period")
            if entry_period is not None and candidate_period > 0:
                diff = abs(entry_period - candidate_period) / candidate_period
                if diff < best_diff:
                    best_diff = diff
                    best_match = entry

        # Only return if reasonably close (within 20%)
        if best_match is not None and best_diff < 0.2:
            return best_match

        # Return first entry if no period match
        return entries[0] if entries else None


# =============================================================================
# Convenience Functions
# =============================================================================


def check_nearby_eb(
    ra: float,
    dec: float,
    candidate_period: float,
    search_radius_arcsec: float = 42.0,
    period_tolerance: float = 0.1,
) -> VetterCheckResult:
    """Convenience function for V06 nearby EB check.

    Args:
        ra: Target RA in degrees
        dec: Target Dec in degrees
        candidate_period: Transit candidate period in days
        search_radius_arcsec: Search radius (default 42")
        period_tolerance: Period match tolerance (default 10%)

    Returns:
        VetterCheckResult
    """
    from bittr_tess_vetter.domain.detection import TransitCandidate

    # Create minimal candidate for period
    candidate = TransitCandidate(
        period=candidate_period,
        t0=0.0,
        duration_hours=3.0,
        depth=0.001,
        snr=10.0,
    )

    check = NearbyEBCheck(
        search_radius_arcsec=search_radius_arcsec,
        period_tolerance=period_tolerance,
    )
    return check.run(candidate, ra=ra, dec=dec)


def check_exofop_disposition(
    tic_id: int,
    candidate_period: float,
    toi: float | None = None,
) -> VetterCheckResult:
    """Convenience function for V07 ExoFOP disposition check.

    Args:
        tic_id: TIC ID to query
        candidate_period: Transit candidate period in days
        toi: Optional specific TOI number

    Returns:
        VetterCheckResult
    """
    from bittr_tess_vetter.domain.detection import TransitCandidate

    # Create minimal candidate for period
    candidate = TransitCandidate(
        period=candidate_period,
        t0=0.0,
        duration_hours=3.0,
        depth=0.001,
        snr=10.0,
    )

    check = ExoFOPDispositionCheck()
    return check.run(candidate, tic_id=tic_id, toi=toi)


def run_catalog_prefilters(
    tic_id: int,
    ra: float,
    dec: float,
    candidate_period: float,
    toi: float | None = None,
) -> tuple[VetterCheckResult, VetterCheckResult]:
    """Run both catalog pre-filter checks (V06 + V07).

    These fast checks can quickly reject known false positives before
    running expensive TRICERATOPS FPP calculation.

    Args:
        tic_id: TIC ID
        ra: Target RA in degrees
        dec: Target Dec in degrees
        candidate_period: Transit candidate period in days
        toi: Optional specific TOI number

    Returns:
        Tuple of (V06 result, V07 result)
    """
    v06_result = check_nearby_eb(ra, dec, candidate_period)
    v07_result = check_exofop_disposition(tic_id, candidate_period, toi)
    return v06_result, v07_result
