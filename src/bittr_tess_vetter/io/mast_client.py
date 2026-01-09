"""MAST client wrapper for TESS light curve queries.

This module provides MASTClient, a wrapper around lightkurve for querying
and downloading TESS light curves from MAST (Mikulski Archive for Space Telescopes).

Usage:
    from bittr_tess_vetter.io import MASTClient

    client = MASTClient()
    results = client.search_lightcurve(tic_id=261136679)
    lc_data = client.download_lightcurve(tic_id=261136679, sector=1)
    target = client.get_target_info(tic_id=261136679)
"""

from __future__ import annotations

import logging
import time as time_module
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from bittr_tess_vetter.api.lightcurve import LightCurveData
from bittr_tess_vetter.api.target import Target

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DownloadPhase(Enum):
    """Phases of the light curve download process."""

    SEARCHING = "searching"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    CACHING = "caching"
    COMPLETE = "complete"


@dataclass
class DownloadProgress:
    """Progress information for light curve downloads.

    This dataclass provides detailed progress information during long-running
    download operations. It can be used to update UI progress bars, log status,
    or send MCP progress notifications.

    Attributes:
        phase: Current phase of the download process
        current_step: Current step number (1-indexed)
        total_steps: Total number of steps in the operation
        percentage: Overall progress percentage (0-100)
        message: Human-readable status message
        sector: Sector being downloaded (if applicable)
        tic_id: TIC ID of the target
        elapsed_seconds: Time elapsed since operation started
        estimated_remaining_seconds: Estimated time remaining (None if unknown)
        bytes_downloaded: Bytes downloaded so far (if available)
        sectors_completed: List of sectors already downloaded (for batch operations)
        sectors_remaining: List of sectors still to download (for batch operations)

    Example:
        >>> def my_callback(progress: DownloadProgress) -> None:
        ...     print(f"{progress.percentage:.0f}% - {progress.message}")
        >>> client = MASTClient()
        >>> lc = client.download_lightcurve(
        ...     tic_id=261136679,
        ...     sector=1,
        ...     progress_callback=my_callback,
        ... )
    """

    phase: DownloadPhase
    current_step: int
    total_steps: int
    percentage: float
    message: str
    tic_id: int
    sector: int | None = None
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: float | None = None
    bytes_downloaded: int | None = None
    sectors_completed: list[int] = field(default_factory=list)
    sectors_remaining: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "phase": self.phase.value,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "percentage": round(self.percentage, 1),
            "message": self.message,
            "tic_id": self.tic_id,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
        }
        if self.sector is not None:
            result["sector"] = self.sector
        if self.estimated_remaining_seconds is not None:
            result["estimated_remaining_seconds"] = round(self.estimated_remaining_seconds, 1)
        if self.bytes_downloaded is not None:
            result["bytes_downloaded"] = self.bytes_downloaded
        if self.sectors_completed:
            result["sectors_completed"] = self.sectors_completed
        if self.sectors_remaining:
            result["sectors_remaining"] = self.sectors_remaining
        return result


class ProgressCallback(Protocol):
    """Protocol for progress callback functions.

    Implementations can log progress, update UI, or send MCP notifications.
    """

    def __call__(self, progress: DownloadProgress) -> None:
        """Handle progress update.

        Args:
            progress: Current progress information
        """
        ...


# TESS quality flag definitions (from TESS Data Release Notes)
# Reference: https://outerspace.stsci.edu/display/TESS/TESS+Pipeline
QUALITY_FLAG_BITS = {
    1: "attitude_tweak",  # Attitude tweak
    2: "safe_mode",  # Safe mode
    4: "coarse_point",  # Coarse point
    8: "earth_point",  # Earth point
    16: "argabrightening",  # Argabrightening event
    32: "desaturation",  # Reaction wheel desaturation
    64: "manual_exclude",  # Manual exclude
    128: "discontinuity",  # Discontinuity corrected
    256: "impulsive_outlier",  # Impulsive outlier
    512: "straylight",  # Stray light
    1024: "crowded_field",  # Crowded aperture (SPOC)
    2048: "scattered_light",  # Scattered light artifact
}

# Default quality mask: exclude severe issues
DEFAULT_QUALITY_MASK = (
    1  # Attitude tweak
    | 2  # Safe mode
    | 4  # Coarse point
    | 8  # Earth point
    | 32  # Desaturation
    | 128  # Discontinuity
    | 256  # Impulsive outlier
)


@dataclass
class SearchResult:
    """Result from a light curve search.

    Attributes:
        tic_id: TIC identifier
        sector: TESS sector number
        author: Pipeline that produced the light curve (e.g., "SPOC", "QLP")
        exptime: Exposure time in seconds
        mission: Mission name (always "TESS")
        distance: Angular distance from query position (arcsec), if applicable
    """

    tic_id: int
    sector: int
    author: str
    exptime: float
    mission: str = "TESS"
    distance: float | None = None


class MASTClientError(Exception):
    """Base exception for MAST client errors."""

    pass


class LightCurveNotFoundError(MASTClientError):
    """Raised when no light curve is found for the specified parameters."""

    pass


class TargetNotFoundError(MASTClientError):
    """Raised when target information cannot be retrieved."""

    pass


class NameResolutionError(MASTClientError):
    """Raised when target name resolution fails."""

    pass


@dataclass
class ResolvedTarget:
    """Result of target name resolution.

    Attributes:
        tic_id: Resolved TESS Input Catalog identifier
        target_input: Original target string provided by user
        separation_arcsec: Angular separation from query position (arcsec)
        ra: Right ascension of resolved target (degrees)
        dec: Declination of resolved target (degrees)
        tmag: TESS magnitude of resolved target
        warning: Optional warning message (e.g., for ambiguous matches)
    """

    tic_id: int
    target_input: str
    separation_arcsec: float
    ra: float | None = None
    dec: float | None = None
    tmag: float | None = None
    warning: str | None = None


class MASTClient:
    """Client for querying TESS light curves from MAST via lightkurve.

    This class wraps the lightkurve library to provide a clean interface
    for searching and downloading TESS light curves, converting them to
    our internal domain models.

    Attributes:
        quality_mask: Bitmask for filtering bad quality data points.
            Default excludes severe issues but keeps minor flags.
        author: Preferred data author/pipeline. Default is "SPOC" for
            Science Processing Operations Center products.
        normalize: Whether to normalize flux to median ~1.0. Default True.

    Example:
        >>> client = MASTClient()
        >>> results = client.search_lightcurve(261136679)
        >>> for r in results:
        ...     print(f"Sector {r.sector}: {r.author}, {r.exptime}s cadence")
        >>> lc = client.download_lightcurve(261136679, sector=1)
        >>> print(f"Downloaded {lc.n_points} points")
    """

    def __init__(
        self,
        quality_mask: int = DEFAULT_QUALITY_MASK,
        author: str | None = "SPOC",
        normalize: bool = True,
    ) -> None:
        """Initialize MAST client.

        Args:
            quality_mask: Bitmask for quality flag filtering.
                Points with quality & quality_mask != 0 are marked invalid.
            author: Preferred pipeline author (SPOC, QLP, TESS-SPOC, etc.)
                Set to None to search all authors.
            normalize: Whether to normalize flux to median ~1.0.
        """
        self.quality_mask = quality_mask
        self.author = author
        self.normalize = normalize
        self._lk_imported = False

    def _ensure_lightkurve(self) -> Any:
        """Lazy import of lightkurve to avoid import-time overhead."""
        if not self._lk_imported:
            try:
                import lightkurve as lk

                self._lk = lk
                self._lk_imported = True
            except ImportError as e:
                raise MASTClientError(
                    "lightkurve is required for MAST queries. Install with: pip install lightkurve"
                ) from e
        return self._lk

    def search_lightcurve(
        self,
        tic_id: int,
        sector: int | None = None,
        author: str | None = None,
    ) -> list[SearchResult]:
        """Search for available TESS light curves for a target.

        Args:
            tic_id: TESS Input Catalog identifier
            sector: Specific sector to search (None for all sectors)
            author: Override default author filter (None uses instance default)

        Returns:
            List of SearchResult objects describing available light curves,
            sorted by sector number.

        Raises:
            MASTClientError: If the search fails due to network or API errors.

        Example:
            >>> client = MASTClient()
            >>> results = client.search_lightcurve(261136679)
            >>> print(f"Found {len(results)} light curves")
        """
        lk = self._ensure_lightkurve()

        search_author = author if author is not None else self.author
        target = f"TIC {tic_id}"

        logger.info(f"Searching MAST for {target}, sector={sector}, author={search_author}")

        try:
            search_result = lk.search_lightcurve(
                target,
                mission="TESS",
                sector=sector,
                author=search_author,
            )
        except Exception as e:
            logger.error(f"MAST search failed for {target}: {e}")
            raise MASTClientError(f"Failed to search MAST for TIC {tic_id}: {e}") from e

        if search_result is None or len(search_result) == 0:
            logger.info(f"No light curves found for {target}")
            return []

        results = []
        for i in range(len(search_result)):
            row = search_result[i]
            try:
                # Extract sector from observation metadata
                # lightkurve stores this in the 'sequence_number' field for TESS
                sector_num = int(row.mission[0].split()[-1]) if hasattr(row, "mission") else 0
                if hasattr(row, "sequence_number"):
                    sector_num = int(row.sequence_number)

                # Handle astropy Quantity objects (have .value attribute)
                # Values may be numpy arrays, so use .item() or index to get scalar
                def _to_float(val: object, default: float = 0.0) -> float:
                    """Convert astropy Quantity or numpy array to float."""
                    if val is None:
                        return default
                    # Get underlying value from Quantity
                    if hasattr(val, "value"):
                        val = val.value
                    # Handle numpy arrays
                    if hasattr(val, "item"):
                        return float(val.item())
                    if hasattr(val, "__getitem__") and hasattr(val, "__len__"):
                        return float(val[0]) if len(val) > 0 else default
                    return float(val)

                exptime_val = _to_float(row.exptime, 120.0) if hasattr(row, "exptime") else 120.0

                distance_val = None
                if hasattr(row, "distance") and row.distance is not None:
                    distance_val = _to_float(row.distance)

                result = SearchResult(
                    tic_id=tic_id,
                    sector=sector_num,
                    author=str(row.author) if hasattr(row, "author") else "Unknown",
                    exptime=exptime_val,
                    mission="TESS",
                    distance=distance_val,
                )
                results.append(result)
            except (ValueError, AttributeError, IndexError) as e:
                logger.warning(f"Failed to parse search result row {i}: {e}")
                continue

        # Sort by sector
        results.sort(key=lambda r: r.sector)
        logger.info(f"Found {len(results)} light curves for {target}")

        return results

    def download_lightcurve(
        self,
        tic_id: int,
        sector: int,
        flux_type: str = "pdcsap",
        quality_mask: int | None = None,
        exptime: float | None = None,
        author: str | None = None,
        progress_callback: Callable[[DownloadProgress], None] | None = None,
    ) -> LightCurveData:
        """Download and process a TESS light curve.

        Args:
            tic_id: TESS Input Catalog identifier
            sector: TESS sector number
            flux_type: Type of flux to use ("pdcsap" or "sap")
                - pdcsap: Pre-search Data Conditioning Simple Aperture Photometry
                  (systematics-corrected, recommended for transit searches)
                - sap: Simple Aperture Photometry (raw photometry)
            quality_mask: Override default quality mask for this download.
                Points with quality & mask != 0 are marked invalid.
            exptime: Exposure time in seconds (e.g., 20 or 120). If provided,
                filters search results to only include products with matching
                exposure time. If None, uses the first available product.
            author: Override default author filter (None uses instance default).
            progress_callback: Optional callback function to receive progress updates.
                The callback receives a DownloadProgress object with current status.
                Useful for long-running downloads to provide user feedback.

        Returns:
            LightCurveData object with normalized flux (median ~1.0)

        Raises:
            LightCurveNotFoundError: If no light curve exists for the parameters.
            MASTClientError: If download fails due to network or processing errors.
            ValueError: If flux_type is not "pdcsap" or "sap".

        Example:
            >>> client = MASTClient()
            >>> lc = client.download_lightcurve(261136679, sector=1)
            >>> print(f"Duration: {lc.duration_days:.1f} days")
            >>> # Request 20-second cadence specifically
            >>> lc_fast = client.download_lightcurve(261136679, sector=1, exptime=20)
            >>> # With progress reporting
            >>> def show_progress(p: DownloadProgress) -> None:
            ...     print(f"[{p.percentage:.0f}%] {p.message}")
            >>> lc = client.download_lightcurve(261136679, sector=1, progress_callback=show_progress)
        """
        if flux_type not in ("pdcsap", "sap"):
            raise ValueError(f"flux_type must be 'pdcsap' or 'sap', got '{flux_type}'")

        lk = self._ensure_lightkurve()
        mask = quality_mask if quality_mask is not None else self.quality_mask
        target = f"TIC {tic_id}"

        # Progress tracking setup
        start_time = time_module.monotonic()
        total_steps = 4  # search, download, process, complete

        def _report_progress(
            phase: DownloadPhase,
            step: int,
            message: str,
        ) -> None:
            """Report progress if callback is provided."""
            if progress_callback is None:
                return
            elapsed = time_module.monotonic() - start_time
            # Estimate remaining time based on current progress
            if step > 0:
                avg_time_per_step = elapsed / step
                remaining_steps = total_steps - step
                est_remaining = avg_time_per_step * remaining_steps
            else:
                est_remaining = None
            progress = DownloadProgress(
                phase=phase,
                current_step=step,
                total_steps=total_steps,
                percentage=(step / total_steps) * 100.0,
                message=message,
                tic_id=tic_id,
                sector=sector,
                elapsed_seconds=elapsed,
                estimated_remaining_seconds=est_remaining,
            )
            progress_callback(progress)

        exptime_str = f", exptime={exptime}s" if exptime is not None else ""
        logger.info(
            f"Downloading light curve for {target}, sector {sector}, flux={flux_type}{exptime_str}"
        )

        # Step 1: Search for the specific sector
        _report_progress(
            DownloadPhase.SEARCHING,
            1,
            f"Searching MAST for TIC {tic_id} sector {sector}...",
        )
        try:
            search_author = author if author is not None else self.author
            search_result = lk.search_lightcurve(
                target,
                mission="TESS",
                sector=sector,
                author=search_author,
            )
        except Exception as e:
            logger.error(f"MAST search failed for {target} sector {sector}: {e}")
            raise MASTClientError(
                f"Failed to search MAST for TIC {tic_id} sector {sector}: {e}"
            ) from e

        if search_result is None or len(search_result) == 0:
            raise LightCurveNotFoundError(f"No light curve found for TIC {tic_id} sector {sector}")

        # Filter by exptime if specified
        if exptime is not None:
            # Filter the search results to find matching exptime
            matching_indices = []
            for i in range(len(search_result)):
                row = search_result[i]
                if hasattr(row, "exptime"):
                    row_exptime = row.exptime
                    # Handle astropy Quantity
                    if hasattr(row_exptime, "value"):
                        row_exptime = row_exptime.value
                    # Handle numpy array
                    if hasattr(row_exptime, "item"):
                        row_exptime = row_exptime.item()
                    elif hasattr(row_exptime, "__getitem__") and hasattr(row_exptime, "__len__"):
                        if len(row_exptime) > 0:
                            row_exptime = row_exptime[0]
                    # Check if exptime matches (with 1s tolerance)
                    if abs(float(row_exptime) - exptime) < 1.0:
                        matching_indices.append(i)

            if not matching_indices:
                available_exptimes = []
                for i in range(len(search_result)):
                    row = search_result[i]
                    if hasattr(row, "exptime"):
                        row_exptime = row.exptime
                        if hasattr(row_exptime, "value"):
                            row_exptime = row_exptime.value
                        if hasattr(row_exptime, "item"):
                            row_exptime = row_exptime.item()
                        elif hasattr(row_exptime, "__getitem__") and hasattr(
                            row_exptime, "__len__"
                        ):
                            if len(row_exptime) > 0:
                                row_exptime = row_exptime[0]
                        available_exptimes.append(float(row_exptime))
                raise LightCurveNotFoundError(
                    f"No light curve found for TIC {tic_id} sector {sector} "
                    f"with exptime={exptime}s. Available exptimes: {available_exptimes}"
                )

            # Use only matching results
            search_result = search_result[matching_indices[0]]

        # Step 2: Download the light curve
        _report_progress(
            DownloadPhase.DOWNLOADING,
            2,
            f"Downloading light curve for TIC {tic_id} sector {sector}...",
        )
        try:
            lc_result = search_result.download()
            # Check if result is a LightCurveCollection (multiple light curves)
            # vs a single LightCurve. Don't use len() - LightCurve has len = n_points
            if type(lc_result).__name__ == "LightCurveCollection":
                lc = lc_result[0]  # Take first light curve from collection
            else:
                lc = lc_result  # Single light curve
        except Exception as e:
            logger.error(f"Download failed for {target} sector {sector}: {e}")
            raise MASTClientError(
                f"Failed to download light curve for TIC {tic_id} sector {sector}: {e}"
            ) from e

        # Step 3: Process the light curve data
        _report_progress(
            DownloadPhase.PROCESSING,
            3,
            f"Processing light curve data for TIC {tic_id} sector {sector}...",
        )

        # Select flux column
        flux_col = "pdcsap_flux" if flux_type == "pdcsap" else "sap_flux"
        flux_err_col = f"{flux_col}_err"

        # Extract data arrays
        try:
            time = np.asarray(lc.time.value, dtype=np.float64)

            # Get flux and flux_err from the appropriate column
            if hasattr(lc, flux_col):
                flux_raw = np.asarray(getattr(lc, flux_col).value, dtype=np.float64)
            else:
                # Fallback to default flux
                flux_raw = np.asarray(lc.flux.value, dtype=np.float64)

            if hasattr(lc, flux_err_col):
                flux_err_raw = np.asarray(getattr(lc, flux_err_col).value, dtype=np.float64)
            elif hasattr(lc, "flux_err"):
                flux_err_raw = np.asarray(lc.flux_err.value, dtype=np.float64)
            else:
                # If no errors available, use small placeholder
                flux_err_raw = np.ones_like(flux_raw) * 1e-4

            # Quality flags
            if hasattr(lc, "quality"):
                quality = np.asarray(lc.quality, dtype=np.int32)
            else:
                quality = np.zeros(len(time), dtype=np.int32)

        except Exception as e:
            logger.error(f"Failed to extract data from light curve: {e}")
            raise MASTClientError(f"Failed to process light curve data: {e}") from e

        # Build validity mask
        # Invalid if: NaN in time/flux, or quality flag set
        valid_mask = (
            ~np.isnan(time)
            & ~np.isnan(flux_raw)
            & ~np.isnan(flux_err_raw)
            & ((quality & mask) == 0)
        ).astype(np.bool_)

        # Normalize flux to median ~1.0
        if self.normalize and np.any(valid_mask):
            median_flux = np.median(flux_raw[valid_mask])
            if median_flux > 0 and np.isfinite(median_flux):
                flux = flux_raw / median_flux
                flux_err = flux_err_raw / median_flux
            else:
                logger.warning(f"Invalid median flux {median_flux}, skipping normalization")
                flux = flux_raw
                flux_err = flux_err_raw
        else:
            flux = flux_raw
            flux_err = flux_err_raw

        # Determine cadence from time differences
        if len(time) > 1:
            # Use median time difference to determine cadence
            dt = np.diff(time[valid_mask]) if np.any(valid_mask) else np.diff(time)
            cadence_days = float(np.median(dt)) if len(dt) > 0 else 0.0
            cadence_seconds = cadence_days * 86400.0
        else:
            cadence_seconds = 120.0  # Default to 2-min cadence

        logger.info(
            f"Downloaded {len(time)} points ({np.sum(valid_mask)} valid), "
            f"cadence={cadence_seconds:.0f}s"
        )

        # Step 4: Complete
        _report_progress(
            DownloadPhase.COMPLETE,
            4,
            f"Download complete: {len(time)} points for TIC {tic_id} sector {sector}",
        )

        return LightCurveData(
            time=time,
            flux=flux,
            flux_err=flux_err,
            quality=quality,
            valid_mask=valid_mask,
            tic_id=tic_id,
            sector=sector,
            cadence_seconds=cadence_seconds,
        )

    def get_target_info(self, tic_id: int) -> Target:
        """Retrieve target information from the TESS Input Catalog.

        Args:
            tic_id: TESS Input Catalog identifier

        Returns:
            Target object with stellar parameters and cross-match IDs.

        Raises:
            TargetNotFoundError: If target is not found in TIC.
            MASTClientError: If query fails due to network or API errors.

        Example:
            >>> client = MASTClient()
            >>> target = client.get_target_info(261136679)
            >>> print(f"Teff={target.stellar.teff}K, R={target.stellar.radius}Rsun")
        """
        self._ensure_lightkurve()  # Ensure lightkurve is available

        logger.info(f"Querying TIC for target {tic_id}")

        try:
            # lightkurve provides access to TIC via search
            # We can also use astroquery directly for more detailed info
            from astroquery.mast import Catalogs

            result = Catalogs.query_criteria(catalog="TIC", ID=tic_id)  # pyright: ignore[reportAttributeAccessIssue]

            if result is None or len(result) == 0:
                raise TargetNotFoundError(f"TIC {tic_id} not found in catalog")

            row = result[0]

            # Convert to dict for Target.from_tic_response
            tic_data = {
                "Teff": self._get_float(row, "Teff"),
                "logg": self._get_float(row, "logg"),
                "rad": self._get_float(row, "rad"),
                "mass": self._get_float(row, "mass"),
                "Tmag": self._get_float(row, "Tmag"),
                "contratio": self._get_float(row, "contratio"),
                "lum": self._get_float(row, "lum"),
                "MH": self._get_float(row, "MH"),
                "ra": self._get_float(row, "ra"),
                "dec": self._get_float(row, "dec"),
                "pmRA": self._get_float(row, "pmRA"),
                "pmDEC": self._get_float(row, "pmDEC"),
                "d": self._get_float(row, "d"),
                "GAIA": self._get_int(row, "GAIA"),
                "TWOMASS": self._get_str(row, "TWOMASS"),
            }

            logger.info(f"Retrieved TIC data for {tic_id}")
            return Target.from_tic_response(tic_id, tic_data)

        except TargetNotFoundError:
            raise
        except ImportError as e:
            logger.error(f"astroquery not available: {e}")
            raise MASTClientError(
                "astroquery is required for TIC queries. Install with: pip install astroquery"
            ) from e
        except Exception as e:
            logger.error(f"TIC query failed for {tic_id}: {e}")
            raise MASTClientError(f"Failed to query TIC for {tic_id}: {e}") from e

    def resolve_target(self, target: str, radius_arcsec: float = 10.0) -> ResolvedTarget:
        """Resolve a target name to a TIC ID.

        Uses MAST's name resolver (via SIMBAD/NED) combined with a TIC cone search
        to find the closest matching TIC entry for the given target.

        Args:
            target: Target identifier - can be:
                - Star name: "Pi Mensae", "HD 209458", "Kepler-10"
                - TIC ID string: "TIC 261136679" or just "261136679"
                - Coordinates: "19:02:43.1 +50:14:28.7" (ICRS)
            radius_arcsec: Search radius in arcseconds for cone match.
                Larger values help with positional uncertainty but may
                return wrong targets in crowded fields.

        Returns:
            ResolvedTarget with TIC ID and match metadata

        Raises:
            NameResolutionError: If resolution fails (name not found, service
                unavailable, no TIC match within radius)
            ValueError: If target string is empty or radius is invalid

        Example:
            >>> client = MASTClient()
            >>> result = client.resolve_target("Pi Mensae")
            >>> print(f"TIC {result.tic_id} at separation {result.separation_arcsec:.2f} arcsec")
            TIC 261136679 at separation 0.05 arcsec
            >>> # Use with other methods
            >>> lc = client.download_lightcurve(result.tic_id, sector=1)
        """
        if not target or not target.strip():
            raise ValueError("Target string cannot be empty")
        if radius_arcsec <= 0:
            raise ValueError(f"radius_arcsec must be positive, got {radius_arcsec}")

        target = target.strip()
        logger.info(f"Resolving target '{target}' with radius {radius_arcsec} arcsec")

        # Fast path: already a TIC ID
        if target.upper().startswith("TIC"):
            tic_str = target.upper().replace("TIC", "").strip()
            try:
                tic_id = int(tic_str)
            except ValueError:
                raise NameResolutionError(
                    f"Invalid TIC ID format: '{target}'. Expected 'TIC <number>'"
                ) from None
            if tic_id <= 0:
                raise ValueError(f"TIC ID must be a positive integer, got: {tic_id}")
            logger.info(f"Direct TIC ID parse: {tic_id}")
            return ResolvedTarget(
                tic_id=tic_id,
                target_input=target,
                separation_arcsec=0.0,
            )

        # Check if target is just a number (bare TIC ID) or negative number
        if target.isdigit() or (target.startswith("-") and target[1:].isdigit()):
            tic_id = int(target)
            if tic_id <= 0:
                raise ValueError(f"TIC ID must be a positive integer, got: {tic_id}")
            logger.info(f"Direct numeric TIC ID: {tic_id}")
            return ResolvedTarget(
                tic_id=tic_id,
                target_input=target,
                separation_arcsec=0.0,
            )

        # Name resolution via MAST
        self._ensure_lightkurve()  # Ensure astroquery is available

        try:
            from astroquery.mast import Catalogs
        except ImportError as e:
            raise NameResolutionError(
                "astroquery is required for name resolution. Install with: pip install astroquery"
            ) from e

        # Convert radius to degrees for MAST query
        radius_deg = radius_arcsec / 3600.0

        try:
            # Query TIC with name resolution - MAST resolves the name first
            result = Catalogs.query_object(target, radius=radius_deg, catalog="TIC")  # pyright: ignore[reportAttributeAccessIssue]
        except Exception as e:
            error_msg = str(e).lower()
            if "could not resolve" in error_msg or "not found" in error_msg:
                raise NameResolutionError(
                    f"Target '{target}' not found. Check spelling or try coordinates."
                ) from e
            if "timeout" in error_msg or "connection" in error_msg:
                raise NameResolutionError(
                    "Name resolution service unavailable. Try again later or use TIC ID directly."
                ) from e
            raise NameResolutionError(f"Name resolution failed for '{target}': {e}") from e

        if result is None or len(result) == 0:
            raise NameResolutionError(
                f"No TIC match found for '{target}' within {radius_arcsec} arcsec. "
                "Try increasing the search radius or check the target name."
            )

        # Find closest match
        separations = np.array(result["dstArcSec"], dtype=np.float64)
        closest_idx = int(np.argmin(separations))
        row = result[closest_idx]

        # Extract match info
        tic_id = int(row["ID"])
        separation = float(separations[closest_idx])
        ra = self._get_float(row, "ra")
        dec = self._get_float(row, "dec")
        tmag = self._get_float(row, "Tmag")

        # Check for ambiguous matches (multiple stars within search radius)
        n_matches = len(result)
        warning = None
        if n_matches > 1:
            # Check how many are within a tighter radius (1 arcsec)
            close_matches = np.sum(separations < 1.0)
            if close_matches > 1:
                warning = (
                    f"Multiple TIC entries ({close_matches}) within 1 arcsec. "
                    f"Returning closest match (TIC {tic_id})."
                )
            else:
                warning = (
                    f"Multiple TIC entries ({n_matches}) within search radius. "
                    f"Returning closest match (TIC {tic_id}, {separation:.2f} arcsec)."
                )

        logger.info(f"Resolved '{target}' to TIC {tic_id} (separation={separation:.3f} arcsec)")

        return ResolvedTarget(
            tic_id=tic_id,
            target_input=target,
            separation_arcsec=separation,
            ra=ra,
            dec=dec,
            tmag=tmag,
            warning=warning,
        )

    @staticmethod
    def _get_float(row: Any, key: str) -> float | None:
        """Safely extract float value from catalog row."""
        try:
            val = row[key]
            if val is None or (hasattr(val, "mask") and val.mask):
                return None
            result = float(val)
            return result if np.isfinite(result) else None
        except (KeyError, ValueError, TypeError):
            return None

    @staticmethod
    def _get_int(row: Any, key: str) -> int | None:
        """Safely extract int value from catalog row."""
        try:
            val = row[key]
            if val is None or (hasattr(val, "mask") and val.mask):
                return None
            return int(val)
        except (KeyError, ValueError, TypeError):
            return None

    @staticmethod
    def _get_str(row: Any, key: str) -> str | None:
        """Safely extract string value from catalog row."""
        try:
            val = row[key]
            if val is None or (hasattr(val, "mask") and val.mask):
                return None
            return str(val)
        except (KeyError, ValueError, TypeError):
            return None

    def get_available_sectors(self, tic_id: int) -> list[int]:
        """Get list of sectors where target has observations.

        Convenience method that wraps search_lightcurve.

        Args:
            tic_id: TESS Input Catalog identifier

        Returns:
            Sorted list of sector numbers with available data.

        Example:
            >>> client = MASTClient()
            >>> sectors = client.get_available_sectors(261136679)
            >>> print(f"Available in sectors: {sectors}")
        """
        results = self.search_lightcurve(tic_id)
        return sorted({r.sector for r in results})

    def download_all_sectors(
        self,
        tic_id: int,
        flux_type: str = "pdcsap",
        sectors: list[int] | None = None,
        progress_callback: Callable[[DownloadProgress], None] | None = None,
    ) -> list[LightCurveData]:
        """Download light curves for all available sectors.

        Args:
            tic_id: TESS Input Catalog identifier
            flux_type: Flux type ("pdcsap" or "sap")
            sectors: Specific sectors to download (None for all available)
            progress_callback: Optional callback function to receive progress updates.
                For batch downloads, reports per-sector progress with sectors_completed
                and sectors_remaining fields populated.

        Returns:
            List of LightCurveData objects, one per sector, sorted by sector.

        Raises:
            LightCurveNotFoundError: If no light curves are available.
            MASTClientError: If downloads fail.

        Example:
            >>> client = MASTClient()
            >>> all_lcs = client.download_all_sectors(261136679)
            >>> total_points = sum(lc.n_points for lc in all_lcs)
            >>> print(f"Downloaded {len(all_lcs)} sectors, {total_points} total points")
            >>> # With progress callback for batch monitoring
            >>> def batch_progress(p: DownloadProgress) -> None:
            ...     print(f"Sector {p.sector}: {p.percentage:.0f}% - "
            ...           f"Completed: {p.sectors_completed}")
            >>> lcs = client.download_all_sectors(261136679, progress_callback=batch_progress)
        """
        start_time = time_module.monotonic()

        if sectors is None:
            sectors = self.get_available_sectors(tic_id)

        if not sectors:
            raise LightCurveNotFoundError(f"No light curves available for TIC {tic_id}")

        sorted_sectors = sorted(sectors)
        n_sectors = len(sorted_sectors)
        logger.info(f"Downloading {n_sectors} sectors for TIC {tic_id}")

        # Track completed and remaining sectors for batch progress
        sectors_completed: list[int] = []
        sectors_remaining = list(sorted_sectors)

        def _report_batch_progress(
            phase: DownloadPhase,
            current_sector: int,
            sector_index: int,
            message: str,
        ) -> None:
            """Report batch download progress if callback is provided."""
            if progress_callback is None:
                return
            elapsed = time_module.monotonic() - start_time
            # Calculate overall progress (each sector contributes 1/n_sectors)
            # Within each sector, there are 4 sub-steps
            overall_progress = (sector_index / n_sectors) * 100.0
            # Estimate remaining time
            if sector_index > 0:
                avg_time_per_sector = elapsed / sector_index
                est_remaining = avg_time_per_sector * (n_sectors - sector_index)
            else:
                est_remaining = None
            progress = DownloadProgress(
                phase=phase,
                current_step=sector_index + 1,
                total_steps=n_sectors,
                percentage=overall_progress,
                message=message,
                tic_id=tic_id,
                sector=current_sector,
                elapsed_seconds=elapsed,
                estimated_remaining_seconds=est_remaining,
                sectors_completed=list(sectors_completed),
                sectors_remaining=list(sectors_remaining),
            )
            progress_callback(progress)

        light_curves = []
        for i, sector in enumerate(sorted_sectors):
            _report_batch_progress(
                DownloadPhase.DOWNLOADING,
                sector,
                i,
                f"Downloading sector {sector} ({i + 1}/{n_sectors}) for TIC {tic_id}...",
            )
            try:
                lc = self.download_lightcurve(tic_id, sector, flux_type)
                light_curves.append(lc)
                sectors_completed.append(sector)
                sectors_remaining.remove(sector)
            except LightCurveNotFoundError as e:
                logger.warning(f"Skipping sector {sector}: {e}")
                sectors_remaining.remove(sector)
                continue
            except MASTClientError as e:
                logger.error(f"Failed to download sector {sector}: {e}")
                raise

        # Report batch completion
        if progress_callback is not None:
            elapsed = time_module.monotonic() - start_time
            progress = DownloadProgress(
                phase=DownloadPhase.COMPLETE,
                current_step=n_sectors,
                total_steps=n_sectors,
                percentage=100.0,
                message=f"Batch download complete: {len(light_curves)}/{n_sectors} sectors for TIC {tic_id}",
                tic_id=tic_id,
                sector=None,
                elapsed_seconds=elapsed,
                estimated_remaining_seconds=0.0,
                sectors_completed=sectors_completed,
                sectors_remaining=[],
            )
            progress_callback(progress)

        if not light_curves:
            raise LightCurveNotFoundError(f"Failed to download any light curves for TIC {tic_id}")

        return light_curves
