"""MAST client wrapper for TESS light curve queries.

This module provides MASTClient, a wrapper around lightkurve for querying
and downloading TESS light curves from MAST (Mikulski Archive for Space Telescopes).

Usage:
    from tess_vetter.platform.io import MASTClient

    client = MASTClient()
    results = client.search_lightcurve(tic_id=261136679)
    lc_data = client.download_lightcurve(tic_id=261136679, sector=1)
    target = client.get_target_info(tic_id=261136679)
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time as time_module
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from tess_vetter.api.lightcurve import LightCurveData, LightCurveProvenance
from tess_vetter.api.target import Target

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_DEFAULT_SEARCH_TIMEOUT_SECONDS = 60.0


def _search_timeout_seconds() -> float:
    raw = os.getenv("BTV_MAST_SEARCH_TIMEOUT_SECONDS")
    if raw is None:
        raw = os.getenv("BTV_MAST_TIMEOUT_SECONDS", str(_DEFAULT_SEARCH_TIMEOUT_SECONDS))
    try:
        value = float(raw)
    except Exception:
        return _DEFAULT_SEARCH_TIMEOUT_SECONDS
    if not np.isfinite(value) or value <= 0.0:
        return _DEFAULT_SEARCH_TIMEOUT_SECONDS
    return value


def _search_lightcurve_with_timeout(lk: Any, **kwargs: Any) -> Any:
    """Run lightkurve.search_lightcurve with a hard timeout.

    lightkurve/astroquery requests can occasionally hang indefinitely on MAST.
    This wrapper bounds wall-time and raises TimeoutError if exceeded.
    """
    timeout_s = _search_timeout_seconds()
    out: dict[str, Any] = {"done": False, "result": None, "error": None}

    def _run() -> None:
        try:
            out["result"] = lk.search_lightcurve(**kwargs)
        except Exception as exc:  # pragma: no cover - exercised through caller path
            out["error"] = exc
        finally:
            out["done"] = True

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)
    if thread.is_alive() or not out["done"]:
        target = kwargs.get("target")
        raise TimeoutError(f"MAST search_lightcurve timed out after {timeout_s:.1f}s for {target}")
    if out["error"] is not None:
        raise out["error"]
    return out["result"]


def _maybe_extract_http_status(exc: BaseException) -> int | None:
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        try:
            if value is not None:
                return int(value)
        except Exception:
            continue
    response = getattr(exc, "response", None)
    if response is not None:
        code = getattr(response, "status_code", None)
        try:
            if code is not None:
                return int(code)
        except Exception:
            pass
    text = str(exc)
    match = re.search(r"\b([45]\d{2})\b", text)
    if match is not None:
        try:
            return int(match.group(1))
        except Exception:
            return None
    return None


def _mast_retry_guidance(exc: BaseException) -> str:
    status = _maybe_extract_http_status(exc)
    if status is not None and 500 <= status <= 599:
        return (
            f" MAST/TIC returned HTTP {status}; this is often transient. "
            "Retry shortly, or use cached/local data (--local-data-path or cache_dir) if available."
        )
    return ""


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


def _extract_flux_array(lc: Any, *, flux_attr: str) -> np.ndarray[Any, np.dtype[np.float64]] | None:
    """Extract a flux-like Quantity/array attribute from a lightkurve LightCurve."""
    if not hasattr(lc, flux_attr):
        return None
    try:
        val = getattr(lc, flux_attr)
        if hasattr(val, "value"):
            val = val.value
        return np.asarray(val, dtype=np.float64)
    except Exception:
        return None


def _select_flux_and_base_valid(
    *,
    lc: Any,
    time: np.ndarray[Any, np.dtype[np.float64]],
    quality: np.ndarray[Any, np.dtype[np.int32]],
    mask: int,
    flux_type: str,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]] | None,
    np.ndarray[Any, np.dtype[np.bool_]],
    str | None,
]:
    """Choose a flux column that yields usable finite samples, with a safe fallback.

    Some products (notably some QLP variants) can have all-NaN in pdcsap_flux/sap_flux
    while still providing a usable `flux` column. Also, some pipelines use quality flags
    incompatible with the default SPOC mask. This helper:
    - tries the requested flux column first
    - falls back through other common columns
    - if flux is finite but quality excludes everything, relaxes the quality cut
      (and records that in the selection_reason).
    """
    requested_flux_col = "pdcsap_flux" if flux_type == "pdcsap" else "sap_flux"
    requested_flux_err_col = f"{requested_flux_col}_err"

    candidates: list[tuple[str, str | None, str]] = [
        (requested_flux_col, requested_flux_err_col, f"requested:{requested_flux_col}"),
        ("pdcsap_flux", "pdcsap_flux_err", "fallback:pdcsap_flux"),
        ("sap_flux", "sap_flux_err", "fallback:sap_flux"),
        ("flux", "flux_err", "fallback:flux"),
    ]

    finite_time = np.isfinite(time)
    finite_quality = (quality & int(mask)) == 0

    best_finite: (
        tuple[
            np.ndarray[Any, np.dtype[np.float64]],
            np.ndarray[Any, np.dtype[np.float64]] | None,
            str,
        ]
        | None
    ) = None

    for flux_attr, flux_err_attr, label in candidates:
        flux_raw = _extract_flux_array(lc, flux_attr=flux_attr)
        if flux_raw is None or flux_raw.shape != time.shape:
            continue
        finite_flux = np.isfinite(flux_raw)
        finite = (finite_time & finite_flux).astype(np.bool_)
        if not np.any(finite):
            continue

        flux_err_raw: np.ndarray[Any, np.dtype[np.float64]] | None = None
        if flux_err_attr is not None:
            flux_err_raw = _extract_flux_array(lc, flux_attr=flux_err_attr)
            if flux_err_raw is not None and flux_err_raw.shape != time.shape:
                flux_err_raw = None

        base_valid = (finite & finite_quality).astype(np.bool_)
        if np.any(base_valid):
            reason = None if label.startswith("requested:") else f"flux_fallback={label}"
            return flux_raw, flux_err_raw, base_valid, reason

        if best_finite is None:
            best_finite = (flux_raw, flux_err_raw, label)

    if best_finite is None:
        raise MASTClientError("No finite flux column found in light curve product")

    flux_raw, flux_err_raw, label = best_finite
    finite = (finite_time & np.isfinite(flux_raw)).astype(np.bool_)
    base_valid = finite.astype(np.bool_)
    return flux_raw, flux_err_raw, base_valid, f"quality_mask_relaxed+{label}"


def _extract_ra_dec_from_lk_meta(obj: Any) -> tuple[float | None, float | None]:
    """Best-effort RA/Dec extraction from a lightkurve LightCurve object."""
    meta = getattr(obj, "meta", None)
    if not isinstance(meta, dict):
        return None, None

    def _coerce(v: object | None) -> float | None:
        if v is None:
            return None
        try:
            # astropy Quantity
            if hasattr(v, "value"):
                v = v.value
        except Exception:
            pass
        try:
            x = float(v)  # type: ignore[arg-type]
        except Exception:
            return None
        return x if np.isfinite(x) else None

    for ra_key, dec_key in [
        ("RA_OBJ", "DEC_OBJ"),
        ("RA", "DEC"),
        ("ra", "dec"),
        ("ra_obj", "dec_obj"),
    ]:
        ra = _coerce(meta.get(ra_key))
        dec = _coerce(meta.get(dec_key))
        if ra is not None and dec is not None:
            return ra, dec

    return None, None


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
        cache_dir: str | None = None,
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
        self.cache_dir = cache_dir
        self._lk_imported = False
        self._cache_index_built = False
        self._cache_dirs_by_tic: dict[str, list[Path]] = {}

    def _ensure_lightkurve(self) -> Any:
        """Lazy import of lightkurve to avoid import-time overhead."""
        if not self._lk_imported:
            try:
                # Best-effort: bound astroquery MAST call latency so a single hung request
                # doesn't stall bulk enrichment for many minutes.
                try:
                    from astroquery.mast import Conf as MastConf

                    MastConf.timeout = float(os.getenv("BTV_MAST_TIMEOUT_SECONDS", "60"))
                except Exception:
                    pass

                if self.cache_dir:
                    # Best-effort: ensure lightkurve reads/writes from a shared cache directory.
                    # This helps bulk enrichment reuse existing local caches across repos.
                    os.environ.setdefault("LIGHTKURVE_CACHE_DIR", str(self.cache_dir))
                import lightkurve as lk

                if self.cache_dir:
                    try:
                        if hasattr(lk, "config") and hasattr(lk.config, "set_cache_dir"):
                            lk.config.set_cache_dir(self.cache_dir)
                    except Exception:
                        # Non-fatal: fall back to env var only.
                        pass

                self._lk = lk
                self._lk_imported = True
            except ImportError as e:
                raise MASTClientError(
                    "lightkurve is required for MAST queries. Install with: pip install lightkurve"
                ) from e
        return self._lk

    def _mast_download_root(self) -> Path | None:
        """Return the cache root that contains `mastDownload/`, if configured."""
        if not self.cache_dir:
            return None
        p = Path(self.cache_dir).expanduser()
        if p.name == "mastDownload":
            return p
        if (p / "mastDownload").exists():
            return p / "mastDownload"
        return None

    def _build_cache_index(self) -> None:
        """Index cached mastDownload directories by TIC (fast lookup for cache-only mode)."""
        if self._cache_index_built:
            return
        self._cache_index_built = True

        root = self._mast_download_root()
        if root is None:
            return

        # Directory names include the 16-digit TIC in both the TESS and HLSP layouts.
        tic_pat = re.compile(r"(\d{16})")

        for subdir in ("TESS", "HLSP"):
            base = root / subdir
            if not base.exists():
                continue
            try:
                with os.scandir(base) as it:
                    for entry in it:
                        if not entry.is_dir():
                            continue
                        m = tic_pat.search(entry.name)
                        if not m:
                            continue
                        tic16 = m.group(1)
                        self._cache_dirs_by_tic.setdefault(tic16, []).append(Path(entry.path))
            except Exception:
                continue

    @staticmethod
    def _parse_sector_from_name(name: str) -> int | None:
        m = re.search(r"-s(\d{4})-", name)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        m = re.search(r"_s(\d{4})_", name)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None

    @staticmethod
    def _infer_exptime_seconds(path: Path) -> float:
        name = path.name.lower()
        if "a_fast" in name or "-fast" in name or "_fast" in name:
            return 20.0
        return 120.0

    def search_lightcurve_cached(
        self,
        tic_id: int,
        sector: int | None = None,
        author: str | None = None,
    ) -> list[SearchResult]:
        """Search cached light curves without any network calls."""
        root = self._mast_download_root()
        if root is None:
            return []
        self._build_cache_index()
        tic16 = f"{int(tic_id):016d}"
        dirs = self._cache_dirs_by_tic.get(tic16, [])
        if not dirs:
            return []

        out: list[SearchResult] = []
        for d in dirs:
            try:
                for f in d.glob("*lc.fits*"):
                    if tic16 not in f.name:
                        continue
                    sec = self._parse_sector_from_name(f.name) or self._parse_sector_from_name(d.name)
                    if sec is None:
                        continue
                    if sector is not None and int(sec) != int(sector):
                        continue
                    out.append(
                        SearchResult(
                            tic_id=int(tic_id),
                            sector=int(sec),
                            author=str(author if author is not None else (self.author or "SPOC")),
                            exptime=float(self._infer_exptime_seconds(f)),
                            mission="TESS",
                            distance=None,
                        )
                    )
            except Exception:
                continue
        out.sort(key=lambda r: int(r.sector))
        return out

    def search_tpf_cached(
        self,
        tic_id: int,
        sector: int | None = None,
        author: str | None = None,
    ) -> list[SearchResult]:
        """Search cached TPF products without any network calls."""
        root = self._mast_download_root()
        if root is None:
            return []
        self._build_cache_index()
        tic16 = f"{int(tic_id):016d}"
        dirs = self._cache_dirs_by_tic.get(tic16, [])
        if not dirs:
            return []

        out: list[SearchResult] = []
        for d in dirs:
            try:
                for f in d.glob("*tp.fits*"):
                    if tic16 not in f.name:
                        continue
                    sec = self._parse_sector_from_name(f.name) or self._parse_sector_from_name(d.name)
                    if sec is None:
                        continue
                    if sector is not None and int(sec) != int(sector):
                        continue
                    out.append(
                        SearchResult(
                            tic_id=int(tic_id),
                            sector=int(sec),
                            author=str(author if author is not None else (self.author or "SPOC")),
                            exptime=float(self._infer_exptime_seconds(f)),
                            mission="TESS",
                            distance=None,
                        )
                    )
            except Exception:
                continue
        out.sort(key=lambda r: int(r.sector))
        return out

    def download_lightcurve_cached(
        self,
        tic_id: int,
        sector: int,
        flux_type: str = "pdcsap",
        *,
        exptime: float | None = None,
    ) -> LightCurveData:
        """Load a cached light curve from disk (no network)."""
        root = self._mast_download_root()
        if root is None:
            raise LightCurveNotFoundError("cache_dir is not configured or has no mastDownload/")
        self._build_cache_index()
        tic16 = f"{int(tic_id):016d}"
        dirs = self._cache_dirs_by_tic.get(tic16, [])
        if not dirs:
            raise LightCurveNotFoundError(f"No cached light curve directories for TIC {tic_id}")

        candidates: list[Path] = []
        for d in dirs:
            try:
                sec = self._parse_sector_from_name(d.name)
                if sec is None or int(sec) != int(sector):
                    continue
                for f in d.glob("*lc.fits*"):
                    if tic16 in f.name:
                        candidates.append(f)
            except Exception:
                continue

        if not candidates:
            raise LightCurveNotFoundError(f"No cached light curve for TIC {tic_id} sector {sector}")

        # Prefer cadence match if requested, else prefer 120s.
        preferred = float(exptime) if exptime is not None else 120.0

        def _cand_key(p: Path) -> tuple[float, int]:
            dt = abs(self._infer_exptime_seconds(p) - preferred)
            # Tie-break: prefer non-fast over fast.
            fast = 1 if self._infer_exptime_seconds(p) < 60.0 else 0
            return (dt, fast)

        path = min(candidates, key=_cand_key)

        lk = self._ensure_lightkurve()
        try:
            lc_any = lk.read(str(path))
        except Exception as e:
            raise MASTClientError(f"Failed to read cached light curve FITS: {path}: {e}") from e

        # Select flux column
        if flux_type == "pdcsap":
            lc = lc_any.PDCSAP_FLUX if hasattr(lc_any, "PDCSAP_FLUX") else lc_any
        elif flux_type == "sap":
            lc = lc_any.SAP_FLUX if hasattr(lc_any, "SAP_FLUX") else lc_any
        else:
            raise ValueError(f"flux_type must be 'pdcsap' or 'sap', got '{flux_type}'")

        try:
            time = np.asarray(lc.time.value, dtype=np.float64)
            flux = np.asarray(lc.flux.value, dtype=np.float64)
            flux_err = (
                np.asarray(lc.flux_err.value, dtype=np.float64)
                if getattr(lc, "flux_err", None) is not None
                else np.full_like(flux, np.nan)
            )
            quality = (
                np.asarray(lc.quality, dtype=np.int32)
                if getattr(lc, "quality", None) is not None
                else np.zeros(time.shape, dtype=np.int32)
            )
        except Exception as e:
            raise MASTClientError(f"Cached light curve parse failed for {path}: {e}") from e

        # Valid mask: finite + quality filtering
        finite = np.isfinite(time) & np.isfinite(flux)
        valid_mask = finite & ((quality & int(self.quality_mask)) == 0)

        if self.normalize:
            med = np.nanmedian(flux[valid_mask]) if np.any(valid_mask) else np.nanmedian(flux)
            if np.isfinite(med) and med != 0:
                flux = flux / med
                flux_err = flux_err / med

        # Estimate cadence
        cadence_seconds = float("nan")
        if time.size > 1:
            dt_days = np.nanmedian(np.diff(time))
            cadence_seconds = float(dt_days) * 86400.0

        ra_deg, dec_deg = _extract_ra_dec_from_lk_meta(lc)

        return LightCurveData(
            time=time,
            flux=flux,
            flux_err=flux_err,
            quality=quality,
            valid_mask=valid_mask,
            tic_id=int(tic_id),
            sector=int(sector),
            cadence_seconds=cadence_seconds,
            provenance=LightCurveProvenance(
                source="MAST/lightkurve(cache_only)",
                selected_author=str(self.author or "SPOC"),
                selected_exptime_seconds=float(self._infer_exptime_seconds(path)),
                preferred_author=str(self.author) if self.author is not None else None,
                requested_exptime_seconds=float(exptime) if exptime is not None else None,
                flux_type=flux_type,
                quality_mask=int(self.quality_mask),
                normalize=bool(self.normalize),
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                selection_reason="cache_only",
                flux_err_kind="provided",
            ),
        )

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
            search_result = _search_lightcurve_with_timeout(
                lk,
                target=target,
                mission="TESS",
                sector=sector,
                author=search_author,
            )
        except Exception as e:
            logger.error(f"MAST search failed for {target}: {e}")
            raise MASTClientError(
                f"Failed to search MAST for TIC {tic_id}: {e}{_mast_retry_guidance(e)}"
            ) from e

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
        presearched_rows: list[Any] | None = None,
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

        # Step 1: Search for the specific sector (or reuse pre-fetched rows).
        search_author = author if author is not None else self.author
        if presearched_rows is not None:
            search_result = list(presearched_rows)
        else:
            _report_progress(
                DownloadPhase.SEARCHING,
                1,
                f"Searching MAST for TIC {tic_id} sector {sector}...",
            )
            try:
                search_result = _search_lightcurve_with_timeout(
                    lk,
                    target=target,
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

        def _coerce_float(val: object) -> float | None:
            if val is None:
                return None
            if hasattr(val, "value"):
                val = val.value
            if hasattr(val, "item"):
                try:
                    return float(val.item())
                except Exception:
                    return None
            if hasattr(val, "__getitem__") and hasattr(val, "__len__"):
                try:
                    if len(val) == 0:
                        return None
                    return float(val[0])
                except Exception:
                    return None
            try:
                return float(val)
            except Exception:
                return None

        def _row_author(row: Any) -> str | None:
            try:
                if hasattr(row, "author") and row.author is not None:
                    return str(row.author)
            except Exception:
                return None
            return None

        def _row_exptime(row: Any) -> float | None:
            if not hasattr(row, "exptime"):
                return None
            return _coerce_float(getattr(row, "exptime", None))

        # Select a single product deterministically (avoid relying on upstream ordering).
        candidate_indices = list(range(len(search_result)))
        requested_exptime = exptime
        if requested_exptime is not None:
            candidate_indices = [
                i
                for i in candidate_indices
                if (_row_exptime(search_result[i]) is not None)
                and abs(float(_row_exptime(search_result[i]) or 0.0) - requested_exptime) < 1.0
            ]

            if not candidate_indices:
                available_exptimes: list[float] = []
                for i in range(len(search_result)):
                    exptime_i = _row_exptime(search_result[i])
                    if exptime_i is not None:
                        available_exptimes.append(float(exptime_i))
                raise LightCurveNotFoundError(
                    f"No light curve found for TIC {tic_id} sector {sector} "
                    f"with exptime={requested_exptime}s. Available exptimes: {available_exptimes}"
                )

        preferred_author = search_author
        preferred_exptime = requested_exptime if requested_exptime is not None else 120.0
        selection_reason = None

        def _selection_key(i: int) -> tuple[int, float, str, float, int]:
            row = search_result[i]
            author_i = _row_author(row) or ""
            exptime_i = _row_exptime(row)

            author_match = 1
            if preferred_author is not None and author_i.lower() == preferred_author.lower():
                author_match = 0

            exptime_delta = (
                abs(float(exptime_i) - preferred_exptime) if exptime_i is not None else 1e9
            )
            exptime_val = float(exptime_i) if exptime_i is not None else 1e9
            return (author_match, exptime_delta, author_i, exptime_val, i)

        selected_index = min(candidate_indices, key=_selection_key)
        if len(candidate_indices) > 1:
            selection_reason = (
                "preferred_author/exptime_tiebreak"
                if preferred_author is not None
                else "exptime_tiebreak"
            )
        selected_row = search_result[selected_index]

        # Step 2: Download the light curve
        _report_progress(
            DownloadPhase.DOWNLOADING,
            2,
            f"Downloading light curve for TIC {tic_id} sector {sector}...",
        )

        def _looks_like_lightcurve(obj: Any) -> bool:
            """Best-effort guard for mocked/invalid download results."""
            try:
                t = obj.time.value
                arr = np.asarray(t)
                return arr.ndim == 1 and arr.size > 0 and np.isfinite(arr).any()
            except Exception:
                return False

        try:
            lc_result = selected_row.download()
            # Compatibility fallback: some mocks (and some upstream wrappers) attach `.download()`
            # to the SearchResult collection rather than the row. Prefer the row, but if the
            # returned object doesn't look like a light curve, fall back to the collection.
            if not _looks_like_lightcurve(lc_result) and hasattr(search_result, "download"):
                lc_result = search_result.download()
            # Check if result is a LightCurveCollection (multiple light curves)
            # vs a single LightCurve. Don't use len() - LightCurve has len = n_points
            lc = lc_result[0] if type(lc_result).__name__ == "LightCurveCollection" else lc_result
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

        # Extract data arrays
        try:
            time = np.asarray(lc.time.value, dtype=np.float64)

            # Quality flags
            if hasattr(lc, "quality"):
                quality = np.asarray(lc.quality, dtype=np.int32)
            else:
                quality = np.zeros(len(time), dtype=np.int32)

        except Exception as e:
            logger.error(f"Failed to extract data from light curve: {e}")
            raise MASTClientError(f"Failed to process light curve data: {e}") from e

        flux_raw, flux_err_raw, base_valid, selection_reason_flux = _select_flux_and_base_valid(
            lc=lc,
            time=time,
            quality=quality,
            mask=int(mask),
            flux_type=str(flux_type),
        )
        if selection_reason_flux is not None:
            selection_reason = (
                selection_reason_flux
                if selection_reason is None
                else f"{selection_reason};{selection_reason_flux}"
            )

        # Normalize flux to median ~1.0 (based on base_valid points).
        median_flux = None
        if self.normalize and np.any(base_valid):
            median_flux = float(np.median(flux_raw[base_valid]))
            if median_flux > 0 and np.isfinite(median_flux):
                flux = flux_raw / median_flux
            else:
                logger.warning(f"Invalid median flux {median_flux}, skipping normalization")
                median_flux = None
                flux = flux_raw
        else:
            flux = flux_raw

        # Flux errors:
        # - If provided by the product, preserve them (and propagate NaNs into valid_mask).
        # - If absent, estimate a representative per-point uncertainty from the flux scatter,
        #   and record this explicitly in provenance.
        flux_err_kind: str = "provided"
        if flux_err_raw is None:
            flux_err_kind = "estimated_missing"
            if np.any(base_valid):
                med = float(np.median(flux[base_valid]))
                mad = float(np.median(np.abs(flux[base_valid] - med)))
                sigma = 1.4826 * mad
                if not np.isfinite(sigma) or sigma <= 0:
                    sigma = float(np.std(flux[base_valid]))
                if not np.isfinite(sigma) or sigma <= 0:
                    sigma = 1e-3
            else:
                sigma = 1e-3
            flux_err = np.full_like(flux, sigma, dtype=np.float64)
        else:
            flux_err = flux_err_raw / median_flux if median_flux is not None else flux_err_raw

        # Final validity mask (requires finite flux_err).
        valid_mask = (base_valid & np.isfinite(flux_err)).astype(np.bool_)

        # Determine cadence from time differences
        if len(time) > 1:
            # Use median time difference to determine cadence.
            # Prefer valid samples, but always drop non-finite dt values.
            dt = np.diff(time[valid_mask]) if np.any(valid_mask) else np.diff(time)
            dt = dt[np.isfinite(dt) & (dt > 0)]
            cadence_days = float(np.median(dt)) if len(dt) > 0 else 0.0
            cadence_seconds = cadence_days * 86400.0 if cadence_days > 0 else 120.0
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

        ra_deg, dec_deg = _extract_ra_dec_from_lk_meta(lc)

        return LightCurveData(
            time=time,
            flux=flux,
            flux_err=flux_err,
            quality=quality,
            valid_mask=valid_mask,
            tic_id=tic_id,
            sector=sector,
            cadence_seconds=cadence_seconds,
            provenance=LightCurveProvenance(
                source="MAST/lightkurve",
                selected_author=_row_author(selected_row),
                selected_exptime_seconds=_row_exptime(selected_row),
                preferred_author=preferred_author,
                requested_exptime_seconds=requested_exptime,
                flux_type=flux_type,
                quality_mask=int(mask),
                normalize=bool(self.normalize),
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                selection_reason=selection_reason,
                flux_err_kind=(
                    "estimated_missing" if flux_err_kind == "estimated_missing" else "provided"
                ),
            ),
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
                # Near-IR photometry (2MASS via TIC)
                "Jmag": self._get_float(row, "Jmag"),
                "e_Jmag": self._get_float(row, "e_Jmag"),
                "Hmag": self._get_float(row, "Hmag"),
                "e_Hmag": self._get_float(row, "e_Hmag"),
                "Kmag": self._get_float(row, "Kmag"),
                "e_Kmag": self._get_float(row, "e_Kmag"),
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
            raise MASTClientError(
                f"Failed to query TIC for {tic_id}: {e}{_mast_retry_guidance(e)}"
            ) from e

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
        pre_rows_by_sector: dict[int, list[Any]] = {}
        if sectors is not None and len(sorted_sectors) > 0:
            # Query once for all sectors and reuse rows to avoid repeated MAST search calls.
            try:
                lk = self._ensure_lightkurve()
                target = f"TIC {tic_id}"
                search_author = self.author
                search_all = _search_lightcurve_with_timeout(
                    lk,
                    target=target,
                    mission="TESS",
                    author=search_author,
                )
                if search_all is not None and len(search_all) > 0:
                    requested_set = {int(s) for s in sorted_sectors}
                    for idx in range(len(search_all)):
                        row = search_all[idx]
                        sec: int | None = None
                        try:
                            if hasattr(row, "sequence_number"):
                                seq = row.sequence_number
                                if seq is not None:
                                    sec = int(seq)
                        except Exception:
                            sec = None
                        if sec is None:
                            try:
                                if hasattr(row, "observation") and row.observation:
                                    match = re.search(r"sector\s*(\d+)", str(row.observation), re.IGNORECASE)
                                    if match:
                                        sec = int(match.group(1))
                            except Exception:
                                sec = None
                        if sec is None or sec not in requested_set:
                            continue
                        pre_rows_by_sector.setdefault(int(sec), []).append(row)
            except Exception:
                pre_rows_by_sector = {}

        for i, sector in enumerate(sorted_sectors):
            _report_batch_progress(
                DownloadPhase.DOWNLOADING,
                sector,
                i,
                f"Downloading sector {sector} ({i + 1}/{n_sectors}) for TIC {tic_id}...",
            )
            try:
                lc = self.download_lightcurve(
                    tic_id,
                    sector,
                    flux_type,
                    presearched_rows=pre_rows_by_sector.get(int(sector)) or None,
                )
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

    def search_tpf(
        self,
        tic_id: int,
        sector: int | None = None,
        author: str | None = None,
    ) -> list[SearchResult]:
        """Search for available TESS Target Pixel Files for a target.

        Args:
            tic_id: TESS Input Catalog identifier
            sector: Specific sector to search (None for all sectors)
            author: Override default author filter (None uses instance default)

        Returns:
            List of SearchResult objects describing available TPFs,
            sorted by sector number.

        Raises:
            MASTClientError: If the search fails due to network or API errors.

        Example:
            >>> client = MASTClient()
            >>> results = client.search_tpf(261136679)
            >>> print(f"Found {len(results)} TPFs")
        """
        lk = self._ensure_lightkurve()

        search_author = author if author is not None else self.author
        target = f"TIC {tic_id}"

        logger.info(f"Searching MAST for TPF {target}, sector={sector}, author={search_author}")

        try:
            search_result = lk.search_targetpixelfile(
                target,
                mission="TESS",
                sector=sector,
                author=search_author,
            )
        except Exception as e:
            logger.error(f"MAST TPF search failed for {target}: {e}")
            raise MASTClientError(f"Failed to search MAST for TPF TIC {tic_id}: {e}") from e

        if search_result is None or len(search_result) == 0:
            logger.info(f"No TPFs found for {target}")
            return []

        results = []
        for i in range(len(search_result)):
            row = search_result[i]
            try:
                sector_num = int(row.mission[0].split()[-1]) if hasattr(row, "mission") else 0
                if hasattr(row, "sequence_number"):
                    sector_num = int(row.sequence_number)

                def _to_float(val: object, default: float = 0.0) -> float:
                    if val is None:
                        return default
                    if hasattr(val, "value"):
                        val = val.value
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
                logger.warning(f"Failed to parse TPF search result row {i}: {e}")
                continue

        results.sort(key=lambda r: r.sector)
        logger.info(f"Found {len(results)} TPFs for {target}")

        return results

    def download_tpf(
        self,
        tic_id: int,
        sector: int,
        exptime: float | None = None,
        author: str | None = None,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray | None, Any | None, np.ndarray | None, np.ndarray | None
    ]:
        """Download and process a TESS Target Pixel File.

        This method downloads a TPF from MAST and returns the raw arrays needed
        to construct a TPFStamp object.

        Args:
            tic_id: TESS Input Catalog identifier
            sector: TESS sector number
            exptime: Exposure time in seconds (e.g., 20 or 120). If provided,
                filters search results to only include products with matching
                exposure time. If None, prefers 120s cadence.
            author: Override default author filter (None uses instance default).

        Returns:
            Tuple of (time, flux, flux_err, wcs, aperture_mask, quality):
                - time: Time array in BTJD, shape (n_cadences,)
                - flux: Flux cube, shape (n_cadences, n_rows, n_cols)
                - flux_err: Flux error cube (optional), same shape as flux
                - wcs: WCS object for coordinate transforms (optional)
                - aperture_mask: Pipeline aperture mask, shape (n_rows, n_cols)
                - quality: Quality flags, shape (n_cadences,)

        Raises:
            LightCurveNotFoundError: If no TPF exists for the parameters.
            MASTClientError: If download fails due to network or processing errors.

        Example:
            >>> client = MASTClient()
            >>> time, flux, flux_err, wcs, aperture, quality = client.download_tpf(261136679, sector=1)
            >>> print(f"TPF shape: {flux.shape}")
        """
        lk = self._ensure_lightkurve()
        target = f"TIC {tic_id}"

        logger.info(f"Downloading TPF for {target}, sector {sector}")

        # Search for the specific sector
        try:
            search_author = author if author is not None else self.author
            search_result = lk.search_targetpixelfile(
                target,
                mission="TESS",
                sector=sector,
                author=search_author,
            )
        except Exception as e:
            logger.error(f"MAST TPF search failed for {target} sector {sector}: {e}")
            raise MASTClientError(
                f"Failed to search MAST for TPF TIC {tic_id} sector {sector}: {e}"
            ) from e

        if search_result is None or len(search_result) == 0:
            raise LightCurveNotFoundError(f"No TPF found for TIC {tic_id} sector {sector}")

        def _coerce_float(val: object) -> float | None:
            if val is None:
                return None
            if hasattr(val, "value"):
                val = val.value
            if hasattr(val, "item"):
                try:
                    return float(val.item())
                except Exception:
                    return None
            if hasattr(val, "__getitem__") and hasattr(val, "__len__"):
                try:
                    if len(val) == 0:
                        return None
                    return float(val[0])
                except Exception:
                    return None
            try:
                return float(val)
            except Exception:
                return None

        preferred_exptime = 120.0 if exptime is None else float(exptime)

        def _matches_exptime(row: Any, target_exptime: float) -> bool:
            if not hasattr(row, "exptime"):
                # If the attribute is missing, we cannot filter reliably; accept only when
                # the caller didn't request a specific cadence.
                return exptime is None
            v = _coerce_float(getattr(row, "exptime", None))
            if v is None:
                return False
            return abs(float(v) - float(target_exptime)) < 1.0

        # Choose product row(s).
        # - If an exptime is requested, prefer those rows.
        # - Otherwise, prefer 120s cadence where available.
        candidates: list[Any] = []
        try:
            for i in range(len(search_result)):
                candidates.append(search_result[i])
        except Exception:
            candidates = []

        if not candidates:
            raise LightCurveNotFoundError(f"No TPF rows found for TIC {tic_id} sector {sector}")

        selected_rows: list[Any]
        if exptime is not None:
            selected_rows = [r for r in candidates if _matches_exptime(r, preferred_exptime)]
        else:
            selected_rows = [r for r in candidates if _matches_exptime(r, 120.0)]
            if not selected_rows:
                selected_rows = candidates

        if not selected_rows:
            raise LightCurveNotFoundError(
                f"No TPF products matched exptime={exptime} for TIC {tic_id} sector {sector}"
            )

        # Prefer closest distance-to-target if available, otherwise take first.
        def _distance_val(row: Any) -> float:
            v = None
            if hasattr(row, "distance"):
                v = _coerce_float(getattr(row, "distance", None))
            return float(v) if v is not None else float("inf")

        selected_rows.sort(key=_distance_val)
        selected = selected_rows[0]

        try:
            tpf = selected.download()
        except Exception as e:
            logger.error(f"TPF download failed for {target} sector {sector}: {e}")
            raise MASTClientError(f"Failed to download TPF TIC {tic_id} sector {sector}: {e}") from e

        if tpf is None:
            raise MASTClientError(f"TPF download returned None for TIC {tic_id} sector {sector}")

        try:
            time_val = getattr(tpf, "time", None)
            if hasattr(time_val, "value"):
                time_val = time_val.value
            time_arr = np.asarray(time_val, dtype=np.float64)

            flux_val = getattr(tpf, "flux", None)
            if hasattr(flux_val, "value"):
                flux_val = flux_val.value
            flux_arr = np.asarray(flux_val, dtype=np.float64)

            flux_err_val = getattr(tpf, "flux_err", None)
            if flux_err_val is None:
                flux_err_arr = None
            else:
                if hasattr(flux_err_val, "value"):
                    flux_err_val = flux_err_val.value
                flux_err_arr = np.asarray(flux_err_val, dtype=np.float64)

            quality_val = getattr(tpf, "quality", None)
            if quality_val is None:
                quality_arr = None
            else:
                if hasattr(quality_val, "value"):
                    quality_val = quality_val.value
                quality_arr = np.asarray(quality_val, dtype=np.int32)

            wcs = getattr(tpf, "wcs", None)

            aperture_mask = None
            if hasattr(tpf, "pipeline_mask") and tpf.pipeline_mask is not None:
                aperture_mask = np.asarray(tpf.pipeline_mask, dtype=np.int32)
        except Exception as e:
            logger.error(f"Failed to extract arrays from downloaded TPF for {target}: {e}")
            raise MASTClientError(f"Failed to process TPF TIC {tic_id} sector {sector}: {e}") from e

        return time_arr, flux_arr, flux_err_arr, wcs, aperture_mask, quality_arr

    def download_tpf_cached(
        self,
        tic_id: int,
        sector: int,
        *,
        exptime: float | None = None,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray | None, Any | None, np.ndarray | None, np.ndarray | None
    ]:
        """Load a cached TPF from disk (no network)."""
        root = self._mast_download_root()
        if root is None:
            raise LightCurveNotFoundError("cache_dir is not configured or has no mastDownload/")
        self._build_cache_index()
        tic16 = f"{int(tic_id):016d}"
        dirs = self._cache_dirs_by_tic.get(tic16, [])
        if not dirs:
            raise LightCurveNotFoundError(f"No cached TPF directories for TIC {tic_id}")

        candidates: list[Path] = []
        for d in dirs:
            try:
                sec = self._parse_sector_from_name(d.name)
                if sec is None or int(sec) != int(sector):
                    continue
                for f in d.glob("*tp.fits*"):
                    if tic16 in f.name:
                        candidates.append(f)
            except Exception:
                continue

        if not candidates:
            raise LightCurveNotFoundError(f"No cached TPF for TIC {tic_id} sector {sector}")

        preferred = float(exptime) if exptime is not None else 120.0

        def _cand_key(p: Path) -> tuple[float, int]:
            dt = abs(self._infer_exptime_seconds(p) - preferred)
            fast = 1 if self._infer_exptime_seconds(p) < 60.0 else 0
            return (dt, fast)

        path = min(candidates, key=_cand_key)
        lk = self._ensure_lightkurve()
        try:
            tpf = lk.read(str(path))
        except Exception as e:
            raise MASTClientError(f"Failed to read cached TPF FITS: {path}: {e}") from e

        # Extract arrays; keep wcs/pipeline_mask when available.
        time = np.asarray(tpf.time.value, dtype=np.float64)
        flux = np.asarray(tpf.flux.value, dtype=np.float64)
        flux_err = (
            np.asarray(tpf.flux_err.value, dtype=np.float64)
            if getattr(tpf, "flux_err", None) is not None
            else None
        )
        wcs = getattr(tpf, "wcs", None)
        aperture_mask = np.asarray(tpf.pipeline_mask, dtype=np.bool_) if hasattr(tpf, "pipeline_mask") else None
        quality = np.asarray(tpf.quality, dtype=np.int32) if hasattr(tpf, "quality") else None
        return time, flux, flux_err, wcs, aperture_mask, quality
