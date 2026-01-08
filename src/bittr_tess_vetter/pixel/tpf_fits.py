"""FITS-preserving Target Pixel File (TPF) handling with WCS support.

This module provides infrastructure for caching TPF data with full FITS/WCS
preservation for pixel-level localization and vetting.

Key Types:
- TPFFitsRef: Reference format for FITS-cached TPF data.
- TPFFitsData: Full TPF data including WCS, aperture mask, quality flags.
- TPFFitsCache: Disk cache with FITS files and JSON sidecar metadata.

TPF FITS Reference Format:
    tpf_fits:<tic_id>:<sector>:<author>

Example:
    tpf_fits:123456789:15:spoc

This identifies the FITS Target Pixel File for TIC 123456789 observed in
sector 15, processed by the SPOC pipeline.

Usage:
    from bittr_tess_vetter.pixel.tpf_fits import TPFFitsRef, TPFFitsCache, TPFFitsData

    # Parse a reference
    ref = TPFFitsRef.from_string("tpf_fits:123456789:15:spoc")
    print(ref.tic_id)  # 123456789

    # Use the cache
    cache = TPFFitsCache(cache_dir=Path("/tmp/tpf_fits_cache"))
    if cache.has(ref):
        data = cache.get(ref)
        print(data.wcs)  # astropy WCS object
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from bittr_tess_vetter.errors import ErrorType, make_error

logger = logging.getLogger(__name__)

# Valid pipeline authors for TESS TPFs
VALID_AUTHORS = frozenset({"spoc", "qlp", "tess-spoc", "tasoc"})


class TPFFitsNotFoundError(Exception):
    """Error raised when FITS TPF data is not available.

    Attributes:
        ref: The TPFFitsRef that was not found.
        envelope: Structured error envelope with error details.
        error_type: The error type (ErrorType.CACHE_MISS).
        message: The error message.
        context: The error context dictionary.
    """

    def __init__(self, ref: TPFFitsRef, message: str | None = None) -> None:
        """Initialize TPFFitsNotFoundError.

        Args:
            ref: The TPFFitsRef that was not found.
            message: Optional custom message. Defaults to a standard message.
        """
        self.ref = ref
        if message is None:
            message = f"TPF FITS not found: {ref.to_string()}"

        self._message = message
        self.envelope = make_error(
            ErrorType.CACHE_MISS,
            message,
            ref=ref.to_string(),
            tic_id=ref.tic_id,
            sector=ref.sector,
            author=ref.author,
        )
        super().__init__(message)

    @property
    def error_type(self) -> ErrorType:
        """Return the error type from the envelope."""
        return self.envelope.type

    @property
    def message(self) -> str:
        """Return the error message."""
        return self._message

    @property
    def context(self) -> dict[str, Any]:
        """Return the error context from the envelope."""
        return self.envelope.context


@dataclass(frozen=True)
class TPFFitsRef:
    """Reference to a FITS-cached Target Pixel File with WCS preserved.

    A TPFFitsRef uniquely identifies a TPF by TIC ID, sector, and pipeline author.
    This is an immutable value object that can be used as a dictionary key.

    Attributes:
        tic_id: TESS Input Catalog identifier.
        sector: TESS observing sector number.
        author: Pipeline author (e.g., "spoc", "qlp").

    Example:
        >>> ref = TPFFitsRef(tic_id=123456789, sector=15, author="spoc")
        >>> ref.to_string()
        'tpf_fits:123456789:15:spoc'
        >>> TPFFitsRef.from_string('tpf_fits:123456789:15:spoc')
        TPFFitsRef(tic_id=123456789, sector=15, author='spoc')
    """

    tic_id: int
    sector: int
    author: str

    PREFIX = "tpf_fits"

    def __post_init__(self) -> None:
        """Validate field values after initialization."""
        if self.tic_id < 1:
            raise ValueError(f"tic_id must be positive, got {self.tic_id}")
        if self.sector < 1:
            raise ValueError(f"sector must be positive, got {self.sector}")
        # Normalize author to lowercase for comparison
        normalized_author = self.author.lower()
        if normalized_author not in VALID_AUTHORS:
            raise ValueError(f"author must be one of {sorted(VALID_AUTHORS)}, got {self.author!r}")
        # Force lowercase for consistency (frozen dataclass workaround)
        object.__setattr__(self, "author", normalized_author)

    def to_string(self) -> str:
        """Convert to string reference format.

        Returns:
            String in format 'tpf_fits:<tic_id>:<sector>:<author>'
        """
        return f"{self.PREFIX}:{self.tic_id}:{self.sector}:{self.author}"

    @classmethod
    def from_string(cls, ref_str: str) -> TPFFitsRef:
        """Parse a TPF FITS reference string.

        Args:
            ref_str: String in format 'tpf_fits:<tic_id>:<sector>:<author>'

        Returns:
            TPFFitsRef instance.

        Raises:
            ValueError: If the string format is invalid.
        """
        parts = ref_str.split(":")
        if len(parts) != 4:
            raise ValueError(
                f"Invalid TPF FITS reference format: {ref_str!r}. "
                f"Expected 'tpf_fits:<tic_id>:<sector>:<author>'"
            )

        prefix, tic_id_str, sector_str, author = parts

        if prefix != cls.PREFIX:
            raise ValueError(
                f"Invalid TPF FITS reference prefix: {prefix!r}. Expected '{cls.PREFIX}'"
            )

        try:
            tic_id = int(tic_id_str)
        except ValueError as e:
            raise ValueError(f"Invalid tic_id: {tic_id_str!r}") from e

        try:
            sector = int(sector_str)
        except ValueError as e:
            raise ValueError(f"Invalid sector: {sector_str!r}") from e

        if not author:
            raise ValueError("author cannot be empty")

        return cls(tic_id=tic_id, sector=sector, author=author)

    def __str__(self) -> str:
        """Return string representation."""
        return self.to_string()


@dataclass
class TPFFitsData:
    """FITS TPF data with WCS and aperture masks preserved.

    This class holds all data necessary for WCS-aware pixel-level analysis
    including the full flux cube, WCS transformation, aperture mask from
    the SPOC pipeline, and quality flags.

    Attributes:
        ref: TPF FITS reference (tic_id/sector/author).
        time: BTJD timestamps, shape (n_cadences,).
        flux: Flux cube, shape (n_cadences, n_rows, n_cols).
        flux_err: Flux error cube, shape (n_cadences, n_rows, n_cols) or None.
        wcs: Astropy WCS object for coordinate transforms.
        aperture_mask: SPOC pipeline aperture mask, shape (n_rows, n_cols).
        quality: Quality flags for each cadence, shape (n_cadences,).
        camera: TESS camera number (1-4).
        ccd: CCD number within the camera (1-4).
        meta: FITS header subset with important keywords.
    """

    ref: TPFFitsRef
    time: np.ndarray[Any, np.dtype[np.floating[Any]]]
    flux: np.ndarray[Any, np.dtype[np.floating[Any]]]
    flux_err: np.ndarray[Any, np.dtype[np.floating[Any]]] | None
    wcs: WCS
    aperture_mask: np.ndarray[Any, np.dtype[np.integer[Any]]]
    quality: np.ndarray[Any, np.dtype[np.integer[Any]]]
    camera: int
    ccd: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_cadences(self) -> int:
        """Number of cadences in the light curve."""
        return int(self.time.shape[0])

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the flux cube (n_cadences, n_rows, n_cols)."""
        return (int(self.flux.shape[0]), int(self.flux.shape[1]), int(self.flux.shape[2]))

    @property
    def aperture_npixels(self) -> int:
        """Number of pixels in the aperture mask."""
        return int(np.sum(self.aperture_mask > 0))


# Keys from the FITS header to preserve in the sidecar JSON
SIDECAR_HEADER_KEYS = frozenset(
    {
        "CAMERA",
        "CCD",
        "CRPIX1",
        "CRPIX2",
        "CRVAL1",
        "CRVAL2",
        "CDELT1",
        "CDELT2",
        "CD1_1",
        "CD1_2",
        "CD2_1",
        "CD2_2",
        "CTYPE1",
        "CTYPE2",
        "RA_OBJ",
        "DEC_OBJ",
        "TELESCOP",
        "INSTRUME",
        "SECTOR",
        "TSTART",
        "TSTOP",
        "EXPOSURE",
        "TIMEDEL",
    }
)


def _compute_wcs_checksum(wcs: WCS) -> str:
    """Compute a checksum for WCS to detect changes.

    Uses a SHA-256 hash of the WCS header representation for stability.

    Args:
        wcs: Astropy WCS object.

    Returns:
        SHA-256 checksum string prefixed with 'sha256:'.
    """
    try:
        # Get the WCS header as a string representation
        header_str = wcs.to_header_string()
        checksum = hashlib.sha256(header_str.encode("utf-8")).hexdigest()
        return f"sha256:{checksum}"
    except Exception:
        # Fallback if WCS serialization fails
        return "sha256:unknown"


def _extract_header_subset(header: fits.Header) -> dict[str, Any]:
    """Extract a subset of FITS header keys for sidecar JSON.

    Args:
        header: FITS header from the TPF.

    Returns:
        Dictionary with selected header values.
    """
    result: dict[str, Any] = {}
    for key in SIDECAR_HEADER_KEYS:
        if key in header:
            value = header[key]
            # Convert numpy types to native Python types for JSON
            if isinstance(value, np.integer):
                value = int(value)
            elif isinstance(value, np.floating):
                value = float(value)
            result[key] = value
    return result


class TPFFitsCache:
    """Disk cache for FITS TPFs with sidecar metadata.

    Stores TPF data as FITS files with a JSON sidecar containing metadata
    such as WCS checksum, header subset, source URL, and cache timestamp.

    The cache preserves the full FITS structure including WCS and aperture
    masks for accurate pixel-level analysis.

    Attributes:
        cache_dir: Directory where cached TPF data is stored.

    Example:
        >>> cache = TPFFitsCache(cache_dir=Path("/tmp/tpf_fits_cache"))
        >>> ref = TPFFitsRef(tic_id=123456789, sector=15, author="spoc")
        >>> if not cache.has(ref):
        ...     data = download_from_mast(ref)
        ...     cache.put(data)
        >>> cached_data = cache.get(ref)
        >>> sidecar = cache.get_sidecar(ref)
    """

    def __init__(self, cache_dir: Path) -> None:
        """Initialize TPFFitsCache.

        Args:
            cache_dir: Directory where cached TPF data will be stored.
                       Created if it doesn't exist.
        """
        self._cache_dir = Path(cache_dir).resolve()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_dir(self) -> Path:
        """Return the cache directory path."""
        return self._cache_dir

    def _base_path(self, ref: TPFFitsRef) -> Path:
        """Get the base file path for a TPFFitsRef (without extension).

        Args:
            ref: TPF FITS reference.

        Returns:
            Base path for cache files.
        """
        filename = f"tpf_fits_{ref.tic_id}_{ref.sector}_{ref.author}"
        return self._cache_dir / filename

    def _fits_path(self, ref: TPFFitsRef) -> Path:
        """Get the FITS file path for a TPFFitsRef.

        Args:
            ref: TPF FITS reference.

        Returns:
            Path to the FITS file.
        """
        return self._base_path(ref).with_suffix(".fits")

    def _sidecar_path(self, ref: TPFFitsRef) -> Path:
        """Get the sidecar JSON path for a TPFFitsRef.

        Args:
            ref: TPF FITS reference.

        Returns:
            Path to the sidecar JSON file.
        """
        return self._base_path(ref).with_suffix(".json")

    def get(self, ref: TPFFitsRef) -> TPFFitsData | None:
        """Get cached TPF FITS data.

        Args:
            ref: TPF FITS reference to look up.

        Returns:
            Cached TPFFitsData if present, None if not cached.
        """
        fits_path = self._fits_path(ref)
        if not fits_path.exists():
            return None

        try:
            with fits.open(fits_path) as hdu_list:
                # Extract data from FITS
                # Primary HDU has metadata
                primary_header = hdu_list[0].header

                # TARGETTABLES or second HDU has the data
                if len(hdu_list) < 2:
                    logger.warning(
                        "Invalid TPF FITS structure for %s: missing data extension",
                        ref.to_string(),
                    )
                    return None

                data_hdu = hdu_list[1]
                data = data_hdu.data

                # Get time, flux, flux_err
                time = np.asarray(data["TIME"], dtype=np.float64)
                flux = np.asarray(data["FLUX"], dtype=np.float64)
                flux_err = None
                if "FLUX_ERR" in data.columns.names:
                    flux_err = np.asarray(data["FLUX_ERR"], dtype=np.float64)

                # Get quality flags
                quality = np.asarray(data["QUALITY"], dtype=np.int32)

                # Get aperture mask from APERTURE extension if present
                aperture_mask: np.ndarray[Any, np.dtype[np.integer[Any]]]
                if len(hdu_list) > 2 and hdu_list[2].name == "APERTURE":
                    aperture_mask = np.asarray(hdu_list[2].data, dtype=np.int32)
                else:
                    # Create default mask (all pixels included)
                    aperture_mask = np.ones(flux.shape[1:], dtype=np.int32)

                # Extract WCS from the data HDU header
                wcs = WCS(data_hdu.header)

                # Get camera and CCD from header
                camera = int(primary_header.get("CAMERA", 1))
                ccd = int(primary_header.get("CCD", 1))

                # Extract metadata
                meta = _extract_header_subset(primary_header)
                # Also add any useful keys from data HDU
                meta.update(_extract_header_subset(data_hdu.header))

                return TPFFitsData(
                    ref=ref,
                    time=time,
                    flux=flux,
                    flux_err=flux_err,
                    wcs=wcs,
                    aperture_mask=aperture_mask,
                    quality=quality,
                    camera=camera,
                    ccd=ccd,
                    meta=meta,
                )

        except (ValueError, KeyError, OSError, fits.VerifyError) as e:
            logger.warning(
                "Failed to load cached TPF FITS for %s: %s",
                ref.to_string(),
                e,
            )
            return None

    def put(
        self,
        data: TPFFitsData,
        source_url: str | None = None,
    ) -> None:
        """Cache TPF FITS data.

        Creates both a FITS file and a JSON sidecar with metadata.

        Args:
            data: TPFFitsData to cache.
            source_url: URL where the TPF was downloaded from.
        """
        fits_path = self._fits_path(data.ref)
        sidecar_path = self._sidecar_path(data.ref)

        # Use temp files for atomic writes
        tmp_fits = fits_path.with_suffix(".tmp.fits")
        tmp_sidecar = sidecar_path.with_suffix(".tmp.json")

        try:
            # Ensure cache directory exists
            self._cache_dir.mkdir(parents=True, exist_ok=True)

            # Build FITS file
            # Primary HDU with metadata
            primary_hdu = fits.PrimaryHDU()
            primary_hdu.header["CAMERA"] = data.camera
            primary_hdu.header["CCD"] = data.ccd
            primary_hdu.header["TICID"] = data.ref.tic_id
            primary_hdu.header["SECTOR"] = data.ref.sector
            primary_hdu.header["AUTHOR"] = data.ref.author
            # Copy metadata
            for key, value in data.meta.items():
                if key not in primary_hdu.header and len(key) <= 8:
                    with contextlib.suppress(ValueError, TypeError):
                        primary_hdu.header[key] = value

            # Data table HDU
            # Create columns
            cols = [
                fits.Column(name="TIME", format="D", array=data.time),
                fits.Column(
                    name="FLUX",
                    format=f"{data.flux.shape[1] * data.flux.shape[2]}D",
                    dim=f"({data.flux.shape[2]},{data.flux.shape[1]})",
                    array=data.flux.reshape(data.flux.shape[0], -1),
                ),
                fits.Column(name="QUALITY", format="J", array=data.quality),
            ]
            if data.flux_err is not None:
                cols.append(
                    fits.Column(
                        name="FLUX_ERR",
                        format=f"{data.flux_err.shape[1] * data.flux_err.shape[2]}D",
                        dim=f"({data.flux_err.shape[2]},{data.flux_err.shape[1]})",
                        array=data.flux_err.reshape(data.flux_err.shape[0], -1),
                    )
                )

            table_hdu = fits.BinTableHDU.from_columns(cols)
            # Add WCS to the table HDU header
            wcs_header = data.wcs.to_header()
            for key in wcs_header:
                if key not in ("", "COMMENT", "HISTORY"):
                    table_hdu.header[key] = wcs_header[key]

            # Aperture mask HDU
            aperture_hdu = fits.ImageHDU(data=data.aperture_mask, name="APERTURE")

            # Write FITS
            hdu_list = fits.HDUList([primary_hdu, table_hdu, aperture_hdu])
            hdu_list.writeto(tmp_fits, overwrite=True)

            # Build sidecar JSON
            sidecar: dict[str, Any] = {
                "tpf_fits_ref": data.ref.to_string(),
                "cached_at": datetime.now(UTC).isoformat(),
                "wcs_checksum": _compute_wcs_checksum(data.wcs),
                "fits_header_subset": _extract_header_subset(fits.Header(data.wcs.to_header())),
                "aperture_mask_npixels": data.aperture_npixels,
                "camera": data.camera,
                "ccd": data.ccd,
                "n_cadences": data.n_cadences,
                "shape": list(data.shape),
            }
            if source_url:
                sidecar["source_url"] = source_url

            # Add any additional metadata from the header subset
            sidecar["fits_header_subset"].update(
                _extract_header_subset(fits.Header([(k, v) for k, v in data.meta.items()]))
            )

            # Write sidecar
            with open(tmp_sidecar, "w") as f:
                json.dump(sidecar, f, indent=2)

            # Atomic move
            tmp_fits.replace(fits_path)
            tmp_sidecar.replace(sidecar_path)

        except OSError as e:
            logger.warning(
                "Failed to cache TPF FITS for %s: %s",
                data.ref.to_string(),
                e,
            )
            # Clean up temp files
            for tmp in (tmp_fits, tmp_sidecar):
                if tmp.exists():
                    with contextlib.suppress(OSError):
                        tmp.unlink()
            raise

    def has(self, ref: TPFFitsRef) -> bool:
        """Check if TPF FITS is cached.

        Args:
            ref: TPF FITS reference to check.

        Returns:
            True if the TPF FITS data is cached, False otherwise.
        """
        return self._fits_path(ref).exists()

    def get_sidecar(self, ref: TPFFitsRef) -> dict[str, Any] | None:
        """Get the sidecar JSON metadata for a cached TPF.

        Args:
            ref: TPF FITS reference.

        Returns:
            Sidecar metadata dictionary if present, None otherwise.
        """
        sidecar_path = self._sidecar_path(ref)
        if not sidecar_path.exists():
            return None

        try:
            with open(sidecar_path) as f:
                result: dict[str, Any] = json.load(f)
                return result
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                "Failed to load sidecar for %s: %s",
                ref.to_string(),
                e,
            )
            return None

    def remove(self, ref: TPFFitsRef) -> bool:
        """Remove cached TPF FITS data and sidecar.

        Args:
            ref: TPF FITS reference to remove.

        Returns:
            True if cache entries were removed, False if they didn't exist.
        """
        fits_path = self._fits_path(ref)
        sidecar_path = self._sidecar_path(ref)

        removed = False
        for path in (fits_path, sidecar_path):
            if path.exists():
                try:
                    path.unlink()
                    removed = True
                except OSError as e:
                    logger.warning(
                        "Failed to remove cached file %s: %s",
                        path,
                        e,
                    )
        return removed

    def clear(self) -> int:
        """Clear all cached TPF FITS data.

        Returns:
            Number of cache entries removed (counting FITS files only).
        """
        count = 0
        for cache_file in self._cache_dir.glob("tpf_fits_*.fits"):
            try:
                cache_file.unlink()
                count += 1
                # Also remove sidecar
                sidecar = cache_file.with_suffix(".json")
                if sidecar.exists():
                    sidecar.unlink()
            except OSError as e:
                logger.debug(
                    "Failed to remove cache file %s during clear: %s",
                    cache_file,
                    e,
                )
        return count

    def list_refs(self) -> list[TPFFitsRef]:
        """List all cached TPF FITS references.

        Returns:
            List of TPFFitsRef objects for cached data.
        """
        refs = []
        for fits_file in self._cache_dir.glob("tpf_fits_*.fits"):
            try:
                # Parse filename: tpf_fits_<tic_id>_<sector>_<author>.fits
                stem = fits_file.stem
                parts = stem.split("_")
                if len(parts) >= 4:
                    # tpf, fits, tic_id, sector, author
                    tic_id = int(parts[2])
                    sector = int(parts[3])
                    author = "_".join(parts[4:])  # Handle authors with underscores
                    refs.append(TPFFitsRef(tic_id=tic_id, sector=sector, author=author))
            except (ValueError, IndexError):
                logger.debug("Failed to parse cache filename: %s", fits_file)
        return refs
