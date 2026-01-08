"""Target Pixel File (TPF) handling for TESS pixel-level data.

This module provides:
- TPFRef: Reference format for addressing TPF data by TIC ID, sector, camera, and CCD.
- TPFCache: Session-scoped cache for TPF data arrays.
- TPFHandler: Abstract handler for TPF acquisition.
- TPFNotFoundError: Error raised when TPF data is not available.

TPF Reference Format:
    tpf:<tic_id>:<sector>:<camera>:<ccd>

Example:
    tpf:123456789:15:2:3

This identifies the Target Pixel File for TIC 123456789 observed in sector 15,
captured by camera 2, CCD 3.

Usage:
    from bittr_tess_vetter.pixel import TPFRef, TPFCache, TPFHandler

    # Parse a reference
    ref = TPFRef.from_string("tpf:123456789:15:2:3")
    print(ref.tic_id)  # 123456789
    print(ref.sector)  # 15

    # Use the cache
    cache = TPFCache(cache_dir=Path("/tmp/tpf_cache"))
    if cache.has(ref):
        data = cache.get(ref)
    else:
        data = handler.fetch(ref)
        cache.put(data)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from bittr_tess_vetter.errors import ErrorType, make_error

logger = logging.getLogger(__name__)


class TPFNotFoundError(Exception):
    """Error raised when TPF data is not available.

    This provides convenient access to the TPF reference that was not found
    and an error envelope with structured error information.

    Attributes:
        ref: The TPFRef that was not found.
        envelope: Structured error envelope with error details.
        error_type: The error type (ErrorType.CACHE_MISS).
        message: The error message.
        context: The error context dictionary.
    """

    def __init__(self, ref: TPFRef, message: str | None = None) -> None:
        """Initialize TPFNotFoundError.

        Args:
            ref: The TPFRef that was not found.
            message: Optional custom message. Defaults to a standard message.
        """
        self.ref = ref
        if message is None:
            message = f"TPF not found: {ref.to_string()}"

        self._message = message
        self.envelope = make_error(
            ErrorType.CACHE_MISS,  # TPF not found is a cache miss variant
            message,
            ref=ref.to_string(),
            tic_id=ref.tic_id,
            sector=ref.sector,
            camera=ref.camera,
            ccd=ref.ccd,
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
class TPFRef:
    """Reference to a Target Pixel File (TPF).

    A TPFRef uniquely identifies a TPF by TIC ID, sector, camera, and CCD.
    This is an immutable value object that can be used as a dictionary key.

    Attributes:
        tic_id: TESS Input Catalog identifier.
        sector: TESS observing sector number.
        camera: TESS camera number (1-4).
        ccd: CCD number within the camera (1-4).

    Example:
        >>> ref = TPFRef(tic_id=123456789, sector=15, camera=2, ccd=3)
        >>> ref.to_string()
        'tpf:123456789:15:2:3'
        >>> TPFRef.from_string('tpf:123456789:15:2:3')
        TPFRef(tic_id=123456789, sector=15, camera=2, ccd=3)
    """

    tic_id: int
    sector: int
    camera: int
    ccd: int

    PREFIX = "tpf"

    def __post_init__(self) -> None:
        """Validate field values after initialization."""
        if self.tic_id < 1:
            raise ValueError(f"tic_id must be positive, got {self.tic_id}")
        if self.sector < 1:
            raise ValueError(f"sector must be positive, got {self.sector}")
        if not 1 <= self.camera <= 4:
            raise ValueError(f"camera must be 1-4, got {self.camera}")
        if not 1 <= self.ccd <= 4:
            raise ValueError(f"ccd must be 1-4, got {self.ccd}")

    def to_string(self) -> str:
        """Convert to string reference format.

        Returns:
            String in format 'tpf:<tic_id>:<sector>:<camera>:<ccd>'
        """
        return f"{self.PREFIX}:{self.tic_id}:{self.sector}:{self.camera}:{self.ccd}"

    @classmethod
    def from_string(cls, ref_str: str) -> TPFRef:
        """Parse a TPF reference string.

        Args:
            ref_str: String in format 'tpf:<tic_id>:<sector>:<camera>:<ccd>'

        Returns:
            TPFRef instance.

        Raises:
            ValueError: If the string format is invalid.
        """
        parts = ref_str.split(":")
        if len(parts) != 5:
            raise ValueError(
                f"Invalid TPF reference format: {ref_str!r}. "
                f"Expected 'tpf:<tic_id>:<sector>:<camera>:<ccd>'"
            )

        prefix, tic_id_str, sector_str, camera_str, ccd_str = parts

        if prefix != cls.PREFIX:
            raise ValueError(f"Invalid TPF reference prefix: {prefix!r}. Expected '{cls.PREFIX}'")

        try:
            tic_id = int(tic_id_str)
        except ValueError as e:
            raise ValueError(f"Invalid tic_id: {tic_id_str!r}") from e

        try:
            sector = int(sector_str)
        except ValueError as e:
            raise ValueError(f"Invalid sector: {sector_str!r}") from e

        try:
            camera = int(camera_str)
        except ValueError as e:
            raise ValueError(f"Invalid camera: {camera_str!r}") from e

        try:
            ccd = int(ccd_str)
        except ValueError as e:
            raise ValueError(f"Invalid ccd: {ccd_str!r}") from e

        return cls(tic_id=tic_id, sector=sector, camera=camera, ccd=ccd)

    def __str__(self) -> str:
        """Return string representation."""
        return self.to_string()


@dataclass(frozen=True)
class TPFData:
    """In-memory representation of a cached Target Pixel File time series.

    Attributes:
        ref: TPF reference (tic_id/sector/camera/ccd).
        time: BTJD timestamps, shape (n_cadences,).
        flux: Flux cube, shape (n_cadences, n_rows, n_cols).
    """

    ref: TPFRef
    time: np.ndarray[Any, Any]
    flux: np.ndarray[Any, Any]


class TPFCache:
    """Session-scoped cache for TPF data.

    Stores TPF numpy arrays on disk in a specified cache directory.
    Files are named based on the TPFRef fields for easy lookup.

    The cache uses numpy's savez format for serialization to preserve
    numpy array dtype and shape exactly.

    Attributes:
        cache_dir: Directory where cached TPF data is stored.

    Example:
        >>> cache = TPFCache(cache_dir=Path("/tmp/tpf_cache"))
        >>> ref = TPFRef(tic_id=123456789, sector=15, camera=2, ccd=3)
        >>> if not cache.has(ref):
        ...     data = fetch_from_mast(ref)
        ...     cache.put(data)
        >>> cached_data = cache.get(ref)
    """

    def __init__(self, cache_dir: Path) -> None:
        """Initialize TPFCache.

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

    def _cache_path(self, ref: TPFRef) -> Path:
        """Get the cache file path for a TPFRef.

        Args:
            ref: TPF reference.

        Returns:
            Path to the cache file.
        """
        filename = f"tpf_{ref.tic_id}_{ref.sector}_{ref.camera}_{ref.ccd}.npz"
        return self._cache_dir / filename

    def get(self, ref: TPFRef) -> TPFData | None:
        """Get cached TPF data (time + flux cube).

        Args:
            ref: TPF reference to look up.

        Returns:
            Cached TPFData if present, None if not cached.
        """
        cache_path = self._cache_path(ref)
        if not cache_path.exists():
            return None

        try:
            with np.load(cache_path) as data:
                flux: np.ndarray[Any, Any] = data["tpf_flux"]
                time: np.ndarray[Any, Any] = data["time"]
                return TPFData(ref=ref, time=time, flux=flux)
        except (ValueError, KeyError, OSError) as e:
            # Corrupted or unreadable cache file
            logger.warning(
                "Failed to load cached TPF data for %s: %s",
                ref.to_string(),
                e,
            )
            return None

    def put(self, tpf: TPFData) -> None:
        """Cache TPF data.

        Args:
            tpf: TPFData containing time + flux cube.
        """
        cache_path = self._cache_path(tpf.ref)
        # Use .tmp suffix, but np.savez adds .npz automatically
        tmp_base = cache_path.with_suffix(".tmp")
        tmp_path = tmp_base.with_suffix(".tmp.npz")  # What savez will actually create

        try:
            # Ensure cache directory exists
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(tmp_base, tpf_flux=tpf.flux, time=tpf.time)  # Creates tmp_base.npz = tmp_path
            tmp_path.replace(cache_path)
        except OSError as e:
            # Clean up temp file on failure
            logger.warning(
                "Failed to cache TPF data for %s: %s",
                tpf.ref.to_string(),
                e,
            )
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError as unlink_err:
                    logger.debug(
                        "Failed to clean up temp file %s: %s",
                        tmp_path,
                        unlink_err,
                    )
            raise

    def has(self, ref: TPFRef) -> bool:
        """Check if TPF is cached.

        Args:
            ref: TPF reference to check.

        Returns:
            True if the TPF data is cached, False otherwise.
        """
        return self._cache_path(ref).exists()

    def remove(self, ref: TPFRef) -> bool:
        """Remove cached TPF data.

        Args:
            ref: TPF reference to remove.

        Returns:
            True if the cache entry was removed, False if it didn't exist.
        """
        cache_path = self._cache_path(ref)
        if cache_path.exists():
            try:
                cache_path.unlink()
                return True
            except OSError as e:
                logger.warning(
                    "Failed to remove cached TPF data for %s: %s",
                    ref.to_string(),
                    e,
                )
                return False
        return False

    def clear(self) -> int:
        """Clear all cached TPF data.

        Returns:
            Number of cache entries removed.
        """
        count = 0
        for cache_file in self._cache_dir.glob("tpf_*.npz"):
            try:
                cache_file.unlink()
                count += 1
            except OSError as e:
                logger.debug(
                    "Failed to remove cache file %s during clear: %s",
                    cache_file,
                    e,
                )
        return count


class TPFHandler(ABC):
    """Abstract handler for TPF acquisition.

    Subclasses implement the fetch method to retrieve TPF data from
    external sources (e.g., MAST). This is an abstract base class;
    the actual MAST implementation is external to this core library.

    Example:
        class MASTTPFHandler(TPFHandler):
            def fetch(self, ref: TPFRef) -> np.ndarray:
                # Implementation using lightkurve or astroquery
                ...
    """

    @abstractmethod
    def fetch(self, ref: TPFRef) -> TPFData:
        """Fetch TPF data.

        Args:
            ref: TPF reference to fetch.

        Returns:
            TPFData containing time + flux cube.

        Raises:
            TPFNotFoundError: If TPF data is not available.
        """
        ...


class CachedTPFHandler(TPFHandler):
    """TPF handler that wraps another handler with caching.

    This handler first checks the cache for TPF data, and only
    fetches from the underlying handler if not cached. Fetched
    data is automatically cached for future requests.

    Attributes:
        cache: TPFCache instance for caching.
        handler: Underlying TPFHandler for fetching.

    Example:
        >>> cache = TPFCache(cache_dir=Path("/tmp/tpf_cache"))
        >>> mast_handler = MASTTPFHandler()
        >>> cached_handler = CachedTPFHandler(cache=cache, handler=mast_handler)
        >>> data = cached_handler.fetch(ref)  # Fetches from MAST, caches result
        >>> data = cached_handler.fetch(ref)  # Returns from cache
    """

    def __init__(self, cache: TPFCache, handler: TPFHandler) -> None:
        """Initialize CachedTPFHandler.

        Args:
            cache: TPFCache instance for caching.
            handler: Underlying TPFHandler for fetching.
        """
        self._cache = cache
        self._handler = handler

    @property
    def cache(self) -> TPFCache:
        """Return the cache instance."""
        return self._cache

    @property
    def handler(self) -> TPFHandler:
        """Return the underlying handler."""
        return self._handler

    def fetch(self, ref: TPFRef) -> TPFData:
        """Fetch TPF data with caching.

        Args:
            ref: TPF reference to fetch.

        Returns:
            Numpy array containing TPF pixel data.

        Raises:
            TPFNotFoundError: If TPF data is not available.
        """
        # Check cache first
        cached = self._cache.get(ref)
        if cached is not None:
            return cached

        # Fetch from underlying handler
        data = self._handler.fetch(ref)

        # Cache the result
        self._cache.put(data)

        return data
