"""Light curve domain models.

This module provides:
- LightCurveData: Internal representation with numpy arrays
- make_data_ref: Helper for generating deterministic cache keys
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from numpy.typing import NDArray


def make_data_ref(tic_id: int, sector: int, flux_type: str = "pdcsap") -> str:
    """Generate deterministic cache key for light curve data.

    Args:
        tic_id: TIC identifier
        sector: TESS sector number
        flux_type: Flux type (pdcsap, sap)

    Returns:
        Cache key in format "lc:{tic_id}:{sector}:{flux_type}"
    """
    return f"lc:{tic_id}:{sector}:{flux_type}"


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


@dataclass(frozen=True)
class LightCurveProvenance:
    """Provenance metadata for LightCurveData.

    This is intended to answer "what exact data product did we ingest and how?".
    It is kept lightweight and pickle-friendly for use with PersistentCache.
    """

    source: str
    selected_author: str | None
    selected_exptime_seconds: float | None
    preferred_author: str | None
    requested_exptime_seconds: float | None
    flux_type: str
    quality_mask: int
    normalize: bool
    selection_reason: str | None = None
    flux_err_kind: Literal["provided", "estimated_missing"] = "provided"


@dataclass
class LightCurveData:
    """Internal representation of light curve data with numpy arrays.

    This is the working representation used within the handler layer.
    Treat this as an internal computation structure.

    Attributes:
        time: BTJD timestamps (float64)
        flux: Normalized flux values (float64, median ~1.0)
        flux_err: Flux uncertainties (float64)
        quality: TESS quality flags (int32)
        valid_mask: Boolean mask for valid data points
        tic_id: TIC identifier
        sector: TESS sector number
        cadence_seconds: Observation cadence in seconds
        provenance: Optional provenance metadata about the source product
    """

    time: NDArray[np.float64]
    flux: NDArray[np.float64]
    flux_err: NDArray[np.float64]
    quality: NDArray[np.int32]
    valid_mask: NDArray[np.bool_]
    tic_id: int
    sector: int
    cadence_seconds: float
    provenance: LightCurveProvenance | None = None

    def __post_init__(self) -> None:
        """Validate array dtypes, shapes, and make arrays immutable."""
        # Validate that inputs are numpy arrays
        arrays: dict[str, np.ndarray[Any, Any]] = {
            "time": self.time,
            "flux": self.flux,
            "flux_err": self.flux_err,
            "quality": self.quality,
            "valid_mask": self.valid_mask,
        }
        for name, arr in arrays.items():
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"{name} must be a numpy array, got {type(arr).__name__}")

        # Dtype checks
        if self.time.dtype != np.float64:
            raise ValueError(f"time must be float64, got {self.time.dtype}")
        if self.flux.dtype != np.float64:
            raise ValueError(f"flux must be float64, got {self.flux.dtype}")
        if self.flux_err.dtype != np.float64:
            raise ValueError(f"flux_err must be float64, got {self.flux_err.dtype}")
        if self.quality.dtype != np.int32:
            raise ValueError(f"quality must be int32, got {self.quality.dtype}")
        if self.valid_mask.dtype != np.bool_:
            raise ValueError(f"valid_mask must be bool, got {self.valid_mask.dtype}")

        # Shape consistency
        n = len(self.time)
        if len(self.flux) != n:
            raise ValueError(f"flux length {len(self.flux)} != time length {n}")
        if len(self.flux_err) != n:
            raise ValueError(f"flux_err length {len(self.flux_err)} != time length {n}")
        if len(self.quality) != n:
            raise ValueError(f"quality length {len(self.quality)} != time length {n}")
        if len(self.valid_mask) != n:
            raise ValueError(f"valid_mask length {len(self.valid_mask)} != time length {n}")

        # Make all arrays read-only to prevent accidental mutation
        # This is especially important for cached data shared across consumers
        for arr in arrays.values():
            arr.flags.writeable = False

    @property
    def n_points(self) -> int:
        """Total number of data points."""
        return len(self.time)

    @property
    def n_valid(self) -> int:
        """Number of valid (unmasked) data points."""
        return int(np.sum(self.valid_mask))

    @property
    def duration_days(self) -> float:
        """Total observation duration in days."""
        if self.n_points == 0:
            return 0.0
        if self.n_valid == 0:
            return 0.0
        finite = self.valid_mask & np.isfinite(self.time)
        if not np.any(finite):
            return 0.0
        t = self.time[finite]
        return float(np.max(t) - np.min(t))

    @property
    def median_flux(self) -> float:
        """Median flux of valid data points."""
        if self.n_valid == 0:
            return float("nan")
        finite = self.valid_mask & np.isfinite(self.flux)
        if not np.any(finite):
            return float("nan")
        return float(np.median(self.flux[finite]))

    @property
    def flux_std(self) -> float:
        """Standard deviation of flux for valid data points."""
        if self.n_valid == 0:
            return float("nan")
        finite = self.valid_mask & np.isfinite(self.flux)
        if not np.any(finite):
            return float("nan")
        return float(np.std(self.flux[finite]))

    @property
    def gap_fraction(self) -> float:
        """Fraction of data that are gaps (invalid points)."""
        if self.n_points == 0:
            return 0.0
        return 1.0 - (self.n_valid / self.n_points)

    @property
    def quality_flags_present(self) -> list[int]:
        """List of unique quality flag values present in data."""
        return sorted({int(q) for q in np.unique(self.quality)})
