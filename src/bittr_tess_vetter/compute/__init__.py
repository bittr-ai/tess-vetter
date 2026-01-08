"""Compute operations for periodograms, transit detection, and detrending.

This module provides astronomical computation primitives for:
- Periodogram analysis (BLS, Lomb-Scargle, auto-selection)
- Transit detection and characterization
- Light curve detrending and normalization
"""

from __future__ import annotations

from typing import Any

# Module-level exports
__all__: list[str] = []

# =============================================================================
# Primitives Module
# =============================================================================
# Type annotation: these may be None if import fails
astro: Any
AstroPrimitives: Any
try:
    from .primitives import AstroPrimitives, astro

    __all__.extend(["astro", "AstroPrimitives"])
except ImportError:
    astro = None
    AstroPrimitives = None

# =============================================================================
# Periodogram Module
# =============================================================================
ls_periodogram: Any
auto_periodogram: Any
compute_bls_model: Any
refine_period: Any
tls_search: Any
tls_search_per_sector: Any
search_planets: Any
detect_sector_gaps: Any
split_by_sectors: Any
merge_candidates: Any
try:
    from .periodogram import (
        auto_periodogram,
        compute_bls_model,
        detect_sector_gaps,
        ls_periodogram,
        merge_candidates,
        refine_period,
        search_planets,
        split_by_sectors,
        tls_search,
        tls_search_per_sector,
    )

    __all__.extend(
        [
            "ls_periodogram",
            "auto_periodogram",
            "compute_bls_model",
            "refine_period",
            "tls_search",
            "tls_search_per_sector",
            "search_planets",
            "detect_sector_gaps",
            "split_by_sectors",
            "merge_candidates",
        ]
    )
except ImportError:
    ls_periodogram = None
    auto_periodogram = None
    compute_bls_model = None
    refine_period = None
    tls_search = None
    tls_search_per_sector = None
    search_planets = None
    detect_sector_gaps = None
    split_by_sectors = None
    merge_candidates = None

# =============================================================================
# Transit Module
# =============================================================================
detect_transit: Any
measure_depth: Any
get_transit_mask: Any
fold_transit: Any
try:
    from .transit import detect_transit, fold_transit, get_transit_mask, measure_depth

    __all__.extend(["detect_transit", "measure_depth", "get_transit_mask", "fold_transit"])
except ImportError:
    detect_transit = None
    measure_depth = None
    get_transit_mask = None
    fold_transit = None

# =============================================================================
# Detrend Module
# =============================================================================
median_detrend: Any
normalize_flux: Any
sigma_clip: Any
flatten: Any
wotan_flatten: Any
flatten_with_wotan: Any
WOTAN_AVAILABLE: Any
try:
    from .detrend import (
        WOTAN_AVAILABLE,
        flatten,
        flatten_with_wotan,
        median_detrend,
        normalize_flux,
        sigma_clip,
        wotan_flatten,
    )

    __all__.extend(
        [
            "median_detrend",
            "normalize_flux",
            "sigma_clip",
            "flatten",
            "wotan_flatten",
            "flatten_with_wotan",
            "WOTAN_AVAILABLE",
        ]
    )
except ImportError:
    median_detrend = None
    normalize_flux = None
    sigma_clip = None
    flatten = None
    wotan_flatten = None
    flatten_with_wotan = None
    WOTAN_AVAILABLE = False

# =============================================================================
# MLX Detection (Optional)
# =============================================================================
MlxTopKScoreResult: Any
smooth_box_template: Any
score_fixed_period: Any
score_top_k_periods: Any
integrated_gradients: Any
MLX_AVAILABLE: Any
try:
    from .mlx_detection import (
        MlxTopKScoreResult,
        integrated_gradients,
        score_fixed_period,
        score_top_k_periods,
        smooth_box_template,
    )

    MLX_AVAILABLE = True
    __all__.extend(
        [
            "MlxTopKScoreResult",
            "smooth_box_template",
            "score_fixed_period",
            "score_top_k_periods",
            "integrated_gradients",
            "MLX_AVAILABLE",
        ]
    )
except ImportError:
    MlxTopKScoreResult = None
    smooth_box_template = None
    score_fixed_period = None
    score_top_k_periods = None
    integrated_gradients = None
    MLX_AVAILABLE = False

# =============================================================================
# Primitives Catalog
# =============================================================================
# Catalog of available primitives for the astro_catalog tool
# Maps primitive names to their metadata (description and status)


class PrimitiveInfo:
    """Metadata for a compute primitive."""

    def __init__(self, description: str, *, implemented: bool = True) -> None:
        self.description = description
        self.implemented = implemented

    @property
    def status(self) -> str:
        """Return 'available' or 'planned' based on implementation status."""
        return "available" if self.implemented else "planned"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API output."""
        return {
            "description": self.description,
            "status": self.status,
        }


PRIMITIVES_CATALOG: dict[str, PrimitiveInfo] = {
    # Core primitives (implemented in primitives.py)
    "astro.periodogram": PrimitiveInfo(
        "Compute Lomb-Scargle periodogram using scipy.signal.lombscargle"
    ),
    "astro.fold": PrimitiveInfo("Phase-fold light curve at given period and epoch (t0)"),
    "astro.detrend": PrimitiveInfo("Remove long-term trends via median filter detrending"),
    "astro.box_model": PrimitiveInfo("Generate simple box transit model (no limb darkening)"),
    # TLS-based transit search (implemented in periodogram.py)
    "tls_search": PrimitiveInfo(
        "Transit Least Squares search - physical transit model with built-in FAP"
    ),
    "search_planets": PrimitiveInfo("Iterative multi-planet search using TLS with transit masking"),
    "ls_periodogram": PrimitiveInfo("Lomb-Scargle periodogram for rotation/variability"),
    "auto_periodogram": PrimitiveInfo(
        "Auto-select TLS (transits) or LS (rotation), returns PeriodogramResult"
    ),
    "refine_period": PrimitiveInfo("Refine period estimate with higher resolution TLS search"),
    "compute_bls_model": PrimitiveInfo("Compute box model for given transit parameters"),
    # Future phase folding operations
    "astro.fold_transit": PrimitiveInfo("Fold light curve centered on transit", implemented=False),
    # Future transit operations
    "astro.detect_transit": PrimitiveInfo(
        "Detect transit signals in light curve", implemented=False
    ),
    "astro.measure_depth": PrimitiveInfo("Measure transit depth and duration", implemented=False),
    "astro.transit_mask": PrimitiveInfo(
        "Create boolean mask for in-transit points", implemented=False
    ),
    # Detrending operations
    "wotan_flatten": PrimitiveInfo(
        "Transit-aware detrending using wotan (biweight, median, spline methods)"
    ),
    "flatten_with_wotan": PrimitiveInfo(
        "Wotan detrending with automatic fallback to median filter"
    ),
    "astro.median_detrend": PrimitiveInfo(
        "Remove median trend from light curve", implemented=False
    ),
    "astro.normalize": PrimitiveInfo("Normalize flux to median or mean", implemented=False),
    "astro.sigma_clip": PrimitiveInfo("Remove outliers using sigma clipping", implemented=False),
    "astro.flatten": PrimitiveInfo(
        "Flatten light curve using spline or polynomial", implemented=False
    ),
    # Future statistical operations
    "astro.cdpp": PrimitiveInfo(
        "Calculate Combined Differential Photometric Precision", implemented=False
    ),
    "astro.scatter": PrimitiveInfo("Calculate point-to-point scatter", implemented=False),
    "astro.snr": PrimitiveInfo("Calculate signal-to-noise ratio", implemented=False),
    # Future utility operations
    "astro.bin": PrimitiveInfo("Bin light curve to reduce noise", implemented=False),
    "astro.interpolate": PrimitiveInfo("Interpolate missing data points", implemented=False),
    "astro.stitch": PrimitiveInfo("Stitch multiple light curve segments", implemented=False),
}

__all__.extend(["PRIMITIVES_CATALOG", "PrimitiveInfo"])


def get_available_primitives() -> list[str]:
    """Return list of currently available (importable) primitives.

    Returns
    -------
    list[str]
        Names of primitives that are currently available for use.
    """
    available = []

    if astro is not None:
        available.append("astro")
    if AstroPrimitives is not None:
        available.append("AstroPrimitives")
    if tls_search is not None:
        available.append("tls_search")
    if search_planets is not None:
        available.append("search_planets")
    if ls_periodogram is not None:
        available.append("ls_periodogram")
    if auto_periodogram is not None:
        available.append("auto_periodogram")
    if compute_bls_model is not None:
        available.append("compute_bls_model")
    if refine_period is not None:
        available.append("refine_period")
    if detect_transit is not None:
        available.append("detect_transit")
    if measure_depth is not None:
        available.append("measure_depth")
    if get_transit_mask is not None:
        available.append("get_transit_mask")
    if fold_transit is not None:
        available.append("fold_transit")
    if median_detrend is not None:
        available.append("median_detrend")
    if normalize_flux is not None:
        available.append("normalize_flux")
    if sigma_clip is not None:
        available.append("sigma_clip")
    if flatten is not None:
        available.append("flatten")
    if wotan_flatten is not None:
        available.append("wotan_flatten")
    if flatten_with_wotan is not None:
        available.append("flatten_with_wotan")

    return available


def list_primitives_catalog() -> dict[str, PrimitiveInfo]:
    """Return the full primitives catalog with metadata.

    Returns
    -------
    dict[str, PrimitiveInfo]
        Mapping of primitive names to their metadata (description and status).
    """
    return PRIMITIVES_CATALOG.copy()


__all__.extend(["get_available_primitives", "list_primitives_catalog"])
