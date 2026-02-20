"""Periodogram computation for transit detection.

This module provides Transit Least Squares (TLS) and Lomb-Scargle periodogram
implementations for detecting periodic transit signals in light curves.

TLS replaces our custom BLS implementation, providing:
- Physical transit model (not box)
- Built-in FAP estimation
- Odd/even depth comparison
- Optimal period grid generation
- Stellar parameter integration from TIC

The Lomb-Scargle periodogram is kept for rotation/variability detection.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import signal

from tess_vetter.domain.detection import PeriodogramPeak, PeriodogramResult

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class PerformancePreset(str, Enum):
    fast = "fast"
    thorough = "thorough"
    deep = "deep"


# =============================================================================
# Sector Gap Detection and Splitting
# =============================================================================


def detect_sector_gaps(
    time: NDArray[np.float64],
    gap_threshold_days: float = 10.0,
) -> NDArray[np.intp]:
    """Find indices where gaps > threshold indicate sector boundaries.

    Args:
        time: Time array, in BTJD (days)
        gap_threshold_days: Minimum gap size to consider as sector boundary (default 10 days)

    Returns:
        Array of indices where gaps occur (i.e., gap is between index i and i+1)
    """
    if len(time) < 2:
        return np.array([], dtype=np.intp)

    gaps = np.diff(time)
    gap_indices = np.where(gaps > gap_threshold_days)[0]
    return gap_indices


def split_by_sectors(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64] | None = None,
    gap_threshold_days: float = 10.0,
) -> list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]]:
    """Split light curve arrays at sector boundaries.

    Detects large gaps in the time array (> gap_threshold_days) and splits
    the data into separate chunks, one per continuous observing sector.

    Args:
        time: Time array, in BTJD (days)
        flux: Normalized flux array (float64, median ~1.0)
        flux_err: Flux uncertainties (float64), or None for equal weights
        gap_threshold_days: Minimum gap size to split on (default 10 days)

    Returns:
        List of (time, flux, flux_err) tuples, one per sector.
        flux_err will be None in each tuple if input flux_err is None.
    """
    gap_indices = detect_sector_gaps(time, gap_threshold_days)

    if len(gap_indices) == 0:
        # Single sector - return as-is
        return [(time, flux, flux_err)]

    sectors: list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]] = []
    start = 0

    for idx in gap_indices:
        # Split at gap_index + 1 (gap is between idx and idx+1)
        end = idx + 1
        sector_time = time[start:end]
        sector_flux = flux[start:end]
        sector_err = flux_err[start:end] if flux_err is not None else None
        sectors.append((sector_time, sector_flux, sector_err))
        start = end

    # Add final sector
    sector_time = time[start:]
    sector_flux = flux[start:]
    sector_err = flux_err[start:] if flux_err is not None else None
    sectors.append((sector_time, sector_flux, sector_err))

    return sectors


def merge_candidates(
    all_results: list[dict],
    period_tolerance: float = 0.02,
) -> list[dict]:
    """Merge candidates from multiple sectors, deduplicate by period.

    Groups detections with similar periods and keeps the best (highest SDE)
    from each group. This prevents the same planet from being reported
    multiple times when detected in different sectors.

    Args:
        all_results: List of TLS result dictionaries from different sectors
        period_tolerance: Fractional tolerance for period matching (default 2%)

    Returns:
        Deduplicated list of candidates, sorted by SDE (best first)
    """
    if not all_results:
        return []

    # Sort by SDE descending to process best detections first
    sorted_results = sorted(all_results, key=lambda x: x.get("sde", 0), reverse=True)

    merged: list[dict] = []
    used_periods: list[float] = []

    for result in sorted_results:
        period = result.get("period", 0)
        if period <= 0:
            continue

        # Check if this period matches any already-selected candidate
        is_duplicate = False
        for existing_period in used_periods:
            relative_diff = abs(period - existing_period) / existing_period
            if relative_diff < period_tolerance:
                is_duplicate = True
                break

        if not is_duplicate:
            merged.append(result)
            used_periods.append(period)

    return merged


def cluster_cross_sector_candidates(
    candidates: list[dict],
    *,
    period_tol_frac: float = 0.003,
    min_sectors: int = 2,
) -> list[dict]:
    """Cluster transit candidates across sectors by period.

    Groups candidates from multiple TESS sectors that share similar orbital
    periods, computing cross-sector agreement metrics. This is useful for
    validating planet candidates by confirming detection consistency.

    Unlike `merge_candidates` (which simply deduplicates), this function:
    - Tracks all sector memberships per family
    - Computes phase scatter across sectors (coherence diagnostic)
    - Sums detection scores for combined significance

    Args:
        candidates: List of candidate dicts, each with:
            - period_days (float): Orbital period in days
            - t0_btjd (float): Mid-transit epoch in BTJD
            - score_z (float): Detection z-score (e.g., from BLS-like search)
            - sector (int): TESS sector number
            - Additional fields are preserved in members
        period_tol_frac: Relative period tolerance for grouping (default 0.3%)
        min_sectors: Minimum sectors to form a family (default 2)

    Returns:
        List of family dicts, sorted by sum_score_z descending, each with:
            - period_days_rep (float): Representative period from best-scoring member
            - n_sectors (int): Number of unique sectors in family
            - sectors (list[int]): Sorted list of sector numbers
            - sum_score_z (float): Sum of score_z across all members
            - phase_scatter_cycles_rms (float): RMS of phase offsets in cycles
              (low values indicate coherent ephemeris across sectors)
            - representative (dict): Full candidate dict from best-scoring member
            - members (list[dict]): All members with sector, score_z, t0_btjd

    Example:
        >>> candidates = [
        ...     {"period_days": 3.0, "t0_btjd": 1500.0, "score_z": 12.5, "sector": 1},
        ...     {"period_days": 3.001, "t0_btjd": 1530.0, "score_z": 11.2, "sector": 4},
        ...     {"period_days": 5.5, "t0_btjd": 1505.0, "score_z": 8.0, "sector": 1},
        ... ]
        >>> families = cluster_cross_sector_candidates(candidates, min_sectors=2)
        >>> len(families)  # Only the 3.0-day family meets min_sectors
        1
        >>> families[0]["n_sectors"]
        2

    Notes:
        - Candidates are first sorted by score_z descending; families adopt
          the period from the highest-scoring member
        - Phase scatter is computed as RMS of phase offsets from the
          best-scoring member's phase, wrapped to [-0.5, +0.5] cycles
    """
    if not candidates:
        return []

    # Sort by score_z descending so best candidate defines each family
    items = sorted(
        candidates,
        key=lambda x: float(x.get("score_z", float("-inf"))),
        reverse=True,
    )

    families: list[dict] = []
    for item in items:
        p = float(item["period_days"])
        assigned = False
        for fam in families:
            pref = float(fam["period_days_rep"])
            if abs(p - pref) / max(pref, 1e-9) <= period_tol_frac:
                fam["members"].append(item)
                assigned = True
                break
        if not assigned:
            families.append(
                {
                    "period_days_rep": p,
                    "members": [item],
                }
            )

    out_families: list[dict] = []
    for fam in families:
        members = fam["members"]
        # Re-sort members by score in case assignment order differed
        members.sort(key=lambda x: float(x.get("score_z", float("-inf"))), reverse=True)
        rep = members[0]
        p = float(rep["period_days"])

        # Phase scatter across sectors
        phases = [(float(m["t0_btjd"]) / p) % 1.0 for m in members]
        ref_phase = phases[0]
        # Wrap phase differences to [-0.5, +0.5]
        dphi = [(((ph - ref_phase) + 0.5) % 1.0) - 0.5 for ph in phases]
        phase_rms = float(np.sqrt(np.mean(np.square(dphi)))) if dphi else float("nan")

        sum_score = float(np.nansum([float(m.get("score_z", float("nan"))) for m in members]))
        sectors_present = sorted({int(m["sector"]) for m in members})

        out_families.append(
            {
                "period_days_rep": p,
                "n_sectors": int(len(sectors_present)),
                "sectors": sectors_present,
                "sum_score_z": sum_score,
                "phase_scatter_cycles_rms": phase_rms,
                "representative": rep,
                "members": [
                    {
                        "sector": int(m["sector"]),
                        "score_z": float(m.get("score_z", float("nan"))),
                        "t0_btjd": float(m["t0_btjd"]),
                        **({"member_key": m["member_key"]} if "member_key" in m else {}),
                    }
                    for m in members
                ],
            }
        )

    # Filter by min_sectors
    out_families = [f for f in out_families if int(f["n_sectors"]) >= min_sectors]

    # Sort by sum_score_z descending, then by phase scatter ascending, then by n_sectors
    out_families.sort(
        key=lambda f: (
            float(f.get("sum_score_z", float("-inf"))),
            -float(f.get("phase_scatter_cycles_rms", float("inf"))),
            float(f.get("n_sectors", 0)),
        ),
        reverse=True,
    )

    return out_families


def tls_search_per_sector(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64] | None = None,
    period_min: float = 0.5,
    period_max: float | None = None,
    tic_id: int | None = None,
    stellar_radius_rsun: float | None = None,
    stellar_mass_msun: float | None = None,
    use_threads: int | None = None,
    gap_threshold_days: float = 10.0,
    min_sector_points: int = 500,
    downsample_factor: int = 1,
) -> dict:
    """TLS search with per-sector strategy for multi-sector data.

    For stitched multi-sector data, long gaps create period aliases that
    can mask short-period planets. This function:
    1. Detects sector boundaries (gaps > gap_threshold_days)
    2. Runs TLS independently on each sector
    3. Merges results, keeping the best detection for each unique period

    Args:
        time: Time array, in BTJD (days)
        flux: Normalized flux array (float64, median ~1.0)
        flux_err: Flux uncertainties (float64), or None for equal weights
        period_min: Minimum period to search, in days
        period_max: Maximum period, in days (default: shortest sector baseline / 2)
        tic_id: Deprecated legacy input (no longer triggers network lookups)
        stellar_radius_rsun: Stellar radius in solar radii (optional)
        stellar_mass_msun: Stellar mass in solar masses (optional)
        use_threads: Number of threads for TLS (default: cpu_count)
        gap_threshold_days: Gap size threshold for sector detection (default 10 days)
        min_sector_points: Minimum points required to search a sector (default 500)
        downsample_factor: Downsample data by this factor for faster search (default: 1)

    Returns:
        Dictionary with best detection results including period, t0, duration,
        depth, SDE, SNR, FAP, and additional per_sector_results metadata.
    """
    # Split data by sectors
    sectors = split_by_sectors(time, flux, flux_err, gap_threshold_days)

    if len(sectors) == 1:
        # Single sector - use standard TLS search
        logger.debug("Single sector detected, using standard TLS search")
        return tls_search(
            time,
            flux,
            flux_err,
            period_min=period_min,
            period_max=period_max,
            tic_id=tic_id,
            stellar_radius_rsun=stellar_radius_rsun,
            stellar_mass_msun=stellar_mass_msun,
            use_threads=use_threads,
            downsample_factor=downsample_factor,
        )

    logger.info(f"Multi-sector data detected: {len(sectors)} sectors")

    # Determine max_period from shortest sector if not provided
    if period_max is None:
        sector_baselines = [
            float(s[0][-1] - s[0][0]) for s in sectors if len(s[0]) >= min_sector_points
        ]
        if sector_baselines:
            # Use shortest sector baseline / 2 to avoid aliases
            shortest_baseline = min(sector_baselines)
            period_max = shortest_baseline / 2
            logger.debug(f"Auto-set period_max={period_max:.1f}d from shortest sector")

    # Run TLS on each sector
    all_results: list[dict] = []
    n_periods_grid_total = 0

    for i, (sector_time, sector_flux, sector_err) in enumerate(sectors):
        if len(sector_time) < min_sector_points:
            logger.debug(f"Skipping sector {i + 1}: insufficient points ({len(sector_time)})")
            continue

        sector_baseline = float(sector_time[-1] - sector_time[0])
        logger.debug(
            f"Searching sector {i + 1}: {len(sector_time)} points, baseline={sector_baseline:.1f}d"
        )

        # Limit max_period to sector baseline / 2
        sector_max_period = (
            min(period_max, sector_baseline / 2) if period_max else sector_baseline / 2
        )

        if sector_max_period <= period_min:
            logger.debug(f"Skipping sector {i + 1}: max_period <= min_period")
            continue

        result = tls_search(
            sector_time,
            sector_flux,
            sector_err,
            period_min=period_min,
            period_max=sector_max_period,
            tic_id=tic_id,
            stellar_radius_rsun=stellar_radius_rsun,
            stellar_mass_msun=stellar_mass_msun,
            use_threads=use_threads,
            downsample_factor=downsample_factor,
        )

        # Add sector metadata
        result["sector_index"] = i
        result["sector_baseline"] = sector_baseline
        result["sector_n_points"] = len(sector_time)
        try:
            n_periods_grid_total += int(result.get("n_periods_grid") or 0)
        except Exception:
            n_periods_grid_total += 0

        # Only keep results with meaningful detections
        if result.get("sde", 0) > 5.0:
            all_results.append(result)
            logger.debug(f"Sector {i + 1}: P={result['period']:.4f}d, SDE={result['sde']:.1f}")

    if not all_results:
        # No significant detections in any sector - return empty result
        logger.info("No significant detections in any sector")
        return {
            "period": float(period_min),
            "t0": float(time[0]),
            "duration_hours": 0.0,
            "depth_ppm": 0.0,
            "sde": 0.0,
            "snr": 0.0,
            "fap": None,
            "rp_rs": None,
            "odd_even_mismatch": 0.0,
            "transit_count": 0,
            "transit_times": [],
            "in_transit_mask": np.zeros(len(time), dtype=bool),
            "_tls_results": None,
            "per_sector_results": [],
            "n_sectors_searched": len(sectors),
        }

    # Merge and deduplicate candidates
    merged = merge_candidates(all_results, period_tolerance=0.02)

    # Best result is first (highest SDE)
    best = merged[0]

    # Build combined in_transit_mask for full time array using best detection
    try:
        from transitleastsquares import transit_mask
    except ImportError as e:
        raise ImportError(
            "Transit detection requires the 'tls' extra. "
            "Install with: pip install 'tess-vetter[tls]'"
        ) from e

    in_transit_combined = transit_mask(
        time,
        best["period"],
        best["duration_hours"] / 24 * 2,  # 2x duration buffer
        best["t0"],
    )

    logger.info(
        f"Best per-sector detection: P={best['period']:.4f}d, "
        f"SDE={best['sde']:.1f}, from sector {best.get('sector_index', 0) + 1}"
    )

    return {
        "period": best["period"],
        "t0": best["t0"],
        "duration_hours": best["duration_hours"],
        "depth_ppm": best["depth_ppm"],
        "sde": best["sde"],
        "snr": best["snr"],
        "fap": best.get("fap"),
        "rp_rs": best.get("rp_rs"),
        "odd_even_mismatch": best.get("odd_even_mismatch", 0.0),
        "transit_count": best.get("transit_count", 0),
        "transit_times": best.get("transit_times", []),
        "in_transit_mask": in_transit_combined,
        "_tls_results": best.get("_tls_results"),
        "n_periods_grid_total": int(n_periods_grid_total),
        "per_sector_results": merged,
        "n_sectors_searched": len(sectors),
        "best_sector_index": best.get("sector_index", 0),
    }


# =============================================================================
# TLS Transit Search
# =============================================================================


def tls_search(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64] | None = None,
    period_min: float = 0.5,
    period_max: float | None = None,
    tic_id: int | None = None,
    stellar_radius_rsun: float | None = None,
    stellar_mass_msun: float | None = None,
    use_threads: int | None = None,
    downsample_factor: int = 1,
) -> dict:
    """Single TLS search. Returns best period and diagnostics.

    Uses Transit Least Squares (Hippke & Heller 2019) for transit detection.
    If stellar params are provided, uses them for better sensitivity.
    `tic_id` is accepted for backward compatibility but is not used for network lookups.

    Args:
        time: Time array, in BTJD (days)
        flux: Normalized flux array (float64, median ~1.0)
        flux_err: Flux uncertainties (float64), or None for equal weights
        period_min: Minimum period to search, in days
        period_max: Maximum period, in days (default: baseline/2)
        tic_id: Deprecated legacy input (no longer triggers network lookups)
        stellar_radius_rsun: Stellar radius in solar radii (optional)
        stellar_mass_msun: Stellar mass in solar masses (optional)
        use_threads: Number of threads for TLS (default: cpu_count)
        downsample_factor: Downsample data by this factor for faster search (default: 1)

    Returns:
        Dictionary with detection results including period, t0, duration,
        depth, SDE, SNR, FAP, and transit mask.
    """
    import warnings

    try:
        from transitleastsquares import transitleastsquares
    except ImportError as e:
        raise ImportError(
            "Transit detection requires the 'tls' extra. "
            "Install with: pip install 'tess-vetter[tls]'"
        ) from e

    # Downsample if requested (for faster coarse search)
    if downsample_factor > 1:
        time = time[::downsample_factor]
        flux = flux[::downsample_factor]
        if flux_err is not None:
            flux_err = flux_err[::downsample_factor]

    # Stellar params: do not perform any network lookups here.
    # The host application should provide stellar params (from TIC/Gaia/etc) explicitly.
    r_star = float(stellar_radius_rsun) if stellar_radius_rsun is not None else 1.0
    m_star = float(stellar_mass_msun) if stellar_mass_msun is not None else 1.0
    if tic_id is not None and (stellar_radius_rsun is None and stellar_mass_msun is None):
        logger.debug("tls_search: ignoring tic_id (no catalog lookup in compute layer)")

    # Auto-calculate max period from baseline if not provided
    baseline = float(time.max() - time.min())
    if period_max is None:
        period_max = baseline / 2

    # Ensure valid period range for TLS
    if period_max <= period_min:
        period_max = period_min * 2
    if period_max <= period_min or baseline < period_min * 2:
        # Not enough data for TLS search, return empty result
        return {
            "period": float(period_min),
            "t0": float(time[0]),
            "duration_hours": 0.0,
            "depth_ppm": 0.0,
            "sde": 0.0,
            "snr": 0.0,
            "fap": None,
            "rp_rs": None,
            "odd_even_mismatch": 0.0,
            "transit_count": 0,
            "transit_times": [],
            "in_transit_mask": np.zeros(len(time), dtype=bool),
            "_tls_results": None,
        }

    # Run TLS with stellar parameters
    try:
        # IMPORTANT: TLS prints progress/version info to stdout by default, which will
        # corrupt stdio-based transports. Force TLS quiet mode.
        # IMPORTANT: TLS prints progress/version info to stdout by default, which can
        # corrupt stdio-based transports. Force TLS quiet mode.
        model = transitleastsquares(time, flux, flux_err, verbose=False)

        # Default to single-threaded unless caller explicitly requests threads.
        effective_threads = 1 if use_threads is None else use_threads

        power_kwargs: dict[str, object] = {
            "period_min": period_min,
            "period_max": period_max,
            "R_star": r_star,
            "R_star_min": r_star * 0.9,
            "R_star_max": r_star * 1.1,
            "M_star": m_star,
            "M_star_min": m_star * 0.9,
            "M_star_max": m_star * 1.1,
            # Disable tqdm progress bar (also writes to stdout by default).
            "show_progress_bar": False,
            "use_threads": effective_threads,
        }

        # TLS can emit benign warnings; keep tool responses stable/clean.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = model.power(**power_kwargs)  # type: ignore[arg-type]
    except (ValueError, RuntimeError) as e:
        logger.warning(f"TLS search failed: {e}")
        return {
            "period": float(period_min),
            "t0": float(time[0]),
            "duration_hours": 0.0,
            "depth_ppm": 0.0,
            "sde": 0.0,
            "snr": 0.0,
            "fap": None,
            "rp_rs": None,
            "odd_even_mismatch": 0.0,
            "transit_count": 0,
            "transit_times": [],
            "in_transit_mask": np.zeros(len(time), dtype=bool),
            "_tls_results": None,
        }

    # Extract depth_mean safely
    depth_mean = 0.0
    if hasattr(results, "depth_mean") and len(results.depth_mean) > 0:
        depth_mean = float(results.depth_mean[0])

    # Calculate depth in ppm
    depth_ppm = float((1 - depth_mean) * 1e6) if depth_mean > 0 else 0.0

    # Handle optional attributes safely
    in_transit_mask = np.zeros(len(time), dtype=bool)
    if hasattr(results, "in_transit") and results.in_transit is not None:
        in_transit_mask = np.array(results.in_transit, dtype=bool)

    n_periods_grid = 0
    try:
        periods_grid = getattr(results, "periods", None)
        if periods_grid is not None:
            n_periods_grid = int(len(periods_grid))
    except Exception:
        n_periods_grid = 0

    transit_times: list[float] = []
    if hasattr(results, "transit_times") and results.transit_times is not None:
        try:
            # transit_times can be an iterable or a single value
            if hasattr(results.transit_times, "__iter__"):
                transit_times = [float(t) for t in results.transit_times if not np.isnan(t)]
        except (TypeError, ValueError):
            pass  # Leave as empty list

    # Helper to safely convert to float, handling NaN
    def safe_float(val: float | None, default: float = 0.0) -> float:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)

    # Helper to safely convert to int, handling NaN
    def safe_int(val: float, default: int = 0) -> int:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return int(val)

    return {
        "period": safe_float(results.period, float(period_min)),
        "t0": safe_float(results.T0, float(time[0])),
        "duration_hours": safe_float(results.duration, 0.0) * 24,
        "depth_ppm": depth_ppm,
        "sde": safe_float(results.SDE, 0.0),
        "snr": safe_float(getattr(results, "snr", 0.0), 0.0),
        "fap": safe_float(getattr(results, "FAP", None), 1.0)
        if hasattr(results, "FAP") and results.FAP
        else None,
        "rp_rs": safe_float(getattr(results, "rp_rs", None), 0.0)
        if hasattr(results, "rp_rs") and results.rp_rs
        else None,
        "odd_even_mismatch": safe_float(getattr(results, "odd_even_mismatch", 0.0), 0.0),
        "transit_count": safe_int(getattr(results, "transit_count", 0), 0),
        "transit_times": transit_times,
        "in_transit_mask": in_transit_mask,
        "n_periods_grid": int(n_periods_grid),
        "_tls_results": results,  # Keep full results for masking
    }


def search_planets(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64] | None = None,
    max_planets: int = 5,
    min_sde: float = 7.0,
    period_min: float = 0.5,
    period_max: float | None = None,
    tic_id: int | None = None,
    stellar_radius_rsun: float | None = None,
    stellar_mass_msun: float | None = None,
    use_threads: int | None = None,
    downsample_factor: int = 1,
    per_sector: bool = True,
) -> list[dict]:
    """Iterative multi-planet search using TLS.

    Finds strongest signal, masks transits, repeats until SDE < threshold.
    This replaces our custom multi_planet.py implementation with ~20 lines.

    For multi-sector stitched data, uses per-sector search strategy by default
    to avoid period aliases from inter-sector gaps masking short-period planets.

    Args:
        time: Time array, in BTJD (days)
        flux: Normalized flux array (float64, median ~1.0)
        flux_err: Flux uncertainties (float64), or None
        max_planets: Maximum number of planets to search for
        min_sde: Minimum SDE to accept a detection
        period_min: Minimum period to search, in days
        period_max: Maximum period, in days (default: baseline/2)
        tic_id: Deprecated legacy input (no longer triggers network lookups)
        stellar_radius_rsun: Stellar radius in solar radii (optional)
        stellar_mass_msun: Stellar mass in solar masses (optional)
        use_threads: Number of threads for TLS (default: cpu_count)
        downsample_factor: Downsample data by this factor for faster search (default: 1)
        per_sector: If True, use per-sector search for multi-sector data (default True)

    Returns:
        List of detection dictionaries for each planet found.
    """
    try:
        from transitleastsquares import transit_mask
    except ImportError as e:
        raise ImportError(
            "Transit detection requires the 'tls' extra. "
            "Install with: pip install 'tess-vetter[tls]'"
        ) from e

    planets = []
    time_work = time.copy()
    flux_work = flux.copy()
    flux_err_work = flux_err.copy() if flux_err is not None else None

    for i in range(max_planets):
        # Use per-sector search by default for multi-sector data
        if per_sector:
            result = tls_search_per_sector(
                time_work,
                flux_work,
                flux_err_work,
                period_min=period_min,
                period_max=period_max,
                tic_id=tic_id,
                stellar_radius_rsun=stellar_radius_rsun,
                stellar_mass_msun=stellar_mass_msun,
                use_threads=use_threads,
                downsample_factor=downsample_factor,
            )
        else:
            result = tls_search(
                time_work,
                flux_work,
                flux_err_work,
                period_min=period_min,
                period_max=period_max,
                tic_id=tic_id,
                stellar_radius_rsun=stellar_radius_rsun,
                stellar_mass_msun=stellar_mass_msun,
                use_threads=use_threads,
                downsample_factor=downsample_factor,
            )

        if result["sde"] < min_sde:
            logger.info(f"Stopping at iteration {i + 1}: SDE {result['sde']:.1f} < {min_sde}")
            break

        result["iteration"] = i + 1
        # Remove internal TLS results before storing
        tls_results = result.pop("_tls_results")
        planets.append(result)

        logger.info(
            f"Planet {i + 1}: P={result['period']:.4f}d, "
            f"SDE={result['sde']:.1f}, depth={result['depth_ppm']:.0f}ppm"
        )

        # Mask detected transits using TLS's transit_mask
        mask = transit_mask(
            time_work,
            tls_results.period,
            tls_results.duration * 2,  # 2x duration buffer
            tls_results.T0,
        )

        time_work = time_work[~mask]
        flux_work = flux_work[~mask]
        if flux_err_work is not None:
            flux_err_work = flux_err_work[~mask]

        # Check if enough data remains
        if len(time_work) < 100:
            logger.info(f"Stopping: insufficient data remaining ({len(time_work)} points)")
            break

    return planets


# =============================================================================
# Lomb-Scargle Periodogram (kept for rotation detection)
# =============================================================================


def ls_periodogram(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    periods: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute Lomb-Scargle periodogram.

    Uses scipy.signal.lombscargle for the computation.
    Best for sinusoidal signals like stellar rotation.

    Args:
        time: Time array, in BTJD (days)
        flux: Normalized flux array (float64, median ~1.0)
        periods: Period grid, in days

    Returns:
        Lomb-Scargle power at each period (normalized)
    """
    # Center flux around zero for Lomb-Scargle
    flux_centered = flux - np.mean(flux)

    # Convert periods to angular frequencies
    angular_frequencies = 2.0 * np.pi / periods

    # Compute Lomb-Scargle periodogram
    power = signal.lombscargle(
        time,
        flux_centered,
        angular_frequencies,
        normalize=True,
    )

    power_arr = np.array(power, dtype=np.float64)
    return np.nan_to_num(power_arr, nan=0.0, posinf=0.0, neginf=0.0)


# =============================================================================
# Helper Functions
# =============================================================================


def _estimate_snr(power: float, powers: NDArray[np.float64]) -> float:
    """Estimate signal-to-noise ratio for a peak.

    Uses the median absolute deviation (MAD) as a robust noise estimate.

    Args:
        power: Power of the peak
        powers: Full power spectrum

    Returns:
        Estimated SNR
    """
    median_power = np.median(powers)
    mad = np.median(np.abs(powers - median_power))
    # MAD to standard deviation conversion factor for Gaussian
    sigma = mad * 1.4826

    if sigma <= 0:
        # Cap at 999.0 to prevent misleading Infinity values
        return 999.0 if power > median_power else 0.0

    # Clamp SNR to non-negative and cap at 999
    snr = (power - median_power) / sigma
    return min(999.0, max(0.0, float(snr)))


def _ls_estimate_t0(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    period: float,
) -> float:
    """Estimate an epoch (t0) for an LS sinusoid peak.

    We define `t0` as the epoch of maximum of the best-fit sinusoid at the
    given period, returned in the same time basis as `time` (BTJD days).

    Notes:
    - For rotation/variability use cases, `t0` is mainly for phase-folding
      visualization; absolute epoch is not physically meaningful.
    - Uses a simple linear least-squares fit to `A*cos(wt) + B*sin(wt)`.
    """
    if len(time) < 2 or not np.isfinite(period) or period <= 0:
        return float(time[0]) if len(time) else 0.0

    w = 2.0 * np.pi / float(period)
    x = np.column_stack([np.cos(w * time), np.sin(w * time)])
    y = flux - np.nanmean(flux)

    # Filter NaNs/infs conservatively (should be rare post-cleaning).
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    if np.sum(mask) < 3:
        return float(time[0])

    coef, *_ = np.linalg.lstsq(x[mask], y[mask], rcond=None)
    a = float(coef[0])
    b = float(coef[1])
    amp = float(np.hypot(a, b))
    if not np.isfinite(amp) or amp <= 0:
        return float(time[0])

    # y ≈ a*cos(wt) + b*sin(wt) = C*cos(wt + phi) with:
    # a = C*cos(phi), b = -C*sin(phi)  => phi = atan2(-b, a)
    phi = float(np.arctan2(-b, a))
    t_max = -phi / w  # time of maximum for the cosine form

    t_ref = float(np.min(time))
    # Return t0 in [t_ref, t_ref + period) for stability/determinism.
    return float(t_ref + ((t_max - t_ref) % period))


# =============================================================================
# Auto Periodogram (Updated to use TLS)
# =============================================================================


def auto_periodogram(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64] | None = None,
    min_period: float = 0.5,
    max_period: float | None = None,
    preset: Literal["fast", "thorough", "deep"] | str = "fast",
    method: Literal["bls", "tls", "ls", "auto"] = "auto",
    min_duration_hours: float = 0.5,
    max_duration_hours: float = 8.0,
    n_peaks: int = 5,
    data_ref: str = "",
    fap_method: Literal["empirical", "analytic"] = "empirical",
    n_bootstrap: int = 100,
    tic_id: int | None = None,
    stellar_radius_rsun: float | None = None,
    stellar_mass_msun: float | None = None,
    max_planets: int = 1,
    use_threads: int | None = None,
    per_sector: bool = True,
    downsample_factor: int = 1,
) -> PeriodogramResult:
    """Automatically compute periodogram with optimal settings.

    Uses TLS for transit detection (replaces BLS) and Lomb-Scargle for
    sinusoidal variations like stellar rotation/variability (not transit detection).

    For multi-sector stitched data, uses per-sector search strategy by default
    to avoid period aliases from inter-sector gaps masking short-period planets.

    Args:
        time: Time array, in BTJD (days)
        flux: Normalized flux array (float64, median ~1.0)
        flux_err: Flux uncertainties (float64), or None
        min_period: Minimum period to search, in days (default 0.5)
        max_period: Maximum period, in days (default: baseline/2)
        preset: Performance preset (ignored for TLS, TLS auto-optimizes)
        method: Periodogram method ("bls"/"tls" for transit, "ls" for rotation/variability)
        min_duration_hours: Minimum transit duration (ignored for TLS)
        max_duration_hours: Maximum transit duration (ignored for TLS)
        n_peaks: Number of top peaks to return (TLS returns 1 per search)
        data_ref: Reference to source light curve data
        fap_method: FAP estimation method (TLS provides built-in FAP)
        n_bootstrap: Bootstrap iterations (ignored, TLS has built-in FAP)
        tic_id: Deprecated legacy input (no longer triggers network lookups)
        stellar_radius_rsun: Stellar radius in solar radii (optional)
        stellar_mass_msun: Stellar mass in solar masses (optional)
        max_planets: Number of planets to search for (>1 enables multi-planet)
        use_threads: Number of threads for TLS (default: cpu_count)
        per_sector: If True, use per-sector search for multi-sector data (default True)
        downsample_factor: Downsample data by this factor for faster search (default: 1)

    Returns:
        PeriodogramResult with top peaks and metadata
    """
    # Defensive cleaning: keep only finite samples and sort by time for stable baselines.
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    flux_err = np.asarray(flux_err, dtype=np.float64) if flux_err is not None else None

    finite = np.isfinite(time) & np.isfinite(flux)
    if flux_err is not None:
        finite &= np.isfinite(flux_err)

    time = time[finite]
    flux = flux[finite]
    if flux_err is not None:
        flux_err = flux_err[finite]

    if len(time) < 3:
        selected_method: Literal["tls", "ls"] = "ls" if method == "ls" else "tls"
        return PeriodogramResult(
            data_ref=data_ref,
            method="ls" if selected_method == "ls" else "tls",
            signal_type="sinusoidal" if selected_method == "ls" else "transit",
            peaks=[],
            best_period=float(min_period),
            best_t0=float(time[0]) if len(time) else 0.0,
            best_duration_hours=None,
            snr=None,
            fap=None,
            n_periods_searched=1 if selected_method == "ls" else 0,
            period_range=(
                float(min_period),
                float(max_period) if max_period is not None else float(min_period),
            ),
        )

    order = np.argsort(time)
    time = time[order]
    flux = flux[order]
    if flux_err is not None:
        flux_err = flux_err[order]

    # Calculate baseline
    baseline_days = float(time[-1] - time[0])

    # Set max_period with Nyquist-like constraint
    max_period = baseline_days / 2.0 if max_period is None else min(max_period, baseline_days / 2.0)

    # Ensure valid period range
    if max_period <= min_period:
        max_period = min_period * 2.0

    # Select method: "bls", "tls", or "auto" all use TLS now
    selected_method: Literal["tls", "ls"] = "ls" if method == "ls" else "tls"

    # Run the appropriate periodogram
    if selected_method == "tls":
        # Use TLS for transit detection
        if max_planets > 1:
            results = search_planets(
                time,
                flux,
                flux_err,
                max_planets=max_planets,
                min_sde=7.0,
                period_min=min_period,
                period_max=max_period,
                tic_id=tic_id,
                stellar_radius_rsun=stellar_radius_rsun,
                stellar_mass_msun=stellar_mass_msun,
                use_threads=use_threads,
                downsample_factor=downsample_factor,
                per_sector=per_sector,
            )
        else:
            # Use per-sector search for multi-sector data by default
            if per_sector:
                single_result = tls_search_per_sector(
                    time,
                    flux,
                    flux_err,
                    period_min=min_period,
                    period_max=max_period,
                    tic_id=tic_id,
                    stellar_radius_rsun=stellar_radius_rsun,
                    stellar_mass_msun=stellar_mass_msun,
                    use_threads=use_threads,
                )
            else:
                single_result = tls_search(
                    time,
                    flux,
                    flux_err,
                    period_min=min_period,
                    period_max=max_period,
                    tic_id=tic_id,
                    stellar_radius_rsun=stellar_radius_rsun,
                    stellar_mass_msun=stellar_mass_msun,
                    use_threads=use_threads,
                    downsample_factor=downsample_factor,
                )
            # Remove internal results
            single_result.pop("_tls_results", None)
            results = [single_result]

        n_periods_searched_tls = 0
        try:
            for row in results:
                if not isinstance(row, dict):
                    continue
                if row.get("n_periods_grid_total") is not None:
                    n_periods_searched_tls += int(row.get("n_periods_grid_total") or 0)
                else:
                    n_periods_searched_tls += int(row.get("n_periods_grid") or 0)
        except Exception:
            n_periods_searched_tls = 0

        # Build PeriodogramPeak objects from TLS results
        peaks: list[PeriodogramPeak] = []
        for result in results:
            # Only include duration if > 0 (valid TLS result)
            duration = result["duration_hours"]
            if duration <= 0 or result.get("sde", 0.0) <= 0:
                continue
            peaks.append(
                PeriodogramPeak(
                    period=result["period"],
                    power=result["sde"],  # Use SDE as "power" metric
                    t0=result["t0"],
                    duration_hours=duration if duration > 0 else None,
                    depth_ppm=float(result.get("depth_ppm", 0.0))
                    if result.get("depth_ppm")
                    else None,
                    snr=result["snr"],
                    fap=result["fap"],
                )
            )

        # Get best peak info
        if peaks:
            best_peak = peaks[0]
            best_period = best_peak.period
            best_t0 = best_peak.t0
            best_duration_hours = best_peak.duration_hours
            best_snr = best_peak.snr
            best_fap = best_peak.fap
        else:
            # No detections
            best_period = min_period
            best_t0 = float(time[0])
            best_duration_hours = None
            best_snr = None
            best_fap = None

        return PeriodogramResult(
            data_ref=data_ref,
            method="tls",
            signal_type="transit",
            peaks=peaks,
            best_period=best_period,
            best_t0=best_t0,
            best_duration_hours=best_duration_hours,
            snr=best_snr,
            fap=best_fap,
            # TLS uses an internal/adaptive grid; we estimate the effective period count from
            # the returned TLS results when available (0 means "unknown").
            n_periods_searched=int(n_periods_searched_tls),
            period_range=(float(min_period), float(max_period)),
        )

    else:
        # Use Lomb-Scargle for sinusoidal signals
        n_periods = 1000  # Default for LS
        periods = np.logspace(
            np.log10(min_period),
            np.log10(max_period),
            n_periods,
            dtype=np.float64,
        )

        power = ls_periodogram(time, flux, periods)

        # Find best peak
        best_idx = int(np.argmax(power))
        best_period = float(periods[best_idx])
        best_power = float(power[best_idx])
        best_snr = _estimate_snr(best_power, power)

        best_t0 = _ls_estimate_t0(time, flux, best_period)

        peaks = [
            PeriodogramPeak(
                period=best_period,
                power=best_power,
                t0=best_t0,
                duration_hours=None,
                depth_ppm=None,
                snr=best_snr,
                fap=None,  # LS doesn't provide a calibrated FAP in this implementation
            )
        ]

        return PeriodogramResult(
            data_ref=data_ref,
            method="ls",
            signal_type="sinusoidal",
            peaks=peaks,
            best_period=best_period,
            best_t0=best_t0,
            best_duration_hours=None,
            snr=best_snr,
            fap=None,
            n_periods_searched=n_periods,
            period_range=(float(min_period), float(max_period)),
        )


# =============================================================================
# Transit Model (kept for visualization)
# =============================================================================


def compute_bls_model(
    time: NDArray[np.float64],
    period: float,
    t0: float,
    duration_hours: float,
    depth: float,
) -> NDArray[np.float64]:
    """Compute BLS box model for given parameters.

    Args:
        time: Time array, in BTJD (days)
        period: Period, in days
        t0: Reference epoch, in BTJD (days)
        duration_hours: Transit duration, in hours
        depth: Transit depth (fractional, e.g., 0.01 for 1%)

    Returns:
        Model flux array (1.0 out of transit, 1.0 - depth in transit)
    """
    duration_days = duration_hours / 24.0
    phase = ((time - t0) % period) / period

    # Transit centered at phase 0
    half_dur_phase = (duration_days / period) / 2.0

    # In transit when phase is near 0 or 1
    in_transit = (phase < half_dur_phase) | (phase > (1.0 - half_dur_phase))

    model = np.ones_like(time)
    model[in_transit] = 1.0 - depth

    return model


# =============================================================================
# Period Refinement (uses TLS for refinement)
# =============================================================================


def refine_period(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64] | None,
    initial_period: float,
    initial_duration: float,
    refine_factor: float = 0.1,
    n_refine: int = 100,
    tic_id: int | None = None,
    stellar_radius_rsun: float | None = None,
    stellar_mass_msun: float | None = None,
) -> tuple[float, float, float]:
    """Refine period estimate with higher resolution search.

    Uses TLS with constrained period range around initial estimate.

    Args:
        time: Time array, in BTJD (days)
        flux: Normalized flux array
        flux_err: Flux uncertainties or None
        initial_period: Initial period estimate, in days
        initial_duration: Initial duration estimate, in hours (ignored by TLS)
        refine_factor: Fractional range around initial period to search
        n_refine: Number of points in refined search (ignored by TLS)
        tic_id: Deprecated legacy input (no longer triggers network lookups)
        stellar_radius_rsun: Stellar radius in solar radii (optional)
        stellar_mass_msun: Stellar mass in solar masses (optional)

    Returns:
        Tuple of (refined_period, refined_t0, refined_power) where
        refined_period and refined_t0 are in days, power is SDE
    """
    # Refined period range
    min_period = initial_period * (1.0 - refine_factor)
    max_period = initial_period * (1.0 + refine_factor)

    # Use TLS for refinement
    result = tls_search(
        time,
        flux,
        flux_err,
        period_min=min_period,
        period_max=max_period,
        tic_id=tic_id,
        stellar_radius_rsun=stellar_radius_rsun,
        stellar_mass_msun=stellar_mass_msun,
    )

    return result["period"], result["t0"], result["sde"]
