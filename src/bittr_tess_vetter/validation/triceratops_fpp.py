"""TRICERATOPS integration for False Positive Probability calculation.

This module provides a wrapper around the TRICERATOPS library for computing
Bayesian False Positive Probability (FPP) for transit candidates. TRICERATOPS
queries Gaia for nearby sources and computes the probability that a detected
signal is a genuine planet vs. an eclipsing binary or blend.

Reference:
    Giacalone, S., et al. 2021, AJ, 161, 24: "Vetting of 384 TESS Objects of
    Interest with TRICERATOPS and Statistical Validation of 12 Planet Candidates"

Example:
    >>> from bittr_tess_vetter.validation.triceratops_fpp import calculate_fpp_handler
    >>> result = calculate_fpp_handler(
    ...     cache=cache,
    ...     tic_id=150428135,
    ...     period=37.426,
    ...     t0=1411.38,
    ...     depth_ppm=780,
    ... )
    >>> print(result["fpp"], result["nfpp"])
    0.003 0.002
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import pickle
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.request import Request, urlopen

import numpy as np
from pydantic import BaseModel, Field

from bittr_tess_vetter.platform.network.timeout import (
    TRICERATOPS_CALC_TIMEOUT,
    NetworkTimeoutError,
    network_timeout,
)

if TYPE_CHECKING:
    from bittr_tess_vetter.platform.io import PersistentCache

logger = logging.getLogger(__name__)

TRILEGAL_POLL_TIMEOUT_SECONDS = 180.0
TRILEGAL_POLL_INTERVAL_SECONDS = 5.0
TRICERATOPS_INIT_TIMEOUT_DEFAULT = 300.0
REPLICATE_SUCCESS_RATE_WARN_THRESHOLD = 0.5
FPP_STAGE_INIT_BUDGET_SECONDS = 500.0
FPP_STAGE_TRILEGAL_BUDGET_SECONDS = 120.0
FPP_STAGE_CALC_PROBS_BUDGET_SECONDS = 600.0

_INSECURE_PERMS_MASK = 0o022  # group/other writable


def _is_secure_pickle_path(path: Path) -> bool:
    try:
        if path.is_symlink():
            return False
        st = path.stat()
        if (st.st_mode & _INSECURE_PERMS_MASK) != 0:
            return False
        getuid = getattr(os, "getuid", None)
        return not (getuid is not None and st.st_uid != getuid())
    except OSError:
        return False


def _best_effort_chmod(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except Exception:
        return


def _extract_target_coordinates(target: Any) -> tuple[float | None, float | None]:
    """Best-effort RA/Dec extraction from TRICERATOPS target objects."""
    for ra_key, dec_key in (("ra", "dec"), ("RA", "Dec")):
        ra = getattr(target, ra_key, None)
        dec = getattr(target, dec_key, None)
        if ra is not None and dec is not None:
            try:
                return float(ra), float(dec)
            except Exception:
                pass

    stars = getattr(target, "stars", None)
    if stars is None:
        return None, None

    def _first_numeric(value: Any) -> float | None:
        if value is None:
            return None
        # pandas-like series
        iloc = getattr(value, "iloc", None)
        if iloc is not None:
            try:
                return float(iloc[0])
            except Exception:
                pass
        values = getattr(value, "values", None)
        if values is not None:
            try:
                if len(values) > 0:
                    return float(values[0])
            except Exception:
                pass
        if isinstance(value, (list, tuple)):
            if len(value) > 0:
                try:
                    return float(value[0])
                except Exception:
                    pass
            return None
        try:
            return float(value)
        except Exception:
            return None

    # DataFrame-like mapping access first, then attribute access.
    try:
        ra_val = _first_numeric(stars["ra"]) if "ra" in stars else None  # type: ignore[index]
    except Exception:
        ra_val = _first_numeric(getattr(stars, "ra", None))
    try:
        dec_val = _first_numeric(stars["dec"]) if "dec" in stars else None  # type: ignore[index]
    except Exception:
        dec_val = _first_numeric(getattr(stars, "dec", None))

    if ra_val is not None and dec_val is not None:
        return ra_val, dec_val
    return None, None


# =============================================================================
# Input Validation Schema
# =============================================================================


class CalculateFppInput(BaseModel):
    """Input schema for calculate_fpp tool."""

    tic_id: int = Field(..., gt=0, description="TESS Input Catalog identifier")
    period: float = Field(
        ...,
        gt=0.1,
        le=1000.0,
        description="Orbital period in days",
    )
    t0: float = Field(..., description="Transit epoch in BTJD")
    depth_ppm: float = Field(
        ...,
        gt=10,
        le=500000,
        description="Transit depth in parts per million (ppm)",
    )
    duration_hours: float | None = Field(
        default=None,
        gt=0.1,
        le=24.0,
        description="Transit duration in hours. Estimated if not provided.",
    )
    sectors: list[int] | None = Field(
        default=None,
        description="Specific sectors to use for aperture modeling",
    )
    flux_type: str = Field(
        default="pdcsap",
        description="Cached flux type to use (default 'pdcsap').",
    )


# =============================================================================
# Result Data Class
# =============================================================================


@dataclass
class FppResult:
    """Result of TRICERATOPS FPP calculation."""

    # Primary outputs
    fpp: float
    """False positive probability (0-1)."""

    nfpp: float
    """Nearby false positive probability (0-1)."""

    # Scenario probabilities
    prob_planet: float
    """P(transiting planet on target)."""

    prob_eb: float
    """P(eclipsing binary on target)."""

    prob_beb: float
    """P(background eclipsing binary)."""

    prob_neb: float
    """P(nearby eclipsing binary)."""

    prob_ntp: float
    """P(nearby transiting planet)."""

    # Context
    n_nearby_sources: int
    """Number of Gaia sources in aperture."""

    brightest_contaminant_dmag: float | None
    """Delta-mag of brightest neighbor relative to target."""

    target_in_crowded_field: bool
    """True if >5 sources within 1 arcmin."""

    # Quality flags
    gaia_query_complete: bool
    """True if Gaia query succeeded."""

    aperture_modeled: bool
    """True if aperture flux fractions were computed."""

    # Metadata
    tic_id: int
    """TESS Input Catalog identifier."""

    sectors_used: list[int]
    """Sectors used for analysis."""

    runtime_seconds: float
    """Total runtime in seconds."""

    disposition: str | None = None
    """Coarse label mapping FPP/NFPP into a human-readable category."""

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "fpp": round(self.fpp, 6),
            "nfpp": round(self.nfpp, 6),
            "disposition": self.disposition,
            "prob_planet": round(self.prob_planet, 6),
            "prob_eb": round(self.prob_eb, 6),
            "prob_beb": round(self.prob_beb, 6),
            "prob_neb": round(self.prob_neb, 6),
            "prob_ntp": round(self.prob_ntp, 6),
            "n_nearby_sources": self.n_nearby_sources,
            "brightest_contaminant_dmag": (
                round(self.brightest_contaminant_dmag, 2)
                if self.brightest_contaminant_dmag is not None
                else None
            ),
            "target_in_crowded_field": self.target_in_crowded_field,
            "gaia_query_complete": self.gaia_query_complete,
            "aperture_modeled": self.aperture_modeled,
            "tic_id": self.tic_id,
            "sectors_used": self.sectors_used,
            "runtime_seconds": round(self.runtime_seconds, 1),
        }


def _get_disposition(fpp: float, nfpp: float) -> str:
    """Map TRICERATOPS FPP/NFPP into a coarse disposition label.

    Kept local to avoid importing `bittr_tess_vetter.api.*` from validation code.
    """
    if fpp < 0.01:
        if nfpp < 0.001:
            return "VALIDATED"
        return "LIKELY_PLANET_NEARBY_UNCERTAIN"
    if fpp < 0.05:
        return "LIKELY_PLANET"
    if fpp < 0.5:
        return "INCONCLUSIVE"
    if fpp < 0.9:
        return "LIKELY_FP"
    return "FALSE_POSITIVE"


# =============================================================================
# TRILEGAL Prefetch (work around TRICERATOPS EmptyDataError)
# =============================================================================


def _normalize_trilegal_url(url: str) -> str:
    """TRILEGAL endpoints often redirect http->https; prefer https to reduce edge cases."""
    if url.startswith("http://"):
        return "https://" + url[len("http://") :]
    return url


def _trilegal_text_to_csv_bytes(text: str) -> bytes:
    """Convert raw TRILEGAL whitespace table text into CSV bytes.

    TRICERATOPS later reads this file via pandas.read_csv(trilegal_fname) with default
    comma separators, so we must output CSV, not whitespace-delimited text.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out_lines: list[str] = []

    header_cols: list[str] | None = None
    for ln in lines:
        if ln.startswith("#TRILEGAL normally terminated"):
            break
        if ln.startswith("#Gc "):
            header_cols = ln[1:].split()
            out_lines.append(",".join(header_cols))
            continue
        if ln.startswith("#"):
            # Ignore other comment lines
            continue
        if header_cols is None:
            # No header yet; skip until we find one.
            continue
        parts = ln.split()
        # Some rows may have trailing spaces; enforce consistent column count.
        if len(parts) != len(header_cols):
            # Be conservative: don't emit malformed rows.
            continue
        out_lines.append(",".join(parts))

    if header_cols is None or len(out_lines) <= 1:
        raise ValueError("TRILEGAL text did not contain a parsable table header/data")

    return ("\n".join(out_lines) + "\n").encode("utf-8")


def _prefetch_trilegal_csv(
    *,
    cache_dir: str | Path,
    tic_id: int,
    trilegal_url: str,
    timeout_seconds: float = TRILEGAL_POLL_TIMEOUT_SECONDS,
    poll_interval_seconds: float = TRILEGAL_POLL_INTERVAL_SECONDS,
) -> str:
    """Poll TRILEGAL output URL until ready, then write a stable CSV under the cache dir.

    This bypasses TRICERATOPS's internal `save_trilegal()` which can raise
    `pandas.errors.EmptyDataError` when the URL exists but is momentarily empty.
    """
    cache_dir_path = Path(cache_dir)
    out_dir = cache_dir_path / "triceratops"
    out_dir.mkdir(parents=True, exist_ok=True)

    url = _normalize_trilegal_url(trilegal_url)
    # Deterministic filename per TIC (latest run wins; TRILEGAL output is time-dependent anyway).
    out_path = out_dir / f"{int(tic_id)}_TRILEGAL.csv"
    # Fast path: if we already have a cached CSV, reuse it and avoid hitting TRILEGAL again.
    # (This is particularly important because TRILEGAL output URLs can expire and return 404.)
    try:
        if out_path.exists() and out_path.stat().st_size > 0:
            return str(out_path)
    except OSError:
        # Ignore cache stat failures and fall back to network poll.
        pass

    deadline = time.time() + float(timeout_seconds)
    last_error: Exception | None = None
    empty_http_response_count = 0
    missing_termination_marker_count = 0
    short_response_count = 0

    while time.time() < deadline:
        try:
            req = Request(url, headers={"User-Agent": "bittr-tess-vetter trilegal-prefetch"})
            with urlopen(req, timeout=30) as resp:  # noqa: S310 (caller controls network policy)
                raw = resp.read()
            txt = raw.decode("utf-8", errors="replace")
            if len(raw) == 0:
                empty_http_response_count += 1
                time.sleep(float(poll_interval_seconds))
                continue
            if "#TRILEGAL normally terminated" not in txt:
                missing_termination_marker_count += 1
                time.sleep(float(poll_interval_seconds))
                continue
            if len(txt) < 1000:
                short_response_count += 1
                time.sleep(float(poll_interval_seconds))
                continue

            out_path.write_bytes(_trilegal_text_to_csv_bytes(txt))
            return str(out_path)
        except Exception as e:
            last_error = e
            time.sleep(float(poll_interval_seconds))

    if last_error is not None:
        if (
            empty_http_response_count > 0
            or missing_termination_marker_count > 0
            or short_response_count > 0
        ):
            raise NetworkTimeoutError(
                operation=(
                    "TRILEGAL_EMPTY_RESPONSE "
                    f"(empty={empty_http_response_count}, "
                    f"missing_marker={missing_termination_marker_count}, "
                    f"short={short_response_count}, "
                    f"last_error={type(last_error).__name__}: {last_error})"
                ),
                timeout_seconds=float(timeout_seconds),
            )
        raise NetworkTimeoutError(
            operation=f"TRILEGAL prefetch (last_error={type(last_error).__name__}: {last_error})",
            timeout_seconds=float(timeout_seconds),
        )
    if empty_http_response_count > 0 or missing_termination_marker_count > 0 or short_response_count > 0:
        raise NetworkTimeoutError(
            operation=(
                "TRILEGAL_EMPTY_RESPONSE "
                f"(empty={empty_http_response_count}, "
                f"missing_marker={missing_termination_marker_count}, "
                f"short={short_response_count})"
            ),
            timeout_seconds=float(timeout_seconds),
        )
    raise NetworkTimeoutError(
        operation="TRILEGAL prefetch (no response)",
        timeout_seconds=float(timeout_seconds),
    )


def _cached_trilegal_csv_path(*, cache_dir: str | Path, tic_id: int) -> Path:
    cache_dir_path = Path(cache_dir)
    out_dir = cache_dir_path / "triceratops"
    return out_dir / f"{int(tic_id)}_TRILEGAL.csv"


def _triceratops_stage_state_path(
    *,
    cache_dir: str | Path,
    tic_id: int,
    sectors_used: list[int],
) -> Path:
    cache_dir_path = Path(cache_dir)
    out_dir = cache_dir_path / "triceratops" / "staging_state"
    out_dir.mkdir(parents=True, exist_ok=True)
    sectors_key = "-".join(str(int(s)) for s in sorted(set(sectors_used)))
    return out_dir / f"tic_{int(tic_id)}__sectors_{sectors_key}.json"


def _write_triceratops_stage_state(
    *,
    cache_dir: str | Path,
    tic_id: int,
    sectors_used: list[int],
    payload: dict[str, Any],
) -> None:
    path = _triceratops_stage_state_path(
        cache_dir=cache_dir,
        tic_id=int(tic_id),
        sectors_used=sectors_used,
    )
    data = dict(payload)
    data["tic_id"] = int(tic_id)
    data["sectors_used"] = [int(s) for s in sectors_used]
    data.setdefault("updated_at_unix", float(time.time()))
    path.write_text(json.dumps(data, sort_keys=True) + "\n", encoding="utf-8")


def stage_triceratops_runtime_artifacts(
    *,
    cache: PersistentCache,
    tic_id: int,
    sectors: list[int] | None = None,
    flux_type: str = "pdcsap",
    timeout_seconds: float | None = None,
    load_cached_target: Callable[..., Any] | None = None,
    save_cached_target: Callable[..., Any] | None = None,
    prefetch_trilegal_csv: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Prepare TRICERATOPS init + TRILEGAL artifacts for offline FPP runs."""
    start_time = time.time()
    cache_dir = getattr(cache, "cache_dir", None) or tempfile.gettempdir()
    deadline = None
    if timeout_seconds is not None and float(timeout_seconds) > 0:
        deadline = start_time + float(timeout_seconds)

    def _remaining_seconds() -> float | None:
        if deadline is None:
            return None
        return float(deadline - time.time())

    def _stage_budget(default_seconds: float) -> float:
        rem = _remaining_seconds()
        if rem is None:
            return float(default_seconds)
        if rem <= 0:
            raise NetworkTimeoutError(operation="TRICERATOPS runtime artifact staging", timeout_seconds=0.0)
        # When an explicit total timeout is provided, allow stage budgets to
        # consume the remaining envelope instead of hard-capping at defaults.
        return float(rem)

    # Ensure sectors are present in cache and resolve effective sector list.
    _lc_data, sectors_used = _gather_light_curves(cache, tic_id, sectors, flux_type)
    if sectors_used is None:
        raise ValueError(f"No cached sectors available for TIC {tic_id}")

    # TRICERATOPS import/bootstrap mirrors calculate_fpp_handler.
    try:
        import lxml  # noqa: F401
    except Exception:
        _set_mechanicalsoup_default_features(features="html.parser")
    from bittr_tess_vetter.ext.triceratops_plus_vendor.triceratops import triceratops as tr

    _load = load_cached_target or _load_cached_triceratops_target
    _save = save_cached_target or _save_cached_triceratops_target
    _prefetch = prefetch_trilegal_csv or _prefetch_trilegal_csv

    _write_triceratops_stage_state(
        cache_dir=cache_dir,
        tic_id=tic_id,
        sectors_used=sectors_used,
        payload={
            "status": "in_progress",
            "stage": "triceratops_init",
            "started_at_unix": float(start_time),
        },
    )

    try:
        target = _load(cache_dir=cache_dir, tic_id=tic_id, sectors_used=sectors_used)
        target_cache_hit = target is not None
        if target is None:
            init_budget = _stage_budget(FPP_STAGE_INIT_BUDGET_SECONDS)
            with network_timeout(float(init_budget), operation=f"TRICERATOPS init (Gaia query) for TIC {tic_id}"):
                target = tr.target(ID=tic_id, sectors=sectors_used, mission="TESS")
            _save(cache_dir=cache_dir, tic_id=tic_id, sectors_used=sectors_used, target=target)

        trilegal_cache_hit = False
        trilegal_csv_path: str | None = None
        if getattr(target, "trilegal_fname", None):
            fname = Path(str(target.trilegal_fname))
            if fname.exists() and fname.stat().st_size > 0:
                trilegal_csv_path = str(fname)

        if trilegal_csv_path is None:
            cached_csv = _cached_trilegal_csv_path(cache_dir=cache_dir, tic_id=tic_id)
            try:
                if cached_csv.exists() and cached_csv.stat().st_size > 0:
                    target.trilegal_fname = str(cached_csv)
                    target.trilegal_url = None
                    trilegal_cache_hit = True
                    trilegal_csv_path = str(cached_csv)
            except OSError:
                pass

        if trilegal_csv_path is None:
            trilegal_url = getattr(target, "trilegal_url", None)
            if not isinstance(trilegal_url, str) or not trilegal_url:
                raise RuntimeError(f"TRICERATOPS target for TIC {tic_id} has no TRILEGAL URL or cached CSV")
            trilegal_budget = _stage_budget(FPP_STAGE_TRILEGAL_BUDGET_SECONDS)
            _write_triceratops_stage_state(
                cache_dir=cache_dir,
                tic_id=tic_id,
                sectors_used=sectors_used,
                payload={
                    "status": "in_progress",
                    "stage": "trilegal_prefetch",
                    "target_cache_hit": bool(target_cache_hit),
                    "trilegal_url": str(trilegal_url),
                    "started_at_unix": float(start_time),
                },
            )
            with network_timeout(float(trilegal_budget), operation=f"TRILEGAL prefetch for TIC {tic_id}"):
                try:
                    trilegal_csv = _prefetch(
                        cache_dir=cache_dir,
                        tic_id=tic_id,
                        trilegal_url=trilegal_url,
                    )
                except Exception as e:
                    msg = str(e)
                    should_retry_with_fresh_url = (
                        "HTTP Error 404" in msg
                        or "404: Not Found" in msg
                        or "TRILEGAL_EMPTY_RESPONSE" in msg
                    )
                    if not should_retry_with_fresh_url:
                        raise

                    try:
                        from bittr_tess_vetter.ext.triceratops_plus_vendor.triceratops.funcs import (
                            query_TRILEGAL,
                        )
                    except Exception:
                        raise
                    ra, dec = _extract_target_coordinates(target)
                    if ra is None or dec is None:
                        raise
                    new_url = query_TRILEGAL(float(ra), float(dec), verbose=0)
                    _write_triceratops_stage_state(
                        cache_dir=cache_dir,
                        tic_id=tic_id,
                        sectors_used=sectors_used,
                        payload={
                            "status": "in_progress",
                            "stage": "trilegal_prefetch",
                            "target_cache_hit": bool(target_cache_hit),
                            "trilegal_url": str(new_url),
                            "retry_reason": "stale_or_empty_trilegal_url",
                            "started_at_unix": float(start_time),
                        },
                    )
                    trilegal_csv = _prefetch(
                        cache_dir=cache_dir,
                        tic_id=tic_id,
                        trilegal_url=str(new_url),
                    )
            target.trilegal_fname = trilegal_csv
            target.trilegal_url = None
            trilegal_csv_path = str(trilegal_csv)

        _save(cache_dir=cache_dir, tic_id=tic_id, sectors_used=sectors_used, target=target)
        result = {
            "tic_id": int(tic_id),
            "sectors_used": [int(s) for s in sectors_used],
            "target_cache_hit": bool(target_cache_hit),
            "trilegal_cache_hit": bool(trilegal_cache_hit),
            "trilegal_csv_path": trilegal_csv_path,
            "runtime_seconds": float(time.time() - start_time),
            "stage_state_path": str(
                _triceratops_stage_state_path(
                    cache_dir=cache_dir,
                    tic_id=tic_id,
                    sectors_used=sectors_used,
                )
            ),
        }
        _write_triceratops_stage_state(
            cache_dir=cache_dir,
            tic_id=tic_id,
            sectors_used=sectors_used,
            payload={
                "status": "ok",
                "stage": "complete",
                "target_cache_hit": bool(target_cache_hit),
                "trilegal_cache_hit": bool(trilegal_cache_hit),
                "trilegal_csv_path": trilegal_csv_path,
                "runtime_seconds": float(time.time() - start_time),
                "started_at_unix": float(start_time),
            },
        )
        return result
    except Exception as exc:
        error_text = str(exc)
        error_code = "TRILEGAL_EMPTY_RESPONSE" if "TRILEGAL_EMPTY_RESPONSE" in error_text else type(exc).__name__
        _write_triceratops_stage_state(
            cache_dir=cache_dir,
            tic_id=tic_id,
            sectors_used=sectors_used,
            payload={
                "status": "failed",
                "stage": "trilegal_prefetch",
                "error_code": error_code,
                "error": error_text,
                "runtime_seconds": float(time.time() - start_time),
                "started_at_unix": float(start_time),
            },
        )
        raise


def _triceratops_target_cache_path(
    *, cache_dir: str | Path, tic_id: int, sectors_used: list[int]
) -> Path:
    cache_dir_path = Path(cache_dir)
    out_dir = cache_dir_path / "triceratops" / "target_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    sectors_key = "-".join(str(int(s)) for s in sorted(set(sectors_used)))
    return out_dir / f"tic_{int(tic_id)}__sectors_{sectors_key}.pkl"


def _load_cached_triceratops_target(
    *,
    cache_dir: str | Path,
    tic_id: int,
    sectors_used: list[int],
) -> Any | None:
    path = _triceratops_target_cache_path(
        cache_dir=cache_dir, tic_id=tic_id, sectors_used=sectors_used
    )
    if not path.exists():
        return None
    if not _is_secure_pickle_path(path):
        with contextlib.suppress(OSError):
            path.unlink()
        return None
    try:
        return pickle.loads(path.read_bytes())
    except Exception:
        with contextlib.suppress(OSError):
            path.unlink()
        return None


def _save_cached_triceratops_target(
    *,
    cache_dir: str | Path,
    tic_id: int,
    sectors_used: list[int],
    target: Any,
) -> None:
    path = _triceratops_target_cache_path(
        cache_dir=cache_dir, tic_id=tic_id, sectors_used=sectors_used
    )
    try:
        payload = pickle.dumps(target)
    except Exception:
        # Best-effort cache; if the object isn't pickleable, skip caching.
        return

    try:
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(payload)
        _best_effort_chmod(tmp, 0o600)
        tmp.replace(path)
        _best_effort_chmod(path, 0o600)
    except Exception:
        with contextlib.suppress(OSError):
            tmp.unlink()  # type: ignore[has-type]
        return


# =============================================================================
# Helper Functions
# =============================================================================


def _estimate_transit_duration(
    period: float,
    stellar_radius: float = 1.0,
    stellar_mass: float = 1.0,
) -> float:
    """Estimate transit duration from orbital parameters.

    Uses circular orbit approximation:
    T_dur ~ (R_star / a) * P / pi

    where a is computed from Kepler's 3rd law.

    Args:
        period: Orbital period in days
        stellar_radius: Stellar radius in solar radii (default 1.0)
        stellar_mass: Stellar mass in solar masses (default 1.0)

    Returns:
        Estimated transit duration in hours
    """
    # Kepler's 3rd law: a^3 = M * P^2 (AU, solar masses, years)
    period_years = period / 365.25
    a_au = (stellar_mass * period_years**2) ** (1 / 3)

    # Convert to solar radii (1 AU ~ 215 R_sun)
    a_rsun = a_au * 215.0

    # Duration approximation for central transit
    # T_dur = (P / pi) * (R_star / a)
    t_dur_days = (period / math.pi) * (stellar_radius / a_rsun)

    # Convert to hours and clamp to reasonable range
    t_dur_hours = t_dur_days * 24.0
    return float(max(0.5, min(t_dur_hours, 12.0)))


def _gather_light_curves(
    cache: PersistentCache,
    tic_id: int,
    sectors: list[int] | None,
    flux_type: str,
) -> tuple[dict[str, Any] | None, list[int]]:
    """Gather light curve data from cache.

    Args:
        cache: Persistent cache containing light curve data
        tic_id: TIC identifier
        sectors: Specific sectors to use, or None for all cached
        flux_type: Cached flux type to include (e.g., "pdcsap", "sap")

    Returns:
        Tuple of (light curve data dict, sectors used)
        Returns (None, []) if no data found
    """
    all_keys = cache.keys()
    tic_prefix = f"lc:{tic_id}:"

    # Find matching cached light curves
    matching_keys: list[tuple[str, int]] = []
    for key in all_keys:
        if key.startswith(tic_prefix) and "stitched" not in key:
            parts = key.split(":")
            if len(parts) >= 4:
                try:
                    sector_num = int(parts[2])
                    key_flux_type = str(parts[3]).split("~", 1)[0]
                    if key_flux_type != str(flux_type):
                        continue
                    if sectors is None or sector_num in sectors:
                        matching_keys.append((key, sector_num))
                except ValueError:
                    continue

    if not matching_keys:
        return None, []

    # Sort by sector
    matching_keys.sort(key=lambda x: x[1])

    # Collect arrays
    time_arrays: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    flux_arrays: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    flux_err_arrays: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    sectors_used: list[int] = []
    seen_sectors: set[int] = set()

    for key, sector_num in matching_keys:
        if sector_num in seen_sectors:
            continue
        lc_data = cache.get(key)
        if lc_data is None:
            continue

        mask = lc_data.valid_mask
        time_arrays.append(lc_data.time[mask].copy())
        flux_arrays.append(lc_data.flux[mask].copy())
        flux_err_arrays.append(lc_data.flux_err[mask].copy())
        sectors_used.append(sector_num)
        seen_sectors.add(sector_num)

    if not time_arrays:
        return None, []

    # Concatenate and sort
    time_all = np.concatenate(time_arrays)
    flux_all = np.concatenate(flux_arrays)
    flux_err_all = np.concatenate(flux_err_arrays)

    sort_idx = np.argsort(time_all)
    return {
        "time": time_all[sort_idx],
        "flux": flux_all[sort_idx],
        "flux_err": flux_err_all[sort_idx],
    }, sectors_used


def _set_mechanicalsoup_default_features(*, features: str) -> None:
    """Best-effort: force MechanicalSoup to use a different default BS4 parser.

    MechanicalSoup defaults to soup_config={'features': 'lxml'} which hard-fails
    when lxml isn't installed. TRICERATOPS pulls MechanicalSoup transitively for
    some upstream interactions, so this can prevent initialization.
    """
    try:
        import mechanicalsoup.browser as ms_browser
        import mechanicalsoup.stateful_browser as ms_stateful
    except Exception:
        return

    for init_fn in (ms_browser.Browser.__init__, ms_stateful.StatefulBrowser.__init__):
        defaults = getattr(init_fn, "__defaults__", None)
        if not defaults:
            continue
        for value in defaults:
            if isinstance(value, dict) and value.get("features") == "lxml":
                value["features"] = features


def _write_external_lc_files(
    external_lcs: list[Any],  # list[ExternalLightCurve] from api.fpp
    temp_dir: Path,
) -> tuple[list[str], list[str]]:
    """Convert external LC arrays to temp .txt files for TRICERATOPS+.

    TRICERATOPS+ expects .txt files with 3 columns, no header:
        time_from_midtransit (days), flux, flux_err

    Args:
        external_lcs: List of ExternalLightCurve dataclass instances.
        temp_dir: Directory to write temporary files.

    Returns:
        Tuple of (file_paths, filters) for use with calc_probs().

    Raises:
        ValueError: If external LC arrays have mismatched lengths or non-finite values.
    """
    file_paths: list[str] = []
    filters: list[str] = []

    for i, lc in enumerate(external_lcs):
        # Validate array lengths
        if len(lc.time_from_midtransit_days) != len(lc.flux):
            raise ValueError(f"External LC {i}: time/flux array length mismatch")
        if len(lc.flux) != len(lc.flux_err):
            raise ValueError(f"External LC {i}: flux/flux_err array length mismatch")

        # Validate finite values
        if not np.all(np.isfinite(lc.flux)):
            raise ValueError(f"External LC {i}: non-finite flux values detected")
        if not np.all(np.isfinite(lc.flux_err)):
            raise ValueError(f"External LC {i}: non-finite flux_err values detected")
        if not np.all(np.isfinite(lc.time_from_midtransit_days)):
            raise ValueError(f"External LC {i}: non-finite time values detected")

        # Write temp file in TRICERATOPS+ expected format
        fpath = temp_dir / f"external_lc_{i}_{lc.filter}.txt"
        data = np.column_stack(
            [
                lc.time_from_midtransit_days,
                lc.flux,
                lc.flux_err,
            ]
        )
        np.savetxt(fpath, data, fmt="%.10f", delimiter=" ")

        file_paths.append(str(fpath))
        filters.append(lc.filter)

    return file_paths, filters


def _normalize_triceratops_filter(filt: str | None) -> str:
    """Normalize filter strings passed into TRICERATOPS(+).

    Vendored TRICERATOPS flux relations support only:
      TESS, Vis, g, r, i, z, J, H, K

    ExoFOP imaging often uses more specific labels (e.g., 'Kcont', 'Brgamma').
    Map common synonyms into the supported set and fall back to 'Vis' to avoid
    upstream crashes (e.g., UnboundLocalError in flux_relation for unknown filt).
    """
    raw = str(filt or "").strip()
    if not raw:
        return "Vis"
    key = "".join(ch.lower() for ch in raw if ch.isalnum())

    canonical = {
        "tess": "TESS",
        "vis": "Vis",
        "g": "g",
        "r": "r",
        "i": "i",
        "z": "z",
        "j": "J",
        "h": "H",
        "k": "K",
    }
    if key in canonical:
        return canonical[key]

    aliases = {
        # Common optical "catch-alls"
        "v": "Vis",
        "vband": "Vis",
        "clear": "Vis",
        "open": "Vis",
        # Narrow K-band variants (treat as K for flux-ratio purposes)
        "kcont": "K",
        "kcontinuum": "K",
        "brgamma": "K",
        "brg": "K",
        "ks": "K",
        "kshort": "K",
        "kprime": "K",
        # Narrow J/H variants (rare, but safe)
        "hcont": "H",
        "jcont": "J",
    }
    if key in aliases:
        return aliases[key]

    # "Kp" is ambiguous (Kepler band vs K-prime). Do not guess silently.
    if key == "kp":
        logger.warning("Ambiguous filter label %r; falling back to 'Vis'", raw)
        return "Vis"

    logger.warning("Unrecognized TRICERATOPS filter %r; falling back to 'Vis'", raw)
    return "Vis"


# =============================================================================
# Replicate Aggregation Helpers
# =============================================================================


def _is_result_degenerate(result: dict[str, Any]) -> bool:
    """Check if an FPP result is degenerate (unusable).

    A result is degenerate if:
    - FPP is not finite
    - posterior_sum_total is not finite or <= 0
    - posterior_prob_nan_count > 0
    """
    if "error" in result:
        return True

    fpp = result.get("fpp")
    if fpp is None or not np.isfinite(float(fpp)):
        return True

    posterior_sum_total = result.get("posterior_sum_total")
    if posterior_sum_total is not None:
        try:
            pst = float(posterior_sum_total)
            if not np.isfinite(pst) or pst <= 0:
                return True
        except (TypeError, ValueError):
            return True

    posterior_prob_nan_count = result.get("posterior_prob_nan_count")
    return posterior_prob_nan_count is not None and int(posterior_prob_nan_count) > 0


def _aggregate_replicate_results(
    results: list[dict[str, Any]],
    *,
    tic_id: int,  # noqa: ARG001
    sectors_used: list[int],  # noqa: ARG001
    total_runtime: float,
) -> dict[str, Any]:
    """Aggregate successful replicate FPP results into a summary.

    Args:
        results: List of result dicts from replicate runs.
        tic_id: TIC ID (for context/logging, not used in aggregation).
        sectors_used: Sectors used (for context/logging, not used in aggregation).
        total_runtime: Total runtime for all replicates.

    Returns a consolidated result with:
    - fpp_median, fpp_p16, fpp_p84 (and same for nfpp)
    - n_success, n_fail, replicates
    - best_run details (lowest FPP run)
    """
    successful = [r for r in results if not _is_result_degenerate(r)]
    n_success = len(successful)
    n_fail = len(results) - n_success

    if n_success == 0:
        # All runs failed - this should be handled by caller
        return {}

    # Extract FPP/NFPP values
    fpps = [float(r["fpp"]) for r in successful]
    nfpps = [float(r["nfpp"]) for r in successful]

    # Compute percentiles
    fpp_median = float(np.median(fpps))
    fpp_p16 = float(np.percentile(fpps, 16))
    fpp_p84 = float(np.percentile(fpps, 84))
    nfpp_median = float(np.median(nfpps))
    nfpp_p16 = float(np.percentile(nfpps, 16))
    nfpp_p84 = float(np.percentile(nfpps, 84))

    # Find best run (lowest FPP)
    best_idx = int(np.argmin(fpps))
    best_run = successful[best_idx]

    # Build aggregated result (use best_run as base, add summary)
    out = dict(best_run)
    out["fpp_summary"] = {
        "median": round(fpp_median, 6),
        "p16": round(fpp_p16, 6),
        "p84": round(fpp_p84, 6),
        "values": [round(f, 6) for f in fpps],
    }
    out["nfpp_summary"] = {
        "median": round(nfpp_median, 6),
        "p16": round(nfpp_p16, 6),
        "p84": round(nfpp_p84, 6),
        "values": [round(f, 6) for f in nfpps],
    }
    out["replicates"] = len(results)
    out["n_success"] = n_success
    out["n_fail"] = n_fail
    total = max(1, n_success + n_fail)
    out["replicate_success_rate"] = round(float(n_success) / float(total), 6)
    if out["replicate_success_rate"] < REPLICATE_SUCCESS_RATE_WARN_THRESHOLD:
        out["warning_note"] = (
            f"High replicate failure rate ({n_fail}/{total}); review replicate_errors/degenerate_reason."
        )
    out["runtime_seconds"] = round(total_runtime, 1)

    out["fpp"] = round(fpp_median, 6)
    out["nfpp"] = round(nfpp_median, 6)
    # `out` starts from `best_run` (lowest-FPP) for detailed scenario fields, but
    # headline quantities should reflect the aggregated/median values.
    out["disposition"] = _get_disposition(float(out["fpp"]), float(out["nfpp"]))
    out["sectors_used"] = sorted({int(s) for s in (out.get("sectors_used") or [])})

    return out


def _extract_single_run_result(
    target: Any,
    *,
    tic_id: int,
    sectors_used: list[int],
    triceratops_engine: str | None,
    n_points_raw: int,
    n_points_windowed: int,
    n_points_used: int,
    half_window_days: float,
    window_duration_mult: float | None,
    max_points: int | None,
    draws: int,
    exptime_days: float,
    flux_err_scalar: float,
    empirical_sigma: float,
    min_flux_err: float,
    use_empirical_noise_floor: bool,
    n_external_lcs: int,
    external_filters_used: list[str],
    run_seed: int | None,
    run_start_time: float,
) -> dict[str, Any]:
    """Extract FPP result from a single calc_probs run.

    Returns a result dict with FPP, NFPP, scenario probs, and diagnostics.
    Used for replicate aggregation.
    """
    fpp = float(target.FPP)
    nfpp = float(target.NFPP)
    probs_raw = getattr(target, "probs", None)

    # Compute total posterior mass and per-scenario probability sums
    posterior_sum_total: float | None = None
    posterior_prob_nan_count: int | None = None
    scenario_prob_sums: dict[str, float] = {}
    scenario_prob_top: list[dict[str, Any]] = []
    try:
        if (
            probs_raw is not None
            and hasattr(probs_raw, "__getitem__")
            and hasattr(probs_raw, "columns")
        ):
            if "scenario" in probs_raw.columns and "prob" in probs_raw.columns:
                scenarios = list(probs_raw["scenario"])
                prob_vals = [float(x) for x in list(probs_raw["prob"])]
                posterior_prob_nan_count = int(
                    sum(1 for v in prob_vals if not np.isfinite(float(v)))
                )
                posterior_sum_total = float(sum(prob_vals))
                for sc, pv in zip(scenarios, prob_vals, strict=False):
                    key = str(sc)
                    scenario_prob_sums[key] = float(scenario_prob_sums.get(key, 0.0) + float(pv))
                scenario_prob_top = [
                    {"scenario": k, "prob": v}
                    for k, v in sorted(
                        scenario_prob_sums.items(), key=lambda kv: kv[1], reverse=True
                    )[:15]
                ]
        elif isinstance(probs_raw, dict):
            scenario_prob_sums = {str(k): float(v) for k, v in probs_raw.items()}
            posterior_prob_nan_count = int(
                sum(1 for v in scenario_prob_sums.values() if not np.isfinite(float(v)))
            )
            posterior_sum_total = float(sum(scenario_prob_sums.values()))
            scenario_prob_top = [
                {"scenario": k, "prob": v}
                for k, v in sorted(scenario_prob_sums.items(), key=lambda kv: kv[1], reverse=True)[
                    :15
                ]
            ]
    except Exception:
        posterior_sum_total = None
        posterior_prob_nan_count = None
        scenario_prob_sums = {}
        scenario_prob_top = []

    n_nearby = len(target.stars) - 1 if hasattr(target, "stars") else 0

    brightest_dmag: float | None = None
    if hasattr(target, "stars") and n_nearby > 0:
        try:
            contaminant_mags = [s["Tmag"] for s in target.stars[1:]]
            target_mag = target.stars[0]["Tmag"]
            brightest_dmag = float(min(contaminant_mags) - target_mag)
        except (KeyError, IndexError, TypeError):
            brightest_dmag = None

    runtime = time.time() - run_start_time

    # TRICERATOPS scenario labels include more than TP/EB/BEB/NEB/NTP, e.g. DTP (diluted
    # transiting planet) and DEB (diluted eclipsing binary). For reporting/guardrails we
    # bucket scenarios into planet-like vs EB-like groups.
    prob_planet = float(
        scenario_prob_sums.get("TP", 0.0)
        + scenario_prob_sums.get("DTP", 0.0)
        + scenario_prob_sums.get("NTP", 0.0)
    )
    prob_eb = float(
        scenario_prob_sums.get("EB", 0.0)
        + scenario_prob_sums.get("DEB", 0.0)
        + scenario_prob_sums.get("BEB", 0.0)
        + scenario_prob_sums.get("NEB", 0.0)
    )
    prob_beb = float(scenario_prob_sums.get("BEB", 0.0))
    prob_neb = float(scenario_prob_sums.get("NEB", 0.0))
    prob_ntp = float(scenario_prob_sums.get("NTP", 0.0))

    result = FppResult(
        fpp=fpp,
        nfpp=nfpp,
        disposition=_get_disposition(fpp, nfpp),
        prob_planet=prob_planet,
        prob_eb=prob_eb,
        prob_beb=prob_beb,
        prob_neb=prob_neb,
        prob_ntp=prob_ntp,
        n_nearby_sources=n_nearby,
        brightest_contaminant_dmag=brightest_dmag,
        target_in_crowded_field=n_nearby > 5,
        gaia_query_complete=True,
        aperture_modeled=True,
        tic_id=tic_id,
        sectors_used=sorted({int(s) for s in sectors_used}),
        runtime_seconds=runtime,
    )

    out = result.to_dict()
    out["engine"] = triceratops_engine
    out["posterior_sum_total"] = posterior_sum_total
    out["posterior_prob_nan_count"] = posterior_prob_nan_count
    out["scenario_prob_top"] = scenario_prob_top
    out["run_seed"] = run_seed
    out["triceratops_runtime"] = {
        "n_points_raw": int(n_points_raw),
        "n_points_windowed": int(n_points_windowed),
        "n_points_used": int(n_points_used),
        "half_window_days": float(half_window_days),
        "window_duration_mult": (
            float(window_duration_mult) if window_duration_mult is not None else None
        ),
        "max_points": int(max_points) if max_points is not None else None,
        "mc_draws": int(draws),
        "exptime_days": float(exptime_days),
        "flux_err_scalar_used": float(flux_err_scalar),
        "empirical_sigma_used": float(empirical_sigma),
        "min_flux_err": float(min_flux_err),
        "use_empirical_noise_floor": bool(use_empirical_noise_floor),
        "n_external_lcs": n_external_lcs,
        "external_filters": external_filters_used,
    }

    # Check for degenerate results
    degenerate: list[str] = []
    if not np.isfinite(float(out.get("fpp", float("nan")))):
        degenerate.append("fpp_not_finite")
    if posterior_sum_total is not None and not np.isfinite(float(posterior_sum_total)):
        degenerate.append("posterior_sum_not_finite")
    if posterior_sum_total is not None and posterior_sum_total <= 0:
        degenerate.append("posterior_sum_total_zero")
    if posterior_prob_nan_count is not None and posterior_prob_nan_count > 0:
        degenerate.append(f"posterior_prob_nan_count={posterior_prob_nan_count}")
    if out.get("prob_planet", 0.0) == 0.0 and out.get("prob_eb", 0.0) == 0.0 and scenario_prob_sums:
        degenerate.append("scenario_probs_missing_expected_keys")
    out["degenerate_reason"] = ",".join(degenerate) if degenerate else None

    return out


# =============================================================================
# Main Handler
# =============================================================================


def calculate_fpp_handler(
    cache: PersistentCache,
    tic_id: int,
    period: float,
    t0: float,
    depth_ppm: float,
    duration_hours: float | None = None,
    sectors: list[int] | None = None,
    flux_type: str = "pdcsap",
    stellar_radius: float | None = None,
    stellar_mass: float | None = None,
    tmag: float | None = None,
    timeout_seconds: float | None = None,
    mc_draws: int | None = None,
    window_duration_mult: float | None = 3.0,
    max_points: int | None = 3000,
    min_flux_err: float = 5e-5,
    use_empirical_noise_floor: bool = True,
    replicates: int | None = None,
    seed: int | None = None,
    external_lightcurves: list[Any] | None = None,
    contrast_curve: Any | None = None,
    *,
    load_cached_target: Callable[..., Any] | None = None,
    save_cached_target: Callable[..., Any] | None = None,
    prefetch_trilegal_csv: Callable[..., Any] | None = None,
    allow_network: bool = True,
    progress_hook: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute TRICERATOPS+ FPP calculation.

    This function:
    1. Loads light curve from cache
    2. Checks target isn't saturated (Tmag < 4)
    3. Estimates duration if not provided
    4. Calls TRICERATOPS+ target.calc_probs() with optional external LCs
    5. Returns FPP, NFPP, scenario probabilities

    Args:
        cache: Persistent cache containing light curve data
        tic_id: TESS Input Catalog identifier
        period: Orbital period in days
        t0: Transit epoch in BTJD
        depth_ppm: Transit depth in parts per million
        duration_hours: Transit duration in hours (estimated if None)
        sectors: Specific sectors to analyze (all cached if None)
        flux_type: Cached flux type to use for the light curve (default "pdcsap")
        stellar_radius: Stellar radius in solar radii (for duration estimation)
        stellar_mass: Stellar mass in solar masses (for duration estimation)
        tmag: TESS magnitude (for saturation check)
        external_lightcurves: Ground-based LCs for multi-band FPP (TRICERATOPS+ feature)
        contrast_curve: High-resolution imaging contrast curve (not yet implemented)

    Returns:
        Dictionary with FPP results or error information
    """
    start_time = time.time()
    cache_dir = getattr(cache, "cache_dir", None) or tempfile.gettempdir()
    deadline = None
    if timeout_seconds is not None and float(timeout_seconds) > 0:
        deadline = start_time + float(timeout_seconds)

    def _remaining_seconds() -> float | None:
        if deadline is None:
            return None
        return float(deadline - time.time())

    def _err(
        message: str,
        *,
        error_type: str,
        stage: str,
        sectors_used_value: list[int] | None = None,
        ) -> dict[str, Any]:
        return {
            "error": message,
            "error_type": error_type,
            "stage": stage,
            "tic_id": tic_id,
            "tmag": tmag,
            "sectors_used": sectors_used_value,
            "runtime_seconds": float(time.time() - start_time),
        }

    def _stage_budget_or_err(
        *,
        stage: str,
        default_seconds: float,
        sectors_used_value: list[int] | None,
    ) -> tuple[float | None, dict[str, Any] | None]:
        rem = _remaining_seconds()
        budget = float(default_seconds)
        if rem is not None:
            if rem <= 0:
                return None, _err(
                    f"FPP calculation timed out before {stage}",
                    error_type="timeout",
                    stage=stage,
                    sectors_used_value=sectors_used_value,
                )
            budget = min(float(default_seconds), float(rem))
        return float(budget), None

    # Step 1: Get light curve from cache
    lc_data, sectors_used = _gather_light_curves(cache, tic_id, sectors, flux_type)
    if lc_data is None:
        return _err(
            f"No cached light curves for TIC {tic_id}. Use load_lightcurve first to download sector data.",
            error_type="cache_miss",
            stage="gather_light_curves",
            sectors_used_value=None,
        )

    # Step 2: Check for saturation
    if tmag is not None and tmag < 4.0:
        return _err(
            f"Target saturated (Tmag={tmag:.1f}), FPP unreliable. "
            "TRICERATOPS aperture modeling fails for saturated stars.",
            error_type="insufficient_data",
            stage="saturation_check",
            sectors_used_value=sectors_used,
        )

    # Step 3: Estimate duration if not provided
    if duration_hours is None:
        rad = stellar_radius if stellar_radius is not None else 1.0
        mass = stellar_mass if stellar_mass is not None else 1.0
        duration_hours = _estimate_transit_duration(period, rad, mass)
        logger.info(f"Estimated transit duration: {duration_hours:.2f} hours")

    # Step 4: Try to import and initialize TRICERATOPS+
    triceratops_engine: str | None = None
    try:
        # If lxml isn't installed, TRICERATOPS's transitive dependency MechanicalSoup
        # may crash while constructing BeautifulSoup(..., 'lxml').
        # Default to stdlib html.parser in that case to keep the tool usable.
        try:
            import lxml  # noqa: F401
        except Exception:
            _set_mechanicalsoup_default_features(features="html.parser")

        # Use vendored TRICERATOPS+ for multi-band FPP support
        from bittr_tess_vetter.ext.triceratops_plus_vendor.triceratops import (
            triceratops as tr,
        )

        triceratops_engine = "triceratops_plus_vendor"
    except ImportError as import_err:
        # Vendored code should always be available; this indicates a packaging issue
        return _err(
            f"TRICERATOPS+ vendored import failed (packaging issue): {import_err}",
            error_type="internal_error",
            stage="import_triceratops",
            sectors_used_value=sectors_used,
        )

    try:
        # Prefer a cached TRICERATOPS target object to avoid repeating slow upstream calls.
        _load = load_cached_target or _load_cached_triceratops_target
        _save = save_cached_target or _save_cached_triceratops_target
        _prefetch = prefetch_trilegal_csv or _prefetch_trilegal_csv

        stage_init_start = time.time()
        target = _load(cache_dir=cache_dir, tic_id=tic_id, sectors_used=sectors_used)
        if target is None:
            if not allow_network:
                return _err(
                    (
                        f"Network disabled and no cached TRICERATOPS target for TIC {tic_id}. "
                        "Run `btv fpp-prepare --network-ok` first."
                    ),
                    error_type="network_disabled",
                    stage="triceratops_init",
                    sectors_used_value=sectors_used,
                )
            init_timeout, init_err = _stage_budget_or_err(
                stage="triceratops_init",
                default_seconds=FPP_STAGE_INIT_BUDGET_SECONDS,
                sectors_used_value=sectors_used,
            )
            if init_err is not None or init_timeout is None:
                return init_err or _err(
                    "FPP calculation timed out before TRICERATOPS initialization",
                    error_type="timeout",
                    stage="triceratops_init",
                    sectors_used_value=sectors_used,
                )
            logger.info(
                "[fpp] Stage triceratops_init: start (TIC=%s sectors=%s budget=%.1fs cache=miss)",
                tic_id,
                len(sectors_used),
                float(init_timeout),
            )

            try:
                with network_timeout(
                    float(init_timeout),
                    operation=f"TRICERATOPS init (Gaia query) for TIC {tic_id}",
                ):
                    target = tr.target(ID=tic_id, sectors=sectors_used, mission="TESS")
                _save(cache_dir=cache_dir, tic_id=tic_id, sectors_used=sectors_used, target=target)
            except NetworkTimeoutError:
                # One retry: some upstream endpoints are bursty (MAST/TessCut/Gaia).
                rem_retry = _remaining_seconds()
                if rem_retry is not None and rem_retry <= 0:
                    raise
                with network_timeout(
                    float(min(float(init_timeout) * 1.5, rem_retry))
                    if rem_retry is not None
                    else float(init_timeout) * 1.5,
                    operation=f"TRICERATOPS init (Gaia query) retry for TIC {tic_id}",
                ):
                    target = tr.target(ID=tic_id, sectors=sectors_used, mission="TESS")
                _save(cache_dir=cache_dir, tic_id=tic_id, sectors_used=sectors_used, target=target)
        else:
            logger.info(
                "[fpp] Stage triceratops_init: cache hit (TIC=%s sectors=%s)",
                tic_id,
                len(sectors_used),
            )
        n_nearby = None
        try:
            stars = getattr(target, "stars", None)
            if isinstance(stars, list):
                n_nearby = len(stars)
        except Exception:
            n_nearby = None
        logger.info(
            "[fpp] Stage triceratops_init: complete (%.1fs nearby_sources=%s)",
            float(time.time() - stage_init_start),
            "unknown" if n_nearby is None else n_nearby,
        )
    except NetworkTimeoutError as e:
        return _err(
            str(e),
            error_type="timeout",
            stage="triceratops_init",
            sectors_used_value=sectors_used,
        )
    except Exception as e:
        logger.exception(f"TRICERATOPS target init failed for TIC {tic_id}")
        return _err(
            f"TRICERATOPS initialization failed: {e}",
            error_type="internal_error",
            stage="triceratops_init",
            sectors_used_value=sectors_used,
        )

        # Step 5: Calculate probabilities
    try:
        # Work around TRICERATOPS's internal TRILEGAL fetch path which can raise
        # pandas EmptyDataError when the output URL exists but is momentarily empty.
        stage_trilegal_start = time.time()
        if getattr(target, "trilegal_fname", None) is None:
            # If we already have a cached TRILEGAL CSV for this TIC, prefer it to avoid
            # re-querying TRILEGAL and to avoid expired TRILEGAL URLs (HTTP 404).
            cached_csv = _cached_trilegal_csv_path(cache_dir=cache_dir, tic_id=tic_id)
            try:
                if cached_csv.exists() and cached_csv.stat().st_size > 0:
                    target.trilegal_fname = str(cached_csv)
                    target.trilegal_url = None
                    logger.info(
                        "[fpp] Stage trilegal_prefetch: cache hit (TIC=%s file=%s)",
                        tic_id,
                        str(cached_csv),
                    )
            except OSError:
                pass

            trilegal_url = getattr(target, "trilegal_url", None)
            if isinstance(trilegal_url, str) and trilegal_url:
                if not allow_network:
                    return _err(
                        (
                            f"Network disabled and no cached TRILEGAL CSV for TIC {tic_id}. "
                            "Run `btv fpp-prepare --network-ok` first."
                        ),
                        error_type="network_disabled",
                        stage="trilegal_prefetch",
                        sectors_used_value=sectors_used,
                    )
                prefetch_timeout, prefetch_err = _stage_budget_or_err(
                    stage="trilegal_prefetch",
                    default_seconds=FPP_STAGE_TRILEGAL_BUDGET_SECONDS,
                    sectors_used_value=sectors_used,
                )
                if prefetch_err is not None or prefetch_timeout is None:
                    return prefetch_err or _err(
                        "FPP calculation timed out before TRILEGAL prefetch",
                        error_type="timeout",
                        stage="trilegal_prefetch",
                        sectors_used_value=sectors_used,
                    )
                logger.info(
                    "[fpp] Stage trilegal_prefetch: start (TIC=%s budget=%.1fs)",
                    tic_id,
                    float(prefetch_timeout),
                )
                with network_timeout(
                    float(prefetch_timeout),
                    operation=f"TRILEGAL prefetch for TIC {tic_id}",
                ):
                    try:
                        trilegal_csv = _prefetch(
                            cache_dir=cache_dir,
                            tic_id=tic_id,
                            trilegal_url=trilegal_url,
                        )
                    except Exception as e:
                        # If the TRILEGAL output URL has expired (commonly HTTP 404),
                        # attempt to re-submit a fresh TRILEGAL query once.
                        msg = str(e)
                        if "HTTP Error 404" in msg or "404: Not Found" in msg:
                            try:
                                from bittr_tess_vetter.ext.triceratops_plus_vendor.triceratops.funcs import (
                                    query_TRILEGAL,
                                )
                            except Exception:
                                raise
                            ra, dec = _extract_target_coordinates(target)
                            if ra is None or dec is None:
                                raise
                            new_url = query_TRILEGAL(float(ra), float(dec), verbose=0)
                            trilegal_csv = _prefetch(
                                cache_dir=cache_dir,
                                tic_id=tic_id,
                                trilegal_url=str(new_url),
                            )
                        else:
                            raise
                target.trilegal_fname = trilegal_csv
                target.trilegal_url = None
        logger.info(
            "[fpp] Stage trilegal_prefetch: complete (%.1fs)",
            float(time.time() - stage_trilegal_start),
        )

        # TRICERATOPS expects time array centered on transit midpoint
        # Fold the light curve to center on transits
        time_arr = np.array(lc_data["time"])
        flux_arr = np.array(lc_data["flux"])
        flux_err_arr = np.array(lc_data["flux_err"])
        n_points_raw = int(len(time_arr))
        n_points_windowed = n_points_raw
        n_points_used = n_points_raw
        try:
            time_sorted = np.sort(time_arr)
            dt = np.diff(time_sorted)
            exptime_days = float(np.nanmedian(dt[np.isfinite(dt)]))
        except Exception:
            exptime_days = 0.00139

        # Fold time to be relative to nearest transit midpoint
        # (time should be in days from transit midpoint)
        phase = ((time_arr - t0) / period + 0.5) % 1 - 0.5
        time_folded = phase * period  # Convert phase back to days from midpoint

        # Use only a local window around transit to keep calc_probs tractable.
        # TRICERATOPS runtime scales strongly with the number of points and the MC draw count.
        dur_days = float(duration_hours) / 24.0
        if window_duration_mult is None:
            half_window_days = float("inf")
        else:
            half_window_days = max(dur_days * float(window_duration_mult), 0.25)
            window_mask = np.abs(time_folded) <= half_window_days
            if np.any(window_mask):
                time_folded = time_folded[window_mask]
                flux_arr = flux_arr[window_mask]
                flux_err_arr = flux_err_arr[window_mask]
        n_points_windowed = int(len(time_folded))

        sort_idx = np.argsort(time_folded)
        time_folded = time_folded[sort_idx]
        flux_arr = flux_arr[sort_idx]
        flux_err_arr = flux_err_arr[sort_idx]

        if max_points is not None and max_points > 0 and len(time_folded) > max_points:
            keep = np.unique(np.linspace(0, len(time_folded) - 1, max_points).astype(int))
            time_folded = time_folded[keep]
            flux_arr = flux_arr[keep]
            flux_err_arr = flux_err_arr[keep]
        n_points_used = int(len(time_folded))

        # Calculate median flux uncertainty for scalar flux_err_0
        flux_err_scalar = float(np.median(flux_err_arr))
        # TRICERATOPS assumes flux_err_0 is a representative per-point uncertainty.
        # For very bright targets, formal pipeline uncertainties can be unrealistically small,
        # driving all scenario evidences (lnZ) toward -inf and yielding NaN probabilities.
        # Use an empirical noise floor from out-of-transit scatter to stabilize.
        empirical_sigma = 0.0
        if use_empirical_noise_floor:
            try:
                oot_mask = np.abs(time_folded) > (dur_days / 2.0)
                oot_flux = flux_arr[oot_mask] if np.any(oot_mask) else flux_arr
                med = float(np.median(oot_flux))
                mad = float(np.median(np.abs(oot_flux - med)))
                empirical_sigma = 1.4826 * mad
            except Exception:
                empirical_sigma = 0.0

        flux_err_scalar = float(max(flux_err_scalar, empirical_sigma, float(min_flux_err)))

        # Convert depth from ppm to fractional for TRICERATOPS
        depth_fractional = depth_ppm / 1e6

        # calc_depths must be called before calc_probs
        # It computes transit depths for all sources in the aperture
        stage_calc_depths_start = time.time()
        logger.info(
            "[fpp] Stage calc_depths: start (TIC=%s)",
            tic_id,
        )
        target.calc_depths(tdepth=depth_fractional)
        calc_depths_runtime = float(time.time() - stage_calc_depths_start)
        logger.info("[fpp] Stage calc_depths: complete (%.1fs)", calc_depths_runtime)

        rem = _remaining_seconds()
        if rem is not None and rem <= 0:
            return _err(
                "FPP calculation timed out before TRICERATOPS calc_probs",
                error_type="timeout",
                stage="triceratops_calc_probs",
                sectors_used_value=sectors_used,
            )

        # Choose MC draw count. TRICERATOPS defaults to N=1e6 which is often too slow
        # for interactive/MCP usage. Use a smaller, budget-aware default.
        draws = int(mc_draws) if mc_draws is not None else 200_000
        if rem is not None:
            if rem < 120:
                draws = min(draws, 50_000)
            elif rem < 300:
                draws = min(draws, 100_000)
            else:
                draws = min(draws, 200_000)
        draws = int(max(10_000, min(draws, 1_000_000)))

        # If the caller provided an overall timeout budget, let calc_probs use the remaining time,
        # but keep a bounded stage budget to avoid silent long hangs.
        calc_timeout = float(TRICERATOPS_CALC_TIMEOUT) if rem is None else float(rem)
        calc_timeout = min(float(calc_timeout), float(FPP_STAGE_CALC_PROBS_BUDGET_SECONDS))
        logger.info(
            "[fpp] Stage triceratops_calc_probs: start (TIC=%s draws=%s max_points=%s budget=%.1fs replicates=%s)",
            tic_id,
            draws,
            max_points,
            float(calc_timeout),
            max(1, int(replicates)) if replicates is not None else 1,
        )

        # Prepare external LC files for TRICERATOPS+ multi-band FPP
        external_lc_files: list[str] | None = None
        filt_lcs: list[str] | None = None
        n_external_lcs = 0
        external_filters_used: list[str] = []

        if external_lightcurves and len(external_lightcurves) > 0:
            n_external_lcs = len(external_lightcurves)
            if n_external_lcs > 4:
                logger.warning(
                    f"TRICERATOPS+ supports up to 4 external LCs; using first 4 of {n_external_lcs}"
                )
                external_lightcurves = external_lightcurves[:4]
                n_external_lcs = 4

        # Determine number of replicates and base seed
        n_replicates = max(1, int(replicates)) if replicates is not None else 1
        base_seed = seed if seed is not None else int(time.time() * 1000) % (2**31)

        # Use temp directory for external LC files / contrast curve file if needed
        temp_dir_obj = (
            tempfile.TemporaryDirectory() if (external_lightcurves or contrast_curve) else None
        )
        replicate_results: list[dict[str, Any]] = []
        replicate_errors: list[dict[str, Any]] = []
        contrast_curve_file: str | None = None
        contrast_curve_filter: str | None = None
        contrast_curve_filter_raw: str | None = None

        stage_calc_probs_start = time.time()
        try:
            if external_lightcurves and temp_dir_obj is not None:
                temp_dir_path = Path(temp_dir_obj.name)
                external_lc_files, filt_lcs = _write_external_lc_files(
                    external_lightcurves, temp_dir_path
                )
                external_filters_used = list(filt_lcs)
                logger.info(
                    f"TRICERATOPS+ multi-band FPP: {n_external_lcs} external LCs "
                    f"in filters {filt_lcs}"
                )

            if contrast_curve is not None and temp_dir_obj is not None:
                temp_dir_path = Path(temp_dir_obj.name)
                try:
                    sep_attr = "separation_arcsec"
                    dmag_attr = "delta_mag"
                    filt_attr = "filter"
                    sep = np.asarray(getattr(contrast_curve, sep_attr), dtype=float)
                    dmag = np.asarray(getattr(contrast_curve, dmag_attr), dtype=float)
                    contrast_curve_filter_raw = getattr(contrast_curve, filt_attr, "Vis")
                    filt = _normalize_triceratops_filter(contrast_curve_filter_raw)
                except Exception:
                    sep = np.array([], dtype=float)
                    dmag = np.array([], dtype=float)
                    filt = "Vis"

                mask = np.isfinite(sep) & np.isfinite(dmag)
                sep = sep[mask]
                dmag = dmag[mask]
                order = np.argsort(sep)
                sep = sep[order]
                dmag = dmag[order]
                if sep.size >= 2:
                    out_path = temp_dir_path / "contrast_curve.txt"
                    # TRICERATOPS expects 2 columns, no header: separation_arcsec, delta_mag
                    # Vendored TRICERATOPS parses with np.loadtxt(..., delimiter=',')
                    np.savetxt(out_path, np.column_stack([sep, dmag]), fmt="%.8f", delimiter=",")
                    contrast_curve_file = str(out_path)
                    contrast_curve_filter = filt
                    logger.info(
                        "Using contrast curve (%s): n=%d, sep=[%.3f..%.3f] arcsec",
                        filt,
                        int(sep.size),
                        float(sep[0]),
                        float(sep[-1]),
                    )

            # Run replicates
            for rep_idx in range(n_replicates):
                # Check timeout budget before each replicate
                rem = _remaining_seconds()
                if rem is not None and rem <= 0:
                    break

                run_seed = base_seed + rep_idx
                run_start_time = time.time()
                with contextlib.suppress(Exception):
                    if progress_hook is not None:
                        progress_hook(
                            {
                                "event": "replicate_start",
                                "replicate_index": int(rep_idx + 1),
                                "replicates_total": int(n_replicates),
                                "seed": int(run_seed),
                            }
                        )

                # Set numpy random seed for this replicate
                np.random.seed(run_seed)

                try:
                    rep_timeout = (
                        float(calc_timeout / max(1, n_replicates - rep_idx))
                        if rem is None
                        else float(rem / max(1, n_replicates - rep_idx))
                    )

                    with network_timeout(
                        float(rep_timeout),
                        operation=f"TRICERATOPS calc_probs for TIC {tic_id} (rep {rep_idx + 1}/{n_replicates})",
                    ):
                        # Pass external LC args to calc_probs (TRICERATOPS+ extension)
                        calc_kwargs: dict[str, Any] = {
                            "time": time_folded,
                            "flux_0": flux_arr,
                            "flux_err_0": flux_err_scalar,
                            "P_orb": period,
                            "N": draws,
                            "exptime": float(exptime_days)
                            if np.isfinite(exptime_days) and exptime_days > 0
                            else 0.00139,
                            "verbose": 0,
                        }
                        if external_lc_files:
                            calc_kwargs["external_lc_files"] = external_lc_files
                            calc_kwargs["filt_lcs"] = filt_lcs
                        if contrast_curve_file is not None:
                            calc_kwargs["contrast_curve_file"] = contrast_curve_file
                            calc_kwargs["filt"] = contrast_curve_filter or "Vis"

                        target.calc_probs(**calc_kwargs)

                    # Extract result for this replicate
                    run_result = _extract_single_run_result(
                        target,
                        tic_id=tic_id,
                        sectors_used=sectors_used,
                        triceratops_engine=triceratops_engine,
                        n_points_raw=n_points_raw,
                        n_points_windowed=n_points_windowed,
                        n_points_used=n_points_used,
                        half_window_days=half_window_days,
                        window_duration_mult=window_duration_mult,
                        max_points=max_points,
                        draws=draws,
                        exptime_days=exptime_days,
                        flux_err_scalar=flux_err_scalar,
                        empirical_sigma=empirical_sigma,
                        min_flux_err=min_flux_err,
                        use_empirical_noise_floor=use_empirical_noise_floor,
                        n_external_lcs=n_external_lcs,
                        external_filters_used=external_filters_used,
                        run_seed=run_seed,
                        run_start_time=run_start_time,
                    )
                    if contrast_curve_file is not None:
                        run_result["contrast_curve"] = {
                            "filter": contrast_curve_filter or "Vis",
                            "filter_raw": str(contrast_curve_filter_raw)
                            if contrast_curve_filter_raw is not None
                            else None,
                            "file": os.path.basename(contrast_curve_file),
                        }
                    replicate_results.append(run_result)
                    with contextlib.suppress(Exception):
                        if progress_hook is not None:
                            progress_hook(
                                {
                                    "event": "replicate_complete",
                                    "replicate_index": int(rep_idx + 1),
                                    "replicates_total": int(n_replicates),
                                    "seed": int(run_seed),
                                    "runtime_seconds": float(time.time() - run_start_time),
                                    "status": "ok",
                                    "fpp": run_result.get("fpp"),
                                }
                            )

                except NetworkTimeoutError as e:
                    replicate_errors.append(
                        {
                            "replicate": rep_idx + 1,
                            "seed": run_seed,
                            "error": str(e),
                            "error_type": "timeout",
                        }
                    )
                    with contextlib.suppress(Exception):
                        if progress_hook is not None:
                            progress_hook(
                                {
                                    "event": "replicate_complete",
                                    "replicate_index": int(rep_idx + 1),
                                    "replicates_total": int(n_replicates),
                                    "seed": int(run_seed),
                                    "runtime_seconds": float(time.time() - run_start_time),
                                    "status": "timeout",
                                    "error": str(e),
                                }
                            )
                except Exception as e:
                    replicate_errors.append(
                        {
                            "replicate": rep_idx + 1,
                            "seed": run_seed,
                            "error": str(e),
                            "error_type": "internal_error",
                        }
                    )
                    with contextlib.suppress(Exception):
                        if progress_hook is not None:
                            progress_hook(
                                {
                                    "event": "replicate_complete",
                                    "replicate_index": int(rep_idx + 1),
                                    "replicates_total": int(n_replicates),
                                    "seed": int(run_seed),
                                    "runtime_seconds": float(time.time() - run_start_time),
                                    "status": "error",
                                    "error": str(e),
                                }
                            )

        finally:
            if temp_dir_obj is not None:
                temp_dir_obj.cleanup()
        logger.info(
            "[fpp] Stage triceratops_calc_probs: complete (%.1fs successes=%s errors=%s)",
            float(time.time() - stage_calc_probs_start),
            len([r for r in replicate_results if not _is_result_degenerate(r)]),
            len(replicate_errors),
        )

    except NetworkTimeoutError as e:
        return _err(
            str(e),
            error_type="timeout",
            stage="triceratops_calc_probs",
            sectors_used_value=sectors_used,
        )
    except Exception as e:
        logger.exception(f"TRICERATOPS calc_probs failed for TIC {tic_id}")
        msg = str(e)
        if "No columns to parse from file" in msg:
            return _err(
                "FPP calculation failed: TRILEGAL returned an empty/malformed response "
                f"({msg}). This is typically an upstream/network issue; retry later.",
                error_type="network_unavailable",
                stage="triceratops_calc_probs",
                sectors_used_value=sectors_used,
            )
        return _err(
            f"FPP calculation failed: {e}",
            error_type="internal_error",
            stage="triceratops_calc_probs",
            sectors_used_value=sectors_used,
        )

    # Step 6: Aggregate replicate results or return degenerate error
    total_runtime = time.time() - start_time

    # Filter to successful (non-degenerate) results
    successful_results = [r for r in replicate_results if not _is_result_degenerate(r)]
    n_success = len(successful_results)
    n_fail = len(replicate_results) - n_success + len(replicate_errors)
    total_attempts = max(1, n_success + n_fail)
    replicate_success_rate = round(float(n_success) / float(total_attempts), 6)
    high_failure_warning = (
        f"High replicate failure rate ({n_fail}/{total_attempts}); review replicate_errors/degenerate_reasons."
        if replicate_success_rate < REPLICATE_SUCCESS_RATE_WARN_THRESHOLD
        else None
    )

    if n_success == 0:
        # All runs failed or were degenerate.
        # Special-case timeout-only failures with clearer semantics/actionable guidance.
        if replicate_errors and all(
            str(e.get("error_type", "")).lower() == "timeout" for e in replicate_errors
        ):
            return {
                "error": (
                    f"All {n_replicates} replicate(s) timed out before producing "
                    "a valid posterior."
                ),
                "error_type": "timeout",
                "stage": "replicate_aggregation",
                "tic_id": tic_id,
                "tmag": tmag,
                "sectors_used": sectors_used,
                "runtime_seconds": round(total_runtime, 1),
                "replicates": n_replicates,
                "n_success": 0,
                "n_fail": n_fail,
                "replicate_success_rate": replicate_success_rate,
                "base_seed": base_seed,
                "replicate_errors": replicate_errors[:5],
                "warning_note": high_failure_warning,
                "actionable_guidance": [
                    "Increase timeout_seconds to allow TRICERATOPS calc_probs to finish.",
                    "Reduce replicates to lower total runtime pressure.",
                    "Retry later if upstream catalog/network services are slow.",
                ],
            }

        # Preserve existing behavior for mixed/non-timeout failures.
        degenerate_reasons: list[str] = []
        for r in replicate_results:
            if r.get("degenerate_reason"):
                degenerate_reasons.append(str(r["degenerate_reason"]))
        for e in replicate_errors:
            degenerate_reasons.append(f"{e.get('error_type', 'error')}:{e.get('error', 'unknown')}")

        return {
            "error": f"All {n_replicates} replicate(s) returned degenerate/invalid posteriors",
            "error_type": "degenerate_posterior",
            "stage": "replicate_aggregation",
            "tic_id": tic_id,
            "tmag": tmag,
            "sectors_used": sectors_used,
            "runtime_seconds": round(total_runtime, 1),
            "replicates": n_replicates,
            "n_success": 0,
            "n_fail": n_fail,
            "replicate_success_rate": replicate_success_rate,
            "base_seed": base_seed,
            "degenerate_reasons": degenerate_reasons[:10],  # Limit to first 10
            "replicate_errors": replicate_errors[:5],  # Limit to first 5
            "warning_note": high_failure_warning,
        }

    # If we have successful results, aggregate them
    if n_replicates > 1 and n_success > 1:
        # Multiple successful replicates - aggregate
        out = _aggregate_replicate_results(
            replicate_results,
            tic_id=tic_id,
            sectors_used=sectors_used,
            total_runtime=total_runtime,
        )
        out["base_seed"] = base_seed
        out["replicate_success_rate"] = replicate_success_rate
        if high_failure_warning is not None:
            out["warning_note"] = high_failure_warning
        if replicate_errors:
            out["replicate_errors"] = replicate_errors[:5]
        return out
    else:
        # Single replicate or only one success - return that result
        out = successful_results[0]
        out["replicates"] = n_replicates
        out["n_success"] = n_success
        out["n_fail"] = n_fail
        out["replicate_success_rate"] = replicate_success_rate
        out["base_seed"] = base_seed
        out["runtime_seconds"] = round(total_runtime, 1)
        if high_failure_warning is not None:
            out["warning_note"] = high_failure_warning
        if replicate_errors:
            out["replicate_errors"] = replicate_errors[:5]

        # For single replicate with degenerate result (shouldn't happen since we filtered),
        # check and return error
        if _is_result_degenerate(out):
            return {
                "error": "FPP calculation returned degenerate posterior",
                "error_type": "degenerate_posterior",
                "stage": "result_validation",
                "tic_id": tic_id,
                "tmag": tmag,
                "sectors_used": sectors_used,
                "runtime_seconds": round(total_runtime, 1),
                "degenerate_reason": out.get("degenerate_reason"),
                "diagnostics": out,
            }

        return out
