"""`btv fpp` command for single-candidate TRICERATOPS FPP estimation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import numpy as np

from bittr_tess_vetter.api.fpp import TUTORIAL_PRESET_OVERRIDES, calculate_fpp
from bittr_tess_vetter.api.fpp_helpers import load_contrast_curve_exofop_tbl
from bittr_tess_vetter.api.stitch import stitch_lightcurve_data
from bittr_tess_vetter.api.transit_masks import get_in_transit_mask, get_out_of_transit_mask, measure_transit_depth
from bittr_tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from bittr_tess_vetter.cli.common_cli import (
    EXIT_DATA_UNAVAILABLE,
    EXIT_INPUT_ERROR,
    EXIT_REMOTE_TIMEOUT,
    EXIT_RUNTIME_ERROR,
    BtvCliError,
    dump_json_output,
    load_json_file,
    parse_extra_params,
    resolve_optional_output_path,
)
from bittr_tess_vetter.cli.stellar_inputs import load_auto_stellar_with_fallback, resolve_stellar_inputs
from bittr_tess_vetter.cli.vet_cli import (
    _detrend_lightcurve_for_vetting,
    _normalize_detrend_method,
    _resolve_candidate_inputs,
    _validate_detrend_args,
)
from bittr_tess_vetter.domain.lightcurve import make_data_ref
from bittr_tess_vetter.platform.io import (
    LightCurveNotFoundError,
    MASTClient,
    PersistentCache,
    TargetNotFoundError,
)

_STANDARD_PRESET_TIMEOUT_SECONDS = 900.0
_MAX_POINTS_RETRY_VALUES = (3000, 2000, 1500, 1000, 750, 500, 300)
_MAX_POINTS_RETRY_LIMIT = 3


def _looks_like_timeout(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    return "timeout" in type(exc).__name__.lower()


def _is_degenerate_fpp_result(result: dict[str, Any]) -> bool:
    """Mirror API helper contract checks for degenerate FPP outputs."""
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
    if posterior_prob_nan_count is not None:
        try:
            return int(posterior_prob_nan_count) > 0
        except (TypeError, ValueError):
            return True
    return False


def _build_retry_guidance(result: dict[str, Any], preset_name: str) -> dict[str, Any] | None:
    if preset_name != "standard" or not _is_degenerate_fpp_result(result):
        return None

    has_error_key = "error" in result
    fpp = result.get("fpp")
    fpp_missing_or_non_finite = fpp is None
    if not fpp_missing_or_non_finite:
        try:
            fpp_missing_or_non_finite = not np.isfinite(float(fpp))
        except (TypeError, ValueError):
            fpp_missing_or_non_finite = True

    posterior_sum_total = result.get("posterior_sum_total")
    posterior_sum_invalid_or_non_positive = False
    if posterior_sum_total is not None:
        try:
            pst = float(posterior_sum_total)
            posterior_sum_invalid_or_non_positive = (not np.isfinite(pst)) or pst <= 0
        except (TypeError, ValueError):
            posterior_sum_invalid_or_non_positive = True

    posterior_prob_nan_count_positive = False
    posterior_prob_nan_count = result.get("posterior_prob_nan_count")
    if posterior_prob_nan_count is not None:
        try:
            posterior_prob_nan_count_positive = int(posterior_prob_nan_count) > 0
        except (TypeError, ValueError):
            posterior_prob_nan_count_positive = True

    return {
        "reason": result.get("degenerate_reason") or "degenerate_fpp_result",
        "preset": "tutorial",
        "overrides": dict(TUTORIAL_PRESET_OVERRIDES),
        "degenerate_checks": {
            "has_error_key": has_error_key,
            "fpp_missing_or_non_finite": fpp_missing_or_non_finite,
            "posterior_sum_invalid_or_non_positive": posterior_sum_invalid_or_non_positive,
            "posterior_prob_nan_count_positive": posterior_prob_nan_count_positive,
        },
    }


def _coerce_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _effective_max_points_for_attempt_zero(*, preset_name: str, overrides: dict[str, Any]) -> int | None:
    if "max_points" in overrides:
        return _coerce_positive_int(overrides.get("max_points"))
    if preset_name == "fast":
        return 1500
    if preset_name == "tutorial":
        return 3000
    return None


def _build_reduced_max_points_schedule(initial_max_points: int | None) -> list[int]:
    if initial_max_points is None:
        return list(_MAX_POINTS_RETRY_VALUES[:_MAX_POINTS_RETRY_LIMIT])

    candidates = [value for value in _MAX_POINTS_RETRY_VALUES if value < initial_max_points]
    if not candidates:
        current = int(initial_max_points)
        while current > 1 and len(candidates) < _MAX_POINTS_RETRY_LIMIT:
            current = max(current // 2, 1)
            if current < initial_max_points and current not in candidates:
                candidates.append(current)
            if current == 1:
                break
    return candidates[:_MAX_POINTS_RETRY_LIMIT]


def _build_cache_for_fpp(
    *,
    tic_id: int,
    sectors: list[int] | None,
    cache_dir: Path | None,
    detrend_cache: bool = False,
    period_days: float = 1.0,
    t0_btjd: float = 0.0,
    duration_hours: float = 1.0,
    depth_ppm: float = 100.0,
    detrend_method: str | None = None,
    detrend_bin_hours: float = 6.0,
    detrend_buffer: float = 2.0,
    detrend_sigma_clip: float = 5.0,
    cache_only_sectors: bool = False,
) -> tuple[PersistentCache, list[int]]:
    client = MASTClient()
    if bool(cache_only_sectors) and sectors is not None:
        lightcurves = []
        for sector in sectors:
            try:
                lightcurves.append(
                    client.download_lightcurve_cached(
                        tic_id=int(tic_id),
                        sector=int(sector),
                        flux_type="pdcsap",
                    )
                )
            except Exception:
                continue
    else:
        lightcurves = client.download_all_sectors(tic_id, flux_type="pdcsap", sectors=sectors)
    if not lightcurves:
        raise LightCurveNotFoundError(f"No sectors available for TIC {tic_id}")

    cache = PersistentCache(cache_dir=cache_dir)
    sectors_loaded: list[int] = []
    for lc_data in lightcurves:
        staged_lc_data = lc_data
        if bool(detrend_cache):
            if detrend_method is None:
                raise BtvCliError(
                    "--detrend-cache requires --detrend",
                    exit_code=EXIT_INPUT_ERROR,
                )
            lc = LightCurve.from_internal(lc_data)
            candidate = Candidate(
                ephemeris=Ephemeris(
                    period_days=float(period_days),
                    t0_btjd=float(t0_btjd),
                    duration_hours=float(duration_hours),
                ),
                depth_ppm=float(depth_ppm),
            )
            detrended_lc, _ = _detrend_lightcurve_for_vetting(
                lc=lc,
                candidate=candidate,
                method=str(detrend_method),
                bin_hours=float(detrend_bin_hours),
                buffer_factor=float(detrend_buffer),
                clip_sigma=float(detrend_sigma_clip),
            )
            staged_lc_data = lc_data.__class__(
                time=np.asarray(detrended_lc.time, dtype=np.float64).copy(),
                flux=np.asarray(detrended_lc.flux, dtype=np.float64).copy(),
                flux_err=np.asarray(detrended_lc.flux_err, dtype=np.float64).copy(),
                quality=np.asarray(lc_data.quality, dtype=np.int32).copy(),
                valid_mask=np.asarray(lc_data.valid_mask, dtype=bool).copy(),
                tic_id=int(lc_data.tic_id),
                sector=int(lc_data.sector),
                cadence_seconds=float(lc_data.cadence_seconds),
                provenance=lc_data.provenance,
            )
        key = make_data_ref(int(tic_id), int(lc_data.sector), "pdcsap")
        cache.put(key, staged_lc_data)
        sectors_loaded.append(int(lc_data.sector))
    return cache, sorted(set(sectors_loaded))


def _execute_fpp(
    *,
    tic_id: int,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    depth_ppm: float,
    sectors: list[int] | None,
    preset: str,
    replicates: int | None,
    seed: int | None,
    timeout_seconds: float | None,
    cache_dir: Path | None,
    stellar_radius: float | None,
    stellar_mass: float | None,
    stellar_tmag: float | None,
    contrast_curve: Any | None,
    overrides: dict[str, Any],
    detrend_cache: bool,
    detrend_method: str | None,
    detrend_bin_hours: float,
    detrend_buffer: float,
    detrend_sigma_clip: float,
    cache_only_sectors: bool = False,
) -> tuple[dict[str, Any], list[int]]:
    cache, sectors_loaded = _build_cache_for_fpp(
        tic_id=tic_id,
        sectors=sectors,
        cache_dir=cache_dir,
        detrend_cache=bool(detrend_cache),
        period_days=float(period_days),
        t0_btjd=float(t0_btjd),
        duration_hours=float(duration_hours),
        depth_ppm=float(depth_ppm),
        detrend_method=detrend_method,
        detrend_bin_hours=float(detrend_bin_hours),
        detrend_buffer=float(detrend_buffer),
        detrend_sigma_clip=float(detrend_sigma_clip),
        cache_only_sectors=bool(cache_only_sectors),
    )
    result = calculate_fpp(
        cache=cache,
        tic_id=tic_id,
        period=period_days,
        t0=t0_btjd,
        depth_ppm=depth_ppm,
        duration_hours=duration_hours,
        sectors=sectors,
        stellar_radius=stellar_radius,
        stellar_mass=stellar_mass,
        tmag=stellar_tmag,
        timeout_seconds=timeout_seconds,
        preset=preset,
        replicates=replicates,
        seed=seed,
        contrast_curve=contrast_curve,
        overrides=overrides,
    )
    return result, sectors_loaded


def _estimate_detrended_depth_ppm(
    *,
    tic_id: int,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    catalog_depth_ppm: float | None,
    sectors: list[int] | None,
    detrend_method: str,
    detrend_bin_hours: float,
    detrend_buffer: float,
    detrend_sigma_clip: float,
    cache_only_sectors: bool = False,
) -> tuple[float | None, dict[str, Any]]:
    client = MASTClient()
    if bool(cache_only_sectors) and sectors is not None:
        lightcurves = []
        for sector in sectors:
            try:
                lightcurves.append(
                    client.download_lightcurve_cached(
                        tic_id=int(tic_id),
                        sector=int(sector),
                        flux_type="pdcsap",
                    )
                )
            except Exception:
                continue
    else:
        lightcurves = client.download_all_sectors(tic_id=int(tic_id), flux_type="pdcsap", sectors=sectors)
    if not lightcurves:
        raise LightCurveNotFoundError(f"No sectors available for TIC {tic_id}")

    stitched = lightcurves[0] if len(lightcurves) == 1 else stitch_lightcurve_data(lightcurves, tic_id=int(tic_id))[0]
    lc = LightCurve.from_internal(stitched)
    candidate = Candidate(
        ephemeris=Ephemeris(
            period_days=float(period_days),
            t0_btjd=float(t0_btjd),
            duration_hours=float(duration_hours),
        ),
        depth_ppm=catalog_depth_ppm,
    )
    detrended_lc, detrend_provenance = _detrend_lightcurve_for_vetting(
        lc=lc,
        candidate=candidate,
        method=str(detrend_method),
        bin_hours=float(detrend_bin_hours),
        buffer_factor=float(detrend_buffer),
        clip_sigma=float(detrend_sigma_clip),
    )
    flux = np.asarray(detrended_lc.flux, dtype=np.float64)
    in_mask = get_in_transit_mask(
        np.asarray(detrended_lc.time, dtype=np.float64),
        float(period_days),
        float(t0_btjd),
        float(duration_hours),
    )
    out_mask = get_out_of_transit_mask(
        np.asarray(detrended_lc.time, dtype=np.float64),
        float(period_days),
        float(t0_btjd),
        float(duration_hours),
        buffer_factor=float(detrend_buffer),
    )
    depth_frac, depth_err_frac = measure_transit_depth(flux, in_mask, out_mask)
    depth_ppm = float(depth_frac * 1_000_000.0)
    depth_err_ppm = float(depth_err_frac * 1_000_000.0)
    if not np.isfinite(depth_ppm) or depth_ppm <= 0.0:
        return None, {
            "method": str(detrend_method),
            "reason": "non_positive_or_non_finite_depth",
            "depth_ppm": depth_ppm,
            "depth_err_ppm": depth_err_ppm,
            "detrend": detrend_provenance,
        }
    return depth_ppm, {
        "method": str(detrend_method),
        "depth_ppm": depth_ppm,
        "depth_err_ppm": depth_err_ppm,
        "detrend": detrend_provenance,
    }


def _extract_report_candidate_inputs(payload: dict[str, Any]) -> tuple[dict[str, Any], list[int] | None]:
    def _dig(root: Any, *path: str) -> Any:
        cur = root
        for key in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return cur

    candidate_blocks = [
        _dig(payload, "inputs_summary", "input_resolution", "inputs"),
        _dig(payload, "inputs_summary", "input_resolution", "resolved"),
        _dig(payload, "report", "inputs_summary", "input_resolution", "inputs"),
        _dig(payload, "report", "inputs_summary", "input_resolution", "resolved"),
        _dig(payload, "provenance", "inputs"),
        _dig(payload, "report", "provenance", "inputs"),
        _dig(payload, "report", "candidate"),
        _dig(payload, "candidate"),
    ]

    candidate: dict[str, Any] = {}
    for block in candidate_blocks:
        if not isinstance(block, dict):
            continue
        for key in ("tic_id", "period_days", "t0_btjd", "duration_hours", "depth_ppm", "toi"):
            value = block.get(key)
            if value is not None and key not in candidate:
                candidate[key] = value

    sectors_candidates = [
        _dig(payload, "provenance", "sectors_used"),
        _dig(payload, "report", "provenance", "sectors_used"),
        _dig(payload, "inputs_summary", "sectors_used"),
        _dig(payload, "report", "inputs_summary", "sectors_used"),
        _dig(payload, "provenance", "inputs", "sectors_loaded"),
        _dig(payload, "provenance", "inputs", "sectors"),
        _dig(payload, "report", "provenance", "inputs", "sectors_loaded"),
        _dig(payload, "report", "provenance", "inputs", "sectors"),
    ]
    sectors_used: list[int] | None = None
    for raw in sectors_candidates:
        if not isinstance(raw, list):
            continue
        parsed: list[int] = []
        for item in raw:
            try:
                parsed.append(int(item))
            except Exception:
                continue
        if parsed:
            sectors_used = list(dict.fromkeys(parsed))
            break
    return candidate, sectors_used


def _load_report_inputs(report_file: Path) -> tuple[dict[str, Any], list[int] | None]:
    payload = load_json_file(report_file, label="report-file")
    return _extract_report_candidate_inputs(payload)


def _load_auto_stellar_inputs(
    tic_id: int,
    toi: str | None = None,
) -> tuple[dict[str, float | None], dict[str, Any]]:
    return load_auto_stellar_with_fallback(tic_id=int(tic_id), toi=toi)


@click.command("fpp")
@click.argument("toi_arg", required=False)
@click.option("--tic-id", type=int, default=None, help="TIC identifier.")
@click.option("--period-days", type=float, default=None, help="Orbital period in days.")
@click.option("--t0-btjd", type=float, default=None, help="Reference epoch in BTJD.")
@click.option("--duration-hours", type=float, default=None, help="Transit duration in hours.")
@click.option("--depth-ppm", type=float, default=None, help="Transit depth in ppm.")
@click.option("--toi", type=str, default=None, help="Optional TOI label (overrides resolved value).")
@click.option(
    "--report-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional prior CLI report/vet JSON to seed candidate inputs and sectors_used.",
)
@click.option(
    "--detrend",
    type=str,
    default=None,
    show_default=True,
    help=(
        "Pre-FPP detrend method. When set, detrending is applied to depth estimation and "
        "to sector flux staged for TRICERATOPS."
    ),
)
@click.option("--detrend-bin-hours", type=float, default=6.0, show_default=True)
@click.option("--detrend-buffer", type=float, default=2.0, show_default=True)
@click.option("--detrend-sigma-clip", type=float, default=5.0, show_default=True)
@click.option(
    "--detrend-cache/--no-detrend-cache",
    default=False,
    show_default=True,
    help=(
        "Stage detrended sector light curves in cache before FPP "
        "(enabled automatically when --detrend is set)."
    ),
)
@click.option(
    "--preset",
    type=click.Choice(["fast", "standard", "tutorial"], case_sensitive=False),
    default="fast",
    show_default=True,
    help="TRICERATOPS runtime preset (standard typically expects a longer timeout budget).",
)
@click.option("--replicates", type=int, default=None, help="Replicate count for FPP aggregation.")
@click.option("--seed", type=int, default=None, help="Base RNG seed.")
@click.option("--override", "overrides", multiple=True, help="Repeat KEY=VALUE TRICERATOPS override entries.")
@click.option("--sectors", multiple=True, type=int, help="Optional sector filters.")
@click.option(
    "--cache-only-sectors/--allow-sector-download",
    default=False,
    show_default=True,
    help="When true, sector loading uses cache-only for selected/report sectors.",
)
@click.option(
    "--timeout-seconds",
    type=float,
    default=None,
    help="Optional timeout budget. If omitted with --preset standard, defaults to 900 seconds.",
)
@click.option(
    "--contrast-curve",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="ExoFOP-style .tbl contrast-curve file for TRICERATOPS companion constraints.",
)
@click.option(
    "--contrast-curve-filter",
    type=str,
    default=None,
    help="Optional band label override for --contrast-curve (for example Kcont, Ks, r).",
)
@click.option(
    "--network-ok/--no-network",
    default=False,
    show_default=True,
    help="Allow network-dependent resolution for TOI inputs.",
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Optional cache directory for FPP light-curve staging.",
)
@click.option("--stellar-radius", type=float, default=None, help="Stellar radius (Rsun).")
@click.option("--stellar-mass", type=float, default=None, help="Stellar mass (Msun).")
@click.option("--stellar-tmag", type=float, default=None, help="TESS magnitude.")
@click.option("--stellar-file", type=str, default=None, help="JSON file with stellar inputs.")
@click.option(
    "--use-stellar-auto/--no-use-stellar-auto",
    default=False,
    show_default=True,
    help="Resolve stellar inputs from TIC when missing from explicit/file inputs.",
)
@click.option(
    "--require-stellar/--no-require-stellar",
    default=False,
    show_default=True,
    help="Fail unless stellar radius and mass resolve.",
)
@click.option(
    "-o",
    "--out",
    "output_path_arg",
    type=str,
    default="-",
    show_default=True,
    help="JSON output path; '-' writes to stdout.",
)
def fpp_command(
    toi_arg: str | None,
    tic_id: int | None,
    period_days: float | None,
    t0_btjd: float | None,
    duration_hours: float | None,
    depth_ppm: float | None,
    toi: str | None,
    report_file: Path | None,
    detrend: str | None,
    detrend_bin_hours: float,
    detrend_buffer: float,
    detrend_sigma_clip: float,
    detrend_cache: bool,
    preset: str,
    replicates: int | None,
    seed: int | None,
    overrides: tuple[str, ...],
    sectors: tuple[int, ...],
    cache_only_sectors: bool,
    timeout_seconds: float | None,
    contrast_curve: Path | None,
    contrast_curve_filter: str | None,
    network_ok: bool,
    cache_dir: Path | None,
    stellar_radius: float | None,
    stellar_mass: float | None,
    stellar_tmag: float | None,
    stellar_file: str | None,
    use_stellar_auto: bool,
    require_stellar: bool,
    output_path_arg: str,
) -> None:
    """Calculate candidate FPP and emit schema-stable JSON."""
    out_path = resolve_optional_output_path(output_path_arg)
    if (
        report_file is None
        and toi_arg is not None
        and toi is not None
        and str(toi_arg).strip() != str(toi).strip()
    ):
        raise BtvCliError(
            "Positional TOI argument and --toi must match when both are provided.",
            exit_code=EXIT_INPUT_ERROR,
        )
    resolved_toi_arg = toi if toi is not None else toi_arg

    report_candidate: dict[str, Any] = {}
    report_sectors_used: list[int] | None = None
    if report_file is not None:
        report_candidate, report_sectors_used = _load_report_inputs(report_file)
        if resolved_toi_arg is not None:
            click.echo(
                "--report-file provided with --toi; preferring report candidate values and using --toi only for missing fields.",
                err=True,
            )

    candidate_tic_id = tic_id if tic_id is not None else report_candidate.get("tic_id")
    candidate_period_days = period_days if period_days is not None else report_candidate.get("period_days")
    candidate_t0_btjd = t0_btjd if t0_btjd is not None else report_candidate.get("t0_btjd")
    candidate_duration_hours = (
        duration_hours if duration_hours is not None else report_candidate.get("duration_hours")
    )
    candidate_depth_ppm = depth_ppm if depth_ppm is not None else report_candidate.get("depth_ppm")

    should_use_toi_resolver = any(
        value is None
        for value in (
            candidate_tic_id,
            candidate_period_days,
            candidate_t0_btjd,
            candidate_duration_hours,
        )
    )
    if candidate_depth_ppm is None and detrend is None:
        should_use_toi_resolver = True
    toi_for_resolution = resolved_toi_arg if should_use_toi_resolver else None

    requested_sectors = [int(s) for s in sectors] if sectors else None
    effective_sectors = requested_sectors if requested_sectors is not None else report_sectors_used
    cache_only_sector_load = bool(cache_only_sectors) and effective_sectors is not None

    (
        resolved_tic_id,
        resolved_period_days,
        resolved_t0_btjd,
        resolved_duration_hours,
        resolved_depth_ppm,
        input_resolution,
    ) = _resolve_candidate_inputs(
        network_ok=network_ok,
        toi=toi_for_resolution,
        tic_id=int(candidate_tic_id) if candidate_tic_id is not None else None,
        period_days=float(candidate_period_days) if candidate_period_days is not None else None,
        t0_btjd=float(candidate_t0_btjd) if candidate_t0_btjd is not None else None,
        duration_hours=float(candidate_duration_hours) if candidate_duration_hours is not None else None,
        depth_ppm=float(candidate_depth_ppm) if candidate_depth_ppm is not None else None,
    )

    detrend_method = _normalize_detrend_method(detrend)
    if detrend_method is not None:
        _validate_detrend_args(
            detrend_bin_hours=float(detrend_bin_hours),
            detrend_buffer=float(detrend_buffer),
            detrend_sigma_clip=float(detrend_sigma_clip),
        )
    detrend_cache_requested = bool(detrend_cache)
    if detrend_cache_requested and detrend_method is None:
        raise BtvCliError("--detrend-cache requires --detrend", exit_code=EXIT_INPUT_ERROR)
    detrend_cache_effective = detrend_cache_requested or (detrend_method is not None)
    if replicates is not None and replicates < 1:
        raise BtvCliError("--replicates must be >= 1", exit_code=EXIT_INPUT_ERROR)
    if use_stellar_auto and not network_ok:
        raise BtvCliError("--use-stellar-auto requires --network-ok", exit_code=EXIT_DATA_UNAVAILABLE)
    if timeout_seconds is not None and float(timeout_seconds) <= 0.0:
        raise BtvCliError("--timeout-seconds must be > 0", exit_code=EXIT_INPUT_ERROR)

    preset_name = str(preset).lower()
    parsed_overrides = parse_extra_params(overrides)
    effective_timeout_seconds = (
        float(timeout_seconds)
        if timeout_seconds is not None
        else (_STANDARD_PRESET_TIMEOUT_SECONDS if preset_name == "standard" else None)
    )
    if timeout_seconds is None and preset_name == "standard":
        click.echo(
            "Using default timeout_seconds=900 for --preset standard.",
            err=True,
        )

    parsed_contrast_curve: Any | None = None
    if contrast_curve is not None:
        try:
            parsed_contrast_curve = load_contrast_curve_exofop_tbl(
                contrast_curve,
                filter=contrast_curve_filter,
            )
        except Exception as exc:
            raise BtvCliError(
                f"Failed to parse --contrast-curve: {exc}",
                exit_code=EXIT_INPUT_ERROR,
            ) from exc

    try:
        resolved_stellar, stellar_resolution = resolve_stellar_inputs(
            tic_id=resolved_tic_id,
            stellar_radius=stellar_radius,
            stellar_mass=stellar_mass,
            stellar_tmag=stellar_tmag,
            stellar_file=stellar_file,
            use_stellar_auto=use_stellar_auto,
            require_stellar=require_stellar,
        auto_loader=(
            (lambda _tic_id: _load_auto_stellar_inputs(_tic_id, toi=resolved_toi_arg))
            if resolved_toi_arg is not None
            else (lambda _tic_id: _load_auto_stellar_inputs(_tic_id))
        )
        if use_stellar_auto
            else None,
        )
    except (TargetNotFoundError, LightCurveNotFoundError) as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_DATA_UNAVAILABLE) from exc
    except Exception as exc:
        if isinstance(exc, BtvCliError):
            raise
        mapped = EXIT_REMOTE_TIMEOUT if _looks_like_timeout(exc) else EXIT_RUNTIME_ERROR
        raise BtvCliError(str(exc), exit_code=mapped) from exc

    depth_source = "catalog"
    depth_ppm_used = resolved_depth_ppm
    detrended_depth_meta: dict[str, Any] | None = None
    if depth_ppm is not None:
        depth_source = "explicit"
        depth_ppm_used = float(depth_ppm)
    elif detrend_method is not None:
        try:
            detrended_depth, detrended_depth_meta = _estimate_detrended_depth_ppm(
                tic_id=resolved_tic_id,
                period_days=resolved_period_days,
                t0_btjd=resolved_t0_btjd,
                duration_hours=resolved_duration_hours,
                catalog_depth_ppm=resolved_depth_ppm,
                sectors=effective_sectors,
                detrend_method=detrend_method,
                detrend_bin_hours=float(detrend_bin_hours),
                detrend_buffer=float(detrend_buffer),
                detrend_sigma_clip=float(detrend_sigma_clip),
                cache_only_sectors=cache_only_sector_load,
            )
            if detrended_depth is not None:
                depth_ppm_used = float(detrended_depth)
                depth_source = "detrended"
        except Exception as exc:
            mapped = EXIT_REMOTE_TIMEOUT if _looks_like_timeout(exc) else EXIT_RUNTIME_ERROR
            if isinstance(exc, LightCurveNotFoundError):
                mapped = EXIT_DATA_UNAVAILABLE
            if resolved_depth_ppm is None:
                raise BtvCliError(str(exc), exit_code=mapped) from exc
            detrended_depth_meta = {
                "method": str(detrend_method),
                "reason": "detrended_depth_failed",
                "error": str(exc),
            }

    if depth_ppm_used is None:
        exit_code = EXIT_DATA_UNAVAILABLE if resolved_toi_arg is not None else EXIT_INPUT_ERROR
        raise BtvCliError(
            "Missing transit depth. Provide --depth-ppm, enable --detrend, or use --toi with depth metadata.",
            exit_code=exit_code,
        )

    initial_max_points = _effective_max_points_for_attempt_zero(
        preset_name=preset_name,
        overrides=parsed_overrides,
    )
    retry_schedule_max_points = _build_reduced_max_points_schedule(initial_max_points)
    explicit_max_points_override = "max_points" in parsed_overrides
    attempts: list[dict[str, Any]] = []
    final_selected_attempt = 1
    fallback_succeeded = False

    try:
        result: dict[str, Any] | None = None
        sectors_loaded: list[int] | None = None
        attempts_max_points: list[int | None] = [initial_max_points, *retry_schedule_max_points]
        for attempt_index, attempt_max_points in enumerate(attempts_max_points, start=1):
            attempt_overrides = dict(parsed_overrides)
            if attempt_index > 1:
                attempt_overrides["max_points"] = attempt_max_points
            result, sectors_loaded = _execute_fpp(
                tic_id=resolved_tic_id,
                period_days=resolved_period_days,
                t0_btjd=resolved_t0_btjd,
                duration_hours=resolved_duration_hours,
                depth_ppm=float(depth_ppm_used),
                sectors=effective_sectors,
                preset=preset_name,
                replicates=replicates,
                seed=seed,
                timeout_seconds=effective_timeout_seconds,
                cache_dir=cache_dir,
                stellar_radius=resolved_stellar.get("radius"),
                stellar_mass=resolved_stellar.get("mass"),
                stellar_tmag=resolved_stellar.get("tmag"),
                contrast_curve=parsed_contrast_curve,
                overrides=attempt_overrides,
                detrend_cache=bool(detrend_cache_effective),
                detrend_method=detrend_method,
                detrend_bin_hours=float(detrend_bin_hours),
                detrend_buffer=float(detrend_buffer),
                detrend_sigma_clip=float(detrend_sigma_clip),
                cache_only_sectors=cache_only_sector_load,
            )
            is_degenerate = _is_degenerate_fpp_result(result)
            attempts.append(
                {
                    "attempt": int(attempt_index),
                    "max_points": attempt_overrides.get("max_points"),
                    "degenerate": bool(is_degenerate),
                    "reason": result.get("degenerate_reason"),
                }
            )
            final_selected_attempt = int(attempt_index)
            if not is_degenerate:
                if attempt_index > 1:
                    fallback_succeeded = True
                break
            if attempt_index >= len(attempts_max_points):
                break
        if result is None or sectors_loaded is None:
            raise BtvCliError("FPP execution did not return a result.", exit_code=EXIT_RUNTIME_ERROR)
    except BtvCliError:
        raise
    except LightCurveNotFoundError as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_DATA_UNAVAILABLE) from exc
    except Exception as exc:
        mapped = EXIT_REMOTE_TIMEOUT if _looks_like_timeout(exc) else EXIT_RUNTIME_ERROR
        raise BtvCliError(str(exc), exit_code=mapped) from exc

    payload: dict[str, Any] = {
        "schema_version": "cli.fpp.v3",
        "fpp_result": result,
        "provenance": {
            "depth_source": depth_source,
            "depth_ppm_used": float(depth_ppm_used),
            "inputs": {
                "tic_id": resolved_tic_id,
                "period_days": resolved_period_days,
                "t0_btjd": resolved_t0_btjd,
                "duration_hours": resolved_duration_hours,
                "depth_ppm": float(depth_ppm_used),
                "depth_ppm_catalog": resolved_depth_ppm,
                "sectors": effective_sectors,
                "sectors_loaded": sectors_loaded,
            },
            "resolved_source": input_resolution.get("source"),
            "resolved_from": input_resolution.get("resolved_from"),
            "stellar": stellar_resolution,
            "detrended_depth": detrended_depth_meta,
            "contrast_curve": {
                "path": str(contrast_curve) if contrast_curve is not None else None,
                "filter": str(parsed_contrast_curve.filter) if parsed_contrast_curve is not None else None,
            },
            "runtime": {
                "preset": preset_name,
                "replicates": replicates,
                "overrides": parsed_overrides,
                "detrend_cache": bool(detrend_cache_effective),
                "detrend_cache_requested": bool(detrend_cache_requested),
                "seed_requested": seed,
                "seed_effective": result.get("base_seed", seed),
                "timeout_seconds_requested": timeout_seconds,
                "timeout_seconds": effective_timeout_seconds,
                "network_ok": bool(network_ok),
                "degenerate_guard": {
                    "guard_triggered": bool(attempts and attempts[0]["degenerate"]),
                    "explicit_max_points_override": bool(explicit_max_points_override),
                    "initial_max_points": initial_max_points,
                    "retry_schedule_max_points": retry_schedule_max_points,
                    "attempts": attempts,
                    "final_selected_attempt": int(final_selected_attempt),
                    "fallback_succeeded": bool(fallback_succeeded),
                },
            },
        },
    }
    retry_guidance = _build_retry_guidance(result=result, preset_name=preset_name)
    if retry_guidance is not None:
        payload["provenance"]["retry_guidance"] = retry_guidance
    dump_json_output(payload, out_path)


__all__ = ["fpp_command"]
