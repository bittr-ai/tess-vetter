"""`btv fpp` command for single-candidate TRICERATOPS FPP estimation."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import numpy as np
from click.core import ParameterSource

from tess_vetter.api.fpp import TUTORIAL_PRESET_OVERRIDES, calculate_fpp
from tess_vetter.api.stitch import stitch_lightcurve_data
from tess_vetter.api.transit_masks import (
    get_in_transit_mask,
    get_out_of_transit_mask,
    measure_transit_depth,
)
from tess_vetter.api.triceratops_cache import (
    load_cached_triceratops_target,
    stage_triceratops_runtime_artifacts,
)
from tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from tess_vetter.cli.common_cli import (
    EXIT_DATA_UNAVAILABLE,
    EXIT_INPUT_ERROR,
    EXIT_REMOTE_TIMEOUT,
    EXIT_RUNTIME_ERROR,
    BtvCliError,
    dump_json_output,
    emit_progress,
    load_json_file,
    parse_extra_params,
    resolve_optional_output_path,
)
from tess_vetter.cli.diagnostics_report_inputs import resolve_inputs_from_report_file
from tess_vetter.cli.fpp_compute_guidance import (
    apply_compute_guidance_to_runtime_knobs,
    build_prepare_compute_guidance,
)
from tess_vetter.cli.stellar_inputs import load_auto_stellar_with_fallback, resolve_stellar_inputs
from tess_vetter.cli.vet_cli import (
    _detrend_lightcurve_for_vetting,
    _normalize_detrend_method,
    _resolve_candidate_inputs,
    _validate_detrend_args,
)
from tess_vetter.contrast_curves import parse_contrast_curve_with_provenance
from tess_vetter.domain.lightcurve import make_data_ref
from tess_vetter.platform.io import (
    LightCurveNotFoundError,
    MASTClient,
    PersistentCache,
    TargetNotFoundError,
)
from tess_vetter.validation.triceratops_fpp import (
    _triceratops_artifact_file_lock,
    droppable_scenario_labels,
    normalize_drop_scenario_labels,
)

_STANDARD_PRESET_TIMEOUT_SECONDS = 900.0
_MAX_POINTS_RETRY_VALUES = (3000, 2000, 1500, 1000, 750, 500, 300)
_MAX_POINTS_RETRY_LIMIT = 3
_VERDICT_TOKEN_PATTERN = re.compile(r"[^A-Z0-9]+")
_LC_KEY_PATTERN = re.compile(r"^lc:(?P<tic>\d+):(?P<sector>\d+):(?P<flux>[a-z0-9_]+)$")
_FPP_PREPARE_SCHEMA_VERSION = "cli.fpp.prepare.v2"
_FPP_PREPARE_SCHEMA_VERSION_COMPAT = frozenset({"cli.fpp.prepare.v1", "cli.fpp.prepare.v2"})

logger = logging.getLogger(__name__)


def _emit_fpp_replicate_progress(command: str, payload: dict[str, Any]) -> None:
    event = str(payload.get("event") or "")
    idx = payload.get("replicate_index")
    total = payload.get("replicates_total")
    seed = payload.get("seed")
    if event == "replicate_start":
        emit_progress(str(command), "replicate", detail=f"{idx}/{total} seed={seed} start")
        return
    if event == "replicate_complete":
        status = str(payload.get("status") or "unknown")
        emit_progress(str(command), "replicate", detail=f"{idx}/{total} seed={seed} status={status}")


def _looks_like_timeout(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    return "timeout" in type(exc).__name__.lower()


def _load_cli_contrast_curve(
    *,
    contrast_curve: Path | None,
    contrast_curve_filter: str | None,
) -> tuple[Any | None, dict[str, Any] | None]:
    if contrast_curve is None:
        return None, None
    try:
        parsed, parse_provenance = parse_contrast_curve_with_provenance(
            contrast_curve,
            filter_name=contrast_curve_filter,
        )
        return parsed, parse_provenance
    except Exception as exc:
        raise BtvCliError(
            f"Failed to parse --contrast-curve: {exc}",
            exit_code=EXIT_INPUT_ERROR,
        ) from exc


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


def _derive_fpp_verdict(result: dict[str, Any]) -> tuple[str, str]:
    disposition = result.get("disposition")
    if isinstance(disposition, str) and disposition.strip():
        token = _VERDICT_TOKEN_PATTERN.sub("_", disposition.strip().upper()).strip("_")
        if not token:
            token = "UNKNOWN"
        return f"FPP_{token}", "$.fpp_result.disposition"

    fpp = result.get("fpp")
    try:
        fpp_value = float(fpp)
    except (TypeError, ValueError):
        fpp_value = None
    if fpp_value is not None and np.isfinite(fpp_value):
        if fpp_value <= 0.01:
            return "FPP_LOW", "$.fpp_result.fpp"
        if fpp_value <= 0.1:
            return "FPP_MODERATE", "$.fpp_result.fpp"
        return "FPP_HIGH", "$.fpp_result.fpp"

    if result.get("error") is not None:
        return "FPP_ERROR", "$.fpp_result.error"
    return "FPP_UNAVAILABLE", "$.fpp_result"


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


def _apply_replicate_output_controls(
    result: dict[str, Any],
    *,
    replicate_detail: str,
    replicate_errors_limit: int,
) -> dict[str, Any]:
    shaped = dict(result)
    analysis = shaped.get("replicate_analysis")
    if isinstance(analysis, dict):
        updated_analysis = dict(analysis)
        updated_analysis["detail_level"] = str(replicate_detail).lower()
        runs = updated_analysis.get("runs")
        if isinstance(runs, list):
            updated_analysis["runs_count"] = int(len(runs))
        if str(replicate_detail).lower() == "compact" and isinstance(runs, list):
            updated_analysis.pop("runs", None)

        errors = updated_analysis.get("errors")
        if (
            isinstance(errors, list)
            and int(replicate_errors_limit) > 0
            and len(errors) > int(replicate_errors_limit)
        ):
            updated_analysis["errors"] = list(errors[: int(replicate_errors_limit)])
            updated_analysis["errors_truncated"] = int(len(errors) - int(replicate_errors_limit))
        shaped["replicate_analysis"] = updated_analysis
    return shaped


def _coerce_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _resolve_drop_scenario_override(
    *,
    parsed_overrides: dict[str, Any],
    explicit_drop_scenarios: tuple[str, ...] | None = None,
) -> list[str]:
    explicit_values = tuple(explicit_drop_scenarios or ())
    selected_value: Any | None = None
    if explicit_values:
        selected_value = list(explicit_values)
    elif "drop_scenario" in parsed_overrides:
        selected_value = parsed_overrides.get("drop_scenario")

    try:
        normalized = normalize_drop_scenario_labels(selected_value)
    except ValueError as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_INPUT_ERROR) from exc

    if explicit_values or "drop_scenario" in parsed_overrides:
        parsed_overrides["drop_scenario"] = list(normalized)
    return normalized


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


def _execute_fpp_with_retry(
    *,
    preset_name: str,
    parsed_overrides: dict[str, Any],
    run_attempt: Callable[[dict[str, Any]], tuple[dict[str, Any], Any]],
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    initial_max_points = _effective_max_points_for_attempt_zero(
        preset_name=preset_name,
        overrides=parsed_overrides,
    )
    retry_schedule_max_points = _build_reduced_max_points_schedule(initial_max_points)
    explicit_max_points_override = "max_points" in parsed_overrides
    attempts: list[dict[str, Any]] = []
    final_selected_attempt = 1
    fallback_succeeded = False
    attempts_max_points: list[int | None] = [initial_max_points, *retry_schedule_max_points]

    result: dict[str, Any] | None = None
    selected_data: Any = None
    for attempt_index, attempt_max_points in enumerate(attempts_max_points, start=1):
        attempt_overrides = dict(parsed_overrides)
        if attempt_index > 1:
            attempt_overrides["max_points"] = attempt_max_points
        result, selected_data = run_attempt(attempt_overrides)
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

    if result is None:
        raise BtvCliError("FPP execution did not return a result.", exit_code=EXIT_RUNTIME_ERROR)
    retry_meta = {
        "explicit_max_points_override": bool(explicit_max_points_override),
        "initial_max_points": initial_max_points,
        "retry_schedule_max_points": retry_schedule_max_points,
        "attempts": attempts,
        "final_selected_attempt": int(final_selected_attempt),
        "fallback_succeeded": bool(fallback_succeeded),
    }
    return result, selected_data, retry_meta


def _new_mast_client(*, cache_dir: Path | None) -> MASTClient:
    if cache_dir is None:
        return MASTClient()
    try:
        return MASTClient(cache_dir=cache_dir)
    except TypeError:
        return MASTClient()


def _cached_sectors_for_tic(
    *,
    cache: PersistentCache,
    tic_id: int,
    flux_type: str = "pdcsap",
) -> list[int]:
    sectors: list[int] = []
    wanted_flux = str(flux_type).lower()
    cache_keys = getattr(cache, "keys", None)
    keys_iterable = cache_keys() if callable(cache_keys) else []
    for key in keys_iterable:
        match = _LC_KEY_PATTERN.match(str(key))
        if match is None:
            continue
        try:
            key_tic = int(match.group("tic"))
            key_sector = int(match.group("sector"))
            key_flux = str(match.group("flux")).lower()
        except Exception:
            continue
        if key_tic == int(tic_id) and key_flux == wanted_flux:
            sectors.append(key_sector)
    return sorted(set(sectors))


def _load_lightcurves_for_fpp(
    *,
    tic_id: int,
    sectors: list[int] | None,
    cache: PersistentCache,
    client: MASTClient,
    cache_only_sectors: bool,
    flux_type: str = "pdcsap",
) -> tuple[list[Any], str]:
    requested = sorted({int(s) for s in sectors}) if sectors is not None else None
    cached_sectors = requested if requested is not None else _cached_sectors_for_tic(
        cache=cache,
        tic_id=int(tic_id),
        flux_type=str(flux_type).lower(),
    )

    lightcurves: list[Any] = []
    loaded_sectors: set[int] = set()
    for sector in cached_sectors:
        key = make_data_ref(int(tic_id), int(sector), str(flux_type).lower())
        cache_get = getattr(cache, "get", None)
        cached_item = cache_get(key) if callable(cache_get) else None
        if cached_item is None:
            continue
        lightcurves.append(cached_item)
        loaded_sectors.add(int(sector))

    if requested is None:
        if lightcurves:
            return lightcurves, "persistent_cache_only"
        fetched = client.download_all_sectors(
            int(tic_id),
            flux_type=str(flux_type).lower(),
            sectors=None,
        )
        return list(fetched), "mast_all_sectors"

    missing = [int(s) for s in requested if int(s) not in loaded_sectors]
    if not missing:
        return lightcurves, "persistent_cache_requested_sectors"

    cached_loaded = 0
    if bool(cache_only_sectors):
        for sector in missing:
            try:
                lc = client.download_lightcurve_cached(
                    tic_id=int(tic_id),
                    sector=int(sector),
                    flux_type=str(flux_type).lower(),
                )
            except Exception:
                continue
            lightcurves.append(lc)
            loaded_sectors.add(int(sector))
            cached_loaded += 1
        if lightcurves:
            return lightcurves, (
                "persistent_plus_lightkurve_cache_requested_sectors"
                if cached_loaded > 0
                else "persistent_cache_requested_sectors"
            )
        return [], "cache_only_requested_sectors_miss"

    remaining = [int(s) for s in requested if int(s) not in loaded_sectors]
    if remaining:
        fetched = client.download_all_sectors(
            int(tic_id),
            flux_type=str(flux_type).lower(),
            sectors=remaining,
        )
        lightcurves.extend(list(fetched))
    return lightcurves, "persistent_plus_mast_requested_sectors"


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
    cache = PersistentCache(cache_dir=cache_dir)
    client = _new_mast_client(cache_dir=cache_dir)
    logger.info(
        "[fpp] Loading light curves (TIC=%s sectors=%s cache_dir=%s cache_only=%s)",
        int(tic_id),
        sectors,
        str(cache.cache_dir),
        bool(cache_only_sectors),
    )
    lightcurves, load_path = _load_lightcurves_for_fpp(
        tic_id=int(tic_id),
        sectors=sectors,
        cache=cache,
        client=client,
        cache_only_sectors=bool(cache_only_sectors),
        flux_type="pdcsap",
    )
    if not lightcurves:
        raise LightCurveNotFoundError(
            f"No sectors available for TIC {tic_id} (load_path={load_path})."
        )
    logger.info(
        "[fpp] Loaded %s sector light curves for TIC %s via %s",
        len(lightcurves),
        int(tic_id),
        load_path,
    )

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
    allow_network: bool = True,
    progress_hook: Callable[[dict[str, Any]], None] | None = None,
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
        allow_network=allow_network,
        progress_hook=progress_hook,
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
    cache_dir: Path | None = None,
    cache_only_sectors: bool = False,
) -> tuple[float | None, dict[str, Any]]:
    cache = PersistentCache(cache_dir=cache_dir)
    client = _new_mast_client(cache_dir=cache_dir)
    lightcurves, load_path = _load_lightcurves_for_fpp(
        tic_id=int(tic_id),
        sectors=sectors,
        cache=cache,
        client=client,
        cache_only_sectors=bool(cache_only_sectors),
        flux_type="pdcsap",
    )
    if not lightcurves:
        raise LightCurveNotFoundError(
            f"No sectors available for TIC {tic_id} (load_path={load_path})."
        )

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
    candidate, sectors_used = _extract_report_candidate_inputs(payload)
    try:
        resolved_from_report = resolve_inputs_from_report_file(str(report_file))
    except BtvCliError:
        return candidate, sectors_used

    report_candidate = {
        "tic_id": int(resolved_from_report.tic_id),
        "period_days": float(resolved_from_report.period_days),
        "t0_btjd": float(resolved_from_report.t0_btjd),
        "duration_hours": float(resolved_from_report.duration_hours),
        "depth_ppm": (
            float(resolved_from_report.depth_ppm) if resolved_from_report.depth_ppm is not None else None
        ),
    }
    for key, value in report_candidate.items():
        if candidate.get(key) is None and value is not None:
            candidate[key] = value
    if sectors_used is None and resolved_from_report.sectors_used is not None:
        sectors_used = [int(s) for s in resolved_from_report.sectors_used]
    return candidate, sectors_used


def _load_auto_stellar_inputs(
    tic_id: int,
    toi: str | None = None,
) -> tuple[dict[str, float | None], dict[str, Any]]:
    return load_auto_stellar_with_fallback(tic_id=int(tic_id), toi=toi)


def _cache_missing_sectors(
    *,
    cache: PersistentCache,
    tic_id: int,
    sectors_loaded: list[int],
) -> list[int]:
    missing: list[int] = []
    for sector in sectors_loaded:
        key = make_data_ref(int(tic_id), int(sector), "pdcsap")
        cache_get = getattr(cache, "get", None)
        cached_item = cache_get(key) if callable(cache_get) else None
        if cached_item is None:
            missing.append(int(sector))
    return missing


def _coerce_manifest_number(payload: dict[str, Any], field: str) -> float:
    raw = payload.get(field)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise BtvCliError(f"Invalid prepare manifest field '{field}'", exit_code=EXIT_INPUT_ERROR) from exc
    if not np.isfinite(value):
        raise BtvCliError(f"Invalid prepare manifest field '{field}'", exit_code=EXIT_INPUT_ERROR)
    return value


def _load_prepare_manifest(path: Path) -> dict[str, Any]:
    payload = load_json_file(path, label="prepare-manifest")
    schema = payload.get("schema_version")
    if schema not in _FPP_PREPARE_SCHEMA_VERSION_COMPAT:
        raise BtvCliError(
            (
                f"Unsupported prepare-manifest schema_version '{schema}'. "
                f"Expected one of {sorted(_FPP_PREPARE_SCHEMA_VERSION_COMPAT)}."
            ),
            exit_code=EXIT_INPUT_ERROR,
        )

    try:
        tic_id = int(payload.get("tic_id"))
    except (TypeError, ValueError) as exc:
        raise BtvCliError("Invalid prepare manifest field 'tic_id'", exit_code=EXIT_INPUT_ERROR) from exc
    if tic_id <= 0:
        raise BtvCliError("Invalid prepare manifest field 'tic_id'", exit_code=EXIT_INPUT_ERROR)

    sectors_raw = payload.get("sectors_loaded")
    if not isinstance(sectors_raw, list):
        raise BtvCliError("Invalid prepare manifest field 'sectors_loaded'", exit_code=EXIT_INPUT_ERROR)
    sectors_loaded: list[int] = []
    for item in sectors_raw:
        try:
            sectors_loaded.append(int(item))
        except (TypeError, ValueError) as exc:
            raise BtvCliError("Invalid prepare manifest field 'sectors_loaded'", exit_code=EXIT_INPUT_ERROR) from exc
    sectors_loaded = sorted(set(sectors_loaded))
    if not sectors_loaded:
        raise BtvCliError("Prepare manifest has empty sectors_loaded", exit_code=EXIT_INPUT_ERROR)

    cache_dir_raw = payload.get("cache_dir")
    cache_dir = Path(str(cache_dir_raw)).expanduser() if cache_dir_raw is not None else None
    if cache_dir is None or not str(cache_dir).strip():
        raise BtvCliError("Invalid prepare manifest field 'cache_dir'", exit_code=EXIT_INPUT_ERROR)

    detrend_payload = payload.get("detrend")
    if detrend_payload is not None and not isinstance(detrend_payload, dict):
        raise BtvCliError("Invalid prepare manifest field 'detrend'", exit_code=EXIT_INPUT_ERROR)
    runtime_artifacts = payload.get("runtime_artifacts")
    if runtime_artifacts is not None and not isinstance(runtime_artifacts, dict):
        raise BtvCliError("Invalid prepare manifest field 'runtime_artifacts'", exit_code=EXIT_INPUT_ERROR)
    compute_insights = payload.get("compute_insights")
    if compute_insights is not None and not isinstance(compute_insights, dict):
        raise BtvCliError("Invalid prepare manifest field 'compute_insights'", exit_code=EXIT_INPUT_ERROR)

    return {
        "schema_version": schema,
        "created_at": payload.get("created_at"),
        "tic_id": tic_id,
        "period_days": _coerce_manifest_number(payload, "period_days"),
        "t0_btjd": _coerce_manifest_number(payload, "t0_btjd"),
        "duration_hours": _coerce_manifest_number(payload, "duration_hours"),
        "depth_ppm_used": _coerce_manifest_number(payload, "depth_ppm_used"),
        "sectors_loaded": sectors_loaded,
        "cache_dir": cache_dir,
        "detrend": detrend_payload if isinstance(detrend_payload, dict) else {},
        "runtime_artifacts": runtime_artifacts if isinstance(runtime_artifacts, dict) else {},
        "compute_insights": compute_insights if isinstance(compute_insights, dict) else None,
    }


def _runtime_artifacts_ready(
    *,
    cache_dir: Path,
    tic_id: int,
    sectors_loaded: list[int],
    stage_state_path: Path | None = None,
) -> tuple[bool, dict[str, Any]]:
    sectors_used = [int(s) for s in sectors_loaded]
    effective_stage_state_path = stage_state_path
    stage_state_required = effective_stage_state_path is not None

    with _triceratops_artifact_file_lock(
        cache_dir=cache_dir,
        tic_id=int(tic_id),
        sectors_used=sectors_used,
        wait=True,
    ):
        target = load_cached_triceratops_target(
            cache_dir=cache_dir,
            tic_id=int(tic_id),
            sectors_used=sectors_used,
        )
        trilegal_ready = False
        trilegal_path: str | None = None
        if target is not None:
            trilegal_fname = getattr(target, "trilegal_fname", None)
            if trilegal_fname is not None:
                trilegal_csv = Path(str(trilegal_fname))
                try:
                    if trilegal_csv.exists() and trilegal_csv.stat().st_size > 0:
                        trilegal_ready = True
                        trilegal_path = str(trilegal_csv)
                except OSError:
                    trilegal_ready = False

        stage_state_status: str | None = None
        stage_state_ok = False
        if stage_state_required and effective_stage_state_path is not None:
            try:
                if effective_stage_state_path.exists():
                    payload = json.loads(effective_stage_state_path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        raw_status = payload.get("status")
                        if isinstance(raw_status, str):
                            stage_state_status = raw_status
                            stage_state_ok = raw_status == "ok"
            except Exception:
                stage_state_ok = False
        else:
            stage_state_ok = True

    ready = (target is not None) and trilegal_ready and (stage_state_ok or not stage_state_required)
    return ready, {
        "target_cached": bool(target is not None),
        "trilegal_cached": bool(trilegal_ready),
        "trilegal_csv_path": trilegal_path,
        "stage_state_path": str(effective_stage_state_path) if effective_stage_state_path is not None else None,
        "stage_state_status": stage_state_status,
        "stage_state_ok": bool(stage_state_ok) if stage_state_required else None,
    }


@click.command("fpp-prepare")
@click.option("--tic-id", type=int, default=None, help="TIC identifier.")
@click.option("--period-days", type=float, default=None, help="Orbital period in days.")
@click.option("--t0-btjd", type=float, default=None, help="Reference epoch in BTJD.")
@click.option("--duration-hours", type=float, default=None, help="Transit duration in hours.")
@click.option("--depth-ppm", type=float, default=None, help="Transit depth in ppm.")
@click.option("--toi", type=str, default=None, help="Optional TOI label for input resolution.")
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
    help="Stage detrended sector light curves in cache before FPP.",
)
@click.option("--sectors", multiple=True, type=int, help="Optional sector filters.")
@click.option(
    "--cache-only-sectors/--allow-sector-download",
    default=False,
    show_default=True,
    help="When true, sector loading uses cache-only for selected/report sectors.",
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
@click.option(
    "--timeout-seconds",
    type=float,
    default=None,
    help=(
        "Optional overall timeout for TRICERATOPS runtime staging. When set, "
        "stage budgets draw from this value."
    ),
)
@click.option(
    "--compute-guidance/--no-compute-guidance",
    default=True,
    show_default=True,
    help="Include optional non-binding compute guidance in the prepared manifest.",
)
@click.option(
    "-o",
    "--out",
    "output_manifest_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output JSON manifest path for staged FPP run.",
)
def fpp_prepare_command(
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
    sectors: tuple[int, ...],
    cache_only_sectors: bool,
    network_ok: bool,
    cache_dir: Path | None,
    timeout_seconds: float | None,
    compute_guidance: bool,
    output_manifest_path: Path,
) -> None:
    """Resolve candidate inputs and stage cache artifacts for FPP compute."""
    click.echo("[fpp-prepare] Resolving candidate inputs...")
    report_candidate: dict[str, Any] = {}
    report_sectors_used: list[int] | None = None
    if report_file is not None:
        report_candidate, report_sectors_used = _load_report_inputs(report_file)
        if toi is not None and network_ok:
            click.echo(
                "[fpp-prepare] --report-file provided with --toi; using --toi for candidate inputs and report only for sectors.",
                err=True,
            )

    use_report_for_candidate_inputs = not (toi is not None and network_ok)
    report_candidate_inputs = report_candidate if use_report_for_candidate_inputs else {}

    candidate_tic_id = tic_id if tic_id is not None else report_candidate_inputs.get("tic_id")
    candidate_period_days = period_days if period_days is not None else report_candidate_inputs.get("period_days")
    candidate_t0_btjd = t0_btjd if t0_btjd is not None else report_candidate_inputs.get("t0_btjd")
    candidate_duration_hours = (
        duration_hours if duration_hours is not None else report_candidate_inputs.get("duration_hours")
    )
    candidate_depth_ppm = depth_ppm if depth_ppm is not None else report_candidate_inputs.get("depth_ppm")

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
    if toi is not None and network_ok:
        should_use_toi_resolver = True
    toi_for_resolution = toi if should_use_toi_resolver else None

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
    if timeout_seconds is not None and float(timeout_seconds) <= 0.0:
        raise BtvCliError("--timeout-seconds must be > 0", exit_code=EXIT_INPUT_ERROR)

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
                cache_dir=cache_dir,
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
        exit_code = EXIT_DATA_UNAVAILABLE if toi is not None else EXIT_INPUT_ERROR
        raise BtvCliError(
            "Missing transit depth. Provide --depth-ppm, enable --detrend, or use --toi with depth metadata.",
            exit_code=exit_code,
        )

    click.echo("[fpp-prepare] Staging sector light curves in cache...")
    try:
        cache, sectors_loaded = _build_cache_for_fpp(
            tic_id=resolved_tic_id,
            sectors=effective_sectors,
            cache_dir=cache_dir,
            detrend_cache=bool(detrend_cache_effective),
            period_days=float(resolved_period_days),
            t0_btjd=float(resolved_t0_btjd),
            duration_hours=float(resolved_duration_hours),
            depth_ppm=float(depth_ppm_used),
            detrend_method=detrend_method,
            detrend_bin_hours=float(detrend_bin_hours),
            detrend_buffer=float(detrend_buffer),
            detrend_sigma_clip=float(detrend_sigma_clip),
            cache_only_sectors=cache_only_sector_load,
        )
    except BtvCliError:
        raise
    except LightCurveNotFoundError as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_DATA_UNAVAILABLE) from exc
    except Exception as exc:
        mapped = EXIT_REMOTE_TIMEOUT if _looks_like_timeout(exc) else EXIT_RUNTIME_ERROR
        raise BtvCliError(str(exc), exit_code=mapped) from exc

    runtime_artifacts: dict[str, Any] = {
        "target_cached": False,
        "trilegal_cached": False,
        "trilegal_csv_path": None,
        "staged_with_network": bool(network_ok),
        "timeout_seconds_requested": float(timeout_seconds) if timeout_seconds is not None else None,
    }
    if network_ok:
        click.echo("[fpp-prepare] Staging TRICERATOPS runtime artifacts...")
        try:
            stage_result = stage_triceratops_runtime_artifacts(
                cache=cache,
                tic_id=int(resolved_tic_id),
                sectors=[int(s) for s in sectors_loaded],
                timeout_seconds=float(timeout_seconds) if timeout_seconds is not None else None,
            )
        except Exception as exc:
            mapped = EXIT_REMOTE_TIMEOUT if _looks_like_timeout(exc) else EXIT_RUNTIME_ERROR
            sectors_key = "-".join(str(int(s)) for s in sorted({int(s) for s in sectors_loaded}))
            stage_state_path = (
                Path(cache.cache_dir) / "triceratops" / "staging_state" / f"tic_{int(resolved_tic_id)}__sectors_{sectors_key}.json"
            )
            hint = "TRILEGAL_EMPTY_RESPONSE" if "TRILEGAL_EMPTY_RESPONSE" in str(exc) else type(exc).__name__
            raise BtvCliError(
                (
                    f"Failed to stage TRICERATOPS runtime artifacts: {exc} "
                    f"[code={hint} stage_state={stage_state_path}]"
                ),
                exit_code=mapped,
            ) from exc
        runtime_artifacts.update(
            {
                "target_cached": True,
                "trilegal_cached": bool(stage_result.get("trilegal_csv_path")),
                "trilegal_csv_path": stage_result.get("trilegal_csv_path"),
                "target_cache_hit": bool(stage_result.get("target_cache_hit", False)),
                "trilegal_cache_hit": bool(stage_result.get("trilegal_cache_hit", False)),
                "runtime_seconds": stage_result.get("runtime_seconds"),
                "stage_state_path": stage_result.get("stage_state_path"),
            }
        )
    else:
        ready, details = _runtime_artifacts_ready(
            cache_dir=Path(cache.cache_dir),
            tic_id=int(resolved_tic_id),
            sectors_loaded=[int(s) for s in sectors_loaded],
        )
        runtime_artifacts.update(details)
        runtime_artifacts["ready_without_network"] = bool(ready)

    manifest: dict[str, Any] = {
        "schema_version": _FPP_PREPARE_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "tic_id": int(resolved_tic_id),
        "period_days": float(resolved_period_days),
        "t0_btjd": float(resolved_t0_btjd),
        "duration_hours": float(resolved_duration_hours),
        "depth_ppm_used": float(depth_ppm_used),
        "sectors_loaded": [int(s) for s in sectors_loaded],
        "cache_dir": str(cache.cache_dir),
        "detrend": {
            "method": detrend_method,
            "cache_applied": bool(detrend_cache_effective),
            "cache_requested": bool(detrend_cache_requested),
            "bin_hours": float(detrend_bin_hours),
            "buffer_factor": float(detrend_buffer),
            "sigma_clip": float(detrend_sigma_clip),
            "depth_source": depth_source,
            "depth_meta": detrended_depth_meta,
        },
        "runtime_artifacts": runtime_artifacts,
        "inputs": {
            "depth_ppm_catalog": resolved_depth_ppm,
            "resolved_source": input_resolution.get("source"),
            "resolved_from": input_resolution.get("resolved_from"),
            "requested_sectors": effective_sectors,
        },
    }
    if compute_guidance:
        manifest["compute_insights"] = build_prepare_compute_guidance(
            sectors_loaded=[int(s) for s in sectors_loaded],
            runtime_artifacts=runtime_artifacts,
        )
    click.echo(f"[fpp-prepare] Writing manifest: {output_manifest_path}")
    dump_json_output(manifest, output_manifest_path)


def _run_fpp_from_prepare_manifest(
    prepare_manifest: Path,
    require_prepared: bool,
    preset: str,
    replicates: int | None,
    apply_compute_guidance: bool,
    preset_was_explicit: bool,
    replicates_was_explicit: bool,
    timeout_was_explicit: bool,
    seed: int | None,
    overrides: tuple[str, ...],
    timeout_seconds: float | None,
    replicate_detail: str,
    replicate_errors_limit: int,
    contrast_curve: Path | None,
    contrast_curve_filter: str | None,
    stellar_radius: float | None,
    stellar_mass: float | None,
    stellar_tmag: float | None,
    stellar_file: str | None,
    use_stellar_auto: bool,
    require_stellar: bool,
    network_ok: bool,
    output_path_arg: str,
    command_label: str = "fpp-run",
) -> None:
    out_path = resolve_optional_output_path(output_path_arg)
    click.echo(f"[{command_label}] Loading prepare manifest: {prepare_manifest}")
    prepared = _load_prepare_manifest(prepare_manifest)
    cache = PersistentCache(cache_dir=prepared["cache_dir"])
    if require_prepared:
        click.echo(f"[{command_label}] Verifying prepared cache artifacts...")
        missing_sectors = _cache_missing_sectors(
            cache=cache,
            tic_id=int(prepared["tic_id"]),
            sectors_loaded=[int(s) for s in prepared["sectors_loaded"]],
        )
        if missing_sectors:
            raise BtvCliError(
                (
                    "Missing prepared cache artifacts for sectors: "
                    + ", ".join(str(s) for s in sorted(set(missing_sectors)))
                ),
                exit_code=EXIT_DATA_UNAVAILABLE,
            )
        runtime_ready, runtime_details = _runtime_artifacts_ready(
            cache_dir=Path(prepared["cache_dir"]),
            tic_id=int(prepared["tic_id"]),
            sectors_loaded=[int(s) for s in prepared["sectors_loaded"]],
            stage_state_path=(
                Path(str(prepared["runtime_artifacts"].get("stage_state_path")))
                if prepared["runtime_artifacts"].get("stage_state_path")
                else None
            ),
        )
        if not runtime_ready:
            raise BtvCliError(
                (
                    "Prepared runtime artifacts missing "
                    f"(target_cached={runtime_details['target_cached']} "
                    f"trilegal_cached={runtime_details['trilegal_cached']} "
                    f"stage_state_ok={runtime_details.get('stage_state_ok')}). "
                    "Run `btv fpp-prepare --network-ok` first."
                ),
                exit_code=EXIT_DATA_UNAVAILABLE,
            )

    if replicates is not None and replicates < 1:
        raise BtvCliError("--replicates must be >= 1", exit_code=EXIT_INPUT_ERROR)
    if replicate_errors_limit < 0:
        raise BtvCliError("--replicate-errors-limit must be >= 0", exit_code=EXIT_INPUT_ERROR)
    if use_stellar_auto and not network_ok:
        raise BtvCliError("--use-stellar-auto requires --network-ok", exit_code=EXIT_DATA_UNAVAILABLE)
    if timeout_seconds is not None and float(timeout_seconds) <= 0.0:
        raise BtvCliError("--timeout-seconds must be > 0", exit_code=EXIT_INPUT_ERROR)

    (
        preset_name,
        effective_replicates,
        effective_timeout_requested,
        compute_guidance_available,
        compute_guidance_applied,
        compute_guidance_source,
    ) = apply_compute_guidance_to_runtime_knobs(
        prepared=prepared,
        apply_compute_guidance=bool(apply_compute_guidance),
        preset=preset,
        replicates=replicates,
        timeout_seconds=timeout_seconds,
        preset_was_explicit=bool(preset_was_explicit),
        replicates_was_explicit=bool(replicates_was_explicit),
        timeout_was_explicit=bool(timeout_was_explicit),
    )
    parsed_overrides = parse_extra_params(overrides)
    _resolve_drop_scenario_override(parsed_overrides=parsed_overrides)
    effective_timeout_seconds = (
        float(effective_timeout_requested)
        if effective_timeout_requested is not None
        else (_STANDARD_PRESET_TIMEOUT_SECONDS if preset_name == "standard" else None)
    )
    if effective_timeout_requested is None and preset_name == "standard":
        click.echo(
            "Using default timeout_seconds=900 for --preset standard.",
            err=True,
        )

    parsed_contrast_curve, contrast_curve_parse_provenance = _load_cli_contrast_curve(
        contrast_curve=contrast_curve,
        contrast_curve_filter=contrast_curve_filter,
    )

    try:
        resolved_stellar, stellar_resolution = resolve_stellar_inputs(
            tic_id=int(prepared["tic_id"]),
            stellar_radius=stellar_radius,
            stellar_mass=stellar_mass,
            stellar_tmag=stellar_tmag,
            stellar_file=stellar_file,
            use_stellar_auto=use_stellar_auto,
            require_stellar=require_stellar,
            auto_loader=(lambda _tic_id: _load_auto_stellar_inputs(_tic_id)) if use_stellar_auto else None,
        )
    except (TargetNotFoundError, LightCurveNotFoundError) as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_DATA_UNAVAILABLE) from exc
    except Exception as exc:
        if isinstance(exc, BtvCliError):
            raise
        mapped = EXIT_REMOTE_TIMEOUT if _looks_like_timeout(exc) else EXIT_RUNTIME_ERROR
        raise BtvCliError(str(exc), exit_code=mapped) from exc

    click.echo(f"[{command_label}] Running FPP compute using prepared cache...")
    try:
        result, selected_sectors, retry_meta = _execute_fpp_with_retry(
            preset_name=preset_name,
            parsed_overrides=parsed_overrides,
            run_attempt=lambda attempt_overrides: (
                calculate_fpp(
                    cache=cache,
                    tic_id=int(prepared["tic_id"]),
                    period=float(prepared["period_days"]),
                    t0=float(prepared["t0_btjd"]),
                    depth_ppm=float(prepared["depth_ppm_used"]),
                    duration_hours=float(prepared["duration_hours"]),
                    sectors=[int(s) for s in prepared["sectors_loaded"]],
                    stellar_radius=resolved_stellar.get("radius"),
                    stellar_mass=resolved_stellar.get("mass"),
                    tmag=resolved_stellar.get("tmag"),
                    timeout_seconds=effective_timeout_seconds,
                    preset=preset_name,
                    replicates=effective_replicates,
                    seed=seed,
                    contrast_curve=parsed_contrast_curve,
                    overrides=attempt_overrides,
                    allow_network=bool(network_ok),
                    progress_hook=lambda payload: _emit_fpp_replicate_progress("fpp-run", payload),
                ),
                [int(s) for s in prepared["sectors_loaded"]],
            ),
        )
    except BtvCliError:
        raise
    except Exception as exc:
        mapped = EXIT_REMOTE_TIMEOUT if _looks_like_timeout(exc) else EXIT_RUNTIME_ERROR
        raise BtvCliError(str(exc), exit_code=mapped) from exc

    result = _apply_replicate_output_controls(
        result,
        replicate_detail=replicate_detail,
        replicate_errors_limit=int(replicate_errors_limit),
    )
    verdict, verdict_source = _derive_fpp_verdict(result)
    payload: dict[str, Any] = {
        "schema_version": "cli.fpp.v3",
        "fpp_result": result,
        "verdict": verdict,
        "verdict_source": verdict_source,
        "result": {
            "fpp_result": result,
            "verdict": verdict,
            "verdict_source": verdict_source,
        },
        "provenance": {
            "depth_source": prepared["detrend"].get("depth_source", "prepared"),
            "depth_ppm_used": float(prepared["depth_ppm_used"]),
            "prepare_manifest": {
                "schema_version": prepared["schema_version"],
                "created_at": prepared["created_at"],
                "path": str(prepare_manifest),
            },
            "inputs": {
                "tic_id": int(prepared["tic_id"]),
                "period_days": float(prepared["period_days"]),
                "t0_btjd": float(prepared["t0_btjd"]),
                "duration_hours": float(prepared["duration_hours"]),
                "depth_ppm": float(prepared["depth_ppm_used"]),
                "depth_ppm_catalog": None,
                "sectors": [int(s) for s in prepared["sectors_loaded"]],
                "sectors_loaded": [int(s) for s in selected_sectors],
            },
            "resolved_source": "prepare_manifest",
            "resolved_from": "prepare_manifest",
            "stellar": stellar_resolution,
            "detrended_depth": prepared["detrend"].get("depth_meta"),
            "contrast_curve": {
                "path": str(contrast_curve) if contrast_curve is not None else None,
                "filter": str(parsed_contrast_curve.filter) if parsed_contrast_curve is not None else None,
                "parse_provenance": contrast_curve_parse_provenance,
            },
            "runtime": {
                "preset": preset_name,
                "replicates": effective_replicates,
                "overrides": parsed_overrides,
                "detrend_cache": bool(prepared["detrend"].get("cache_applied", False)),
                "detrend_cache_requested": bool(prepared["detrend"].get("cache_requested", False)),
                "seed_requested": seed,
                "seed_effective": result.get("base_seed", seed),
                "timeout_seconds_requested": timeout_seconds,
                "timeout_seconds": effective_timeout_seconds,
                "compute_guidance_available": bool(compute_guidance_available),
                "compute_guidance_applied": bool(compute_guidance_applied),
                "compute_guidance_source": compute_guidance_source,
                "requested_runtime_knobs": {
                    "preset": str(preset).lower(),
                    "replicates": replicates,
                    "timeout_seconds": timeout_seconds,
                    "overrides": parse_extra_params(overrides),
                },
                "effective_runtime_knobs": {
                    "preset": preset_name,
                    "replicates": effective_replicates,
                    "timeout_seconds": effective_timeout_seconds,
                    "overrides": parsed_overrides,
                },
                "replicate_detail": str(replicate_detail).lower(),
                "replicate_errors_limit": int(replicate_errors_limit),
                "network_ok": bool(network_ok),
                "require_prepared": bool(require_prepared),
                "degenerate_guard": {
                    "guard_triggered": bool(retry_meta["attempts"] and retry_meta["attempts"][0]["degenerate"]),
                    "explicit_max_points_override": bool(retry_meta["explicit_max_points_override"]),
                    "initial_max_points": retry_meta["initial_max_points"],
                    "retry_schedule_max_points": retry_meta["retry_schedule_max_points"],
                    "attempts": retry_meta["attempts"],
                    "final_selected_attempt": int(retry_meta["final_selected_attempt"]),
                    "fallback_succeeded": bool(retry_meta["fallback_succeeded"]),
                },
            },
        },
    }
    retry_guidance = _build_retry_guidance(result=result, preset_name=preset_name)
    if retry_guidance is not None:
        payload["provenance"]["retry_guidance"] = retry_guidance
    if result.get("error") is not None:
        click.echo(
            (
                f"[{command_label}] Warning: FPP returned error payload "
                f"(stage={result.get('stage')}, error_type={result.get('error_type')})."
            ),
            err=True,
        )
    click.echo(f"[{command_label}] Writing FPP output...")
    dump_json_output(payload, out_path)


@click.command("fpp-run")
@click.option(
    "--prepare-manifest",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Manifest JSON produced by btv fpp-prepare.",
)
@click.option(
    "--require-prepared/--allow-missing-prepared",
    default=True,
    show_default=True,
    help="Fail fast when staged cache artifacts referenced by the manifest are missing.",
)
@click.option(
    "--preset",
    type=click.Choice(["fast", "standard", "tutorial"], case_sensitive=False),
    default="fast",
    show_default=True,
    help="TRICERATOPS runtime preset (standard typically expects a longer timeout budget).",
)
@click.option("--replicates", type=int, default=None, help="Replicate count for FPP aggregation.")
@click.option(
    "--apply-compute-guidance/--no-apply-compute-guidance",
    default=False,
    show_default=True,
    help="Apply optional prepare-manifest compute guidance only to unset runtime knobs.",
)
@click.option("--seed", type=int, default=None, help="Base RNG seed.")
@click.option("--override", "overrides", multiple=True, help="Repeat KEY=VALUE TRICERATOPS override entries.")
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
    help="ExoFOP-style .tbl/.dat contrast-curve file for TRICERATOPS companion constraints.",
)
@click.option(
    "--contrast-curve-filter",
    type=str,
    default=None,
    help="Optional band label override for --contrast-curve (for example Kcont, Ks, r).",
)
@click.option(
    "--replicate-detail",
    type=click.Choice(["full", "compact"], case_sensitive=False),
    default="full",
    show_default=True,
    help="Replicate payload detail level in output JSON.",
)
@click.option(
    "--replicate-errors-limit",
    type=int,
    default=0,
    show_default=True,
    help="Maximum replicate error records retained in output (0 keeps all).",
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
    "--network-ok/--no-network",
    default=False,
    show_default=True,
    help="Allow network-dependent stellar auto resolution when requested.",
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
def fpp_run_command(
    prepare_manifest: Path,
    require_prepared: bool,
    preset: str,
    replicates: int | None,
    apply_compute_guidance: bool,
    seed: int | None,
    overrides: tuple[str, ...],
    timeout_seconds: float | None,
    contrast_curve: Path | None,
    contrast_curve_filter: str | None,
    replicate_detail: str,
    replicate_errors_limit: int,
    stellar_radius: float | None,
    stellar_mass: float | None,
    stellar_tmag: float | None,
    stellar_file: str | None,
    use_stellar_auto: bool,
    require_stellar: bool,
    network_ok: bool,
    output_path_arg: str,
) -> None:
    """Run FPP compute from a prepared staging manifest."""
    ctx = click.get_current_context()
    _run_fpp_from_prepare_manifest(
        prepare_manifest=prepare_manifest,
        require_prepared=require_prepared,
        preset=preset,
        replicates=replicates,
        apply_compute_guidance=bool(apply_compute_guidance),
        preset_was_explicit=ctx.get_parameter_source("preset") is not ParameterSource.DEFAULT,
        replicates_was_explicit=ctx.get_parameter_source("replicates") is not ParameterSource.DEFAULT,
        timeout_was_explicit=ctx.get_parameter_source("timeout_seconds") is not ParameterSource.DEFAULT,
        seed=seed,
        overrides=overrides,
        timeout_seconds=timeout_seconds,
        replicate_detail=str(replicate_detail).lower(),
        replicate_errors_limit=int(replicate_errors_limit),
        contrast_curve=contrast_curve,
        contrast_curve_filter=contrast_curve_filter,
        stellar_radius=stellar_radius,
        stellar_mass=stellar_mass,
        stellar_tmag=stellar_tmag,
        stellar_file=stellar_file,
        use_stellar_auto=use_stellar_auto,
        require_stellar=require_stellar,
        network_ok=network_ok,
        output_path_arg=output_path_arg,
        command_label="fpp-run",
    )


@click.command("fpp")
@click.argument("toi_arg", required=False)
@click.option(
    "--prepare-manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Manifest JSON produced by btv fpp-prepare. "
        "When provided, btv fpp runs in prepared-manifest mode (same compute path as btv fpp-run)."
    ),
)
@click.option(
    "--require-prepared/--allow-missing-prepared",
    default=True,
    show_default=True,
    help="Only used with --prepare-manifest. Fail fast when prepared artifacts are missing.",
)
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
@click.option(
    "--apply-compute-guidance/--no-apply-compute-guidance",
    default=False,
    show_default=True,
    help="Only used with --prepare-manifest. Apply optional guidance to unset runtime knobs.",
)
@click.option("--seed", type=int, default=None, help="Base RNG seed.")
@click.option("--override", "overrides", multiple=True, help="Repeat KEY=VALUE TRICERATOPS override entries.")
@click.option(
    "--drop-scenario",
    "drop_scenarios",
    multiple=True,
    type=str,
    help=(
        "Repeatable TRICERATOPS hard-exclusion scenario label. "
        f"Droppable options: {', '.join(droppable_scenario_labels())}. TP is not allowed."
    ),
)
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
    help="ExoFOP-style .tbl/.dat contrast-curve file for TRICERATOPS companion constraints.",
)
@click.option(
    "--contrast-curve-filter",
    type=str,
    default=None,
    help="Optional band label override for --contrast-curve (for example Kcont, Ks, r).",
)
@click.option(
    "--replicate-detail",
    type=click.Choice(["full", "compact"], case_sensitive=False),
    default="full",
    show_default=True,
    help="Replicate payload detail level in output JSON.",
)
@click.option(
    "--replicate-errors-limit",
    type=int,
    default=0,
    show_default=True,
    help="Maximum replicate error records retained in output (0 keeps all).",
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
    prepare_manifest: Path | None,
    require_prepared: bool,
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
    apply_compute_guidance: bool,
    seed: int | None,
    overrides: tuple[str, ...],
    drop_scenarios: tuple[str, ...],
    sectors: tuple[int, ...],
    cache_only_sectors: bool,
    timeout_seconds: float | None,
    contrast_curve: Path | None,
    contrast_curve_filter: str | None,
    replicate_detail: str,
    replicate_errors_limit: int,
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
    ctx = click.get_current_context()
    out_path = resolve_optional_output_path(output_path_arg)
    if prepare_manifest is not None:
        conflicting_fields: list[str] = []
        if toi_arg is not None:
            conflicting_fields.append("TOI_ARG")
        if tic_id is not None:
            conflicting_fields.append("--tic-id")
        if period_days is not None:
            conflicting_fields.append("--period-days")
        if t0_btjd is not None:
            conflicting_fields.append("--t0-btjd")
        if duration_hours is not None:
            conflicting_fields.append("--duration-hours")
        if depth_ppm is not None:
            conflicting_fields.append("--depth-ppm")
        if toi is not None:
            conflicting_fields.append("--toi")
        if report_file is not None:
            conflicting_fields.append("--report-file")
        if detrend is not None:
            conflicting_fields.append("--detrend")
        if bool(detrend_cache):
            conflicting_fields.append("--detrend-cache")
        if bool(sectors):
            conflicting_fields.append("--sectors")
        if bool(cache_only_sectors):
            conflicting_fields.append("--cache-only-sectors")
        if cache_dir is not None:
            conflicting_fields.append("--cache-dir")
        if bool(drop_scenarios):
            conflicting_fields.append("--drop-scenario")

        if conflicting_fields:
            raise BtvCliError(
                (
                    "--prepare-manifest cannot be combined with direct candidate/staging options: "
                    + ", ".join(conflicting_fields)
                    + ". Use `btv fpp-run --prepare-manifest ...` semantics, or remove --prepare-manifest."
                ),
                exit_code=EXIT_INPUT_ERROR,
            )
        # Route through the prepared-manifest execution path.
        return _run_fpp_from_prepare_manifest(
            prepare_manifest=prepare_manifest,
            require_prepared=bool(require_prepared),
            preset=preset,
            replicates=replicates,
            apply_compute_guidance=bool(apply_compute_guidance),
            preset_was_explicit=ctx.get_parameter_source("preset") is not ParameterSource.DEFAULT,
            replicates_was_explicit=ctx.get_parameter_source("replicates") is not ParameterSource.DEFAULT,
            timeout_was_explicit=ctx.get_parameter_source("timeout_seconds") is not ParameterSource.DEFAULT,
            seed=seed,
            overrides=overrides,
            timeout_seconds=timeout_seconds,
            replicate_detail=str(replicate_detail).lower(),
            replicate_errors_limit=int(replicate_errors_limit),
            contrast_curve=contrast_curve,
            contrast_curve_filter=contrast_curve_filter,
            stellar_radius=stellar_radius,
            stellar_mass=stellar_mass,
            stellar_tmag=stellar_tmag,
            stellar_file=stellar_file,
            use_stellar_auto=use_stellar_auto,
            require_stellar=require_stellar,
            network_ok=network_ok,
            output_path_arg=output_path_arg,
            command_label="fpp",
        )

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
        if resolved_toi_arg is not None and network_ok:
            click.echo(
                "--report-file provided with --toi; using --toi for candidate inputs and report only for sectors.",
                err=True,
            )

    use_report_for_candidate_inputs = not (resolved_toi_arg is not None and network_ok)
    report_candidate_inputs = report_candidate if use_report_for_candidate_inputs else {}

    candidate_tic_id = tic_id if tic_id is not None else report_candidate_inputs.get("tic_id")
    candidate_period_days = period_days if period_days is not None else report_candidate_inputs.get("period_days")
    candidate_t0_btjd = t0_btjd if t0_btjd is not None else report_candidate_inputs.get("t0_btjd")
    candidate_duration_hours = (
        duration_hours if duration_hours is not None else report_candidate_inputs.get("duration_hours")
    )
    candidate_depth_ppm = depth_ppm if depth_ppm is not None else report_candidate_inputs.get("depth_ppm")

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
    if resolved_toi_arg is not None and network_ok:
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
    if replicate_errors_limit < 0:
        raise BtvCliError("--replicate-errors-limit must be >= 0", exit_code=EXIT_INPUT_ERROR)
    if use_stellar_auto and not network_ok:
        raise BtvCliError("--use-stellar-auto requires --network-ok", exit_code=EXIT_DATA_UNAVAILABLE)
    if timeout_seconds is not None and float(timeout_seconds) <= 0.0:
        raise BtvCliError("--timeout-seconds must be > 0", exit_code=EXIT_INPUT_ERROR)

    preset_name = str(preset).lower()
    parsed_overrides = parse_extra_params(overrides)
    _resolve_drop_scenario_override(
        parsed_overrides=parsed_overrides,
        explicit_drop_scenarios=drop_scenarios,
    )
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

    parsed_contrast_curve, contrast_curve_parse_provenance = _load_cli_contrast_curve(
        contrast_curve=contrast_curve,
        contrast_curve_filter=contrast_curve_filter,
    )

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
                (
                    (lambda _tic_id: _load_auto_stellar_inputs(_tic_id, toi=resolved_toi_arg))
                    if resolved_toi_arg is not None
                    else (lambda _tic_id: _load_auto_stellar_inputs(_tic_id))
                )
                if use_stellar_auto
                else None
            ),
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
                cache_dir=cache_dir,
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

    try:
        result, sectors_loaded, retry_meta = _execute_fpp_with_retry(
            preset_name=preset_name,
            parsed_overrides=parsed_overrides,
            run_attempt=lambda attempt_overrides: _execute_fpp(
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
                allow_network=bool(network_ok),
                progress_hook=lambda payload: _emit_fpp_replicate_progress("fpp", payload),
            ),
        )
    except BtvCliError:
        raise
    except LightCurveNotFoundError as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_DATA_UNAVAILABLE) from exc
    except Exception as exc:
        mapped = EXIT_REMOTE_TIMEOUT if _looks_like_timeout(exc) else EXIT_RUNTIME_ERROR
        raise BtvCliError(str(exc), exit_code=mapped) from exc

    result = _apply_replicate_output_controls(
        result,
        replicate_detail=str(replicate_detail).lower(),
        replicate_errors_limit=int(replicate_errors_limit),
    )
    verdict, verdict_source = _derive_fpp_verdict(result)
    payload: dict[str, Any] = {
        "schema_version": "cli.fpp.v3",
        "fpp_result": result,
        "verdict": verdict,
        "verdict_source": verdict_source,
        "result": {
            "fpp_result": result,
            "verdict": verdict,
            "verdict_source": verdict_source,
        },
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
                "parse_provenance": contrast_curve_parse_provenance,
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
                "compute_guidance_available": False,
                "compute_guidance_applied": False,
                "compute_guidance_source": None,
                "requested_runtime_knobs": {
                    "preset": str(preset).lower(),
                    "replicates": replicates,
                    "timeout_seconds": timeout_seconds,
                    "overrides": parse_extra_params(overrides),
                },
                "effective_runtime_knobs": {
                    "preset": preset_name,
                    "replicates": replicates,
                    "timeout_seconds": effective_timeout_seconds,
                    "overrides": parsed_overrides,
                },
                "replicate_detail": str(replicate_detail).lower(),
                "replicate_errors_limit": int(replicate_errors_limit),
                "network_ok": bool(network_ok),
                "degenerate_guard": {
                    "guard_triggered": bool(retry_meta["attempts"] and retry_meta["attempts"][0]["degenerate"]),
                    "explicit_max_points_override": bool(retry_meta["explicit_max_points_override"]),
                    "initial_max_points": retry_meta["initial_max_points"],
                    "retry_schedule_max_points": retry_meta["retry_schedule_max_points"],
                    "attempts": retry_meta["attempts"],
                    "final_selected_attempt": int(retry_meta["final_selected_attempt"]),
                    "fallback_succeeded": bool(retry_meta["fallback_succeeded"]),
                },
            },
        },
    }
    retry_guidance = _build_retry_guidance(result=result, preset_name=preset_name)
    if retry_guidance is not None:
        payload["provenance"]["retry_guidance"] = retry_guidance
    if result.get("error") is not None:
        click.echo(
            (
                "Warning: FPP returned error payload "
                f"(stage={result.get('stage')}, error_type={result.get('error_type')})."
            ),
            err=True,
        )
    dump_json_output(payload, out_path)


__all__ = ["fpp_command", "fpp_prepare_command", "fpp_run_command"]
