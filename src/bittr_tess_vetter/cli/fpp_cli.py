"""`btv fpp` command for single-candidate TRICERATOPS FPP estimation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import numpy as np

from bittr_tess_vetter.api.fpp import calculate_fpp
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
    resolve_optional_output_path,
)
from bittr_tess_vetter.cli.stellar_inputs import resolve_stellar_inputs
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


def _looks_like_timeout(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    return "timeout" in type(exc).__name__.lower()


def _build_cache_for_fpp(
    *,
    tic_id: int,
    sectors: list[int] | None,
    cache_dir: Path | None,
) -> tuple[PersistentCache, list[int]]:
    client = MASTClient()
    lightcurves = client.download_all_sectors(tic_id, flux_type="pdcsap", sectors=sectors)
    if not lightcurves:
        raise LightCurveNotFoundError(f"No sectors available for TIC {tic_id}")

    cache = PersistentCache(cache_dir=cache_dir)
    sectors_loaded: list[int] = []
    for lc_data in lightcurves:
        key = make_data_ref(int(tic_id), int(lc_data.sector), "pdcsap")
        cache.put(key, lc_data)
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
) -> tuple[dict[str, Any], list[int]]:
    cache, sectors_loaded = _build_cache_for_fpp(
        tic_id=tic_id,
        sectors=sectors,
        cache_dir=cache_dir,
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
) -> tuple[float | None, dict[str, Any]]:
    client = MASTClient()
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


def _load_auto_stellar_inputs(tic_id: int) -> dict[str, float | None]:
    target = MASTClient().get_target_info(int(tic_id))
    return {
        "radius": target.stellar.radius,
        "mass": target.stellar.mass,
        "tmag": target.stellar.tmag,
    }


@click.command("fpp")
@click.option("--tic-id", type=int, default=None, help="TIC identifier.")
@click.option("--period-days", type=float, default=None, help="Orbital period in days.")
@click.option("--t0-btjd", type=float, default=None, help="Reference epoch in BTJD.")
@click.option("--duration-hours", type=float, default=None, help="Transit duration in hours.")
@click.option("--depth-ppm", type=float, default=None, help="Transit depth in ppm.")
@click.option("--toi", type=str, default=None, help="Optional TOI label (overrides resolved value).")
@click.option(
    "--detrend",
    type=str,
    default=None,
    show_default=True,
    help="Pre-FPP detrend method used for depth estimation when --depth-ppm is missing.",
)
@click.option("--detrend-bin-hours", type=float, default=6.0, show_default=True)
@click.option("--detrend-buffer", type=float, default=2.0, show_default=True)
@click.option("--detrend-sigma-clip", type=float, default=5.0, show_default=True)
@click.option(
    "--preset",
    type=click.Choice(["fast", "standard"], case_sensitive=False),
    default="fast",
    show_default=True,
    help="TRICERATOPS runtime preset.",
)
@click.option("--replicates", type=int, default=None, help="Replicate count for FPP aggregation.")
@click.option("--seed", type=int, default=None, help="Base RNG seed.")
@click.option("--sectors", multiple=True, type=int, help="Optional sector filters.")
@click.option("--timeout-seconds", type=float, default=None, help="Optional timeout budget.")
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
    "--out",
    "output_path_arg",
    type=str,
    default="-",
    show_default=True,
    help="JSON output path; '-' writes to stdout.",
)
def fpp_command(
    tic_id: int | None,
    period_days: float | None,
    t0_btjd: float | None,
    duration_hours: float | None,
    depth_ppm: float | None,
    toi: str | None,
    detrend: str | None,
    detrend_bin_hours: float,
    detrend_buffer: float,
    detrend_sigma_clip: float,
    preset: str,
    replicates: int | None,
    seed: int | None,
    sectors: tuple[int, ...],
    timeout_seconds: float | None,
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

    (
        resolved_tic_id,
        resolved_period_days,
        resolved_t0_btjd,
        resolved_duration_hours,
        resolved_depth_ppm,
        input_resolution,
    ) = _resolve_candidate_inputs(
        network_ok=network_ok,
        toi=toi,
        tic_id=tic_id,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        depth_ppm=depth_ppm,
    )

    detrend_method = _normalize_detrend_method(detrend)
    if detrend_method is not None:
        _validate_detrend_args(
            detrend_bin_hours=float(detrend_bin_hours),
            detrend_buffer=float(detrend_buffer),
            detrend_sigma_clip=float(detrend_sigma_clip),
        )
    if replicates is not None and replicates < 1:
        raise BtvCliError("--replicates must be >= 1", exit_code=EXIT_INPUT_ERROR)
    if use_stellar_auto and not network_ok:
        raise BtvCliError("--use-stellar-auto requires --network-ok", exit_code=EXIT_DATA_UNAVAILABLE)

    try:
        resolved_stellar, stellar_resolution = resolve_stellar_inputs(
            tic_id=resolved_tic_id,
            stellar_radius=stellar_radius,
            stellar_mass=stellar_mass,
            stellar_tmag=stellar_tmag,
            stellar_file=stellar_file,
            use_stellar_auto=use_stellar_auto,
            require_stellar=require_stellar,
            auto_loader=_load_auto_stellar_inputs if use_stellar_auto else None,
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
                sectors=list(sectors) if sectors else None,
                detrend_method=detrend_method,
                detrend_bin_hours=float(detrend_bin_hours),
                detrend_buffer=float(detrend_buffer),
                detrend_sigma_clip=float(detrend_sigma_clip),
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

    try:
        result, sectors_loaded = _execute_fpp(
            tic_id=resolved_tic_id,
            period_days=resolved_period_days,
            t0_btjd=resolved_t0_btjd,
            duration_hours=resolved_duration_hours,
            depth_ppm=float(depth_ppm_used),
            sectors=list(sectors) if sectors else None,
            preset=str(preset).lower(),
            replicates=replicates,
            seed=seed,
            timeout_seconds=timeout_seconds,
            cache_dir=cache_dir,
            stellar_radius=resolved_stellar.get("radius"),
            stellar_mass=resolved_stellar.get("mass"),
            stellar_tmag=resolved_stellar.get("tmag"),
        )
    except BtvCliError:
        raise
    except LightCurveNotFoundError as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_DATA_UNAVAILABLE) from exc
    except Exception as exc:
        mapped = EXIT_REMOTE_TIMEOUT if _looks_like_timeout(exc) else EXIT_RUNTIME_ERROR
        raise BtvCliError(str(exc), exit_code=mapped) from exc

    payload: dict[str, Any] = {
        "schema_version": "cli.fpp.v2",
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
                "sectors": list(sectors) if sectors else None,
                "sectors_loaded": sectors_loaded,
            },
            "resolved_source": input_resolution.get("source"),
            "resolved_from": input_resolution.get("resolved_from"),
            "stellar": stellar_resolution,
            "detrended_depth": detrended_depth_meta,
            "runtime": {
                "preset": str(preset).lower(),
                "replicates": replicates,
                "seed_requested": seed,
                "seed_effective": result.get("base_seed", seed),
                "timeout_seconds": timeout_seconds,
                "network_ok": bool(network_ok),
            },
        },
    }
    dump_json_output(payload, out_path)


__all__ = ["fpp_command"]
