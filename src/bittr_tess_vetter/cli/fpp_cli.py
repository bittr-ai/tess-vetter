"""`btv fpp` command for single-candidate TRICERATOPS FPP estimation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from bittr_tess_vetter.api.fpp import calculate_fpp
from bittr_tess_vetter.cli.common_cli import (
    EXIT_DATA_UNAVAILABLE,
    EXIT_INPUT_ERROR,
    EXIT_REMOTE_TIMEOUT,
    EXIT_RUNTIME_ERROR,
    BtvCliError,
    dump_json_output,
    resolve_optional_output_path,
)
from bittr_tess_vetter.cli.vet_cli import _resolve_candidate_inputs
from bittr_tess_vetter.domain.lightcurve import make_data_ref
from bittr_tess_vetter.platform.io import LightCurveNotFoundError, MASTClient, PersistentCache


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
        timeout_seconds=timeout_seconds,
        preset=preset,
        replicates=replicates,
        seed=seed,
    )
    return result, sectors_loaded


@click.command("fpp")
@click.option("--tic-id", type=int, default=None, help="TIC identifier.")
@click.option("--period-days", type=float, default=None, help="Orbital period in days.")
@click.option("--t0-btjd", type=float, default=None, help="Reference epoch in BTJD.")
@click.option("--duration-hours", type=float, default=None, help="Transit duration in hours.")
@click.option("--depth-ppm", type=float, default=None, help="Transit depth in ppm.")
@click.option("--toi", type=str, default=None, help="Optional TOI label (overrides resolved value).")
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
    preset: str,
    replicates: int | None,
    seed: int | None,
    sectors: tuple[int, ...],
    timeout_seconds: float | None,
    network_ok: bool,
    cache_dir: Path | None,
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

    if resolved_depth_ppm is None:
        exit_code = EXIT_DATA_UNAVAILABLE if toi is not None else EXIT_INPUT_ERROR
        raise BtvCliError(
            "Missing transit depth. Provide --depth-ppm or --toi with depth metadata.",
            exit_code=exit_code,
        )
    if replicates is not None and replicates < 1:
        raise BtvCliError("--replicates must be >= 1", exit_code=EXIT_INPUT_ERROR)

    try:
        result, sectors_loaded = _execute_fpp(
            tic_id=resolved_tic_id,
            period_days=resolved_period_days,
            t0_btjd=resolved_t0_btjd,
            duration_hours=resolved_duration_hours,
            depth_ppm=resolved_depth_ppm,
            sectors=list(sectors) if sectors else None,
            preset=str(preset).lower(),
            replicates=replicates,
            seed=seed,
            timeout_seconds=timeout_seconds,
            cache_dir=cache_dir,
        )
    except BtvCliError:
        raise
    except LightCurveNotFoundError as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_DATA_UNAVAILABLE) from exc
    except Exception as exc:
        mapped = EXIT_REMOTE_TIMEOUT if _looks_like_timeout(exc) else EXIT_RUNTIME_ERROR
        raise BtvCliError(str(exc), exit_code=mapped) from exc

    payload: dict[str, Any] = {
        "schema_version": "cli.fpp.v1",
        "fpp_result": result,
        "provenance": {
            "inputs": {
                "tic_id": resolved_tic_id,
                "period_days": resolved_period_days,
                "t0_btjd": resolved_t0_btjd,
                "duration_hours": resolved_duration_hours,
                "depth_ppm": resolved_depth_ppm,
                "sectors": list(sectors) if sectors else None,
                "sectors_loaded": sectors_loaded,
            },
            "resolved_source": input_resolution.get("source"),
            "resolved_from": input_resolution.get("resolved_from"),
            "runtime": {
                "preset": str(preset).lower(),
                "replicates": replicates,
                "seed": result.get("base_seed", seed),
                "seed_requested": seed,
                "timeout_seconds": timeout_seconds,
                "network_ok": bool(network_ok),
            },
        },
    }
    dump_json_output(payload, out_path)


__all__ = ["fpp_command"]
