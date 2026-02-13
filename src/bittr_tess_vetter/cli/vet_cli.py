"""`btv vet` command for single-candidate vetting."""

from __future__ import annotations

import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import click

from bittr_tess_vetter.api.pipeline import PipelineConfig
from bittr_tess_vetter.api.stitch import stitch_lightcurve_data
from bittr_tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from bittr_tess_vetter.api.vet import vet_candidate
from bittr_tess_vetter.cli.common_cli import (
    EXIT_INPUT_ERROR,
    EXIT_LIGHTCURVE_NOT_FOUND,
    EXIT_PROGRESS_ERROR,
    EXIT_REMOTE_TIMEOUT,
    EXIT_RUNTIME_ERROR,
    BtvCliError,
    dump_json_output,
    parse_extra_params,
    resolve_optional_output_path,
)
from bittr_tess_vetter.cli.progress_metadata import (
    ProgressIOError,
    build_single_candidate_progress,
    decide_resume_for_single_candidate,
    read_progress_metadata,
    write_progress_metadata_atomic,
)
from bittr_tess_vetter.platform.io.mast_client import LightCurveNotFoundError, MASTClient


def _looks_like_timeout(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    return "timeout" in type(exc).__name__.lower()


def _progress_path_from_args(
    out_path: Path | None,
    progress_path_arg: str | None,
    resume: bool,
) -> Path | None:
    if progress_path_arg:
        return Path(progress_path_arg)
    if resume:
        if out_path is None:
            raise BtvCliError(
                "--resume requires --out to be a file path or explicit --progress-path",
                exit_code=EXIT_INPUT_ERROR,
            )
        return out_path.with_suffix(out_path.suffix + ".progress.json")
    return None


def _execute_vet(
    *,
    tic_id: int,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    depth_ppm: float | None,
    preset: str,
    checks: list[str] | None,
    network_ok: bool,
    sectors: list[int] | None,
    flux_type: str,
    pipeline_config: PipelineConfig,
) -> dict[str, Any]:
    client = MASTClient()
    lightcurves = client.download_all_sectors(tic_id, flux_type=flux_type, sectors=sectors)
    if not lightcurves:
        raise LightCurveNotFoundError(f"No sectors available for TIC {tic_id}")

    if len(lightcurves) == 1:
        stitched_lc = lightcurves[0]
    else:
        stitched_lc, _ = stitch_lightcurve_data(lightcurves, tic_id=tic_id)

    lc = LightCurve.from_internal(stitched_lc)
    candidate = Candidate(
        ephemeris=Ephemeris(
            period_days=period_days,
            t0_btjd=t0_btjd,
            duration_hours=duration_hours,
        ),
        depth_ppm=depth_ppm,
    )

    bundle = vet_candidate(
        lc=lc,
        candidate=candidate,
        network=network_ok,
        tic_id=tic_id,
        preset=preset,
        checks=checks,
        pipeline_config=pipeline_config,
    )

    return bundle.model_dump(mode="json")


@click.command("vet")
@click.option("--tic-id", type=int, required=True, help="TIC identifier.")
@click.option("--period-days", type=float, required=True, help="Orbital period in days.")
@click.option("--t0-btjd", type=float, required=True, help="Reference epoch in BTJD.")
@click.option("--duration-hours", type=float, required=True, help="Transit duration in hours.")
@click.option("--depth-ppm", type=float, default=None, help="Transit depth in ppm.")
@click.option(
    "--preset",
    type=click.Choice(["default", "extended"], case_sensitive=False),
    default="default",
    show_default=True,
    help="Vetting preset.",
)
@click.option("--check", "checks", multiple=True, help="Repeat for specific check IDs.")
@click.option(
    "--network-ok/--no-network",
    default=False,
    show_default=True,
    help="Allow network-dependent checks.",
)
@click.option("--sectors", multiple=True, type=int, help="Optional sector filters.")
@click.option(
    "--flux-type",
    type=click.Choice(["pdcsap", "sap"], case_sensitive=False),
    default="pdcsap",
    show_default=True,
)
@click.option("--timeout-seconds", type=float, default=None)
@click.option("--random-seed", type=int, default=None)
@click.option("--extra-param", "extra_params", multiple=True, help="Repeat KEY=VALUE entries.")
@click.option("--fail-fast/--no-fail-fast", default=False, show_default=True)
@click.option("--emit-warnings/--no-emit-warnings", default=False, show_default=True)
@click.option(
    "--out",
    "output_path_arg",
    type=str,
    default="-",
    show_default=True,
    help="JSON output path; '-' writes to stdout.",
)
@click.option("--progress-path", type=str, default=None, help="Optional progress metadata path.")
@click.option("--resume", is_flag=True, default=False, help="Skip when completed output already exists.")
def vet_command(
    tic_id: int,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    depth_ppm: float | None,
    preset: str,
    checks: tuple[str, ...],
    network_ok: bool,
    sectors: tuple[int, ...],
    flux_type: str,
    timeout_seconds: float | None,
    random_seed: int | None,
    extra_params: tuple[str, ...],
    fail_fast: bool,
    emit_warnings: bool,
    output_path_arg: str,
    progress_path: str | None,
    resume: bool,
) -> None:
    """Run candidate vetting and emit `VettingBundleResult` JSON."""
    out_path = resolve_optional_output_path(output_path_arg)
    progress_file = _progress_path_from_args(out_path, progress_path, resume)

    candidate_meta = {
        "tic_id": tic_id,
        "period_days": period_days,
        "t0_btjd": t0_btjd,
        "duration_hours": duration_hours,
    }

    if resume:
        existing = None
        if progress_file is not None:
            try:
                existing = read_progress_metadata(progress_file)
            except ProgressIOError as exc:
                raise BtvCliError(str(exc), exit_code=EXIT_PROGRESS_ERROR) from exc

        decision = decide_resume_for_single_candidate(
            command="vet",
            candidate=candidate_meta,
            resume=True,
            output_exists=bool(out_path and out_path.exists()),
            progress=existing,
        )
        if decision["resume"]:
            if progress_file is not None:
                skipped = build_single_candidate_progress(
                    command="vet",
                    output_path=str(out_path or "stdout"),
                    candidate=candidate_meta,
                    resume=True,
                    status="skipped_resume",
                )
                try:
                    write_progress_metadata_atomic(progress_file, skipped)
                except ProgressIOError as exc:
                    raise BtvCliError(str(exc), exit_code=EXIT_PROGRESS_ERROR) from exc
            return

    started = time.monotonic()
    if progress_file is not None:
        running = build_single_candidate_progress(
            command="vet",
            output_path=str(out_path or "stdout"),
            candidate=candidate_meta,
            resume=resume,
            status="running",
        )
        try:
            write_progress_metadata_atomic(progress_file, running)
        except ProgressIOError as exc:
            raise BtvCliError(str(exc), exit_code=EXIT_PROGRESS_ERROR) from exc

    config = PipelineConfig(
        timeout_seconds=timeout_seconds,
        random_seed=random_seed,
        emit_warnings=emit_warnings,
        fail_fast=fail_fast,
        extra_params=parse_extra_params(extra_params),
    )

    try:
        payload = _execute_vet(
            tic_id=tic_id,
            period_days=period_days,
            t0_btjd=t0_btjd,
            duration_hours=duration_hours,
            depth_ppm=depth_ppm,
            preset=str(preset).lower(),
            checks=list(checks) if checks else None,
            network_ok=network_ok,
            sectors=list(sectors) if sectors else None,
            flux_type=str(flux_type).lower(),
            pipeline_config=config,
        )
        dump_json_output(payload, out_path)

        if progress_file is not None:
            completed = build_single_candidate_progress(
                command="vet",
                output_path=str(out_path or "stdout"),
                candidate=candidate_meta,
                resume=resume,
                status="completed",
                wall_time_seconds=time.monotonic() - started,
            )
            write_progress_metadata_atomic(progress_file, completed)
    except BtvCliError:
        raise
    except ProgressIOError as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_PROGRESS_ERROR) from exc
    except LightCurveNotFoundError as exc:
        if progress_file is not None:
            errored = build_single_candidate_progress(
                command="vet",
                output_path=str(out_path or "stdout"),
                candidate=candidate_meta,
                resume=resume,
                status="error",
                wall_time_seconds=time.monotonic() - started,
                error_class=type(exc).__name__,
                error_message=str(exc),
            )
            with suppress(ProgressIOError):
                write_progress_metadata_atomic(progress_file, errored)
        raise BtvCliError(str(exc), exit_code=EXIT_LIGHTCURVE_NOT_FOUND) from exc
    except Exception as exc:
        if progress_file is not None:
            errored = build_single_candidate_progress(
                command="vet",
                output_path=str(out_path or "stdout"),
                candidate=candidate_meta,
                resume=resume,
                status="error",
                wall_time_seconds=time.monotonic() - started,
                error_class=type(exc).__name__,
                error_message=str(exc),
            )
            with suppress(ProgressIOError):
                write_progress_metadata_atomic(progress_file, errored)
        mapped = EXIT_REMOTE_TIMEOUT if _looks_like_timeout(exc) else EXIT_RUNTIME_ERROR
        raise BtvCliError(str(exc), exit_code=mapped) from exc


__all__ = ["vet_command"]
