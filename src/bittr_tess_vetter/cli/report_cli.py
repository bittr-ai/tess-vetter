"""`btv report` command for single-candidate report payload generation."""

from __future__ import annotations

import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import click
from pydantic import ValidationError

from bittr_tess_vetter.api.generate_report import EnrichmentConfig, generate_report
from bittr_tess_vetter.api.pipeline import PipelineConfig
from bittr_tess_vetter.api.report_vet_reuse import (
    build_cli_report_payload,
    coerce_vetting_bundle,
    validate_vet_artifact_candidate_match,
)
from bittr_tess_vetter.cli.common_cli import (
    EXIT_DATA_UNAVAILABLE,
    EXIT_INPUT_ERROR,
    EXIT_PROGRESS_ERROR,
    EXIT_REMOTE_TIMEOUT,
    EXIT_RUNTIME_ERROR,
    BtvCliError,
    dump_json_output,
    emit_progress,
    load_json_file,
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
from bittr_tess_vetter.cli.report_seed import (
    load_report_seed,
    resolve_candidate_inputs_with_report_seed,
)
from bittr_tess_vetter.platform.io.mast_client import LightCurveNotFoundError, MASTClient


def _looks_like_timeout(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    name = type(exc).__name__.lower()
    return "timeout" in name


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


def _load_custom_views(custom_view_file: str | None) -> dict[str, Any] | None:
    if not custom_view_file:
        return None
    return load_json_file(Path(custom_view_file), label="custom-view-file")


def _resolve_report_candidate_inputs(
    *,
    network_ok: bool,
    report_file: str | None,
    toi: str | None,
    tic_id: int | None,
    period_days: float | None,
    t0_btjd: float | None,
    duration_hours: float | None,
    depth_ppm: float | None,
) -> tuple[int, float, float, float, float | None, dict[str, Any]]:
    """Resolve report candidate inputs with backward-compatible TOI label behavior.

    Legacy behavior allowed `--toi` as a display-only label when explicit numeric
    ephemeris inputs were already provided. Preserve that path without requiring
    network lookups.
    """
    report_seed = load_report_seed(report_file) if report_file is not None else None
    return resolve_candidate_inputs_with_report_seed(
        network_ok=network_ok,
        toi=toi,
        tic_id=tic_id,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        depth_ppm=depth_ppm,
        report_seed=report_seed,
    )


def _execute_report(
    *,
    tic_id: int,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    depth_ppm: float | None,
    toi: str | None,
    sectors: list[int] | None,
    cache_dir: Path | None,
    flux_type: str,
    include_html: bool,
    include_enrichment: bool,
    custom_views: dict[str, Any] | None,
    pipeline_config: PipelineConfig,
    vet_result: dict[str, Any] | None,
    vet_result_path: str | None,
    resolved_inputs: dict[str, Any],
    diagnostic_artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    enrichment_cfg = EnrichmentConfig() if include_enrichment else None
    result = generate_report(
        tic_id=tic_id,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        depth_ppm=depth_ppm,
        toi=toi,
        sectors=sectors,
        flux_type=flux_type,
        mast_client=(MASTClient(cache_dir=str(cache_dir)) if cache_dir is not None else None),
        include_html=include_html,
        include_enrichment=include_enrichment,
        enrichment_config=enrichment_cfg,
        custom_views=custom_views,
        pipeline_config=pipeline_config,
        vet_result=vet_result,
    )
    if result.vet_artifact_reuse is not None:
        vet_artifact = result.vet_artifact_reuse.to_json(
            provided=True,
            path=vet_result_path,
        )
    else:
        vet_artifact = {
            "provided": False,
            "path": None,
            "reused": False,
            "missing_fields": [],
            "incremental_compute": [],
        }
    return {
        "report_json": build_cli_report_payload(
            report_json=result.report_json,
            vet_artifact=vet_artifact,
            sectors_used=result.sectors_used,
            resolved_inputs=resolved_inputs,
            diagnostic_artifacts=diagnostic_artifacts,
        ),
        "plot_data_json": result.plot_data_json,
        "html": result.html,
    }


def _load_diagnostic_artifacts(paths: tuple[str, ...]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for path in paths:
        payload = load_json_file(Path(path), label="diagnostic-json file")
        schema_version = payload.get("schema_version")
        result = payload.get("result")
        artifacts.append(
            {
                "path": str(path),
                "schema_version": str(schema_version) if schema_version is not None else None,
                "verdict": payload.get("verdict"),
                "verdict_source": payload.get("verdict_source"),
                "has_result": isinstance(result, dict),
            }
        )
    return artifacts


def _apply_report_canonical_verdict(report_payload: dict[str, Any]) -> None:
    report_block = report_payload.get("report")
    summary_block: dict[str, Any] = {}
    if isinstance(report_block, dict) and isinstance(report_block.get("summary"), dict):
        summary_block = report_block["summary"]

    summary_verdict = summary_block.get("verdict")
    summary_verdict_source = summary_block.get("verdict_source")

    verdict = summary_verdict if summary_verdict is not None else report_payload.get("verdict")
    verdict_source = report_payload.get("verdict_source")
    if summary_verdict is not None:
        verdict_source = (
            summary_verdict_source
            if summary_verdict_source is not None
            else "$.report.summary.verdict"
        )

    report_payload["verdict"] = verdict
    report_payload["verdict_source"] = verdict_source

    result_payload = report_payload.get("result")
    if not isinstance(result_payload, dict):
        result_payload = {}
    result_payload["verdict"] = verdict
    result_payload["verdict_source"] = verdict_source
    report_payload["result"] = result_payload


@click.command("report")
@click.argument("toi_arg", required=False)
@click.option("--tic-id", type=int, default=None, help="TIC identifier.")
@click.option("--period-days", type=float, default=None, help="Orbital period in days.")
@click.option("--t0-btjd", type=float, default=None, help="Reference epoch in BTJD.")
@click.option("--duration-hours", type=float, default=None, help="Transit duration in hours.")
@click.option("--depth-ppm", type=float, default=None, help="Transit depth in ppm.")
@click.option(
    "--toi",
    type=str,
    default=None,
    help="Optional TOI label; with --network-ok resolves missing candidate inputs from ExoFOP.",
)
@click.option(
    "--network-ok/--no-network",
    default=False,
    show_default=True,
    help="Allow network-dependent TOI resolution.",
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Optional cache directory for MAST/lightkurve products.",
)
@click.option("--sectors", multiple=True, type=int, help="Optional sector filters.")
@click.option(
    "--flux-type",
    type=click.Choice(["pdcsap", "sap"], case_sensitive=False),
    default="pdcsap",
    show_default=True,
)
@click.option("--include-html", is_flag=True, default=False, help="Include standalone HTML output.")
@click.option("--include-enrichment", is_flag=True, default=False, help="Include enrichment blocks.")
@click.option("--custom-view-file", type=str, default=None, help="JSON file for custom views contract.")
@click.option("--vet-result", type=str, default=None, help="Optional prior vet artifact JSON for check reuse.")
@click.option("--report-file", type=str, default=None, help="Optional prior report JSON to seed candidate inputs.")
@click.option(
    "--diagnostic-json",
    "diagnostic_json_paths",
    multiple=True,
    type=str,
    help="Optional diagnostic CLI JSON artifact(s) to pass through into report summary.",
)
@click.option("--timeout-seconds", type=float, default=None)
@click.option("--random-seed", type=int, default=None)
@click.option("--extra-param", "extra_params", multiple=True, help="Repeat KEY=VALUE entries.")
@click.option("--fail-fast/--no-fail-fast", default=False, show_default=True)
@click.option("--emit-warnings/--no-emit-warnings", default=False, show_default=True)
@click.option(
    "-o",
    "--out",
    "output_path_arg",
    type=str,
    default="-",
    show_default=True,
    help="Report payload JSON output path; '-' writes to stdout.",
)
@click.option(
    "--plot-data-out",
    type=str,
    default=None,
    help="Optional path for separate plot-data JSON artifact.",
)
@click.option("--html-out", type=str, default=None, help="Optional path to write HTML when --include-html is set.")
@click.option("--progress-path", type=str, default=None, help="Optional progress metadata path.")
@click.option("--resume", is_flag=True, default=False, help="Skip when completed output already exists.")
def report_command(
    toi_arg: str | None,
    tic_id: int | None,
    period_days: float | None,
    t0_btjd: float | None,
    duration_hours: float | None,
    depth_ppm: float | None,
    toi: str | None,
    network_ok: bool,
    cache_dir: Path | None,
    sectors: tuple[int, ...],
    flux_type: str,
    include_html: bool,
    include_enrichment: bool,
    custom_view_file: str | None,
    vet_result: str | None,
    report_file: str | None,
    diagnostic_json_paths: tuple[str, ...],
    timeout_seconds: float | None,
    random_seed: int | None,
    extra_params: tuple[str, ...],
    fail_fast: bool,
    emit_warnings: bool,
    output_path_arg: str,
    plot_data_out: str | None,
    html_out: str | None,
    progress_path: str | None,
    resume: bool,
) -> None:
    """Generate report payload JSON and optional HTML from candidate inputs."""
    if toi_arg is not None and toi is not None and str(toi_arg).strip() != str(toi).strip():
        raise BtvCliError(
            "Positional TOI argument and --toi must match when both are provided.",
            exit_code=EXIT_INPUT_ERROR,
        )
    resolved_toi = toi if toi is not None else toi_arg
    (
        resolved_tic_id,
        resolved_period_days,
        resolved_t0_btjd,
        resolved_duration_hours,
        resolved_depth_ppm,
        _input_resolution,
    ) = _resolve_report_candidate_inputs(
        network_ok=network_ok,
        report_file=report_file,
        toi=resolved_toi,
        tic_id=tic_id,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        depth_ppm=depth_ppm,
    )

    out_path = resolve_optional_output_path(output_path_arg)
    progress_file = _progress_path_from_args(out_path, progress_path, resume)
    if plot_data_out:
        plot_data_path = Path(plot_data_out)
    elif out_path is not None:
        plot_data_path = out_path.with_suffix(out_path.suffix + ".plot_data.json")
    else:
        raise BtvCliError(
            "--out - requires --plot-data-out so plot data is not dropped",
            exit_code=EXIT_INPUT_ERROR,
        )
    html_path = Path(html_out) if html_out else None

    candidate_meta = {
        "tic_id": resolved_tic_id,
        "period_days": resolved_period_days,
        "t0_btjd": resolved_t0_btjd,
        "duration_hours": resolved_duration_hours,
    }

    if include_html and html_path is None and out_path is not None:
        html_path = out_path.with_suffix(".html")

    if resume:
        existing = None
        if progress_file is not None:
            try:
                existing = read_progress_metadata(progress_file)
            except ProgressIOError as exc:
                raise BtvCliError(str(exc), exit_code=EXIT_PROGRESS_ERROR) from exc

        output_exists = bool(out_path and out_path.exists())
        decision = decide_resume_for_single_candidate(
            command="report",
            candidate=candidate_meta,
            resume=True,
            output_exists=output_exists,
            progress=existing,
        )
        if decision["resume"]:
            if progress_file is not None:
                skipped = build_single_candidate_progress(
                    command="report",
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
    emit_progress("report", "start")

    config = PipelineConfig(
        timeout_seconds=timeout_seconds,
        random_seed=random_seed,
        emit_warnings=emit_warnings,
        fail_fast=fail_fast,
        extra_params=parse_extra_params(extra_params),
    )
    vet_result_payload: dict[str, Any] | None = None
    diagnostic_artifacts = _load_diagnostic_artifacts(diagnostic_json_paths)
    if vet_result:
        vet_result_payload = load_json_file(Path(vet_result), label="vet result file")
        try:
            vet_bundle = coerce_vetting_bundle(vet_result_payload)
            validate_vet_artifact_candidate_match(
                vet_bundle=vet_bundle,
                tic_id=resolved_tic_id,
                period_days=resolved_period_days,
                t0_btjd=resolved_t0_btjd,
                duration_hours=resolved_duration_hours,
            )
        except ValidationError as exc:
            raise BtvCliError(
                f"Invalid vet result file schema: {exc}",
                exit_code=EXIT_INPUT_ERROR,
            ) from exc
        except ValueError as exc:
            raise BtvCliError(str(exc), exit_code=EXIT_INPUT_ERROR) from exc

    try:
        if progress_file is not None:
            running = build_single_candidate_progress(
                command="report",
                output_path=str(out_path or "stdout"),
                candidate=candidate_meta,
                resume=resume,
                status="running",
            )
            write_progress_metadata_atomic(progress_file, running)

        output = _execute_report(
            tic_id=resolved_tic_id,
            period_days=resolved_period_days,
            t0_btjd=resolved_t0_btjd,
            duration_hours=resolved_duration_hours,
            depth_ppm=resolved_depth_ppm,
            toi=resolved_toi,
            sectors=list(sectors) if sectors else None,
            cache_dir=cache_dir,
            flux_type=str(flux_type).lower(),
            include_html=include_html,
            include_enrichment=include_enrichment,
            custom_views=_load_custom_views(custom_view_file),
            pipeline_config=config,
            vet_result=vet_result_payload,
            vet_result_path=vet_result,
            resolved_inputs={
                "tic_id": int(resolved_tic_id),
                "period_days": float(resolved_period_days),
                "t0_btjd": float(resolved_t0_btjd),
                "duration_hours": float(resolved_duration_hours),
                "depth_ppm": float(resolved_depth_ppm) if resolved_depth_ppm is not None else None,
            },
            diagnostic_artifacts=diagnostic_artifacts,
        )
        _apply_report_canonical_verdict(output["report_json"])

        dump_json_output(output["report_json"], out_path)
        dump_json_output(output["plot_data_json"], plot_data_path)

        html = output.get("html")
        if include_html and isinstance(html, str) and html_path is not None:
            html_path.parent.mkdir(parents=True, exist_ok=True)
            html_path.write_text(html, encoding="utf-8")

        if progress_file is not None:
            completed = build_single_candidate_progress(
                command="report",
                output_path=str(out_path or "stdout"),
                candidate=candidate_meta,
                resume=resume,
                status="completed",
                wall_time_seconds=time.monotonic() - started,
            )
            write_progress_metadata_atomic(progress_file, completed)
        emit_progress("report", "completed")
    except BtvCliError:
        raise
    except ProgressIOError as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_PROGRESS_ERROR) from exc
    except LightCurveNotFoundError as exc:
        if progress_file is not None:
            errored = build_single_candidate_progress(
                command="report",
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
        raise BtvCliError(str(exc), exit_code=EXIT_DATA_UNAVAILABLE) from exc
    except Exception as exc:
        mapped = EXIT_REMOTE_TIMEOUT if _looks_like_timeout(exc) else EXIT_RUNTIME_ERROR
        if progress_file is not None:
            errored = build_single_candidate_progress(
                command="report",
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
        raise BtvCliError(str(exc), exit_code=mapped) from exc


__all__ = ["report_command"]
