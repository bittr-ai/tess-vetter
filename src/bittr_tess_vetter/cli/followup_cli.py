"""`btv followup` command for follow-up artifact/notes context."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol

import click

from bittr_tess_vetter.cli.common_cli import (
    EXIT_INPUT_ERROR,
    EXIT_RUNTIME_ERROR,
    BtvCliError,
    dump_json_output,
    resolve_optional_output_path,
)
from bittr_tess_vetter.cli.diagnostics_report_inputs import resolve_inputs_from_report_file
from bittr_tess_vetter.cli.vet_cli import _resolve_candidate_inputs


@dataclass(frozen=True)
class FollowupRequest:
    tic_id: int
    toi: str | None
    network_ok: bool
    cache_dir: Path | None
    render_images: bool
    include_raw_spectra: bool
    max_files: int | None
    skip_notes: bool
    notes_file: Path | None


class FollowupExecutor(Protocol):
    def __call__(self, request: FollowupRequest) -> dict[str, Any]:
        ...


def _resolve_tic_and_inputs(
    *,
    tic_id: int | None,
    toi: str | None,
    network_ok: bool,
) -> tuple[int, dict[str, Any]]:
    if toi is not None:
        (
            resolved_tic_id,
            _period_days,
            _t0_btjd,
            _duration_hours,
            _depth_ppm,
            input_resolution,
        ) = _resolve_candidate_inputs(
            network_ok=bool(network_ok),
            toi=toi,
            tic_id=tic_id,
            period_days=None,
            t0_btjd=None,
            duration_hours=None,
            depth_ppm=None,
        )
        return int(resolved_tic_id), input_resolution

    if tic_id is None:
        raise BtvCliError(
            "Missing TIC identifier. Provide --tic-id or --toi.",
            exit_code=EXIT_INPUT_ERROR,
        )
    return int(tic_id), {"source": "cli", "resolved_from": "cli", "inputs": {"tic_id": int(tic_id)}}


def _derive_followup_verdict(*, n_files: int, n_notes: int) -> tuple[str, str]:
    if int(n_files) > 0:
        return "FOLLOWUP_AVAILABLE", "$.summary.n_files"
    if int(n_notes) > 0:
        return "FOLLOWUP_NOTES_ONLY", "$.summary.n_vetting_notes"
    return "FOLLOWUP_MISSING", "$.summary.n_files"


def _read_notes_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    notes = [line.strip() for line in text.splitlines() if line.strip()]
    return notes


def _safe_rel(path: Path, *, root: Path | None) -> str:
    if root is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def _fallback_followup_executor(request: FollowupRequest) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    warnings: list[str] = []

    manifest_path: Path | None = None
    if request.cache_dir is not None:
        manifest_path = request.cache_dir / "exofop" / f"tic_{int(request.tic_id)}" / "manifest.json"
        if manifest_path.exists():
            try:
                import json

                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                raw_files = manifest.get("files") if isinstance(manifest, dict) else None
                if isinstance(raw_files, list):
                    for row in raw_files:
                        if not isinstance(row, dict):
                            continue
                        kind = str(row.get("type") or "Unknown")
                        if (not request.include_raw_spectra) and kind.lower() == "spectrum":
                            continue
                        files.append(
                            {
                                "file_id": row.get("file_id"),
                                "filename": row.get("filename"),
                                "type": kind,
                                "date_utc": row.get("date_utc"),
                                "description": row.get("description"),
                                "path": row.get("path"),
                                "sha256": row.get("sha256"),
                                "bytes": row.get("bytes"),
                            }
                        )
            except Exception as exc:
                warnings.append(f"Failed to read cached manifest: {exc}")

    if request.max_files is not None:
        files = files[: int(request.max_files)]

    vetting_notes: list[str] = []
    notes_source: str | None = None
    if not request.skip_notes and request.notes_file is not None:
        try:
            vetting_notes = _read_notes_file(request.notes_file)
            notes_source = str(request.notes_file)
        except Exception as exc:
            warnings.append(f"Failed to read notes file: {exc}")

    capabilities = {
        "image_rendering": {
            "requested": bool(request.render_images),
            "available": False,
            "used": False,
            "reason": "system_image_tools_not_required",
        },
        "spectra_content": {
            "mode": "raw" if bool(request.include_raw_spectra) else "headers_only",
            "raw_available": False,
            "reason": "system_spectra_tools_not_required",
        },
    }

    if request.render_images:
        warnings.append("Image rendering requested but system rendering tools are not enabled in fallback executor.")

    return {
        "files": files,
        "vetting_notes": vetting_notes,
        "summary": {
            "n_files": int(len(files)),
            "n_vetting_notes": int(len(vetting_notes)),
            "files_source": "cache_manifest" if manifest_path is not None and manifest_path.exists() else "none",
            "notes_source": notes_source,
        },
        "provenance_extra": {
            "cache_manifest": str(manifest_path) if manifest_path is not None else None,
            "warnings": warnings,
            "capabilities": capabilities,
        },
    }


def _resolve_followup_executor() -> FollowupExecutor:
    try:
        module = import_module("bittr_tess_vetter.followup")
    except Exception:
        return _fallback_followup_executor

    candidate = getattr(module, "run_followup", None)
    if callable(candidate):
        return candidate
    return _fallback_followup_executor


@click.command("followup")
@click.argument("toi_arg", required=False)
@click.option("--toi", type=str, default=None, help="Optional TOI label.")
@click.option("--tic-id", type=int, default=None, help="TIC identifier.")
@click.option("--report-file", type=str, default=None, help="Optional report JSON path for candidate inputs.")
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
    help="Optional cache directory for follow-up artifacts.",
)
@click.option("--render-images/--no-render-images", default=False, show_default=True)
@click.option("--include-raw-spectra/--headers-only", default=False, show_default=True)
@click.option("--max-files", type=int, default=None, help="Optional maximum number of files.")
@click.option("--skip-notes", is_flag=True, default=False, help="Skip notes ingestion.")
@click.option(
    "--notes-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional plaintext notes file.",
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
def followup_command(
    toi_arg: str | None,
    toi: str | None,
    tic_id: int | None,
    report_file: str | None,
    network_ok: bool,
    cache_dir: Path | None,
    render_images: bool,
    include_raw_spectra: bool,
    max_files: int | None,
    skip_notes: bool,
    notes_file: Path | None,
    output_path_arg: str,
) -> None:
    """Assemble follow-up files and vetting notes into a stable JSON contract."""
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
    if max_files is not None and int(max_files) < 1:
        raise BtvCliError("--max-files must be >= 1", exit_code=EXIT_INPUT_ERROR)
    if skip_notes and notes_file is not None:
        raise BtvCliError("--skip-notes cannot be combined with --notes-file", exit_code=EXIT_INPUT_ERROR)

    resolved_toi_arg = toi if toi is not None else toi_arg
    report_file_path: str | None = None

    if report_file is not None:
        if resolved_toi_arg is not None:
            click.echo(
                "Warning: --report-file provided; ignoring --toi and using report-file candidate inputs.",
                err=True,
            )
            resolved_toi_arg = None
        resolved_from_report = resolve_inputs_from_report_file(str(report_file))
        resolved_tic_id = int(resolved_from_report.tic_id)
        resolved_toi_arg = resolved_from_report.toi
        input_resolution = dict(resolved_from_report.input_resolution)
        report_file_path = str(resolved_from_report.report_file_path)
    else:
        resolved_tic_id, input_resolution = _resolve_tic_and_inputs(
            tic_id=tic_id,
            toi=resolved_toi_arg,
            network_ok=bool(network_ok),
        )

    request = FollowupRequest(
        tic_id=int(resolved_tic_id),
        toi=resolved_toi_arg,
        network_ok=bool(network_ok),
        cache_dir=cache_dir,
        render_images=bool(render_images),
        include_raw_spectra=bool(include_raw_spectra),
        max_files=max_files,
        skip_notes=bool(skip_notes),
        notes_file=notes_file,
    )

    try:
        executor = _resolve_followup_executor()
        result_payload = executor(request)
    except BtvCliError:
        raise
    except Exception as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_RUNTIME_ERROR) from exc

    files = list(result_payload.get("files") or [])
    vetting_notes = [str(item) for item in list(result_payload.get("vetting_notes") or [])]
    summary = dict(result_payload.get("summary") or {})
    n_files = int(summary.get("n_files") if summary.get("n_files") is not None else len(files))
    n_vetting_notes = int(
        summary.get("n_vetting_notes") if summary.get("n_vetting_notes") is not None else len(vetting_notes)
    )
    summary["n_files"] = n_files
    summary["n_vetting_notes"] = n_vetting_notes

    verdict, verdict_source = _derive_followup_verdict(n_files=n_files, n_notes=n_vetting_notes)
    provenance_extra = dict(result_payload.get("provenance_extra") or {})
    options = {
        "network_ok": bool(network_ok),
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "render_images": bool(render_images),
        "include_raw_spectra": bool(include_raw_spectra),
        "max_files": int(max_files) if max_files is not None else None,
        "skip_notes": bool(skip_notes),
        "notes_file": str(notes_file) if notes_file is not None else None,
    }

    payload = {
        "schema_version": "cli.followup.v1",
        "result": {
            "files": files,
            "vetting_notes": vetting_notes,
            "summary": summary,
            "verdict": verdict,
            "verdict_source": verdict_source,
        },
        "files": files,
        "vetting_notes": vetting_notes,
        "summary": summary,
        "verdict": verdict,
        "verdict_source": verdict_source,
        "inputs_summary": {
            "tic_id": int(resolved_tic_id),
            "toi": resolved_toi_arg,
            "input_resolution": input_resolution,
        },
        "provenance": {
            "inputs_source": "report_file" if report_file_path is not None else str(input_resolution.get("source")),
            "report_file": report_file_path,
            "options": options,
            "cache_dir": str(cache_dir.resolve()) if cache_dir is not None else None,
            "capabilities": provenance_extra.get("capabilities")
            or {
                "image_rendering": {"requested": bool(render_images), "available": False, "used": False},
                "spectra_content": {
                    "mode": "raw" if bool(include_raw_spectra) else "headers_only",
                    "raw_available": False,
                },
            },
            "cache_manifest": (
                _safe_rel(Path(provenance_extra["cache_manifest"]), root=cache_dir)
                if provenance_extra.get("cache_manifest")
                else None
            ),
            "warnings": [str(item) for item in list(provenance_extra.get("warnings") or [])],
        },
    }
    dump_json_output(payload, out_path)


__all__ = ["FollowupRequest", "followup_command"]
