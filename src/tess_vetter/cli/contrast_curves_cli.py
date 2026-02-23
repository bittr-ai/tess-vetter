"""CLI commands for ExoFOP contrast-curve discovery and summary generation."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import click

from tess_vetter.cli.common_cli import (
    EXIT_DATA_UNAVAILABLE,
    EXIT_INPUT_ERROR,
    EXIT_RUNTIME_ERROR,
    BtvCliError,
    dump_json_output,
    resolve_optional_output_path,
)
from tess_vetter.contrast_curves import (
    ContrastCurveParseError,
    build_ruling_summary,
    combine_normalized_curves,
    derive_contrast_verdict,
    normalize_contrast_curve,
    parse_contrast_curve_with_provenance,
)
from tess_vetter.exofop.client import ExoFopClient
from tess_vetter.exofop.types import ExoFopFileRow, ExoFopSelectors

_CLI_CONTRAST_CURVES_SCHEMA = "cli.contrast_curves.v2"
_CLI_CONTRAST_CURVE_SUMMARY_SCHEMA = "cli.contrast_curve_summary.v1"

_LIKELY_FILENAME_RE = re.compile(r"(?i).*\.(tbl|dat|csv|txt)$")
_LIKELY_TEXT_RE = re.compile(r"(?i)(contrast|sensitivity|speckle|ao|high[- ]res|imaging)")


def _resolve_toi_argument(toi_arg: str | None, toi: str | None) -> str | None:
    if toi_arg is not None and toi is not None and str(toi_arg).strip() != str(toi).strip():
        raise BtvCliError(
            "Positional TOI argument and --toi must match when both are provided.",
            exit_code=EXIT_INPUT_ERROR,
        )
    resolved = toi if toi is not None else toi_arg
    if resolved is None:
        return None
    text = str(resolved).strip()
    return text or None


def _resolve_target(
    *,
    client: ExoFopClient,
    tic_id: int | None,
    toi_value: str | None,
    network_ok: bool,
) -> tuple[int, dict[str, Any]]:
    if toi_value is not None:
        if not network_ok:
            raise BtvCliError(
                "--toi requires --network-ok to resolve ExoFOP inputs",
                exit_code=EXIT_DATA_UNAVAILABLE,
            )
        target = client.resolve_target(toi_value)
        resolved_tic_id = int(target.tic_id)
        return resolved_tic_id, {
            "source": "toi_catalog",
            "resolved_from": "exofop_client.resolve_target",
            "inputs": {"toi": toi_value, "tic_id": resolved_tic_id},
        }

    if tic_id is None:
        raise BtvCliError("Missing TIC identifier. Provide --tic-id or --toi.", exit_code=EXIT_INPUT_ERROR)

    resolved_tic_id = int(tic_id)
    return resolved_tic_id, {
        "source": "cli",
        "resolved_from": "cli",
        "inputs": {"tic_id": resolved_tic_id},
    }


def _is_likely_contrast_row(row: ExoFopFileRow) -> bool:
    filename = str(row.filename or "")
    description = str(row.description or "")
    file_type = str(row.type or "")
    if str(file_type).lower() != "image":
        return False
    # Prefer extension-based eligibility over brittle keyword-only matching.
    if _LIKELY_FILENAME_RE.search(filename):
        return True
    return bool(_LIKELY_TEXT_RE.search(description))


def _selector_for_likely_contrast_files(*, max_files: int) -> ExoFopSelectors:
    return ExoFopSelectors(
        types={"Image"},
        # Download by type (Image) and parse downstream, rather than relying on brittle name heuristics.
        filename_regex=None,
        max_files=int(max_files),
    )


def _infer_filter_label(*, filename: str, description: str | None) -> str | None:
    text = " ".join([filename, description or ""]).lower()
    if "562" in text:
        return "562nm"
    if "832" in text:
        return "832nm"
    if "kcont" in text:
        return "Kcont"
    if "ks" in text:
        return "Ks"
    if "kp" in text:
        return "Kp"
    if "i-band" in text or re.search(r"\bi\b", text):
        return "i"
    return None


def _bandpass_family(filter_label: str | None) -> str:
    if filter_label is None:
        return "unknown"
    f = filter_label.lower()
    if "562" in f or f in {"v", "r"}:
        return "optical_blue"
    if "832" in f or f in {"i", "z"}:
        return "optical_red"
    if "k" in f or "h" in f or "j" in f or "nir" in f:
        return "nir"
    return "unknown"


def _compatibility_score(filter_label: str | None) -> float:
    family = _bandpass_family(filter_label)
    if family == "optical_red":
        return 1.0
    if family == "nir":
        return 0.9
    if family == "optical_blue":
        return 0.7
    return 0.5


def _summarize_single_curve(
    *, path: Path, filter_name: str | None, pixel_scale_arcsec_per_px: float | None = None
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    parsed, parse_provenance = parse_contrast_curve_with_provenance(
        path,
        filter_name=filter_name,
        pixel_scale_arcsec_per_px=pixel_scale_arcsec_per_px,
    )
    normalized = normalize_contrast_curve(parsed)
    ruling_summary = build_ruling_summary(normalized)
    return (
        {
            "path": str(path),
            "filter": str(parsed.filter),
            "parse_provenance": parse_provenance,
            "sensitivity": normalized,
        },
        normalized,
        ruling_summary,
    )


@click.command("contrast-curve-summary")
@click.argument("toi_arg", required=False)
@click.option("--toi", type=str, default=None, help="Optional TOI label.")
@click.option("--tic-id", type=int, default=None, help="Optional TIC identifier for metadata.")
@click.option(
    "--file",
    "contrast_curve_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Local contrast-curve file (.tbl/.dat/.csv or generic delimited numeric table).",
)
@click.option("--filter", "filter_name", type=str, default=None, help="Optional filter label override.")
@click.option(
    "--pixel-scale-arcsec-per-px",
    type=float,
    default=None,
    help="Optional pixel scale override for FITS image extraction (arcsec/pixel).",
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
def contrast_curve_summary_command(
    toi_arg: str | None,
    toi: str | None,
    tic_id: int | None,
    contrast_curve_file: Path,
    filter_name: str | None,
    pixel_scale_arcsec_per_px: float | None,
    output_path_arg: str,
) -> None:
    """Parse a local contrast-curve file and emit normalized sensitivity summary."""
    out_path = resolve_optional_output_path(output_path_arg)
    resolved_toi = _resolve_toi_argument(toi_arg, toi)

    try:
        summary_block, normalized, ruling_summary = _summarize_single_curve(
            path=contrast_curve_file,
            filter_name=filter_name,
            pixel_scale_arcsec_per_px=pixel_scale_arcsec_per_px,
        )
    except (FileNotFoundError, ContrastCurveParseError) as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_INPUT_ERROR) from exc
    except Exception as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_RUNTIME_ERROR) from exc

    verdict, verdict_source = derive_contrast_verdict(ruling_summary)
    payload = {
        "schema_version": _CLI_CONTRAST_CURVE_SUMMARY_SCHEMA,
        "contrast_curve_summary": {
            "file": summary_block,
            "ruling_summary": ruling_summary,
        },
        "sensitivity": normalized,
        "ruling_summary": ruling_summary,
        "verdict": verdict,
        "verdict_source": verdict_source,
        "result": {
            "contrast_curve_summary": {
                "file": summary_block,
                "ruling_summary": ruling_summary,
            },
            "sensitivity": normalized,
            "ruling_summary": ruling_summary,
            "verdict": verdict,
            "verdict_source": verdict_source,
        },
        "inputs_summary": {
            "toi": resolved_toi,
            "tic_id": int(tic_id) if tic_id is not None else None,
            "contrast_curve_file": str(contrast_curve_file),
            "filter": filter_name,
            "pixel_scale_arcsec_per_px": pixel_scale_arcsec_per_px,
        },
        "provenance": {
            "parser": "parse_contrast_curve_with_provenance",
            "source": "local_file",
            "parse_provenance": summary_block.get("parse_provenance", {}),
        },
    }
    dump_json_output(payload, out_path)


@click.command("contrast-curves")
@click.argument("toi_arg", required=False)
@click.option("--toi", type=str, default=None, help="Optional TOI label.")
@click.option("--tic-id", type=int, default=None, help="TIC identifier.")
@click.option(
    "--network-ok/--no-network",
    default=False,
    show_default=True,
    help="Allow network-dependent TOI resolution and ExoFOP downloads.",
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path(".cache"),
    show_default=True,
    help="Cache directory for ExoFOP index/files.",
)
@click.option(
    "--cookie-jar",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional Mozilla/Netscape cookie jar for ExoFOP authenticated downloads.",
)
@click.option("--force-refresh", is_flag=True, default=False, help="Refresh ExoFOP file indices and archives.")
@click.option("--max-files", type=int, default=12, show_default=True, help="Max likely files to fetch.")
@click.option(
    "--download-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Optional directory to copy discovered contrast-curve files for downstream use.",
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
def contrast_curves_command(
    toi_arg: str | None,
    toi: str | None,
    tic_id: int | None,
    network_ok: bool,
    cache_dir: Path,
    cookie_jar: Path | None,
    force_refresh: bool,
    max_files: int,
    download_dir: Path | None,
    output_path_arg: str,
) -> None:
    """Discover and summarize likely ExoFOP contrast-curve files for a target."""
    out_path = resolve_optional_output_path(output_path_arg)
    resolved_toi = _resolve_toi_argument(toi_arg, toi)

    if int(max_files) < 1:
        raise BtvCliError("--max-files must be >= 1", exit_code=EXIT_INPUT_ERROR)

    try:
        client = ExoFopClient(cache_dir=cache_dir, cookie_jar_path=cookie_jar)
        resolved_tic_id, input_resolution = _resolve_target(
            client=client,
            tic_id=tic_id,
            toi_value=resolved_toi,
            network_ok=bool(network_ok),
        )
        file_rows = client.file_list(tic_id=int(resolved_tic_id), force_refresh=bool(force_refresh))
        likely_rows = [row for row in file_rows if _is_likely_contrast_row(row)]

        fetch_result = client.fetch_files(
            target=int(resolved_tic_id),
            selectors=_selector_for_likely_contrast_files(max_files=int(max_files)),
            force_refresh=bool(force_refresh),
        )

        row_by_basename = {Path(row.filename).name: row for row in file_rows}
        observations: list[dict[str, Any]] = []
        normalized_curves: list[dict[str, Any]] = []
        parse_failures: list[dict[str, str]] = []

        for downloaded_path in sorted(fetch_result.files_downloaded):
            basename = downloaded_path.name
            row = row_by_basename.get(basename)
            filter_label = _infer_filter_label(
                filename=basename,
                description=(row.description if row is not None else None),
            )
            try:
                parsed_block, normalized, ruling_summary = _summarize_single_curve(
                    path=downloaded_path,
                    filter_name=filter_label,
                )
            except Exception as exc:
                parse_failures.append({"file": str(downloaded_path), "error": str(exc)})
                continue
            local_file = downloaded_path
            if download_dir is not None:
                download_dir.mkdir(parents=True, exist_ok=True)
                local_file = download_dir / basename
                if local_file != downloaded_path:
                    shutil.copy2(downloaded_path, local_file)

            observations.append(
                {
                    "filename": basename,
                    "file_id": int(row.file_id) if row is not None else None,
                    "type": str(row.type) if row is not None else None,
                    "tag": str(row.tag) if (row is not None and row.tag is not None) else None,
                    "description": str(row.description) if (row is not None and row.description is not None) else None,
                    "instrument": str(row.tag) if (row is not None and row.tag is not None) else None,
                    "filter": parsed_block.get("filter"),
                    "bandpass_family": _bandpass_family(parsed_block.get("filter")),
                    "tess_bandpass_compatibility": _compatibility_score(parsed_block.get("filter")),
                    "file": parsed_block,
                    "file_local": str(local_file),
                    "ruling_summary": ruling_summary,
                }
            )
            normalized_curves.append(normalized)

        combined_exclusion = combine_normalized_curves(normalized_curves)
        ruling_summary = (
            build_ruling_summary(combined_exclusion)
            if combined_exclusion is not None
            else {
                "status": "no_data",
                "coverage_quality": "none",
                "notes": ["No parseable ExoFOP contrast-curve files were found."],
            }
        )

        if observations:
            def _obs_key(obs: dict[str, Any]) -> tuple[float, float]:
                compat = float(obs.get("tess_bandpass_compatibility") or 0.0)
                d05 = float((obs.get("ruling_summary") or {}).get("max_delta_mag_at_0p5_arcsec") or 0.0)
                return (compat, d05)

            strongest_idx = max(range(len(observations)), key=lambda idx: _obs_key(observations[idx]))
            primary_obs = observations[strongest_idx]
            families = {str(obs.get("bandpass_family")) for obs in observations}
            run_multiple = len(families - {"unknown"}) > 1
            all_candidates = []
            for idx, obs in enumerate(observations):
                rsum = obs.get("ruling_summary") if isinstance(obs.get("ruling_summary"), dict) else {}
                all_candidates.append(
                    {
                        "observation_index": idx,
                        "file_local": obs.get("file_local"),
                        "filter": obs.get("filter"),
                        "instrument": obs.get("instrument"),
                        "delta_mag_at_0p5": rsum.get("max_delta_mag_at_0p5_arcsec"),
                        "delta_mag_at_1p0": rsum.get("max_delta_mag_at_1p0_arcsec"),
                        "tess_bandpass_compatibility": obs.get("tess_bandpass_compatibility"),
                    }
                )
            fpp_recommendations = {
                "primary": {
                    "observation_index": int(strongest_idx),
                    "file_local": primary_obs.get("file_local"),
                    "filter": primary_obs.get("filter"),
                    "instrument": primary_obs.get("instrument"),
                    "reason": "Best tess-bandpass compatibility with deepest 0.5 arcsec exclusion.",
                },
                "all_candidates": all_candidates,
                "run_multiple": bool(run_multiple),
                "run_multiple_reason": (
                    "Multiple non-redundant bandpass families available."
                    if run_multiple
                    else None
                ),
                "selection_policy_version": "v1",
            }
        else:
            fpp_recommendations = {
                "primary": None,
                "all_candidates": [],
                "run_multiple": False,
                "run_multiple_reason": None,
                "selection_policy_version": "v1",
            }

        verdict, verdict_source = derive_contrast_verdict(ruling_summary)
        availability = "available" if observations else "none"
        primary = fpp_recommendations.get("primary") if isinstance(fpp_recommendations, dict) else None

        payload: dict[str, Any] = {
            "schema_version": _CLI_CONTRAST_CURVES_SCHEMA,
            "tic_id": int(resolved_tic_id),
            "toi": resolved_toi,
            "observations": observations,
            "combined_exclusion": combined_exclusion,
            "ruling_summary": ruling_summary,
            "fpp_recommendations": fpp_recommendations,
            "summary": {
                "availability": availability,
                "n_observations": int(len(observations)),
                "filter": primary.get("filter") if isinstance(primary, dict) else None,
                "quality": ruling_summary.get("quality_assessment"),
                "depth0p5": ruling_summary.get("max_delta_mag_at_0p5_arcsec"),
                "depth1p0": ruling_summary.get("max_delta_mag_at_1p0_arcsec"),
                "selected_curve": {
                    "id": (
                        f"obs_{int(primary.get('observation_index'))}"
                        if isinstance(primary, dict) and primary.get("observation_index") is not None
                        else None
                    ),
                    "source": "exofop",
                    "filter": primary.get("filter") if isinstance(primary, dict) else None,
                    "quality": ruling_summary.get("quality_assessment"),
                    "depth0p5": ruling_summary.get("max_delta_mag_at_0p5_arcsec"),
                    "depth1p0": ruling_summary.get("max_delta_mag_at_1p0_arcsec"),
                }
                if isinstance(primary, dict)
                else None,
            },
            "verdict": verdict,
            "verdict_source": verdict_source,
            "result": {
                "tic_id": int(resolved_tic_id),
                "toi": resolved_toi,
                "observations": observations,
                "combined_exclusion": combined_exclusion,
                "ruling_summary": ruling_summary,
                "fpp_recommendations": fpp_recommendations,
                "summary": {
                    "availability": availability,
                    "n_observations": int(len(observations)),
                    "filter": primary.get("filter") if isinstance(primary, dict) else None,
                    "quality": ruling_summary.get("quality_assessment"),
                    "depth0p5": ruling_summary.get("max_delta_mag_at_0p5_arcsec"),
                    "depth1p0": ruling_summary.get("max_delta_mag_at_1p0_arcsec"),
                    "selected_curve": {
                        "id": (
                            f"obs_{int(primary.get('observation_index'))}"
                            if isinstance(primary, dict) and primary.get("observation_index") is not None
                            else None
                        ),
                        "source": "exofop",
                        "filter": primary.get("filter") if isinstance(primary, dict) else None,
                        "quality": ruling_summary.get("quality_assessment"),
                        "depth0p5": ruling_summary.get("max_delta_mag_at_0p5_arcsec"),
                        "depth1p0": ruling_summary.get("max_delta_mag_at_1p0_arcsec"),
                    }
                    if isinstance(primary, dict)
                    else None,
                },
                "verdict": verdict,
                "verdict_source": verdict_source,
            },
            "inputs_summary": {
                "toi": resolved_toi,
                "tic_id": int(resolved_tic_id),
                "input_resolution": input_resolution,
            },
            "provenance": {
                "cache_root": str(fetch_result.cache_root),
                "manifest_path": str(fetch_result.manifest_path),
                "warnings": list(fetch_result.warnings),
                "files_skipped": list(fetch_result.files_skipped),
                "parse_failures": parse_failures,
                "n_file_rows": int(len(file_rows)),
                "n_likely_rows": int(len(likely_rows)),
                "n_downloaded": int(len(fetch_result.files_downloaded)),
                "n_parsed": int(len(observations)),
                "selectors": {
                    "types": ["Image"],
                    "filename_regex": _selector_for_likely_contrast_files(max_files=1).filename_regex,
                    "max_files": int(max_files),
                },
                "download_dir": str(download_dir) if download_dir is not None else None,
                "network_ok": bool(network_ok),
                "force_refresh": bool(force_refresh),
            },
        }
    except BtvCliError:
        raise
    except Exception as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_RUNTIME_ERROR) from exc

    dump_json_output(payload, out_path)


__all__ = ["contrast_curve_summary_command", "contrast_curves_command"]
