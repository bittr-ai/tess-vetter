"""`btv dilution` command for host-dilution scenario diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import click

from bittr_tess_vetter.api import stellar_dilution as stellar_dilution_api
from bittr_tess_vetter.cli.common_cli import (
    EXIT_DATA_UNAVAILABLE,
    EXIT_INPUT_ERROR,
    EXIT_RUNTIME_ERROR,
    BtvCliError,
    dump_json_output,
    emit_progress,
    load_json_file,
    resolve_optional_output_path,
)
from bittr_tess_vetter.cli.diagnostics_report_inputs import resolve_inputs_from_report_file
from bittr_tess_vetter.cli.stellar_inputs import load_auto_stellar_with_fallback
from bittr_tess_vetter.cli.vet_cli import _resolve_candidate_inputs


def _required_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BtvCliError(f"{label} must be a JSON object", exit_code=EXIT_INPUT_ERROR)
    return cast(dict[str, Any], value)


def _required_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise BtvCliError(f"{label} must be an integer", exit_code=EXIT_INPUT_ERROR)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise BtvCliError(f"{label} must be an integer", exit_code=EXIT_INPUT_ERROR) from exc


def _optional_int(value: Any, *, label: str) -> int | None:
    if value is None:
        return None
    return _required_int(value, label=label)


def _required_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise BtvCliError(f"{label} must be a number", exit_code=EXIT_INPUT_ERROR)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise BtvCliError(f"{label} must be a number", exit_code=EXIT_INPUT_ERROR) from exc


def _optional_float(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    return _required_float(value, label=label)


@dataclass(frozen=True)
class _HostHypothesisInputs:
    tic_id: int | None
    primary_g_mag: float | None
    primary_radius_rsun: float | None
    companions: list[tuple[int, float, float | None, float | None]]
    host_ambiguous: bool | None


@dataclass(frozen=True)
class _ReferenceSourceEntry:
    source_id: int | None
    tic_id: int | None
    separation_arcsec: float | None
    g_mag: float | None
    radius_rsun: float | None
    is_target: bool


def _parse_host_profile(payload: dict[str, Any]) -> _HostHypothesisInputs:
    primary_raw = payload.get("primary", {})
    primary = _required_mapping(primary_raw, label="host profile primary")
    tic_id = _optional_int(primary.get("tic_id"), label="primary.tic_id")
    primary_g_mag = _optional_float(primary.get("g_mag"), label="primary.g_mag")
    primary_radius_rsun = _optional_float(primary.get("radius_rsun"), label="primary.radius_rsun")

    companions_raw = payload.get("companions", [])
    if not isinstance(companions_raw, list):
        raise BtvCliError("companions must be a JSON list", exit_code=EXIT_INPUT_ERROR)

    companions: list[tuple[int, float, float | None, float | None]] = []
    for idx, item in enumerate(companions_raw):
        comp = _required_mapping(item, label=f"companions[{idx}]")
        source_id = _required_int(comp.get("source_id"), label=f"companions[{idx}].source_id")
        separation_arcsec = _required_float(
            comp.get("separation_arcsec"),
            label=f"companions[{idx}].separation_arcsec",
        )
        g_mag = _optional_float(comp.get("g_mag"), label=f"companions[{idx}].g_mag")
        radius_rsun = _optional_float(comp.get("radius_rsun"), label=f"companions[{idx}].radius_rsun")
        companions.append((source_id, separation_arcsec, g_mag, radius_rsun))

    host_ambiguous_raw = payload.get("host_ambiguous")
    if host_ambiguous_raw is None:
        host_ambiguous = None
    elif isinstance(host_ambiguous_raw, bool):
        host_ambiguous = bool(host_ambiguous_raw)
    else:
        raise BtvCliError("host_ambiguous must be a boolean", exit_code=EXIT_INPUT_ERROR)

    return _HostHypothesisInputs(
        tic_id=tic_id,
        primary_g_mag=primary_g_mag,
        primary_radius_rsun=primary_radius_rsun,
        companions=companions,
        host_ambiguous=host_ambiguous,
    )


def _parse_reference_source_id(value: Any, *, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise BtvCliError(f"{label} must be an integer or numeric string", exit_code=EXIT_INPUT_ERROR)
    if isinstance(value, int):
        return int(value)
    text = str(value).strip()
    if text == "":
        return None
    if ":" in text:
        maybe_id = text.split(":", 1)[1].strip()
        if maybe_id.lstrip("+-").isdigit():
            return int(maybe_id)
    if text.lstrip("+-").isdigit():
        return int(text)
    raise BtvCliError(f"{label} must be an integer or numeric string", exit_code=EXIT_INPUT_ERROR)


def _parse_reference_sources(payload: dict[str, Any]) -> _HostHypothesisInputs:
    schema_version = payload.get("schema_version")
    if schema_version != "reference_sources.v1":
        raise BtvCliError(
            "reference sources file schema_version must be 'reference_sources.v1'",
            exit_code=EXIT_INPUT_ERROR,
        )

    rows_raw = payload.get("reference_sources")
    if not isinstance(rows_raw, list):
        raise BtvCliError(
            "reference sources schema error: top-level 'reference_sources' must be a list",
            exit_code=EXIT_INPUT_ERROR,
        )

    entries: list[_ReferenceSourceEntry] = []
    for idx, raw in enumerate(rows_raw):
        row = _required_mapping(raw, label=f"reference_sources[{idx}]")
        source_id = _parse_reference_source_id(row.get("source_id"), label=f"reference_sources[{idx}].source_id")
        tic_id = _optional_int(row.get("tic_id"), label=f"reference_sources[{idx}].tic_id")
        separation_arcsec = _optional_float(
            row.get("separation_arcsec"),
            label=f"reference_sources[{idx}].separation_arcsec",
        )
        g_mag = _optional_float(row.get("g_mag"), label=f"reference_sources[{idx}].g_mag")
        radius_rsun = _optional_float(row.get("radius_rsun"), label=f"reference_sources[{idx}].radius_rsun")
        meta = row.get("meta")
        if isinstance(meta, dict):
            if separation_arcsec is None and meta.get("separation_arcsec") is not None:
                separation_arcsec = _optional_float(
                    meta.get("separation_arcsec"),
                    label=f"reference_sources[{idx}].meta.separation_arcsec",
                )
            if g_mag is None and meta.get("phot_g_mean_mag") is not None:
                g_mag = _optional_float(
                    meta.get("phot_g_mean_mag"),
                    label=f"reference_sources[{idx}].meta.phot_g_mean_mag",
                )

        is_target_raw = row.get("is_target")
        if is_target_raw is not None and not isinstance(is_target_raw, bool):
            raise BtvCliError(
                f"reference_sources[{idx}].is_target must be a boolean",
                exit_code=EXIT_INPUT_ERROR,
            )
        role_text = str(row.get("role", "")).strip().lower()
        source_id_raw = row.get("source_id")
        source_id_text = str(source_id_raw).strip().lower() if source_id_raw is not None else ""
        is_target = (
            bool(is_target_raw)
            or role_text in {"target", "primary", "host"}
            or (tic_id is not None)
            or source_id_text.startswith("tic:")
        )

        entries.append(
            _ReferenceSourceEntry(
                source_id=source_id,
                tic_id=tic_id,
                separation_arcsec=separation_arcsec,
                g_mag=g_mag,
                radius_rsun=radius_rsun,
                is_target=is_target,
            )
        )

    target_indices = [idx for idx, entry in enumerate(entries) if entry.is_target]
    if len(target_indices) > 1:
        raise BtvCliError(
            "reference sources schema error: expected at most one target source",
            exit_code=EXIT_INPUT_ERROR,
        )

    primary_tic_id: int | None = None
    primary_g_mag: float | None = None
    primary_radius_rsun: float | None = None
    if target_indices:
        primary_entry = entries[target_indices[0]]
        primary_tic_id = primary_entry.tic_id if primary_entry.tic_id is not None else primary_entry.source_id
        primary_g_mag = primary_entry.g_mag
        primary_radius_rsun = primary_entry.radius_rsun

    companions: list[tuple[int, float, float | None, float | None]] = []
    for idx, entry in enumerate(entries):
        if entry.is_target:
            continue
        if entry.source_id is None:
            raise BtvCliError(
                f"reference sources schema error: reference_sources[{idx}].source_id is required for companions",
                exit_code=EXIT_INPUT_ERROR,
            )
        if entry.separation_arcsec is None:
            raise BtvCliError(
                f"reference sources schema error: reference_sources[{idx}].separation_arcsec is required for companions",
                exit_code=EXIT_INPUT_ERROR,
            )
        companions.append((entry.source_id, entry.separation_arcsec, entry.g_mag, entry.radius_rsun))

    host_ambiguous_raw = payload.get("host_ambiguous")
    if host_ambiguous_raw is None:
        host_ambiguous = None
    elif isinstance(host_ambiguous_raw, bool):
        host_ambiguous = bool(host_ambiguous_raw)
    else:
        raise BtvCliError("reference sources host_ambiguous must be a boolean", exit_code=EXIT_INPUT_ERROR)

    return _HostHypothesisInputs(
        tic_id=primary_tic_id,
        primary_g_mag=primary_g_mag,
        primary_radius_rsun=primary_radius_rsun,
        companions=companions,
        host_ambiguous=host_ambiguous,
    )


def _merge_host_inputs(
    *,
    host_profile: _HostHypothesisInputs | None,
    reference_sources: _HostHypothesisInputs | None,
    cli_tic_id: int | None,
) -> tuple[int, float | None, float | None, list[tuple[int, float, float | None, float | None]], bool]:
    tic_id = (
        host_profile.tic_id
        if host_profile is not None and host_profile.tic_id is not None
        else reference_sources.tic_id
        if reference_sources is not None and reference_sources.tic_id is not None
        else cli_tic_id
    )
    if tic_id is None:
        raise BtvCliError(
            "Unable to resolve host TIC ID. Provide primary.tic_id in --host-profile-file, "
            "target tic_id/source_id in --reference-sources-file, or --tic-id.",
            exit_code=EXIT_INPUT_ERROR,
        )

    primary_g_mag = (
        host_profile.primary_g_mag
        if host_profile is not None and host_profile.primary_g_mag is not None
        else reference_sources.primary_g_mag if reference_sources is not None else None
    )
    primary_radius_rsun = (
        host_profile.primary_radius_rsun
        if host_profile is not None and host_profile.primary_radius_rsun is not None
        else reference_sources.primary_radius_rsun if reference_sources is not None else None
    )

    companions: list[tuple[int, float, float | None, float | None]] = []
    by_source_id: dict[int, tuple[int, float, float | None, float | None]] = {}
    for item in (host_profile.companions if host_profile is not None else []):
        by_source_id[int(item[0])] = item
    for item in (reference_sources.companions if reference_sources is not None else []):
        by_source_id.setdefault(int(item[0]), item)
    companions.extend(by_source_id.values())

    if host_profile is not None and host_profile.host_ambiguous is not None:
        host_ambiguous = bool(host_profile.host_ambiguous)
    elif reference_sources is not None and reference_sources.host_ambiguous is not None:
        host_ambiguous = bool(reference_sources.host_ambiguous)
    else:
        host_ambiguous = len(companions) > 0

    return int(tic_id), primary_g_mag, primary_radius_rsun, companions, bool(host_ambiguous)


def _resolve_observed_depth(
    *,
    depth_ppm: float | None,
    report_file: str | None,
    network_ok: bool,
    toi: str | None,
    tic_id: int | None,
    period_days: float | None,
    t0_btjd: float | None,
    duration_hours: float | None,
) -> tuple[float, dict[str, Any]]:
    if depth_ppm is not None:
        return float(depth_ppm), {
            "source": "cli",
            "resolved_from": "cli",
            "inputs": {"depth_ppm": float(depth_ppm)},
        }
    if report_file is not None:
        resolved_from_report = resolve_inputs_from_report_file(str(report_file))
        if resolved_from_report.depth_ppm is None:
            raise BtvCliError(
                "Observed depth unavailable in report file. Provide --depth-ppm.",
                exit_code=EXIT_DATA_UNAVAILABLE,
            )
        return float(resolved_from_report.depth_ppm), dict(resolved_from_report.input_resolution)

    (
        _resolved_tic_id,
        _resolved_period_days,
        _resolved_t0_btjd,
        _resolved_duration_hours,
        resolved_depth_ppm,
        input_resolution,
    ) = _resolve_candidate_inputs(
        network_ok=bool(network_ok),
        toi=toi,
        tic_id=tic_id,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        depth_ppm=None,
    )
    if resolved_depth_ppm is None:
        raise BtvCliError(
            "Observed depth unavailable from candidate resolution. Provide --depth-ppm.",
            exit_code=EXIT_DATA_UNAVAILABLE,
        )
    return float(resolved_depth_ppm), input_resolution


def _derive_dilution_verdict(physics_flags_payload: Any) -> tuple[str | None, str | None]:
    if not isinstance(physics_flags_payload, dict):
        return None, None
    requires_resolved_followup = bool(physics_flags_payload.get("requires_resolved_followup"))
    if requires_resolved_followup:
        return "REVIEW_WITH_DILUTION", "$.physics_flags.requires_resolved_followup"
    planet_radius_inconsistent = bool(physics_flags_payload.get("planet_radius_inconsistent"))
    if planet_radius_inconsistent:
        return "IMPLAUSIBLE_PRIMARY_SCENARIO", "$.physics_flags.planet_radius_inconsistent"
    return (
        "DILUTION_PRIMARY_PLAUSIBLE",
        "$.physics_flags.requires_resolved_followup|$.physics_flags.planet_radius_inconsistent",
    )


def _build_dilution_reliability_summary(
    *,
    physics_flags_payload: dict[str, Any],
    n_plausible_scenarios: int,
    verdict: str | None,
    multiplicity_risk: dict[str, Any] | None,
) -> dict[str, Any]:
    requires_followup = bool(physics_flags_payload.get("requires_resolved_followup"))
    radius_inconsistent = bool(physics_flags_payload.get("planet_radius_inconsistent"))
    status = "RELIABLE"
    action_hint = "DILUTION_PRIMARY_PLAUSIBLE"
    if requires_followup:
        status = "REVIEW_REQUIRED"
        action_hint = "REVIEW_WITH_DILUTION"
    if radius_inconsistent:
        status = "IMPLAUSIBLE"
        action_hint = "IMPLAUSIBLE_PRIMARY_SCENARIO"
    if verdict is not None:
        action_hint = str(verdict)
    return {
        "status": status,
        "action_hint": action_hint,
        "requires_resolved_followup": requires_followup,
        "planet_radius_inconsistent": radius_inconsistent,
        "n_plausible_scenarios": int(n_plausible_scenarios),
        "multiplicity_risk": multiplicity_risk,
    }


@click.command("dilution")
@click.argument("toi_arg", required=False)
@click.option("--tic-id", type=int, default=None, help="TIC identifier for candidate input resolution.")
@click.option("--period-days", type=float, default=None, help="Orbital period in days for candidate input resolution.")
@click.option("--t0-btjd", type=float, default=None, help="Reference epoch in BTJD for candidate input resolution.")
@click.option("--duration-hours", type=float, default=None, help="Transit duration in hours for candidate input resolution.")
@click.option("--depth-ppm", type=float, default=None, help="Observed transit depth in ppm.")
@click.option("--toi", type=str, default=None, help="Optional TOI label for candidate input resolution.")
@click.option("--report-file", type=str, default=None, help="Optional report JSON path for candidate inputs.")
@click.option(
    "--network-ok/--no-network",
    default=False,
    show_default=True,
    help="Allow network-dependent TOI resolution.",
)
@click.option(
    "--host-profile-file",
    type=str,
    default=None,
    help="JSON file with primary/companions host profile.",
)
@click.option(
    "--reference-sources-file",
    type=str,
    default=None,
    help="JSON file with schema_version=reference_sources.v1 for companion hypotheses.",
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
def dilution_command(
    toi_arg: str | None,
    tic_id: int | None,
    period_days: float | None,
    t0_btjd: float | None,
    duration_hours: float | None,
    depth_ppm: float | None,
    toi: str | None,
    report_file: str | None,
    network_ok: bool,
    host_profile_file: str | None,
    reference_sources_file: str | None,
    output_path_arg: str,
) -> None:
    """Compute host-dilution scenarios and implied-size physics flags.

    Output includes ``reliability_summary`` for machine routing; when
    ``--reference-sources-file`` includes ``multiplicity_risk``, it is threaded
    into the summary.
    """
    out_path = resolve_optional_output_path(output_path_arg)
    if toi_arg is not None and toi is not None and str(toi_arg).strip() != str(toi).strip():
        raise BtvCliError(
            "Positional TOI argument and --toi must match when both are provided.",
            exit_code=EXIT_INPUT_ERROR,
        )
    resolved_toi_arg = toi if toi is not None else toi_arg
    if not host_profile_file and not reference_sources_file:
        raise BtvCliError(
            "Provide at least one: --host-profile-file or --reference-sources-file",
            exit_code=EXIT_INPUT_ERROR,
        )
    report_file_path: str | None = None
    report_tic_id: int | None = None
    if report_file is not None:
        if (
            resolved_toi_arg is not None
            or tic_id is not None
            or period_days is not None
            or t0_btjd is not None
            or duration_hours is not None
        ):
            click.echo(
                "Warning: --report-file provided; ignoring candidate input flags and using report-file candidate inputs.",
                err=True,
            )
        resolved_from_report = resolve_inputs_from_report_file(str(report_file))
        report_file_path = str(resolved_from_report.report_file_path)
        report_tic_id = int(resolved_from_report.tic_id)
        resolved_toi_arg = None

    host_profile_path: Path | None = None
    host_profile_inputs: _HostHypothesisInputs | None = None
    if host_profile_file:
        host_profile_path = Path(host_profile_file)
        host_profile_payload = load_json_file(host_profile_path, label="host profile file")
        host_profile_inputs = _parse_host_profile(host_profile_payload)

    reference_sources_path: Path | None = None
    reference_source_inputs: _HostHypothesisInputs | None = None
    reference_sources_multiplicity_risk: dict[str, Any] | None = None
    if reference_sources_file:
        reference_sources_path = Path(reference_sources_file)
        reference_sources_payload = load_json_file(reference_sources_path, label="reference sources file")
        reference_source_inputs = _parse_reference_sources(reference_sources_payload)
        if isinstance(reference_sources_payload.get("multiplicity_risk"), dict):
            reference_sources_multiplicity_risk = dict(reference_sources_payload.get("multiplicity_risk") or {})

    profile_tic_id, primary_g_mag, primary_radius_rsun, companions_profile, host_ambiguous = _merge_host_inputs(
        host_profile=host_profile_inputs,
        reference_sources=reference_source_inputs,
        cli_tic_id=report_tic_id if report_tic_id is not None else tic_id,
    )
    stellar_radius_resolution: dict[str, Any] = {
        "attempted": False,
        "resolved_from": None,
        "radius_rsun": primary_radius_rsun,
        "meta": None,
        "error": None,
    }

    emit_progress("dilution", "start")
    try:
        observed_depth_ppm, input_resolution = _resolve_observed_depth(
            depth_ppm=depth_ppm,
            report_file=report_file_path,
            network_ok=bool(network_ok),
            toi=resolved_toi_arg,
            tic_id=(report_tic_id if report_tic_id is not None else tic_id),
            period_days=period_days,
            t0_btjd=t0_btjd,
            duration_hours=duration_hours,
        )
        if primary_radius_rsun is None and bool(network_ok):
            stellar_radius_resolution["attempted"] = True
            try:
                auto_stellar_values, auto_stellar_meta = load_auto_stellar_with_fallback(
                    tic_id=int(profile_tic_id),
                    toi=resolved_toi_arg,
                )
                stellar_radius_resolution["meta"] = auto_stellar_meta
                auto_radius = auto_stellar_values.get("radius")
                if auto_radius is not None:
                    primary_radius_rsun = float(auto_radius)
                    stellar_radius_resolution["resolved_from"] = "auto_stellar_fallback"
                    stellar_radius_resolution["radius_rsun"] = float(auto_radius)
            except Exception as exc:
                # Fail open: dilution depth plausibility remains meaningful even
                # when radius auto-resolution is unavailable.
                stellar_radius_resolution["error"] = str(exc)

        companion_hypothesis_inputs: list[tuple[int, float, float | None, float | None, float | None]] = []
        for source_id, separation_arcsec, g_mag, radius_rsun in companions_profile:
            delta_mag = (float(g_mag) - float(primary_g_mag)) if (g_mag is not None and primary_g_mag is not None) else None
            companion_hypothesis_inputs.append(
                (
                    int(source_id),
                    float(separation_arcsec),
                    g_mag,
                    delta_mag,
                    radius_rsun,
                )
            )

        primary_h, companions_h = stellar_dilution_api.build_host_hypotheses_from_profile(
            tic_id=int(profile_tic_id),
            primary_g_mag=primary_g_mag,
            primary_radius_rsun=primary_radius_rsun,
            close_bright_companions=cast(
                list[tuple[int, float, float, float | None, float | None]],
                companion_hypothesis_inputs,
            ),
        )
        scenarios = stellar_dilution_api.compute_dilution_scenarios(
            observed_depth_ppm=float(observed_depth_ppm),
            primary=primary_h,
            companions=companions_h,
        )
        physics_flags = stellar_dilution_api.evaluate_physics_flags(
            scenarios,
            host_ambiguous=bool(host_ambiguous),
        )
    except BtvCliError:
        raise
    except Exception as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_RUNTIME_ERROR) from exc

    scenarios_payload = [scenario.to_dict() for scenario in scenarios]
    physics_flags_payload = physics_flags.to_dict()
    verdict, verdict_source = _derive_dilution_verdict(physics_flags_payload)
    reliability_summary = _build_dilution_reliability_summary(
        physics_flags_payload=physics_flags_payload,
        n_plausible_scenarios=int(physics_flags.n_plausible_scenarios),
        verdict=verdict,
        multiplicity_risk=reference_sources_multiplicity_risk,
    )
    payload = {
        "schema_version": "cli.dilution.v1",
        "result": {
            "scenarios": scenarios_payload,
            "physics_flags": physics_flags_payload,
            "n_plausible_scenarios": int(physics_flags.n_plausible_scenarios),
            "reliability_summary": reliability_summary,
            "verdict": verdict,
            "verdict_source": verdict_source,
        },
        "scenarios": scenarios_payload,
        "physics_flags": physics_flags_payload,
        "n_plausible_scenarios": int(physics_flags.n_plausible_scenarios),
        "reliability_summary": reliability_summary,
        "verdict": verdict,
        "verdict_source": verdict_source,
        "inputs_summary": {
            "input_resolution": input_resolution,
        },
        "provenance": {
            "inputs_source": "report_file" if report_file_path is not None else "cli",
            "report_file": report_file_path,
            "host_profile_path": str(host_profile_path) if host_profile_path is not None else None,
            "reference_sources_path": (
                str(reference_sources_path) if reference_sources_path is not None else None
            ),
            "host_ambiguous": bool(host_ambiguous),
            "observed_depth_ppm": float(observed_depth_ppm),
            "primary_radius_resolution": stellar_radius_resolution,
            "reference_sources_multiplicity_risk": reference_sources_multiplicity_risk,
            "reliability_summary": reliability_summary,
        },
    }
    dump_json_output(payload, out_path)
    emit_progress("dilution", "completed")


__all__ = ["dilution_command"]
