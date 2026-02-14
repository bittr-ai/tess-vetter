"""`btv dilution` command for host-dilution scenario diagnostics."""

from __future__ import annotations

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
    load_json_file,
    resolve_optional_output_path,
)
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


def _parse_host_profile(
    payload: dict[str, Any],
) -> tuple[int, float | None, float | None, list[tuple[int, float, float | None, float | None]], bool]:
    primary_raw = payload.get("primary")
    primary = _required_mapping(primary_raw, label="host profile primary")
    tic_id = _required_int(primary.get("tic_id"), label="primary.tic_id")
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
        host_ambiguous = len(companions) > 0
    elif isinstance(host_ambiguous_raw, bool):
        host_ambiguous = bool(host_ambiguous_raw)
    else:
        raise BtvCliError("host_ambiguous must be a boolean", exit_code=EXIT_INPUT_ERROR)

    return tic_id, primary_g_mag, primary_radius_rsun, companions, host_ambiguous


def _resolve_observed_depth(
    *,
    depth_ppm: float | None,
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


@click.command("dilution")
@click.option("--tic-id", type=int, default=None, help="TIC identifier for candidate input resolution.")
@click.option("--period-days", type=float, default=None, help="Orbital period in days for candidate input resolution.")
@click.option("--t0-btjd", type=float, default=None, help="Reference epoch in BTJD for candidate input resolution.")
@click.option("--duration-hours", type=float, default=None, help="Transit duration in hours for candidate input resolution.")
@click.option("--depth-ppm", type=float, default=None, help="Observed transit depth in ppm.")
@click.option("--toi", type=str, default=None, help="Optional TOI label for candidate input resolution.")
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
    "--out",
    "output_path_arg",
    type=str,
    default="-",
    show_default=True,
    help="JSON output path; '-' writes to stdout.",
)
def dilution_command(
    tic_id: int | None,
    period_days: float | None,
    t0_btjd: float | None,
    duration_hours: float | None,
    depth_ppm: float | None,
    toi: str | None,
    network_ok: bool,
    host_profile_file: str | None,
    output_path_arg: str,
) -> None:
    """Compute host-dilution scenarios and implied-size physics flags."""
    out_path = resolve_optional_output_path(output_path_arg)
    if not host_profile_file:
        raise BtvCliError("Missing required option: --host-profile-file", exit_code=EXIT_INPUT_ERROR)

    host_profile_path = Path(host_profile_file)
    host_profile_payload = load_json_file(host_profile_path, label="host profile file")
    profile_tic_id, primary_g_mag, primary_radius_rsun, companions_profile, host_ambiguous = _parse_host_profile(
        host_profile_payload
    )

    try:
        observed_depth_ppm, input_resolution = _resolve_observed_depth(
            depth_ppm=depth_ppm,
            network_ok=bool(network_ok),
            toi=toi,
            tic_id=tic_id,
            period_days=period_days,
            t0_btjd=t0_btjd,
            duration_hours=duration_hours,
        )

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

    payload = {
        "schema_version": "cli.dilution.v1",
        "scenarios": [scenario.to_dict() for scenario in scenarios],
        "physics_flags": physics_flags.to_dict(),
        "inputs_summary": {
            "input_resolution": input_resolution,
        },
        "provenance": {
            "host_profile_path": str(host_profile_path),
            "host_ambiguous": bool(host_ambiguous),
            "observed_depth_ppm": float(observed_depth_ppm),
        },
    }
    dump_json_output(payload, out_path)


__all__ = ["dilution_command"]
