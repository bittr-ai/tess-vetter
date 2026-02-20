"""`btv resolve-neighbors` command for standardized reference source payloads."""

from __future__ import annotations

from typing import Any

import click

from tess_vetter.cli.common_cli import (
    EXIT_DATA_UNAVAILABLE,
    EXIT_INPUT_ERROR,
    EXIT_REMOTE_TIMEOUT,
    BtvCliError,
    dump_json_output,
    emit_progress,
    resolve_optional_output_path,
)
from tess_vetter.cli.diagnostics_report_inputs import resolve_inputs_from_report_file
from tess_vetter.cli.reference_sources import REFERENCE_SOURCES_SCHEMA_VERSION
from tess_vetter.platform.catalogs.gaia_client import query_gaia_by_position_sync
from tess_vetter.platform.catalogs.toi_resolution import (
    LookupStatus,
    lookup_tic_coordinates,
    resolve_toi_to_tic_ephemeris_depth,
)

_RUWE_ELEVATED_THRESHOLD = 1.4


def _coerce_optional_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _coerce_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return bool(value)
    return None


def _compute_multiplicity_risk(*, reference_sources: list[dict[str, Any]], gaia_status: str) -> dict[str, Any]:
    if gaia_status != "ok":
        return {
            "status": "UNKNOWN",
            "reasons": ["GAIA_UNAVAILABLE"],
            "ruwe_threshold": float(_RUWE_ELEVATED_THRESHOLD),
            "target_ruwe": None,
            "target_non_single_star": None,
            "target_duplicated_source": None,
            "n_neighbors": int(max(0, len(reference_sources) - 1)),
            "n_neighbors_ruwe_elevated": 0,
            "n_neighbors_nss_or_duplicated": 0,
        }

    target = reference_sources[0] if reference_sources else {}
    target_meta = target.get("meta") if isinstance(target.get("meta"), dict) else {}
    target_ruwe = _coerce_optional_float(target_meta.get("ruwe"))
    target_nss = _coerce_optional_bool(target_meta.get("non_single_star"))
    target_dup = _coerce_optional_bool(target_meta.get("duplicated_source"))

    reasons: list[str] = []
    severity = 0  # 0=LOW, 1=ELEVATED, 2=HIGH
    if target_nss:
        reasons.append("TARGET_NON_SINGLE_STAR")
        severity = max(severity, 2)
    if target_dup:
        reasons.append("TARGET_DUPLICATED_SOURCE")
        severity = max(severity, 1)
    if target_ruwe is not None and target_ruwe > _RUWE_ELEVATED_THRESHOLD:
        reasons.append("TARGET_RUWE_ELEVATED")
        severity = max(severity, 1)

    n_neighbors = 0
    n_neighbors_ruwe_elevated = 0
    n_neighbors_nss_or_duplicated = 0
    for source in reference_sources[1:]:
        n_neighbors += 1
        meta = source.get("meta") if isinstance(source.get("meta"), dict) else {}
        ruwe = _coerce_optional_float(meta.get("ruwe"))
        nss = _coerce_optional_bool(meta.get("non_single_star"))
        dup = _coerce_optional_bool(meta.get("duplicated_source"))
        if ruwe is not None and ruwe > _RUWE_ELEVATED_THRESHOLD:
            n_neighbors_ruwe_elevated += 1
        if nss or dup:
            n_neighbors_nss_or_duplicated += 1

    if n_neighbors_ruwe_elevated > 0:
        reasons.append("NEIGHBOR_RUWE_ELEVATED")
        severity = max(severity, 1)
    if n_neighbors_nss_or_duplicated > 0:
        reasons.append("NEIGHBOR_NSS_OR_DUPLICATED")
        severity = max(severity, 1)

    if not reasons:
        reasons = ["NO_MULTIPLICITY_FLAGS"]

    status = "LOW"
    if severity >= 2:
        status = "HIGH"
    elif severity == 1:
        status = "ELEVATED"
    return {
        "status": status,
        "reasons": reasons,
        "ruwe_threshold": float(_RUWE_ELEVATED_THRESHOLD),
        "target_ruwe": target_ruwe,
        "target_non_single_star": target_nss,
        "target_duplicated_source": target_dup,
        "n_neighbors": int(n_neighbors),
        "n_neighbors_ruwe_elevated": int(n_neighbors_ruwe_elevated),
        "n_neighbors_nss_or_duplicated": int(n_neighbors_nss_or_duplicated),
    }


def _derive_resolve_neighbors_verdict(payload: dict[str, Any]) -> tuple[str, str]:
    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
    gaia_resolution = (
        provenance.get("gaia_resolution") if isinstance(provenance.get("gaia_resolution"), dict) else {}
    )
    gaia_status = str(gaia_resolution.get("status") or "")
    if gaia_status == "ok":
        n_neighbors_added = gaia_resolution.get("n_neighbors_added")
        try:
            neighbors = int(n_neighbors_added)
        except (TypeError, ValueError):
            neighbors = max(0, len(payload.get("reference_sources", [])) - 1)
        if neighbors > 0:
            return "NEIGHBORS_RESOLVED", "$.provenance.gaia_resolution.n_neighbors_added"
        return "TARGET_ONLY", "$.provenance.gaia_resolution.n_neighbors_added"

    if gaia_status in {"skipped_no_network", "error_fallback_target_only"}:
        return "DATA_UNAVAILABLE", "$.provenance.gaia_resolution.status"
    return "DATA_UNAVAILABLE", "$.provenance.gaia_resolution.status"


def _resolve_tic_id(*, tic_id: int | None, toi: str | None, network_ok: bool) -> tuple[int, dict[str, Any] | None]:
    if tic_id is not None:
        return int(tic_id), None
    if toi is None:
        raise BtvCliError("Provide --tic-id or --toi", exit_code=EXIT_INPUT_ERROR)
    if not network_ok:
        raise BtvCliError("--toi resolution requires --network-ok", exit_code=EXIT_INPUT_ERROR)

    resolved = resolve_toi_to_tic_ephemeris_depth(toi)
    toi_resolution = {
        "status": resolved.status.value,
        "toi_query": resolved.toi_query,
        "tic_id": resolved.tic_id,
        "matched_toi": resolved.matched_toi,
        "message": resolved.message,
        "missing_fields": list(resolved.missing_fields),
        "source_record": resolved.source_record.model_dump(mode="json", exclude_none=True),
    }
    if resolved.status != LookupStatus.OK or resolved.tic_id is None:
        code = EXIT_REMOTE_TIMEOUT if resolved.status == LookupStatus.TIMEOUT else EXIT_DATA_UNAVAILABLE
        raise BtvCliError(
            resolved.message or f"Unable to resolve TIC from TOI {toi}",
            exit_code=code,
        )
    return int(resolved.tic_id), toi_resolution


def _resolve_target_coordinates(
    *,
    tic_id: int,
    ra_deg: float | None,
    dec_deg: float | None,
    network_ok: bool,
) -> tuple[float, float, str, dict[str, Any] | None]:
    if ra_deg is not None and dec_deg is not None:
        return float(ra_deg), float(dec_deg), "cli", None
    if not network_ok:
        raise BtvCliError(
            "Target coordinates unavailable. Provide --ra-deg and --dec-deg when --no-network is set.",
            exit_code=EXIT_INPUT_ERROR,
        )

    coord_result = lookup_tic_coordinates(tic_id=int(tic_id))
    coord_meta = {
        "status": coord_result.status.value,
        "message": coord_result.message,
        "source_record": coord_result.source_record.model_dump(mode="json", exclude_none=True)
        if coord_result.source_record is not None
        else None,
        "attempts": [
            attempt.model_dump(mode="json", exclude_none=True) for attempt in list(coord_result.attempts)
        ],
    }
    if coord_result.status == LookupStatus.OK and coord_result.ra_deg is not None and coord_result.dec_deg is not None:
        return float(coord_result.ra_deg), float(coord_result.dec_deg), "tic_lookup", coord_meta
    if coord_result.status == LookupStatus.TIMEOUT:
        raise BtvCliError(
            coord_result.message or f"TIC coordinate lookup timed out for TIC {tic_id}",
            exit_code=EXIT_REMOTE_TIMEOUT,
        )
    raise BtvCliError(
        coord_result.message or f"TIC coordinate lookup failed for TIC {tic_id}",
        exit_code=EXIT_DATA_UNAVAILABLE,
    )


def _execute_resolve_neighbors(
    *,
    tic_id: int,
    toi: str | None,
    ra_deg: float | None,
    dec_deg: float | None,
    radius_arcsec: float,
    max_neighbors: int,
    network_ok: bool,
    toi_resolution: dict[str, Any] | None,
) -> dict[str, Any]:
    target_ra, target_dec, coordinate_source, coordinate_resolution = _resolve_target_coordinates(
        tic_id=int(tic_id),
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        network_ok=bool(network_ok),
    )

    reference_sources: list[dict[str, Any]] = [
        {
            "name": f"Target TIC {int(tic_id)}",
            "source_id": f"tic:{int(tic_id)}",
            "tic_id": int(tic_id),
            "role": "target",
            "ra": float(target_ra),
            "dec": float(target_dec),
            "g_mag": None,
            "separation_arcsec": 0.0,
            "meta": {"source": str(coordinate_source)},
        }
    ]

    gaia_resolution: dict[str, Any] = {
        "attempted": bool(network_ok),
        "status": "skipped_no_network" if not network_ok else "ok",
        "message": None,
        "source_record": None,
        "n_neighbors_added": 0,
    }
    if network_ok:
        try:
            gaia_result = query_gaia_by_position_sync(
                float(target_ra),
                float(target_dec),
                radius_arcsec=float(radius_arcsec),
            )
            gaia_resolution["source_record"] = gaia_result.source_record.model_dump(
                mode="json", exclude_none=True
            )

            if gaia_result.source is not None:
                reference_sources[0]["ra"] = float(gaia_result.source.ra)
                reference_sources[0]["dec"] = float(gaia_result.source.dec)
                reference_sources[0]["g_mag"] = gaia_result.source.phot_g_mean_mag
                reference_sources[0]["meta"] = {
                    "source": "gaia_dr3_primary",
                    "gaia_source_id": int(gaia_result.source.source_id),
                    "phot_g_mean_mag": gaia_result.source.phot_g_mean_mag,
                    "ruwe": gaia_result.source.ruwe,
                    "non_single_star": getattr(gaia_result.source, "non_single_star", None),
                    "duplicated_source": getattr(gaia_result.source, "duplicated_source", None),
                }

            added = 0
            for neighbor in list(gaia_result.neighbors):
                if added >= int(max_neighbors):
                    break
                reference_sources.append(
                    {
                        "name": f"Gaia {int(neighbor.source_id)}",
                        "source_id": f"gaia:{int(neighbor.source_id)}",
                        "role": "companion",
                        "ra": float(neighbor.ra),
                        "dec": float(neighbor.dec),
                        "separation_arcsec": float(neighbor.separation_arcsec),
                        "g_mag": neighbor.phot_g_mean_mag,
                        "meta": {
                            "source": "gaia_dr3_neighbor",
                            "separation_arcsec": float(neighbor.separation_arcsec),
                            "phot_g_mean_mag": neighbor.phot_g_mean_mag,
                            "delta_mag": neighbor.delta_mag,
                            "ruwe": neighbor.ruwe,
                            "non_single_star": getattr(neighbor, "non_single_star", None),
                            "duplicated_source": getattr(neighbor, "duplicated_source", None),
                        },
                    }
                )
                added += 1
            gaia_resolution["n_neighbors_added"] = int(added)
        except Exception as exc:
            gaia_resolution["status"] = "error_fallback_target_only"
            gaia_resolution["message"] = f"{type(exc).__name__}: {exc}"

    multiplicity_risk = _compute_multiplicity_risk(
        reference_sources=reference_sources,
        gaia_status=str(gaia_resolution.get("status")),
    )
    payload: dict[str, Any] = {
        "schema_version": REFERENCE_SOURCES_SCHEMA_VERSION,
        "reference_sources": reference_sources,
        "multiplicity_risk": multiplicity_risk,
        "target": {
            "tic_id": int(tic_id),
            "toi": toi,
            "ra_deg": float(reference_sources[0]["ra"]),
            "dec_deg": float(reference_sources[0]["dec"]),
        },
        "provenance": {
            "radius_arcsec": float(radius_arcsec),
            "max_neighbors": int(max_neighbors),
            "network_ok": bool(network_ok),
            "coordinate_source": str(coordinate_source),
            "coordinate_resolution": coordinate_resolution,
            "gaia_resolution": gaia_resolution,
            "multiplicity_risk": multiplicity_risk,
            "toi_resolution": toi_resolution,
        },
    }
    verdict, verdict_source = _derive_resolve_neighbors_verdict(payload)
    payload["verdict"] = verdict
    payload["verdict_source"] = verdict_source
    payload["result"] = {
        "verdict": verdict,
        "verdict_source": verdict_source,
    }
    return payload


@click.command("resolve-neighbors")
@click.argument("toi_arg", required=False)
@click.option("--tic-id", type=int, default=None, help="TIC identifier.")
@click.option("--toi", type=str, default=None, help="Optional TOI label to resolve TIC.")
@click.option("--report-file", type=str, default=None, help="Optional report JSON path for candidate inputs.")
@click.option("--ra-deg", type=float, default=None, help="Optional target right ascension in degrees.")
@click.option("--dec-deg", type=float, default=None, help="Optional target declination in degrees.")
@click.option("--radius-arcsec", type=float, default=60.0, show_default=True, help="Gaia cone radius.")
@click.option("--max-neighbors", type=int, default=10, show_default=True, help="Maximum Gaia neighbors.")
@click.option(
    "--network-ok/--no-network",
    default=False,
    show_default=True,
    help="Allow network-dependent TOI, coordinate, and Gaia lookups.",
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
def resolve_neighbors_command(
    toi_arg: str | None,
    tic_id: int | None,
    toi: str | None,
    report_file: str | None,
    ra_deg: float | None,
    dec_deg: float | None,
    radius_arcsec: float,
    max_neighbors: int,
    network_ok: bool,
    output_path_arg: str,
) -> None:
    """Resolve target and nearby Gaia sources into ``reference_sources.v1`` payload.

    Output includes ``multiplicity_risk`` (RUWE/NSS/duplicated-source rollup)
    for downstream host-localization and dilution policy routing.
    """
    out_path = resolve_optional_output_path(output_path_arg)
    if toi_arg is not None and toi is not None and str(toi_arg).strip() != str(toi).strip():
        raise BtvCliError(
            "Positional TOI argument and --toi must match when both are provided.",
            exit_code=EXIT_INPUT_ERROR,
        )
    resolved_toi_arg = toi if toi is not None else toi_arg
    if (ra_deg is None) != (dec_deg is None):
        raise BtvCliError("Provide both --ra-deg and --dec-deg together.", exit_code=EXIT_INPUT_ERROR)
    if float(radius_arcsec) <= 0.0:
        raise BtvCliError("--radius-arcsec must be > 0", exit_code=EXIT_INPUT_ERROR)
    if int(max_neighbors) < 0:
        raise BtvCliError("--max-neighbors must be >= 0", exit_code=EXIT_INPUT_ERROR)
    if report_file is not None and (tic_id is not None or resolved_toi_arg is not None):
        click.echo(
            "Warning: --report-file provided; ignoring --tic-id/--toi and using report-file candidate inputs.",
            err=True,
        )
    if report_file is None and tic_id is not None and resolved_toi_arg is not None:
        raise BtvCliError("Provide either --tic-id or --toi, not both.", exit_code=EXIT_INPUT_ERROR)

    if report_file is not None:
        resolved_from_report = resolve_inputs_from_report_file(str(report_file))
        resolved_tic = int(resolved_from_report.tic_id)
        toi_resolution = {
            "status": "ok",
            "source": "report_file",
            "report_file": str(resolved_from_report.report_file_path),
            "tic_id": int(resolved_from_report.tic_id),
        }
        resolved_toi = resolved_toi_arg
    else:
        resolved_tic, toi_resolution = _resolve_tic_id(
            tic_id=tic_id,
            toi=resolved_toi_arg,
            network_ok=bool(network_ok),
        )
        resolved_toi = resolved_toi_arg

    emit_progress("resolve-neighbors", "start")
    payload = _execute_resolve_neighbors(
        tic_id=int(resolved_tic),
        toi=resolved_toi,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        radius_arcsec=float(radius_arcsec),
        max_neighbors=int(max_neighbors),
        network_ok=bool(network_ok),
        toi_resolution=toi_resolution,
    )
    dump_json_output(payload, out_path)
    emit_progress("resolve-neighbors", "completed")


__all__ = ["resolve_neighbors_command"]
