"""`btv resolve-neighbors` command for standardized reference source payloads."""

from __future__ import annotations

from typing import Any

import click

from bittr_tess_vetter.cli.common_cli import (
    EXIT_DATA_UNAVAILABLE,
    EXIT_INPUT_ERROR,
    EXIT_REMOTE_TIMEOUT,
    BtvCliError,
    dump_json_output,
    resolve_optional_output_path,
)
from bittr_tess_vetter.cli.reference_sources import REFERENCE_SOURCES_SCHEMA_VERSION
from bittr_tess_vetter.platform.catalogs.gaia_client import query_gaia_by_position_sync
from bittr_tess_vetter.platform.catalogs.toi_resolution import (
    LookupStatus,
    lookup_tic_coordinates,
    resolve_toi_to_tic_ephemeris_depth,
)


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
                        },
                    }
                )
                added += 1
            gaia_resolution["n_neighbors_added"] = int(added)
        except Exception as exc:
            gaia_resolution["status"] = "error_fallback_target_only"
            gaia_resolution["message"] = f"{type(exc).__name__}: {exc}"

    return {
        "schema_version": REFERENCE_SOURCES_SCHEMA_VERSION,
        "reference_sources": reference_sources,
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
            "toi_resolution": toi_resolution,
        },
    }


@click.command("resolve-neighbors")
@click.option("--tic-id", type=int, default=None, help="TIC identifier.")
@click.option("--toi", type=str, default=None, help="Optional TOI label to resolve TIC.")
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
    "--out",
    "output_path_arg",
    type=str,
    default="-",
    show_default=True,
    help="JSON output path; '-' writes to stdout.",
)
def resolve_neighbors_command(
    tic_id: int | None,
    toi: str | None,
    ra_deg: float | None,
    dec_deg: float | None,
    radius_arcsec: float,
    max_neighbors: int,
    network_ok: bool,
    output_path_arg: str,
) -> None:
    """Resolve target and nearby Gaia sources into ``reference_sources.v1`` payload."""
    out_path = resolve_optional_output_path(output_path_arg)
    if (ra_deg is None) != (dec_deg is None):
        raise BtvCliError("Provide both --ra-deg and --dec-deg together.", exit_code=EXIT_INPUT_ERROR)
    if float(radius_arcsec) <= 0.0:
        raise BtvCliError("--radius-arcsec must be > 0", exit_code=EXIT_INPUT_ERROR)
    if int(max_neighbors) < 0:
        raise BtvCliError("--max-neighbors must be >= 0", exit_code=EXIT_INPUT_ERROR)
    if tic_id is not None and toi is not None:
        raise BtvCliError("Provide either --tic-id or --toi, not both.", exit_code=EXIT_INPUT_ERROR)

    resolved_tic, toi_resolution = _resolve_tic_id(
        tic_id=tic_id,
        toi=toi,
        network_ok=bool(network_ok),
    )

    payload = _execute_resolve_neighbors(
        tic_id=int(resolved_tic),
        toi=toi,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        radius_arcsec=float(radius_arcsec),
        max_neighbors=int(max_neighbors),
        network_ok=bool(network_ok),
        toi_resolution=toi_resolution,
    )
    dump_json_output(payload, out_path)


__all__ = ["resolve_neighbors_command"]
