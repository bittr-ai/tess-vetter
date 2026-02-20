"""`btv resolve-stellar` command for stellar parameter resolution with fallback."""

from __future__ import annotations

from typing import Any

import click

from tess_vetter.cli.common_cli import (
    EXIT_DATA_UNAVAILABLE,
    EXIT_INPUT_ERROR,
    EXIT_REMOTE_TIMEOUT,
    BtvCliError,
    dump_json_output,
    resolve_optional_output_path,
)
from tess_vetter.cli.stellar_inputs import load_auto_stellar_with_fallback
from tess_vetter.platform.catalogs.toi_resolution import (
    LookupStatus,
    resolve_toi_to_tic_ephemeris_depth,
)


@click.command("resolve-stellar")
@click.option("--tic-id", type=int, default=None, help="TIC identifier.")
@click.option("--toi", type=str, default=None, help="Optional TOI label to resolve TIC via ExoFOP.")
@click.option(
    "-o",
    "--out",
    "output_path_arg",
    type=str,
    default="-",
    show_default=True,
    help="JSON output path; '-' writes to stdout.",
)
def resolve_stellar_command(
    tic_id: int | None,
    toi: str | None,
    output_path_arg: str,
) -> None:
    """Resolve stellar radius/mass/tmag with TIC->ExoFOP fallback and provenance."""
    out_path = resolve_optional_output_path(output_path_arg)
    if tic_id is None and toi is None:
        raise BtvCliError("Provide --tic-id or --toi", exit_code=EXIT_INPUT_ERROR)

    effective_tic = tic_id
    toi_resolution: dict[str, Any] | None = None
    if effective_tic is None and toi is not None:
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
        effective_tic = int(resolved.tic_id)

    assert effective_tic is not None
    values, meta = load_auto_stellar_with_fallback(tic_id=int(effective_tic), toi=toi)
    payload: dict[str, Any] = {
        "schema_version": "cli.resolve-stellar.v1",
        "tic_id": int(effective_tic),
        "toi": toi,
        "stellar": dict(values),
        "provenance": {
            "resolution": meta,
            "toi_resolution": toi_resolution,
        },
    }

    if bool(meta.get("echo_of_tic")):
        payload["note"] = (
            "ExoFOP stellar values match TIC-derived values for overlapping fields; "
            "treat this as source continuity, not independent confirmation."
        )

    if values.get("radius") is None or values.get("mass") is None:
        warnings = payload.setdefault("warnings", [])
        if isinstance(warnings, list):
            warnings.append("Stellar radius/mass unresolved; provide explicit overrides for downstream FPP.")

    dump_json_output(payload, out_path)


__all__ = ["resolve_stellar_command"]

