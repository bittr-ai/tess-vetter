"""`btv fetch` command for long-term light-curve cache prewarm."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from bittr_tess_vetter.cli.common_cli import (
    EXIT_DATA_UNAVAILABLE,
    EXIT_INPUT_ERROR,
    EXIT_REMOTE_TIMEOUT,
    EXIT_RUNTIME_ERROR,
    BtvCliError,
    dump_json_output,
    resolve_optional_output_path,
)
from bittr_tess_vetter.cli.diagnostics_report_inputs import (
    choose_effective_sectors,
    resolve_inputs_from_report_file,
)
from bittr_tess_vetter.domain.lightcurve import make_data_ref
from bittr_tess_vetter.platform.catalogs.toi_resolution import LookupStatus, resolve_toi_to_tic_ephemeris_depth
from bittr_tess_vetter.platform.io import LightCurveNotFoundError, MASTClient, PersistentCache


def _resolution_error_to_exit(status: LookupStatus) -> int:
    if status == LookupStatus.TIMEOUT:
        return EXIT_REMOTE_TIMEOUT
    if status == LookupStatus.DATA_UNAVAILABLE:
        return EXIT_DATA_UNAVAILABLE
    return EXIT_RUNTIME_ERROR


def _resolve_tic_and_inputs(
    *,
    tic_id: int | None,
    toi: str | None,
    report_file: Path | None,
    network_ok: bool,
) -> tuple[int, dict[str, Any], str | None, list[int] | None]:
    if report_file is not None:
        resolved = resolve_inputs_from_report_file(str(report_file))
        return (
            int(resolved.tic_id),
            dict(resolved.input_resolution),
            str(resolved.report_file_path),
            [int(s) for s in resolved.sectors_used] if resolved.sectors_used is not None else None,
        )

    if toi is not None:
        if not network_ok:
            raise BtvCliError(
                "--toi requires --network-ok to resolve ExoFOP inputs",
                exit_code=EXIT_DATA_UNAVAILABLE,
            )
        toi_result = resolve_toi_to_tic_ephemeris_depth(str(toi))
        if toi_result.tic_id is None:
            raise BtvCliError(
                toi_result.message or f"Failed to resolve TOI {toi}",
                exit_code=_resolution_error_to_exit(toi_result.status),
            )
        errors: list[str] = []
        if toi_result.status != LookupStatus.OK:
            errors.append(toi_result.message or f"TOI resolution degraded for {toi}")
        resolved_tic = int(toi_result.tic_id)
        overrides: list[str] = []
        if tic_id is not None and int(tic_id) != resolved_tic:
            overrides.append("tic_id")
            resolved_tic = int(tic_id)
        input_resolution = {
            "source": "toi_catalog",
            "resolved_from": "exofop_toi_table",
            "inputs": {"tic_id": int(resolved_tic), "toi": str(toi)},
            "overrides": overrides,
            "errors": errors,
        }
        return int(resolved_tic), input_resolution, None, None

    if tic_id is None:
        raise BtvCliError(
            "Missing target input. Provide one of --tic-id, --toi, or --report-file.",
            exit_code=EXIT_INPUT_ERROR,
        )
    return (
        int(tic_id),
        {"source": "cli", "resolved_from": "cli", "inputs": {"tic_id": int(tic_id)}},
        None,
        None,
    )


def _load_lightcurves_for_fetch(
    *,
    client: MASTClient,
    tic_id: int,
    sectors: list[int] | None,
    sectors_explicit: bool,
    flux_type: str,
    network_ok: bool,
) -> tuple[list[Any], str]:
    normalized_flux_type = str(flux_type).lower()

    if sectors_explicit:
        if not sectors:
            raise BtvCliError("Internal error: explicit sectors requested without sector ids.")
        if network_ok:
            lightcurves = client.download_all_sectors(
                tic_id=int(tic_id),
                flux_type=normalized_flux_type,
                sectors=[int(s) for s in sectors],
            )
            if not lightcurves:
                raise LightCurveNotFoundError(f"No sectors available for TIC {int(tic_id)}")
            return sorted(lightcurves, key=lambda lc: int(getattr(lc, "sector", 0))), "mast_filtered_explicit_sectors"

        cached_lightcurves: list[Any] = []
        missing: list[int] = []
        for sector in sectors:
            try:
                cached_lightcurves.append(
                    client.download_lightcurve_cached(
                        tic_id=int(tic_id),
                        sector=int(sector),
                        flux_type=normalized_flux_type,
                    )
                )
            except Exception:
                missing.append(int(sector))
        if missing:
            raise BtvCliError(
                (
                    f"Cache-only sector load failed for TIC {int(tic_id)}. "
                    f"Missing cached light curve for sector(s): {', '.join(str(s) for s in sorted(set(missing)))}."
                ),
                exit_code=EXIT_DATA_UNAVAILABLE,
            )
        return sorted(cached_lightcurves, key=lambda lc: int(getattr(lc, "sector", 0))), "cache_only_explicit_sectors"

    if sectors:
        if network_ok:
            lightcurves = client.download_all_sectors(
                tic_id=int(tic_id),
                flux_type=normalized_flux_type,
                sectors=[int(s) for s in sectors],
            )
            if not lightcurves:
                raise LightCurveNotFoundError(f"No sectors available for TIC {int(tic_id)}")
            return sorted(lightcurves, key=lambda lc: int(getattr(lc, "sector", 0))), "mast_filtered"

        cached_lightcurves = []
        missing = []
        for sector in sectors:
            try:
                cached_lightcurves.append(
                    client.download_lightcurve_cached(
                        tic_id=int(tic_id),
                        sector=int(sector),
                        flux_type=normalized_flux_type,
                    )
                )
            except Exception:
                missing.append(int(sector))
        if missing:
            raise BtvCliError(
                (
                    f"Cache-only sector load failed for TIC {int(tic_id)}. "
                    f"Missing cached light curve for sector(s): {', '.join(str(s) for s in sorted(set(missing)))}."
                ),
                exit_code=EXIT_DATA_UNAVAILABLE,
            )
        return sorted(cached_lightcurves, key=lambda lc: int(getattr(lc, "sector", 0))), "cache_only_filtered"

    if network_ok:
        lightcurves = client.download_all_sectors(
            tic_id=int(tic_id),
            flux_type=normalized_flux_type,
            sectors=None,
        )
        if not lightcurves:
            raise LightCurveNotFoundError(f"No sectors available for TIC {int(tic_id)}")
        return sorted(lightcurves, key=lambda lc: int(getattr(lc, "sector", 0))), "mast_discovery"

    cached_rows = client.search_lightcurve_cached(tic_id=int(tic_id))
    cached_sectors = sorted({int(row.sector) for row in cached_rows if getattr(row, "sector", None) is not None})
    if not cached_sectors:
        raise BtvCliError(
            (
                f"No cached sectors available for TIC {int(tic_id)} with --no-network. "
                "Provide --sectors for known cached sectors or enable --network-ok."
            ),
            exit_code=EXIT_DATA_UNAVAILABLE,
        )
    cached_lightcurves = [
        client.download_lightcurve_cached(
            tic_id=int(tic_id),
            sector=int(sector),
            flux_type=normalized_flux_type,
        )
        for sector in cached_sectors
    ]
    return sorted(cached_lightcurves, key=lambda lc: int(getattr(lc, "sector", 0))), "cache_discovery"


@click.command("fetch")
@click.option("--toi", type=str, default=None, help="Optional TOI label.")
@click.option("--tic-id", type=int, default=None, help="TIC identifier.")
@click.option(
    "--report-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional report JSON path for target inputs and sectors.",
)
@click.option(
    "--network-ok/--no-network",
    default=False,
    show_default=True,
    help="Allow network downloads for sector discovery and fetch.",
)
@click.option("--sectors", multiple=True, type=int, help="Optional sector filters.")
@click.option(
    "--flux-type",
    type=click.Choice(["pdcsap", "sap"], case_sensitive=False),
    default="pdcsap",
    show_default=True,
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Optional cache directory used by MAST and persistent cache stores.",
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
def fetch_command(
    toi: str | None,
    tic_id: int | None,
    report_file: Path | None,
    network_ok: bool,
    sectors: tuple[int, ...],
    flux_type: str,
    cache_dir: Path | None,
    output_path_arg: str,
) -> None:
    """Prewarm long-term cache with sector light curves and cache keys."""
    out_path = resolve_optional_output_path(output_path_arg)

    if report_file is not None and toi is not None:
        click.echo(
            "Warning: --report-file provided; ignoring --toi and using report-file target inputs.",
            err=True,
        )

    resolved_tic_id, input_resolution, report_file_path, report_sectors_used = _resolve_tic_and_inputs(
        tic_id=tic_id,
        toi=toi,
        report_file=report_file,
        network_ok=bool(network_ok),
    )

    effective_sectors, sectors_explicit, sector_selection_source = choose_effective_sectors(
        sectors_arg=sectors,
        report_sectors_used=report_sectors_used,
    )
    normalized_flux_type = str(flux_type).lower()

    try:
        client = MASTClient(cache_dir=str(cache_dir) if cache_dir is not None else None)
        lightcurves, sector_load_path = _load_lightcurves_for_fetch(
            client=client,
            tic_id=int(resolved_tic_id),
            sectors=effective_sectors,
            sectors_explicit=bool(sectors_explicit),
            flux_type=normalized_flux_type,
            network_ok=bool(network_ok),
        )

        cache = PersistentCache(cache_dir=cache_dir)
        staged: list[tuple[int, str]] = []
        for lc_data in lightcurves:
            sector = int(getattr(lc_data, "sector"))
            key = make_data_ref(int(resolved_tic_id), int(sector), normalized_flux_type)
            cache.put(key, lc_data)
            staged.append((sector, key))
    except LightCurveNotFoundError as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_DATA_UNAVAILABLE) from exc
    except BtvCliError:
        raise
    except Exception as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_RUNTIME_ERROR) from exc

    staged_sorted = sorted(staged, key=lambda row: (int(row[0]), str(row[1])))
    cached_keys = [row[1] for row in staged_sorted]
    sectors_cached = sorted({int(row[0]) for row in staged_sorted})

    options = {
        "network_ok": bool(network_ok),
        "sectors": [int(s) for s in effective_sectors] if effective_sectors else None,
        "flux_type": normalized_flux_type,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
    }
    payload = {
        "schema_version": "cli.fetch.v1",
        "result": {
            "cached_keys": cached_keys,
            "sectors": sectors_cached,
        },
        "cached_keys": cached_keys,
        "sectors": sectors_cached,
        "inputs_summary": {
            "tic_id": int(resolved_tic_id),
            "toi": toi,
            "input_resolution": input_resolution,
            "sectors_used": sectors_cached,
        },
        "provenance": {
            "n_cached": len(cached_keys),
            "inputs_source": "report_file" if report_file_path is not None else str(input_resolution.get("source")),
            "report_file": report_file_path,
            "sector_selection_source": sector_selection_source,
            "sector_load_path": sector_load_path,
            "options": options,
        },
    }
    dump_json_output(payload, out_path)


__all__ = ["fetch_command"]
