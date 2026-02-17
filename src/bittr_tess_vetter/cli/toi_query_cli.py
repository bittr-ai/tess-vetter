from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import click

from bittr_tess_vetter.cli.common_cli import resolve_optional_output_path
from bittr_tess_vetter.platform.catalogs.exofop_toi_table import (
    _DISPOSITION_ALIASES,
    _NUMERIC_FIELD_ALIASES,
    fetch_exofop_toi_table,
    query_exofop_toi_rows,
)


def _parse_multi_csv(values: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for value in values:
        for part in str(value).split(","):
            text = part.strip()
            if text:
                out.append(text)
    return out


def _normalize_requested_columns(
    *,
    requested_columns: list[str],
    headers: list[str],
    rows: list[dict[str, str]],
) -> list[str]:
    if requested_columns:
        return requested_columns
    if headers:
        return list(headers)
    discovered: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            discovered.append(key)
    return discovered


def _write_text_output(*, text: str, out_path: Path | None) -> None:
    if out_path is None:
        click.echo(text, nl=False)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


@click.command(name="toi-query")
@click.option("--radius-min", type=float, default=None)
@click.option("--radius-max", type=float, default=None)
@click.option("--teff-min", type=float, default=None)
@click.option("--teff-max", type=float, default=None)
@click.option("--snr-min", type=float, default=None)
@click.option("--snr-max", type=float, default=None)
@click.option("--tmag-min", type=float, default=None)
@click.option("--tmag-max", type=float, default=None)
@click.option("--period-min", type=float, default=None)
@click.option("--period-max", type=float, default=None)
@click.option("--depth-min", type=float, default=None)
@click.option("--depth-max", type=float, default=None)
@click.option("--duration-min", type=float, default=None)
@click.option("--duration-max", type=float, default=None)
@click.option(
    "--disposition",
    "include_dispositions_raw",
    multiple=True,
    help="Disposition(s) to include (repeat and/or comma-separate).",
)
@click.option(
    "--exclude-disposition",
    "exclude_dispositions_raw",
    multiple=True,
    help="Disposition(s) to exclude (repeat and/or comma-separate).",
)
@click.option(
    "--exclude-known-planets",
    is_flag=True,
    default=False,
    help="Exclude known planets based on TOI disposition.",
)
@click.option(
    "--exclude-false-positives",
    is_flag=True,
    default=False,
    help="Exclude false positives based on TOI disposition.",
)
@click.option("--sort-by", type=str, default=None, help="Sort key (e.g., period, snr, toi).")
@click.option("--sort-descending", is_flag=True, default=False)
@click.option("--max-results", type=int, default=None)
@click.option("--format", "output_format", type=click.Choice(["json", "csv"]), default="json")
@click.option(
    "--columns",
    "columns_raw",
    multiple=True,
    help="Columns to include (repeat and/or comma-separate).",
)
@click.option(
    "--cache-ttl",
    type=int,
    default=6 * 3600,
    show_default=True,
    help="ExoFOP TOI table cache TTL in seconds.",
)
@click.option(
    "-o",
    "--output",
    "output_path_arg",
    type=str,
    default=None,
    help="Output path (default: stdout). Use '-' for stdout.",
)
def toi_query_command(
    radius_min: float | None,
    radius_max: float | None,
    teff_min: float | None,
    teff_max: float | None,
    snr_min: float | None,
    snr_max: float | None,
    tmag_min: float | None,
    tmag_max: float | None,
    period_min: float | None,
    period_max: float | None,
    depth_min: float | None,
    depth_max: float | None,
    duration_min: float | None,
    duration_max: float | None,
    include_dispositions_raw: tuple[str, ...],
    exclude_dispositions_raw: tuple[str, ...],
    exclude_known_planets: bool,
    exclude_false_positives: bool,
    sort_by: str | None,
    sort_descending: bool,
    max_results: int | None,
    output_format: str,
    columns_raw: tuple[str, ...],
    cache_ttl: int,
    output_path_arg: str | None,
) -> None:
    """Query ExoFOP TOI table with range/disposition filters."""
    if max_results is not None and int(max_results) < 0:
        raise click.ClickException("--max-results must be >= 0")
    if int(cache_ttl) < 0:
        raise click.ClickException("--cache-ttl must be >= 0")

    include_dispositions = set(_parse_multi_csv(include_dispositions_raw))
    exclude_dispositions = set(_parse_multi_csv(exclude_dispositions_raw))
    selected_columns = _parse_multi_csv(columns_raw)
    out_path = resolve_optional_output_path(output_path_arg)

    table = fetch_exofop_toi_table(cache_ttl_seconds=int(cache_ttl))
    query_result = query_exofop_toi_rows(
        table,
        radius_min=radius_min,
        radius_max=radius_max,
        teff_min=teff_min,
        teff_max=teff_max,
        snr_min=snr_min,
        snr_max=snr_max,
        tmag_min=tmag_min,
        tmag_max=tmag_max,
        period_min=period_min,
        period_max=period_max,
        depth_min=depth_min,
        depth_max=depth_max,
        duration_min=duration_min,
        duration_max=duration_max,
        include_dispositions=include_dispositions,
        exclude_dispositions=exclude_dispositions,
        exclude_known_planets=exclude_known_planets,
        exclude_false_positives=exclude_false_positives,
        sort_by=sort_by,
        sort_descending=sort_descending,
        max_results=max_results,
    )
    columns = _normalize_requested_columns(
        requested_columns=selected_columns,
        headers=table.headers,
        rows=query_result.rows,
    )
    projected_rows = [{column: str(row.get(column, "")) for column in columns} for row in query_result.rows]

    query_payload: dict[str, Any] = {
        "numeric_ranges": {
            "radius": {"min": radius_min, "max": radius_max},
            "teff": {"min": teff_min, "max": teff_max},
            "snr": {"min": snr_min, "max": snr_max},
            "tmag": {"min": tmag_min, "max": tmag_max},
            "period": {"min": period_min, "max": period_max},
            "depth": {"min": depth_min, "max": depth_max},
            "duration": {"min": duration_min, "max": duration_max},
        },
        "include_dispositions": sorted(include_dispositions),
        "exclude_dispositions": sorted(exclude_dispositions),
        "exclude_known_planets": bool(exclude_known_planets),
        "exclude_false_positives": bool(exclude_false_positives),
        "sort_by": sort_by,
        "sort_descending": bool(sort_descending),
        "max_results": max_results,
        "columns": columns,
    }
    source_stats = {
        "fetched_at_unix": table.fetched_at_unix,
        "headers_count": len(table.headers),
        "source_rows": query_result.stats.source_rows,
        "matched_rows_before_limit": query_result.stats.matched_rows_before_limit,
        "returned_rows": query_result.stats.returned_rows,
        "skipped_non_numeric_rows": query_result.stats.skipped_non_numeric_rows,
        "filtered_by_disposition_rows": query_result.stats.filtered_by_disposition_rows,
        "disposition_keys_checked": list(_DISPOSITION_ALIASES),
        "numeric_field_aliases": {key: list(val) for key, val in _NUMERIC_FIELD_ALIASES.items()},
    }

    if output_format == "json":
        payload = {
            "schema_version": "cli.toi_query.v1",
            "query": query_payload,
            "source_stats": source_stats,
            "results": projected_rows,
        }
        _write_text_output(text=json.dumps(payload, sort_keys=True, indent=2) + "\n", out_path=out_path)
        return

    lines: list[str] = []
    # DictWriter expects a file-like object; keep this tiny/portable with a shim.
    class _Sink(list[str]):
        def write(self, value: str) -> int:
            self.append(value)
            return len(value)

    sink = _Sink()
    writer = csv.DictWriter(sink, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in projected_rows:
        writer.writerow(row)
    lines.extend(sink)
    _write_text_output(text="".join(lines), out_path=out_path)
