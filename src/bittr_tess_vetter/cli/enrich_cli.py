"""CLI entrypoint for batch enrichment of TESS transit candidates.

This module provides the `btv enrich` command for processing worklists
of transit candidates and generating enriched JSONL output.

Usage:
    btv enrich --in worklist.jsonl --out enriched.jsonl [options]

Example:
    btv enrich -i candidates.jsonl -o enriched.jsonl --bulk --network-ok --limit 100
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import click

from bittr_tess_vetter.cli.activity_cli import activity_command
from bittr_tess_vetter.cli.common_cli import (
    EXIT_INPUT_ERROR,
    EXIT_RUNTIME_ERROR,
    BtvCliError,
)
from bittr_tess_vetter.cli.describe_checks_cli import describe_checks_command
from bittr_tess_vetter.cli.detrend_grid_cli import detrend_grid_command
from bittr_tess_vetter.cli.dilution_cli import dilution_command
from bittr_tess_vetter.cli.ephemeris_reliability_cli import ephemeris_reliability_command
from bittr_tess_vetter.cli.fpp_cli import fpp_command
from bittr_tess_vetter.cli.localize_cli import localize_command
from bittr_tess_vetter.cli.localize_host_cli import localize_host_command
from bittr_tess_vetter.cli.measure_sectors_cli import measure_sectors_command
from bittr_tess_vetter.cli.model_compete_cli import model_compete_command
from bittr_tess_vetter.cli.periodogram_cli import periodogram_command
from bittr_tess_vetter.cli.resolve_stellar_cli import resolve_stellar_command
from bittr_tess_vetter.cli.report_cli import report_command
from bittr_tess_vetter.cli.resolve_neighbors_cli import resolve_neighbors_command
from bittr_tess_vetter.cli.systematics_proxy_cli import systematics_proxy_command
from bittr_tess_vetter.cli.timing_cli import timing_command
from bittr_tess_vetter.cli.transit_fit_cli import fit_command
from bittr_tess_vetter.cli.vet_cli import vet_command
from bittr_tess_vetter.features import FeatureConfig


def _stream_worklist(input_path: Path) -> Iterator[dict[str, Any]]:
    """Stream-read a JSONL worklist file.

    Args:
        input_path: Path to the input JSONL worklist file.

    Yields:
        Candidate dictionaries from the worklist.

    Raises:
        click.ClickException: If the file cannot be read.
    """
    if not input_path.exists():
        raise click.ClickException(f"Input file not found: {input_path}")

    try:
        with input_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    click.echo(
                        f"Warning: Skipping malformed JSON at line {line_num}: {e}",
                        err=True,
                    )
    except OSError as e:
        raise click.ClickException(f"Cannot read input file: {e}") from e


@click.group()
@click.version_option(package_name="bittr-tess-vetter")
def cli() -> None:
    """bittr-tess-vetter CLI for TESS transit vetting."""
    pass


cli.add_command(vet_command)
cli.add_command(fpp_command)
cli.add_command(report_command)
cli.add_command(resolve_stellar_command)
cli.add_command(resolve_neighbors_command)
cli.add_command(describe_checks_command)
cli.add_command(measure_sectors_command)
cli.add_command(detrend_grid_command)
cli.add_command(localize_command)
cli.add_command(localize_host_command)
cli.add_command(fit_command)
cli.add_command(periodogram_command)
cli.add_command(model_compete_command)
cli.add_command(ephemeris_reliability_command)
cli.add_command(activity_command)
cli.add_command(timing_command)
cli.add_command(systematics_proxy_command)
cli.add_command(dilution_command)


@cli.command()
@click.option(
    "--in",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Input worklist JSONL path.",
)
@click.option(
    "--out",
    "-o",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output enriched JSONL path.",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Skip already-processed candidate_keys in output.",
)
@click.option(
    "--bulk",
    is_flag=True,
    default=False,
    help="Enable bulk mode (skip slow diagnostics).",
)
@click.option(
    "--network-ok",
    is_flag=True,
    default=False,
    help="Allow network calls (e.g., catalog lookups).",
)
@click.option(
    "--allow-20s",
    is_flag=True,
    default=False,
    help="Allow 20-second cadence fallback.",
)
@click.option(
    "--no-download",
    is_flag=True,
    default=False,
    help="Use cached data only.",
)
@click.option(
    "--require-tpf",
    is_flag=True,
    default=False,
    help="Fail items that cannot load TPF (for pixel-based checks).",
)
@click.option(
    "--t0-refine",
    is_flag=True,
    default=False,
    help="Enable T0 refinement.",
)
@click.option(
    "--limit",
    "-n",
    type=int,
    default=None,
    help="Process only first N candidates (for testing).",
)
@click.option(
    "--progress-interval",
    type=int,
    default=10,
    show_default=True,
    help="Write progress update every N candidates.",
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Optional lightkurve/MAST cache directory for downloads and reuse.",
)
@click.option(
    "--local-data-path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Path to local data directory (contains tic{id}/ subdirs with CSV files).",
)
def enrich(
    input_path: Path,
    output_path: Path,
    resume: bool,
    bulk: bool,
    network_ok: bool,
    allow_20s: bool,
    no_download: bool,
    require_tpf: bool,
    t0_refine: bool,
    limit: int | None,
    progress_interval: int,
    cache_dir: Path | None,
    local_data_path: Path | None,
) -> None:
    """Enrich a worklist of transit candidates with vetting features.

    Reads candidates from the input JSONL worklist, runs the vetting pipeline
    on each candidate, and writes enriched rows to the output JSONL file.

    Each input row must have: tic_id, period_days, t0_btjd, duration_hours.
    depth_ppm is optional (recommended for full feature coverage).

    Output rows contain the original ephemeris plus all extracted features.
    """
    from bittr_tess_vetter.pipeline import enrich_worklist

    # Build feature config from CLI flags
    config = FeatureConfig(
        bulk_mode=bulk,
        network_ok=network_ok,
        allow_20s=allow_20s,
        no_download=no_download,
        require_tpf=require_tpf,
        enable_t0_refine=t0_refine,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_data_path=str(local_data_path) if local_data_path else None,
    )

    click.echo(f"Input:  {input_path}")
    click.echo(f"Output: {output_path}")
    click.echo(f"Config: bulk={bulk}, network_ok={network_ok}, allow_20s={allow_20s}")
    if cache_dir:
        click.echo(f"Cache dir: {cache_dir}")
    if local_data_path:
        click.echo(f"Local data: {local_data_path}")
    if resume:
        click.echo("Resume mode: skipping already-processed candidates")
    if limit:
        click.echo(f"Limit: processing first {limit} candidates only")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    summary = enrich_worklist(
        worklist_iter=_stream_worklist(input_path),
        output_path=output_path,
        config=config,
        resume=resume,
        limit=limit,
        progress_interval=progress_interval,
    )
    click.echo(f"Done in {time.time() - start:.1f}s")

    # Print summary
    click.echo("\n" + "=" * 50)
    click.echo("Enrichment Summary")
    click.echo("=" * 50)
    click.echo(f"Total input:    {summary.total_input}")
    click.echo(f"Processed:      {summary.processed}")
    click.echo(f"Skipped resume: {summary.skipped_resume}")
    click.echo(f"Errors:         {summary.errors}")
    click.echo(f"Wall time:      {summary.wall_time_seconds:.1f}s")
    if summary.error_class_counts:
        click.echo("Error classes:")
        for cls, count in sorted(summary.error_class_counts.items()):
            click.echo(f"  {cls}: {count}")


def main() -> int:
    """Main entry point for the CLI."""
    try:
        argv = sys.argv[1:]
        if argv and argv[0].startswith("-") and argv[0] not in {"--help", "--version"}:
            argv = ["enrich", *argv]
        cli(args=argv, standalone_mode=False)
        return 0
    except BtvCliError as e:
        e.show()
        return e.exit_code
    except click.ClickException as e:
        e.show()
        return EXIT_INPUT_ERROR
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(main())
