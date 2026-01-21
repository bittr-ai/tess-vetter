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

from bittr_tess_vetter.features import FeatureConfig


def _stream_existing_keys(output_path: Path) -> set[str]:
    """Stream-read existing output file to collect candidate keys.

    This function reads the output file line-by-line to build a set of
    already-processed candidate keys without loading the entire file
    into memory.

    Args:
        output_path: Path to the existing output JSONL file.

    Returns:
        Set of candidate_key strings already in the output file.
    """
    existing_keys: set[str] = set()
    if not output_path.exists():
        return existing_keys

    try:
        with output_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if "candidate_key" in row:
                        existing_keys.add(row["candidate_key"])
                except json.JSONDecodeError:
                    click.echo(
                        f"Warning: Skipping malformed JSON at line {line_num} in {output_path}",
                        err=True,
                    )
    except OSError as e:
        click.echo(f"Warning: Could not read {output_path}: {e}", err=True)

    return existing_keys


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


def _write_enriched_row(output_path: Path, row: dict[str, Any]) -> None:
    """Append an enriched row to the output file with file locking.

    Uses filelock for concurrent-safe append operations.

    Args:
        output_path: Path to the output JSONL file.
        row: Enriched row dictionary to write.
    """
    from filelock import FileLock

    lock_path = output_path.with_suffix(output_path.suffix + ".lock")
    lock = FileLock(str(lock_path), timeout=30)

    with lock, output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, separators=(",", ":")))
        f.write("\n")


def _make_candidate_key(row: dict[str, Any]) -> str:
    """Generate candidate key from a worklist row.

    Args:
        row: Candidate dictionary with tic_id, period_days, t0_btjd.

    Returns:
        Candidate key in format "tic_id|period_days|t0_btjd".
    """
    return f"{row['tic_id']}|{row['period_days']}|{row['t0_btjd']}"


@click.group()
@click.version_option(package_name="bittr-tess-vetter")
def cli() -> None:
    """bittr-tess-vetter CLI for TESS transit vetting."""
    pass


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
    t0_refine: bool,
    limit: int | None,
    local_data_path: Path | None,
) -> None:
    """Enrich a worklist of transit candidates with vetting features.

    Reads candidates from the input JSONL worklist, runs the vetting pipeline
    on each candidate, and writes enriched rows to the output JSONL file.

    Each input row must have: tic_id, period_days, t0_btjd, duration_hours, depth_ppm.

    Output rows contain the original ephemeris plus all extracted features.
    """
    from bittr_tess_vetter.pipeline import EnrichmentSummary, enrich_candidate

    # Build feature config from CLI flags
    config = FeatureConfig(
        bulk_mode=bulk,
        network_ok=network_ok,
        allow_20s=allow_20s,
        no_download=no_download,
        enable_t0_refine=t0_refine,
        local_data_path=str(local_data_path) if local_data_path else None,
    )

    click.echo(f"Input:  {input_path}")
    click.echo(f"Output: {output_path}")
    click.echo(f"Config: bulk={bulk}, network_ok={network_ok}, allow_20s={allow_20s}")
    if local_data_path:
        click.echo(f"Local data: {local_data_path}")
    if resume:
        click.echo("Resume mode: skipping already-processed candidates")
    if limit:
        click.echo(f"Limit: processing first {limit} candidates only")

    # Build skip set if resuming
    skip_keys: set[str] = set()
    if resume:
        click.echo("Scanning existing output for resume...")
        skip_keys = _stream_existing_keys(output_path)
        click.echo(f"Found {len(skip_keys)} existing candidate keys to skip")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process worklist
    start_time = time.time()
    total_input = 0
    processed = 0
    skipped_resume = 0
    errors = 0
    error_class_counts: dict[str, int] = {}
    progress_interval = 10

    for row in _stream_worklist(input_path):
        total_input += 1

        # Check limit
        if limit and (processed + skipped_resume) >= limit:
            click.echo(f"Reached limit of {limit} candidates")
            break

        # Extract required fields
        try:
            tic_id = int(row["tic_id"])
            period_days = float(row["period_days"])
            t0_btjd = float(row["t0_btjd"])
            duration_hours = float(row["duration_hours"])
            depth_ppm = float(row["depth_ppm"])
        except (KeyError, ValueError, TypeError) as e:
            click.echo(
                f"Warning: Skipping row {total_input} due to missing/invalid field: {e}", err=True
            )
            errors += 1
            error_class = type(e).__name__
            error_class_counts[error_class] = error_class_counts.get(error_class, 0) + 1
            continue

        # Generate candidate key
        candidate_key = _make_candidate_key(row)

        # Skip if already processed (resume mode)
        if candidate_key in skip_keys:
            skipped_resume += 1
            continue

        toi = row.get("toi")
        if toi is not None:
            toi = str(toi)

        # Enrich candidate
        try:
            _raw_evidence, enriched_row = enrich_candidate(
                tic_id=tic_id,
                toi=toi,
                period_days=period_days,
                t0_btjd=t0_btjd,
                duration_hours=duration_hours,
                depth_ppm=depth_ppm,
                config=config,
            )
            # Write enriched row (type is EnrichedRow which is a TypedDict)
            _write_enriched_row(output_path, dict(enriched_row))
            processed += 1
        except NotImplementedError:
            # Expected during stub phase - write placeholder row
            placeholder = {
                "candidate_key": candidate_key,
                "tic_id": tic_id,
                "period_days": period_days,
                "t0_btjd": t0_btjd,
                "duration_hours": duration_hours,
                "depth_ppm": depth_ppm,
                "status": "ERROR",
                "error_class": "NotImplementedError",
                "error": "enrich_candidate is not yet implemented",
            }
            _write_enriched_row(output_path, placeholder)
            errors += 1
            error_class_counts["NotImplementedError"] = (
                error_class_counts.get("NotImplementedError", 0) + 1
            )
        except Exception as e:
            # Log error and continue
            error_class = type(e).__name__
            error_class_counts[error_class] = error_class_counts.get(error_class, 0) + 1
            errors += 1
            click.echo(f"Error processing {candidate_key}: {error_class}: {e}", err=True)
            # Write error row
            error_row = {
                "candidate_key": candidate_key,
                "tic_id": tic_id,
                "period_days": period_days,
                "t0_btjd": t0_btjd,
                "duration_hours": duration_hours,
                "depth_ppm": depth_ppm,
                "status": "ERROR",
                "error_class": error_class,
                "error": str(e),
            }
            _write_enriched_row(output_path, error_row)

        # Progress update
        count = processed + errors + skipped_resume
        if count % progress_interval == 0:
            elapsed = time.time() - start_time
            rate = count / elapsed if elapsed > 0 else 0
            click.echo(f"Progress: {count} candidates ({rate:.1f}/s)")

    wall_time = time.time() - start_time

    # Build summary
    summary = EnrichmentSummary(
        total_input=total_input,
        processed=processed,
        skipped_resume=skipped_resume,
        errors=errors,
        wall_time_seconds=wall_time,
        error_class_counts=error_class_counts,
    )

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
        cli()
        return 0
    except click.ClickException as e:
        e.show()
        return 1
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
