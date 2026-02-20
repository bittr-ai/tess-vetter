"""`btv describe-checks` command."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import click


def _build_meta(*, output_format: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "format": output_format,
        "timestamp": datetime.now(UTC).isoformat(),
        "check_count": len(checks),
    }


@click.command("describe-checks")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output format.",
)
def describe_checks_command(output_format: str) -> None:
    """Describe available vetting checks and requirements."""
    from tess_vetter.api.pipeline import describe_checks, list_checks

    fmt = str(output_format).lower()
    checks = list_checks()
    meta = _build_meta(output_format=fmt, checks=checks)
    if fmt == "json":
        payload: dict[str, Any] = {
            "checks": checks,
            "meta": meta,
        }
        click.echo(json.dumps(payload, sort_keys=True, indent=2))
        return
    click.echo("meta:")
    click.echo(f"  format: {meta['format']}")
    click.echo(f"  timestamp: {meta['timestamp']}")
    click.echo(f"  check_count: {meta['check_count']}")
    click.echo("")
    click.echo(describe_checks())


__all__ = ["describe_checks_command"]
