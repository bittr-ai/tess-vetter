"""`btv pipeline` command group for composable workflow execution."""

from __future__ import annotations

from pathlib import Path

import click

from bittr_tess_vetter.cli.common_cli import EXIT_INPUT_ERROR, BtvCliError
from bittr_tess_vetter.pipeline_composition.executor import run_composition
from bittr_tess_vetter.pipeline_composition.registry import get_profile, list_profiles
from bittr_tess_vetter.pipeline_composition.schema import load_composition_file


@click.group("pipeline")
def pipeline_group() -> None:
    """Run composable multi-step vetting pipelines."""


@pipeline_group.command("run")
@click.option("--profile", type=str, default=None, help="Built-in profile id.")
@click.option(
    "--composition-file",
    type=str,
    default=None,
    help="Composition JSON/YAML file path; use '-' for stdin.",
)
@click.option("--toi", "tois", multiple=True, required=True, help="TOI identifier(s). Repeatable.")
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Output directory for run artifacts.",
)
@click.option(
    "--network-ok/--no-network",
    default=False,
    show_default=True,
    help="Allow network-dependent command paths.",
)
@click.option(
    "--continue-on-error/--fail-fast",
    default=False,
    show_default=True,
    help="Continue remaining steps/TOIs after step errors.",
)
@click.option("--max-workers", type=int, default=4, show_default=True)
@click.option("--resume", is_flag=True, default=False, help="Reuse completed step checkpoints.")
def pipeline_run_command(
    profile: str | None,
    composition_file: str | None,
    tois: tuple[str, ...],
    out_dir: Path,
    network_ok: bool,
    continue_on_error: bool,
    max_workers: int,
    resume: bool,
) -> None:
    """Execute a built-in profile or injected composition for explicit TOIs."""
    if (profile is None and composition_file is None) or (profile is not None and composition_file is not None):
        raise BtvCliError(
            "Provide exactly one of --profile or --composition-file.",
            exit_code=EXIT_INPUT_ERROR,
        )

    if profile is not None:
        composition = get_profile(profile)
    else:
        assert composition_file is not None
        composition = load_composition_file(composition_file)

    result = run_composition(
        composition=composition,
        tois=[str(x) for x in tois],
        out_dir=Path(out_dir),
        network_ok=bool(network_ok),
        continue_on_error=bool(continue_on_error),
        max_workers=int(max_workers),
        resume=bool(resume),
    )
    manifest = result["manifest"]
    click.echo(
        (
            f"Pipeline run complete: n_tois={manifest['counts']['n_tois']} "
            f"ok={manifest['counts']['n_ok']} partial={manifest['counts']['n_partial']} "
            f"failed={manifest['counts']['n_failed']} out_dir={out_dir}"
        )
    )


@pipeline_group.command("profiles")
def pipeline_profiles_command() -> None:
    """List built-in pipeline profiles."""
    names = list_profiles()
    for name in names:
        click.echo(name)


__all__ = ["pipeline_group", "pipeline_profiles_command", "pipeline_run_command"]
