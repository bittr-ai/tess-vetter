"""`btv doctor` command for environment and dependency preflight checks."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from importlib import metadata
from typing import Any

import click

from tess_vetter.cli.common_cli import EXIT_DATA_UNAVAILABLE, BtvCliError


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    ok: bool
    detail: str
    remediation: str | None = None


_PROFILE_DEPENDENCIES: dict[str, list[tuple[str, str, str]]] = {
    "vet": [
        (
            "lightkurve",
            "`lightkurve` is available for vetting flows that query/load light curves.",
            "pip install lightkurve",
        ),
    ],
    "detection": [
        (
            "transitleastsquares",
            "`transitleastsquares` is available for TLS period search.",
            "pip install 'tess-vetter[tls]'",
        ),
        (
            "wotan",
            "`wotan` is available for advanced detrending.",
            "pip install 'tess-vetter[wotan]'",
        ),
    ],
    "fpp": [
        (
            "triceratops",
            "`triceratops` is available for FPP workflows.",
            "pip install 'tess-vetter[triceratops]'",
        ),
        (
            "lightkurve",
            "`lightkurve` is available for FPP light-curve access.",
            "pip install 'tess-vetter[triceratops]'",
        ),
    ],
}
_PROFILE_DEPENDENCIES["full"] = sorted(
    {dep for deps in _PROFILE_DEPENDENCIES.values() for dep in deps},
    key=lambda item: item[0],
)


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _script_available(script_name: str) -> bool:
    try:
        entries = metadata.entry_points(group="console_scripts")
    except Exception:
        return False
    return any(ep.name == script_name for ep in entries)


def _run_doctor_checks(profile: str) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = [
        DoctorCheck(
            name="console_script:btv",
            ok=_script_available("btv"),
            detail="`btv` console script is installed.",
            remediation="Reinstall package: pip install -U tess-vetter",
        ),
        DoctorCheck(
            name="console_script:tess-vetter",
            ok=_script_available("tess-vetter"),
            detail="`tess-vetter` console script alias is installed.",
            remediation="Upgrade package: pip install -U tess-vetter",
        ),
        DoctorCheck(
            name="module:tess_vetter",
            ok=_module_available("tess_vetter"),
            detail="Python module `tess_vetter` is importable.",
            remediation="Reinstall package into active environment: pip install -U tess-vetter",
        ),
    ]
    for module_name, detail, remediation in _PROFILE_DEPENDENCIES[profile]:
        checks.append(
            DoctorCheck(
                name=f"module:{module_name}",
                ok=_module_available(module_name),
                detail=detail,
                remediation=f"Install runtime dependency: {remediation}",
            )
        )
    return checks


@click.command("doctor")
@click.option(
    "--profile",
    type=click.Choice(["vet", "detection", "fpp", "full"], case_sensitive=False),
    default="vet",
    show_default=True,
    help="Dependency preflight profile.",
)
@click.option("--json", "json_output", is_flag=True, default=False, help="Emit JSON output.")
def doctor_command(profile: str, json_output: bool) -> None:
    """Run CLI and runtime dependency preflight checks."""
    normalized_profile = str(profile).strip().lower()
    checks = _run_doctor_checks(normalized_profile)
    ready = all(check.ok for check in checks)

    if json_output:
        payload: dict[str, Any] = {
            "profile": normalized_profile,
            "ready": ready,
            "checks": [
                {
                    "name": check.name,
                    "ok": check.ok,
                    "detail": check.detail,
                    "remediation": check.remediation,
                }
                for check in checks
            ],
        }
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        click.echo(f"Doctor profile: {normalized_profile}")
        for check in checks:
            status = "OK" if check.ok else "MISSING"
            click.echo(f"[{status}] {check.name} - {check.detail}")
            if not check.ok and check.remediation:
                click.echo(f"       fix: {check.remediation}")

    if not ready:
        raise BtvCliError(
            "Environment is not ready. Fix missing dependencies and re-run `btv doctor`.",
            exit_code=EXIT_DATA_UNAVAILABLE,
        )
