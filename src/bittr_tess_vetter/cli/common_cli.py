"""Shared helpers for click-based `btv` commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

EXIT_OK = 0
EXIT_INPUT_ERROR = 1
EXIT_RUNTIME_ERROR = 2
EXIT_PROGRESS_ERROR = 3
EXIT_DATA_UNAVAILABLE = 4
EXIT_REMOTE_TIMEOUT = 5


class BtvCliError(click.ClickException):
    """Click exception with explicit exit-code control."""

    def __init__(self, message: str, *, exit_code: int = EXIT_INPUT_ERROR) -> None:
        super().__init__(message)
        self.exit_code = int(exit_code)


def parse_extra_params(items: tuple[str, ...]) -> dict[str, Any]:
    """Parse repeated `KEY=VALUE` pairs into a JSON-friendly dictionary."""
    parsed: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise BtvCliError(
                f"Invalid --extra-param '{item}'. Expected KEY=VALUE.",
                exit_code=EXIT_INPUT_ERROR,
            )
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise BtvCliError(
                f"Invalid --extra-param '{item}'. Key cannot be empty.",
                exit_code=EXIT_INPUT_ERROR,
            )

        value_text = raw_value.strip()
        if not value_text:
            parsed[key] = ""
            continue

        try:
            parsed[key] = json.loads(value_text)
        except json.JSONDecodeError:
            parsed[key] = value_text

    return parsed


def dump_json_output(payload: dict[str, Any], out_path: Path | None) -> None:
    """Write JSON payload to file or stdout."""
    text = json.dumps(payload, sort_keys=True, indent=2)
    if out_path is None:
        click.echo(text)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")


def load_json_file(path: Path, *, label: str) -> dict[str, Any]:
    """Load an object JSON file with user-facing errors."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise BtvCliError(f"{label} not found: {path}") from exc
    except OSError as exc:
        raise BtvCliError(f"Cannot read {label}: {exc}") from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BtvCliError(f"Malformed JSON in {label}: {exc}") from exc

    if not isinstance(payload, dict):
        raise BtvCliError(f"{label} must be a JSON object")
    return payload


def resolve_optional_output_path(output_arg: str | None) -> Path | None:
    """Map '-', empty, or None to stdout; otherwise return filesystem path."""
    if output_arg is None:
        return None
    value = str(output_arg).strip()
    if value in {"", "-"}:
        return None
    return Path(value)
