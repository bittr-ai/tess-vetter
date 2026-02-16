"""Built-in profile registry for pipeline compositions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bittr_tess_vetter.cli.common_cli import EXIT_INPUT_ERROR, BtvCliError
from bittr_tess_vetter.pipeline_composition.schema import (
    CompositionSpec,
    load_composition_file,
    validate_composition_payload,
)


def _profiles_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "pipeline_profiles"


def list_profiles() -> list[str]:
    names: list[str] = []
    for path in sorted(_profiles_dir().glob("*")):
        if path.suffix.lower() in {".json", ".yaml", ".yml"}:
            names.append(path.stem)
    return names


def _load_profile_payload(path: Path) -> dict[str, Any]:
    spec = load_composition_file(str(path))
    return dict(spec.raw)


def get_profile(profile_id: str) -> CompositionSpec:
    base = _profiles_dir()
    for suffix in (".json", ".yaml", ".yml"):
        candidate = base / f"{profile_id}{suffix}"
        if candidate.exists():
            return load_composition_file(str(candidate))
    available = ", ".join(list_profiles())
    raise BtvCliError(
        f"Unknown pipeline profile '{profile_id}'. Available: {available}",
        exit_code=EXIT_INPUT_ERROR,
    )


def validate_profile(profile_payload: dict[str, Any], *, source: str = "profile") -> CompositionSpec:
    return validate_composition_payload(profile_payload, source=source)


__all__ = ["get_profile", "list_profiles", "validate_profile"]
