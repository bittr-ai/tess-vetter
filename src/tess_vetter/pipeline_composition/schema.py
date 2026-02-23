"""Schema and loading helpers for workflow pipeline compositions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tess_vetter.cli.common_cli import EXIT_INPUT_ERROR, BtvCliError

SCHEMA_VERSION = "pipeline.composition.v1"


@dataclass(frozen=True)
class StepSpec:
    id: str
    op: str
    inputs: dict[str, Any]
    ports: dict[str, Any]
    outputs: dict[str, Any]
    on_error: str


@dataclass(frozen=True)
class CompositionSpec:
    schema_version: str
    id: str
    description: str
    defaults: dict[str, Any]
    steps: list[StepSpec]
    final_mapping: dict[str, Any]
    raw: dict[str, Any]


def _normalize_op_name(name: str) -> str:
    return str(name).strip().replace("-", "_")


def _load_text(path: str) -> str:
    if path == "-":
        import sys

        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def _parse_composition_text(text: str, *, source: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise BtvCliError(
                f"Invalid JSON for composition {source}. YAML requires PyYAML to be installed.",
                exit_code=EXIT_INPUT_ERROR,
            ) from exc
        try:
            loaded = yaml.safe_load(text)
        except Exception as exc:
            raise BtvCliError(
                f"Invalid YAML for composition {source}: {exc}",
                exit_code=EXIT_INPUT_ERROR,
            ) from exc
        if not isinstance(loaded, dict):
            raise BtvCliError(
                f"Composition {source} must parse to an object.",
                exit_code=EXIT_INPUT_ERROR,
            ) from None
        return dict(loaded)
    if not isinstance(payload, dict):
        raise BtvCliError(
            f"Composition {source} must parse to an object.",
            exit_code=EXIT_INPUT_ERROR,
        )
    return dict(payload)


def validate_composition_payload(payload: dict[str, Any], *, source: str = "payload") -> CompositionSpec:
    schema_version = str(payload.get("schema_version") or "")
    if schema_version != SCHEMA_VERSION:
        raise BtvCliError(
            f"Composition {source} schema_version must be '{SCHEMA_VERSION}'.",
            exit_code=EXIT_INPUT_ERROR,
        )

    comp_id = str(payload.get("id") or "").strip()
    if not comp_id:
        raise BtvCliError(f"Composition {source} requires non-empty id.", exit_code=EXIT_INPUT_ERROR)

    description = str(payload.get("description") or "").strip()
    defaults_raw = payload.get("defaults")
    defaults = dict(defaults_raw) if isinstance(defaults_raw, dict) else {}

    steps_raw = payload.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        raise BtvCliError(
            f"Composition {source} requires a non-empty steps list.",
            exit_code=EXIT_INPUT_ERROR,
        )

    step_ids: set[str] = set()
    steps: list[StepSpec] = []
    for idx, step_raw in enumerate(steps_raw):
        if not isinstance(step_raw, dict):
            raise BtvCliError(
                f"Composition {source} steps[{idx}] must be an object.",
                exit_code=EXIT_INPUT_ERROR,
            )
        step_id = str(step_raw.get("id") or "").strip()
        if not step_id:
            raise BtvCliError(
                f"Composition {source} steps[{idx}] requires non-empty id.",
                exit_code=EXIT_INPUT_ERROR,
            )
        if step_id in step_ids:
            raise BtvCliError(
                f"Composition {source} contains duplicate step id '{step_id}'.",
                exit_code=EXIT_INPUT_ERROR,
            )
        step_ids.add(step_id)

        op = _normalize_op_name(str(step_raw.get("op") or ""))
        if not op:
            raise BtvCliError(
                f"Composition {source} steps[{idx}] requires non-empty op.",
                exit_code=EXIT_INPUT_ERROR,
            )

        inputs_raw = step_raw.get("inputs")
        inputs = dict(inputs_raw) if isinstance(inputs_raw, dict) else {}
        ports_raw = step_raw.get("ports")
        ports = dict(ports_raw) if isinstance(ports_raw, dict) else {}
        outputs_raw = step_raw.get("outputs")
        outputs = dict(outputs_raw) if isinstance(outputs_raw, dict) else {}
        on_error = str(step_raw.get("on_error") or "fail").strip().lower()
        if on_error not in {"fail", "continue"}:
            raise BtvCliError(
                f"Composition {source} step '{step_id}' on_error must be 'fail' or 'continue'.",
                exit_code=EXIT_INPUT_ERROR,
            )
        steps.append(
            StepSpec(
                id=step_id,
                op=op,
                inputs=inputs,
                ports=ports,
                outputs=outputs,
                on_error=on_error,
            )
        )

    final_mapping_raw = payload.get("final_mapping")
    final_mapping = dict(final_mapping_raw) if isinstance(final_mapping_raw, dict) else {}

    return CompositionSpec(
        schema_version=schema_version,
        id=comp_id,
        description=description,
        defaults=defaults,
        steps=steps,
        final_mapping=final_mapping,
        raw=payload,
    )


def load_composition_file(path: str) -> CompositionSpec:
    text = _load_text(path)
    payload = _parse_composition_text(text, source=path)
    return validate_composition_payload(payload, source=path)


def composition_digest(comp: CompositionSpec) -> str:
    import hashlib

    canonical = json.dumps(comp.raw, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "SCHEMA_VERSION",
    "CompositionSpec",
    "StepSpec",
    "composition_digest",
    "load_composition_file",
    "validate_composition_payload",
]
