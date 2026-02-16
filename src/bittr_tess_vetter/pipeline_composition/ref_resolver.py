"""Reference and port resolution for pipeline compositions."""

from __future__ import annotations

import re
from typing import Any

from bittr_tess_vetter.cli.common_cli import EXIT_INPUT_ERROR, BtvCliError

_JSON_PATH_TOKEN = re.compile(r"([^[.]+)|(\[(\d+)\])")


def _extract_json_path(data: Any, path: str) -> Any:
    raw = path.strip()
    if raw.startswith("steps."):
        raw = raw[len("steps.") :]
    if raw.startswith("$."):
        raw = raw[2:]
    elif raw.startswith("$"):
        raw = raw[1:]
    current = data
    for part in raw.split("."):
        if part == "":
            continue
        segment = part
        for match in _JSON_PATH_TOKEN.finditer(segment):
            key = match.group(1)
            idx = match.group(3)
            if key is not None:
                if not isinstance(current, dict) or key not in current:
                    raise KeyError(f"Missing key '{key}'")
                current = current[key]
            elif idx is not None:
                i = int(idx)
                if not isinstance(current, list) or i < 0 or i >= len(current):
                    raise KeyError(f"Missing list index [{i}]")
                current = current[i]
    return current


def resolve_value(
    value: Any,
    *,
    step_outputs: dict[str, dict[str, Any]],
    step_ports: dict[str, dict[str, Any]],
) -> Any:
    if isinstance(value, dict):
        if "port" in value and len(value) == 1:
            port_ref = str(value.get("port") or "").strip()
            if "." not in port_ref:
                raise BtvCliError(
                    f"Invalid port reference '{port_ref}'. Expected '<step_id>.<port_name>'.",
                    exit_code=EXIT_INPUT_ERROR,
                )
            step_id, port_name = port_ref.split(".", 1)
            if step_id not in step_ports or port_name not in step_ports[step_id]:
                raise BtvCliError(
                    f"Unknown port reference '{port_ref}'.",
                    exit_code=EXIT_INPUT_ERROR,
                )
            return step_ports[step_id][port_name]

        if "report_from" in value and len(value) == 1:
            step_id = str(value.get("report_from") or "").strip()
            payload = step_outputs.get(step_id, {})
            output_path = payload.get("_step_output_path")
            if not output_path:
                raise BtvCliError(
                    f"report_from references unknown step '{step_id}'.",
                    exit_code=EXIT_INPUT_ERROR,
                )
            return str(output_path)

        if "$ref" in value and len(value) == 1:
            ref = str(value.get("$ref") or "").strip()
            if not ref.startswith("steps."):
                raise BtvCliError(
                    f"Unsupported $ref '{ref}'. Expected prefix 'steps.'.",
                    exit_code=EXIT_INPUT_ERROR,
                )
            parts = ref.split(".", 2)
            if len(parts) < 3:
                raise BtvCliError(
                    f"Invalid $ref '{ref}'.",
                    exit_code=EXIT_INPUT_ERROR,
                )
            step_id = parts[1]
            subpath = parts[2]
            if step_id not in step_outputs:
                raise BtvCliError(
                    f"$ref references unknown step '{step_id}'.",
                    exit_code=EXIT_INPUT_ERROR,
                )
            try:
                return _extract_json_path(step_outputs[step_id], subpath)
            except KeyError as exc:
                raise BtvCliError(
                    f"$ref '{ref}' failed: {exc}",
                    exit_code=EXIT_INPUT_ERROR,
                ) from exc

        return {
            k: resolve_value(v, step_outputs=step_outputs, step_ports=step_ports)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [resolve_value(v, step_outputs=step_outputs, step_ports=step_ports) for v in value]

    return value


def extract_path(data: Any, path: str) -> Any:
    return _extract_json_path(data, path)


__all__ = ["extract_path", "resolve_value"]
