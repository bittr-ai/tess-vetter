"""Custom-view JSON Pointer path parsing and allowlist checks."""

from __future__ import annotations

from typing import Any

_ALLOWED_ROOTS = ("summary", "plot_data")


def iter_custom_view_paths(custom_views_payload: dict[str, Any]) -> list[str]:
    """Return all data path references across authored custom views."""
    paths: list[str] = []
    for view in custom_views_payload.get("views", []):
        chart = view.get("chart")
        if not isinstance(chart, dict):
            continue
        series = chart.get("series")
        if not isinstance(series, list):
            continue
        for item in series:
            if not isinstance(item, dict):
                continue
            for key in ("x", "y", "y_err"):
                ref = item.get(key)
                if not isinstance(ref, dict):
                    continue
                path = ref.get("path")
                if isinstance(path, str):
                    paths.append(path)
    return paths


def is_allowed_custom_view_path(path: str) -> bool:
    """Accept only JSON Pointer paths rooted at /summary or /plot_data."""
    if not path or not path.startswith("/"):
        return False
    segments = path.split("/")
    if len(segments) < 2:
        return False
    if segments[0] != "":
        return False
    root = segments[1]
    if root not in _ALLOWED_ROOTS:
        return False
    return root != "payload_meta"


def resolve_custom_view_path(payload: dict[str, Any], path: str) -> Any:
    """Resolve an RFC 6901 JSON Pointer path against the report payload map."""
    if not path.startswith("/"):
        raise KeyError(path)

    def _unescape(token: str) -> str:
        return token.replace("~1", "/").replace("~0", "~")

    segments = [_unescape(token) for token in path.split("/")[1:]]
    current: Any = payload
    for segment in segments:
        if not isinstance(current, dict) or segment not in current:
            raise KeyError(path)
        current = current[segment]
    return current
