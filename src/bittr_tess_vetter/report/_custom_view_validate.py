"""Validation helpers for custom-view contract enforcement."""

from __future__ import annotations

from typing import Any

from bittr_tess_vetter.report._custom_view_paths import (
    is_allowed_custom_view_path,
    iter_custom_view_paths,
    resolve_custom_view_path,
)
from bittr_tess_vetter.report.custom_views_schema import CustomViewsModel


def validate_custom_views_payload(
    custom_views: dict[str, Any], *, summary: dict[str, Any], plot_data: dict[str, Any]
) -> CustomViewsModel:
    """Validate custom views and degrade path-invalid entries to unavailable."""
    model = CustomViewsModel.model_validate(custom_views)
    payload_scope = {
        "summary": summary,
        "plot_data": plot_data,
    }
    custom_views_payload = model.model_dump(exclude_none=True)
    for view in custom_views_payload.get("views", []):
        quality = view.setdefault("quality", {})
        flags = quality.setdefault("flags", [])
        has_path_issue = False
        for path in iter_custom_view_paths({"views": [view]}):
            if not is_allowed_custom_view_path(path):
                has_path_issue = True
                flags.append("INVALID_PATH")
                continue
            try:
                resolve_custom_view_path(payload_scope, path)
            except KeyError:
                has_path_issue = True
                flags.append("UNRESOLVED_PATH")
        if has_path_issue:
            quality["status"] = "unavailable"
            quality["flags"] = sorted({str(flag) for flag in flags})
    return CustomViewsModel.model_validate(custom_views_payload)
