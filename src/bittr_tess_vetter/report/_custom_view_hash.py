"""Deterministic custom-view hashing helpers."""

from __future__ import annotations

from typing import Any

from bittr_tess_vetter.report._serialization_utils import _canonical_sha256


def _sorted_custom_views_payload(custom_views_payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize custom view ordering by id before canonical serialization."""
    views = list(custom_views_payload.get("views", []))
    sorted_views = sorted(views, key=lambda view: str(view.get("id", "")))
    return {
        "version": custom_views_payload.get("version", ""),
        "views": sorted_views,
    }


def custom_view_hashes_by_id(custom_views_payload: dict[str, Any]) -> dict[str, str]:
    """Return deterministic SHA256 hashes keyed by custom view id."""
    sorted_payload = _sorted_custom_views_payload(custom_views_payload)
    result: dict[str, str] = {}
    for view in sorted_payload["views"]:
        view_id = str(view["id"])
        result[view_id] = _canonical_sha256(view)
    return result


def custom_views_hash(custom_views_payload: dict[str, Any]) -> str:
    """Return deterministic SHA256 hash for the full custom-views block."""
    return _canonical_sha256(_sorted_custom_views_payload(custom_views_payload))
