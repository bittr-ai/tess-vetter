from __future__ import annotations

import copy

import pytest

from tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from tess_vetter.report import build_report
from tess_vetter.report._custom_view_hash import custom_view_hashes_by_id, custom_views_hash


def _make_minimal_lc() -> LightCurve:
    return LightCurve(
        time=[0.0, 0.1, 0.2, 0.3],
        flux=[1.0, 0.999, 1.001, 1.0],
        flux_err=[0.0001, 0.0001, 0.0001, 0.0001],
    )


def _make_candidate() -> Candidate:
    return Candidate(
        ephemeris=Ephemeris(period_days=1.0, t0_btjd=0.0, duration_hours=1.0),
        depth_ppm=500.0,
    )


def test_custom_view_hashes_are_deterministic_and_sorted_by_view_id() -> None:
    report = build_report(_make_minimal_lc(), _make_candidate(), include_additional_plots=False)
    report.custom_views = {
        "version": "1",
        "views": [
            {
                "id": "b-view",
                "title": "B",
                "producer": {"source": "agent"},
                "mode": "ad_hoc",
                "chart": {
                    "type": "line",
                    "series": [
                        {
                            "x": {"path": "/plot_data/phase_folded/phase"},
                            "y": {"path": "/plot_data/phase_folded/flux"},
                        }
                    ],
                    "options": {},
                },
                "quality": {"min_points_required": 3, "status": "ok", "flags": []},
            },
            {
                "id": "a-view",
                "title": "A",
                "producer": {"source": "user"},
                "mode": "deterministic",
                "chart": {
                    "type": "scatter",
                    "series": [
                        {
                            "x": {"path": "/plot_data/phase_folded/phase"},
                            "y": {"path": "/plot_data/phase_folded/flux"},
                        }
                    ],
                    "options": {"x_range": [0.0, 1.0]},
                },
                "quality": {"min_points_required": 3, "status": "degraded", "flags": []},
            },
        ],
    }

    payload = report.to_json()
    custom_views_payload = payload["custom_views"]
    custom_views_meta = payload["payload_meta"]

    assert [view["id"] for view in custom_views_payload["views"]] == ["b-view", "a-view"]
    assert list(custom_views_meta["custom_view_hashes_by_id"].keys()) == ["a-view", "b-view"]
    assert custom_views_meta["custom_views_hash"] == custom_views_hash(custom_views_payload)
    assert custom_views_meta["custom_view_hashes_by_id"] == custom_view_hashes_by_id(
        custom_views_payload
    )
    assert custom_views_meta["custom_views_includes_ad_hoc"] is True

    reversed_payload = copy.deepcopy(custom_views_payload)
    reversed_payload["views"] = list(reversed(reversed_payload["views"]))
    assert custom_views_hash(reversed_payload) == custom_views_meta["custom_views_hash"]
    assert custom_view_hashes_by_id(reversed_payload) == custom_views_meta["custom_view_hashes_by_id"]


def test_custom_views_invalid_paths_degrade_to_unavailable() -> None:
    report = build_report(_make_minimal_lc(), _make_candidate(), include_additional_plots=False)
    report.custom_views = {
        "version": "1",
        "views": [
            {
                "id": "bad-path",
                "title": "Blocked",
                "producer": {"source": "agent"},
                "mode": "deterministic",
                "chart": {
                    "type": "line",
                    "series": [
                        {
                            "x": {"path": "/payload_meta/summary_hash"},
                            "y": {"path": "/plot_data/phase_folded/flux"},
                        }
                    ],
                    "options": {},
                },
                "quality": {"min_points_required": 3, "status": "ok", "flags": []},
            }
        ],
    }

    payload = report.to_json()
    view = payload["custom_views"]["views"][0]
    assert view["quality"]["status"] == "unavailable"
    assert "INVALID_PATH" in view["quality"]["flags"]


def test_custom_views_unresolved_allowed_paths_degrade_to_unavailable() -> None:
    report = build_report(_make_minimal_lc(), _make_candidate(), include_additional_plots=False)
    report.custom_views = {
        "version": "1",
        "views": [
            {
                "id": "missing-path",
                "title": "Missing",
                "producer": {"source": "agent"},
                "mode": "deterministic",
                "chart": {
                    "type": "line",
                    "series": [
                        {
                            "x": {"path": "/plot_data/not_a_real_key"},
                            "y": {"path": "/plot_data/phase_folded/flux"},
                        }
                    ],
                    "options": {},
                },
                "quality": {"min_points_required": 3, "status": "ok", "flags": []},
            }
        ],
    }

    payload = report.to_json()
    view = payload["custom_views"]["views"][0]
    assert view["quality"]["status"] == "unavailable"
    assert "UNRESOLVED_PATH" in view["quality"]["flags"]


def test_custom_views_reject_duplicate_ids() -> None:
    report = build_report(_make_minimal_lc(), _make_candidate(), include_additional_plots=False)
    report.custom_views = {
        "version": "1",
        "views": [
            {
                "id": "dup",
                "title": "A",
                "producer": {"source": "agent"},
                "mode": "deterministic",
                "chart": {
                    "type": "line",
                    "series": [
                        {
                            "x": {"path": "/plot_data/phase_folded/phase"},
                            "y": {"path": "/plot_data/phase_folded/flux"},
                        }
                    ],
                    "options": {},
                },
                "quality": {"min_points_required": 3, "status": "ok", "flags": []},
            },
            {
                "id": "dup",
                "title": "B",
                "producer": {"source": "user"},
                "mode": "ad_hoc",
                "chart": {
                    "type": "bar",
                    "series": [
                        {
                            "x": {"path": "/plot_data/phase_folded/phase"},
                            "y": {"path": "/plot_data/phase_folded/flux"},
                        }
                    ],
                    "options": {},
                },
                "quality": {"min_points_required": 3, "status": "ok", "flags": []},
            },
        ],
    }

    with pytest.raises(ValueError):
        report.to_json()


def test_custom_views_reject_non_finite_range_values() -> None:
    report = build_report(_make_minimal_lc(), _make_candidate(), include_additional_plots=False)
    report.custom_views = {
        "version": "1",
        "views": [
            {
                "id": "bad-range",
                "title": "Bad",
                "producer": {"source": "system"},
                "mode": "deterministic",
                "chart": {
                    "type": "histogram",
                    "series": [
                        {
                            "x": {"path": "/plot_data/phase_folded/flux"},
                            "y": {"path": "/plot_data/phase_folded/phase"},
                        }
                    ],
                    "options": {"x_range": [0.0, float("inf")]},
                },
                "quality": {"min_points_required": 3, "status": "ok", "flags": []},
            }
        ],
    }

    with pytest.raises(ValueError):
        report.to_json()
