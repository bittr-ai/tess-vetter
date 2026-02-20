from __future__ import annotations

from tess_vetter.features.aggregates.host import build_host_plausibility_summary


def test_host_feasible_best_selects_lowest_depth_correction_among_feasible() -> None:
    out = build_host_plausibility_summary(
        {
            "requires_resolved_followup": True,
            "physically_impossible_source_ids": ["gaia:3"],
            "rationale": "demo",
            "scenarios": [
                {
                    "source_id": "gaia:1",
                    "flux_fraction": 0.7,
                    "true_depth_ppm": 300.0,
                    "depth_correction_factor": 1.5,
                    "physically_impossible": False,
                },
                {
                    "source_id": "gaia:2",
                    "flux_fraction": 0.2,
                    "true_depth_ppm": 900.0,
                    "depth_correction_factor": 1.2,
                    "physically_impossible": False,
                },
                {
                    "source_id": "gaia:3",
                    "flux_fraction": 0.1,
                    "true_depth_ppm": 2000.0,
                    "depth_correction_factor": 1.1,
                    "physically_impossible": False,
                },
            ],
        }
    )
    assert out["host_requires_resolved_followup"] is True
    assert out["host_physically_impossible_count"] == 1
    assert out["host_feasible_best_source_id"] == "gaia:2"
    assert out["host_feasible_best_flux_fraction"] == 0.2
    assert out["host_feasible_best_true_depth_ppm"] == 900.0


def test_host_feasible_best_respects_per_scenario_impossible_flag() -> None:
    out = build_host_plausibility_summary(
        {
            "physically_impossible_source_ids": [],
            "scenarios": [
                {
                    "source_id": "gaia:1",
                    "depth_correction_factor": 1.1,
                    "physically_impossible": True,
                },
                {
                    "source_id": "gaia:2",
                    "depth_correction_factor": 1.3,
                    "physically_impossible": False,
                },
            ],
        }
    )
    assert out["host_feasible_best_source_id"] == "gaia:2"


def test_host_feasible_best_ignores_missing_depth_correction_factor() -> None:
    out = build_host_plausibility_summary(
        {
            "scenarios": [
                {"source_id": "gaia:1", "depth_correction_factor": None},
                {"source_id": "gaia:2", "depth_correction_factor": 2.0},
            ]
        }
    )
    assert out["host_feasible_best_source_id"] == "gaia:2"

