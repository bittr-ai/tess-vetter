from __future__ import annotations

from bittr_tess_vetter.features.aggregates.localization import build_localization_summary


def test_localization_summary_empty_when_missing() -> None:
    assert build_localization_summary(None) == {}
    assert build_localization_summary({}) == {}


def test_localization_summary_normalizes_verdict_and_confidence_flags() -> None:
    out = build_localization_summary(
        {
            "verdict": "on-target",
            "target_distance_arcsec": 2.5,
            "uncertainty_semimajor_arcsec": 3.0,
            "warnings": [],
            "host_ambiguous_within_1pix": False,
        }
    )
    assert out["localization_verdict"] == "ON_TARGET"
    assert out["localization_low_confidence"] is False
    assert out["localization_target_distance_arcsec"] == 2.5
    assert out["localization_uncertainty_semimajor_arcsec"] == 3.0
    assert out["host_ambiguous_within_1pix"] is False


def test_localization_low_confidence_triggers_on_warnings_or_large_uncertainty() -> None:
    out_warn = build_localization_summary(
        {
            "verdict": "ON_TARGET",
            "uncertainty_semimajor_arcsec": 3.0,
            "warnings": ["fit_failed"],
        }
    )
    assert out_warn["localization_low_confidence"] is True

    out_unc = build_localization_summary(
        {
            "verdict": "ON_TARGET",
            "uncertainty_semimajor_arcsec": 11.0,
            "warnings": [],
        }
    )
    assert out_unc["localization_low_confidence"] is True


def test_localization_v09_reliability_prefers_explicit_boolean() -> None:
    out = build_localization_summary(
        {"verdict": "ON_TARGET", "warnings": []},
        v09={"distance_to_target_pixels": 0.2, "localization_reliable": False, "warnings": []},
    )
    assert out["v09_localization_reliable"] is False


def test_localization_v09_reliability_inferred_from_distance_and_warnings() -> None:
    out_good = build_localization_summary(
        {"verdict": "ON_TARGET", "warnings": []},
        v09={"distance_to_target_pixels": 0.8, "warnings": []},
    )
    assert out_good["v09_localization_reliable"] is True

    out_bad = build_localization_summary(
        {"verdict": "ON_TARGET", "warnings": []},
        v09={"distance_to_target_pixels": 1.2, "warnings": []},
    )
    assert out_bad["v09_localization_reliable"] is False

    out_warn = build_localization_summary(
        {"verdict": "ON_TARGET", "warnings": []},
        v09={"distance_to_target_pixels": 0.2, "warnings": ["noisy"]},
    )
    assert out_warn["v09_localization_reliable"] is False

