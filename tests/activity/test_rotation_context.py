from __future__ import annotations

from tess_vetter.activity.rotation_context import build_rotation_context, estimate_v_eq_kms


def test_estimate_v_eq_kms_returns_none_for_missing_inputs() -> None:
    assert estimate_v_eq_kms(stellar_radius_rsun=None, rotation_period_days=3.0) is None
    assert estimate_v_eq_kms(stellar_radius_rsun=1.2, rotation_period_days=None) is None


def test_build_rotation_context_ready_when_inputs_present() -> None:
    payload = build_rotation_context(
        rotation_period_days=3.0,
        stellar_radius_rsun=1.5,
        rotation_period_source_path="activity.rotation_period",
        stellar_radius_source_path="stellar_auto.radius",
        rotation_period_source_authority="activity_lomb_scargle",
        stellar_radius_source_authority="tic_mast",
    )
    assert payload["status"] == "READY"
    assert payload["v_eq_est_kms"] is not None
    assert payload["quality_flags"] == []
    assert payload["provenance"]["rotation_period_source_path"] == "activity.rotation_period"
    assert payload["provenance"]["stellar_radius_source_authority"] == "tic_mast"


def test_build_rotation_context_incomplete_when_inputs_missing() -> None:
    payload = build_rotation_context(rotation_period_days=5.0, stellar_radius_rsun=None)
    assert payload["status"] == "INCOMPLETE_INPUTS"
    assert payload["v_eq_est_kms"] is None
    assert "MISSING_STELLAR_RADIUS" in payload["quality_flags"]
