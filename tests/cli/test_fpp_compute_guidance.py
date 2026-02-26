from __future__ import annotations

from tess_vetter.cli.fpp_compute_guidance import (
    apply_compute_guidance_to_runtime_knobs,
    build_prepare_compute_guidance,
)


def test_build_prepare_compute_guidance_is_deterministic_and_non_binding() -> None:
    runtime_artifacts = {
        "target_cached": True,
        "trilegal_cached": True,
        "staged_with_network": True,
    }

    first = build_prepare_compute_guidance(
        sectors_loaded=[14, 15],
        runtime_artifacts=runtime_artifacts,
    )
    second = build_prepare_compute_guidance(
        sectors_loaded=[14, 15],
        runtime_artifacts=runtime_artifacts,
    )

    assert first == second
    assert first["non_binding"] is True
    assert first["recommendation"] == {
        "preset": "standard",
        "replicates": 3,
        "timeout_seconds": 900.0,
    }
    assert first["evidence"] == {
        "sectors_loaded": 2,
        "runtime_artifacts_ready": True,
        "staged_with_network": True,
    }
    assert "runtime_estimate_seconds" not in first["recommendation"]


def test_apply_compute_guidance_respects_explicit_overrides() -> None:
    prepared = {
        "compute_insights": {
            "recommendation": {
                "preset": "standard",
                "replicates": 7,
                "timeout_seconds": 999,
            }
        }
    }

    (
        preset_name,
        replicates,
        timeout_seconds,
        guidance_available,
        guidance_applied,
        guidance_source,
    ) = apply_compute_guidance_to_runtime_knobs(
        prepared=prepared,
        apply_compute_guidance=True,
        preset="tutorial",
        replicates=2,
        timeout_seconds=123.0,
        preset_was_explicit=True,
        replicates_was_explicit=True,
        timeout_was_explicit=True,
    )

    assert preset_name == "tutorial"
    assert replicates == 2
    assert timeout_seconds == 123.0
    assert guidance_available is True
    assert guidance_applied is False
    assert guidance_source == "prepare_manifest"


def test_apply_compute_guidance_only_sets_unset_knobs() -> None:
    prepared = {
        "compute_insights": {
            "recommendation": {
                "preset": "standard",
                "replicates": 7,
                "timeout_seconds": 999,
            }
        }
    }

    (
        preset_name,
        replicates,
        timeout_seconds,
        _guidance_available,
        guidance_applied,
        _guidance_source,
    ) = apply_compute_guidance_to_runtime_knobs(
        prepared=prepared,
        apply_compute_guidance=True,
        preset="fast",
        replicates=None,
        timeout_seconds=None,
        preset_was_explicit=False,
        replicates_was_explicit=False,
        timeout_was_explicit=False,
    )

    assert preset_name == "standard"
    assert replicates == 7
    assert timeout_seconds == 999.0
    assert guidance_applied is True


def test_apply_compute_guidance_preserves_explicit_timeout_while_applying_other_knobs() -> None:
    prepared = {
        "compute_insights": {
            "recommendation": {
                "preset": "standard",
                "replicates": 7,
                "timeout_seconds": 999,
            }
        }
    }

    (
        preset_name,
        replicates,
        timeout_seconds,
        guidance_available,
        guidance_applied,
        guidance_source,
    ) = apply_compute_guidance_to_runtime_knobs(
        prepared=prepared,
        apply_compute_guidance=True,
        preset="fast",
        replicates=None,
        timeout_seconds=123.0,
        preset_was_explicit=False,
        replicates_was_explicit=False,
        timeout_was_explicit=True,
    )

    assert preset_name == "standard"
    assert replicates == 7
    assert timeout_seconds == 123.0
    assert guidance_available is True
    assert guidance_applied is True
    assert guidance_source == "prepare_manifest"
