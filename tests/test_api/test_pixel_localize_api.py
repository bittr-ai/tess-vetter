from __future__ import annotations


def test_pixel_localize_boundary_contract_constants_are_stable() -> None:
    from tess_vetter.api import pixel_localize as px

    assert px.PIXEL_LOCALIZE_BOUNDARY_SCHEMA_VERSION == 1
    assert px.PIXEL_LOCALIZE_MARGIN_RESOLVE_THRESHOLD == px.MARGIN_RESOLVE_THRESHOLD
    assert px.PIXEL_LOCALIZE_ACTION_HINT_REVIEW_MARGIN_THRESHOLD == 10.0
    assert px.PIXEL_LOCALIZE_BASELINE_CENTROID_SHIFT_THRESHOLD_PIXELS_DEFAULT == 0.5


def test_pixel_localize_boundary_contract_schema_is_stable() -> None:
    from tess_vetter.api import pixel_localize as px

    contract = px.PIXEL_LOCALIZE_BOUNDARY_CONTRACT
    assert contract["schema_version"] == 1
    assert contract["margin_resolve_threshold"] == px.PIXEL_LOCALIZE_MARGIN_RESOLVE_THRESHOLD
    assert (
        contract["action_hint_review_margin_threshold"]
        == px.PIXEL_LOCALIZE_ACTION_HINT_REVIEW_MARGIN_THRESHOLD
    )
    assert (
        contract["baseline_centroid_shift_threshold_pixels_default"]
        == px.PIXEL_LOCALIZE_BASELINE_CENTROID_SHIFT_THRESHOLD_PIXELS_DEFAULT
    )
    assert contract["verdicts"] == ("ON_TARGET", "OFF_TARGET", "AMBIGUOUS", "INVALID")
    assert contract["interpretation_codes"] == ("INSUFFICIENT_DISCRIMINATION",)
    assert contract["action_hints"] == (
        "DEFER_HOST_ASSIGNMENT",
        "REVIEW_WITH_DILUTION",
        "HOST_ON_TARGET_SUPPORTED",
        "HOST_OFF_TARGET_CANDIDATE_REVIEW",
    )
    assert contract["reliability_flags"] == (
        "NON_PHYSICAL_PRF_BEST_FIT",
        "BASELINE_SENSITIVE_LOCALIZATION",
        "HIGH_CADENCE_DROPOUT",
    )


def test_pixel_localize_contract_exports_are_stable() -> None:
    from tess_vetter.api import pixel_localize as px

    expected_exports = {
        "PIXEL_LOCALIZE_BOUNDARY_SCHEMA_VERSION",
        "PIXEL_LOCALIZE_MARGIN_RESOLVE_THRESHOLD",
        "PIXEL_LOCALIZE_ACTION_HINT_REVIEW_MARGIN_THRESHOLD",
        "PIXEL_LOCALIZE_BASELINE_CENTROID_SHIFT_THRESHOLD_PIXELS_DEFAULT",
        "PIXEL_LOCALIZE_INTERPRETATION_INSUFFICIENT_DISCRIMINATION",
        "PIXEL_LOCALIZE_ACTION_HINT_DEFER_HOST_ASSIGNMENT",
        "PIXEL_LOCALIZE_ACTION_HINT_REVIEW_WITH_DILUTION",
        "PIXEL_LOCALIZE_ACTION_HINT_HOST_ON_TARGET_SUPPORTED",
        "PIXEL_LOCALIZE_ACTION_HINT_HOST_OFF_TARGET_CANDIDATE_REVIEW",
        "PIXEL_LOCALIZE_RELIABILITY_FLAG_NON_PHYSICAL_PRF_BEST_FIT",
        "PIXEL_LOCALIZE_RELIABILITY_FLAG_BASELINE_SENSITIVE_LOCALIZATION",
        "PIXEL_LOCALIZE_RELIABILITY_FLAG_HIGH_CADENCE_DROPOUT",
        "PIXEL_LOCALIZE_VERDICTS",
        "PIXEL_LOCALIZE_INTERPRETATION_CODES",
        "PIXEL_LOCALIZE_ACTION_HINTS",
        "PIXEL_LOCALIZE_RELIABILITY_FLAGS",
        "PixelLocalizeStatus",
        "PixelLocalizeVerdict",
        "PixelLocalizeInterpretationCode",
        "PixelLocalizeActionHint",
        "PixelLocalizeReliabilityFlag",
        "PixelLocalizeBoundaryContract",
        "PIXEL_LOCALIZE_BOUNDARY_CONTRACT",
    }
    assert expected_exports.issubset(set(px.__all__))
