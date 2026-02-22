from __future__ import annotations

from tess_vetter.api.contracts import callable_input_schema_from_signature
from tess_vetter.api.wcs_localization import (
    COMPUTE_DIFFERENCE_IMAGE_CENTROID_DIAGNOSTICS_CALL_SCHEMA,
    LOCALIZE_TRANSIT_SOURCE_CALL_SCHEMA,
    WCS_LOCALIZATION_BASELINE_MODE_GLOBAL,
    WCS_LOCALIZATION_BASELINE_MODE_LOCAL,
    WCS_LOCALIZATION_BASELINE_MODES,
    WCS_LOCALIZATION_BOUNDARY_CONTRACT,
    WCS_LOCALIZATION_BOUNDARY_SCHEMA_VERSION,
    WCS_LOCALIZATION_METHOD_CENTROID,
    WCS_LOCALIZATION_METHOD_GAUSSIAN_FIT,
    WCS_LOCALIZATION_METHODS,
    WCS_LOCALIZATION_VERDICTS,
    compute_difference_image_centroid_diagnostics,
    localize_transit_source,
)


def test_wcs_localization_boundary_constants_are_stable() -> None:
    assert WCS_LOCALIZATION_BOUNDARY_SCHEMA_VERSION == 1
    assert WCS_LOCALIZATION_METHOD_CENTROID == "centroid"
    assert WCS_LOCALIZATION_METHOD_GAUSSIAN_FIT == "gaussian_fit"
    assert WCS_LOCALIZATION_METHODS == ("centroid", "gaussian_fit")
    assert WCS_LOCALIZATION_BASELINE_MODE_LOCAL == "local"
    assert WCS_LOCALIZATION_BASELINE_MODE_GLOBAL == "global"
    assert WCS_LOCALIZATION_BASELINE_MODES == ("local", "global")
    assert WCS_LOCALIZATION_VERDICTS == ("ON_TARGET", "OFF_TARGET", "AMBIGUOUS", "INVALID")


def test_wcs_localization_boundary_contract_is_stable() -> None:
    contract = WCS_LOCALIZATION_BOUNDARY_CONTRACT
    assert contract["schema_version"] == 1
    assert contract["methods"] == ("centroid", "gaussian_fit")
    assert contract["baseline_modes"] == ("local", "global")
    assert contract["verdicts"] == ("ON_TARGET", "OFF_TARGET", "AMBIGUOUS", "INVALID")


def test_wcs_localization_call_schema_constants_match_helpers() -> None:
    assert callable_input_schema_from_signature(
        localize_transit_source
    ) == LOCALIZE_TRANSIT_SOURCE_CALL_SCHEMA
    assert (
        callable_input_schema_from_signature(compute_difference_image_centroid_diagnostics)
    ) == COMPUTE_DIFFERENCE_IMAGE_CENTROID_DIAGNOSTICS_CALL_SCHEMA


def test_wcs_localization_call_schema_snapshots_are_stable() -> None:
    assert LOCALIZE_TRANSIT_SOURCE_CALL_SCHEMA == {
        "type": "object",
        "properties": {
            "tpf_fits": {},
            "period": {},
            "t0": {},
            "duration_hours": {},
            "reference_sources": {},
            "method": {},
            "bootstrap_draws": {},
            "bootstrap_seed": {},
            "oot_margin_mult": {},
            "oot_window_mult": {},
        },
        "required": ["duration_hours", "period", "t0", "tpf_fits"],
        "additionalProperties": False,
    }
    assert COMPUTE_DIFFERENCE_IMAGE_CENTROID_DIAGNOSTICS_CALL_SCHEMA == {
        "type": "object",
        "properties": {
            "tpf_fits": {},
            "period": {},
            "t0": {},
            "duration_hours": {},
            "oot_margin_mult": {},
            "oot_window_mult": {},
            "method": {},
        },
        "required": ["duration_hours", "period", "t0", "tpf_fits"],
        "additionalProperties": False,
    }


def test_wcs_localization_contract_exports_are_stable() -> None:
    from tess_vetter.api import wcs_localization as wl

    expected_exports = {
        "WCS_LOCALIZATION_BOUNDARY_SCHEMA_VERSION",
        "WCS_LOCALIZATION_METHOD_CENTROID",
        "WCS_LOCALIZATION_METHOD_GAUSSIAN_FIT",
        "WCS_LOCALIZATION_METHODS",
        "WCS_LOCALIZATION_BASELINE_MODE_LOCAL",
        "WCS_LOCALIZATION_BASELINE_MODE_GLOBAL",
        "WCS_LOCALIZATION_BASELINE_MODES",
        "WCS_LOCALIZATION_VERDICTS",
        "WCSLocalizationMethod",
        "WCSLocalizationBaselineMode",
        "WCSLocalizationVerdict",
        "WCSLocalizationBoundaryContract",
        "WCS_LOCALIZATION_BOUNDARY_CONTRACT",
        "LOCALIZE_TRANSIT_SOURCE_CALL_SCHEMA",
        "COMPUTE_DIFFERENCE_IMAGE_CENTROID_DIAGNOSTICS_CALL_SCHEMA",
    }
    assert expected_exports.issubset(set(wl.__all__))
