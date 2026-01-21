"""Unit tests for the aggregates subpackage."""

from bittr_tess_vetter.features.aggregates import (
    FAMILY_GHOST_RELIABILITY,
    FAMILY_HOST_PLAUSIBILITY,
    FAMILY_PIXEL_TIMESERIES,
    FAMILY_TPF_LOCALIZATION,
    CheckPresenceFlags,
    GhostSectorInput,
    HostPlausibilityInput,
    HostScenario,
    LocalizationInput,
    V09Metrics,
    build_aggregates,
    build_ghost_summary,
    build_host_plausibility_summary,
    build_localization_summary,
    compute_missing_families,
    normalize_verdict,
)

# =============================================================================
# test_normalize_verdict
# =============================================================================


class TestNormalizeVerdict:
    """Tests for normalize_verdict function."""

    def test_none_returns_none(self) -> None:
        """None input returns None."""
        assert normalize_verdict(None) is None

    def test_empty_string_returns_none(self) -> None:
        """Empty string returns None."""
        assert normalize_verdict("") is None

    def test_canonical_verdict_preserved(self) -> None:
        """Canonical ON_TARGET verdict is preserved as-is."""
        assert normalize_verdict("ON_TARGET") == "ON_TARGET"

    def test_case_insensitive(self) -> None:
        """Lowercase verdict is normalized to uppercase."""
        assert normalize_verdict("on_target") == "ON_TARGET"

    def test_legacy_alias_unambiguous(self) -> None:
        """Legacy UNAMBIGUOUS alias maps to ON_TARGET."""
        assert normalize_verdict("UNAMBIGUOUS") == "ON_TARGET"

    def test_unrecognized_returns_no_evidence(self) -> None:
        """Unrecognized garbage string returns NO_EVIDENCE."""
        assert normalize_verdict("garbage_string") == "NO_EVIDENCE"

    def test_all_canonical_verdicts(self) -> None:
        """All canonical verdicts are correctly normalized."""
        assert normalize_verdict("OFF_TARGET") == "OFF_TARGET"
        assert normalize_verdict("AMBIGUOUS") == "AMBIGUOUS"
        assert normalize_verdict("INVALID") == "INVALID"
        assert normalize_verdict("NO_EVIDENCE") == "NO_EVIDENCE"

    def test_mixed_case(self) -> None:
        """Mixed case is handled correctly."""
        assert normalize_verdict("Off_Target") == "OFF_TARGET"
        assert normalize_verdict("ambiguous") == "AMBIGUOUS"

    def test_track_unknown(self) -> None:
        """Unknown values can be tracked via optional list."""
        unknowns: list[str] = []
        result = normalize_verdict("some_random_value", track_unknown=unknowns)
        assert result == "NO_EVIDENCE"
        assert "some_random_value" in unknowns


# =============================================================================
# test_build_ghost_summary
# =============================================================================


class TestBuildGhostSummary:
    """Tests for build_ghost_summary function."""

    def test_empty_input_returns_empty_summary(self) -> None:
        """Empty input list returns empty GhostSummary."""
        result = build_ghost_summary([])
        assert result == {}

    def test_none_input_returns_empty_summary(self) -> None:
        """None input returns empty GhostSummary."""
        result = build_ghost_summary(None)
        assert result == {}

    def test_single_sector_with_all_values(self) -> None:
        """Single sector with all values returns correct median/max (same value)."""
        sector: GhostSectorInput = {
            "sector": 1,
            "ghost_like_score_adjusted": 0.5,
            "scattered_light_risk": 0.3,
            "aperture_sign_consistent": True,
        }
        result = build_ghost_summary([sector])

        assert result.get("ghost_like_score_adjusted_median") == 0.5
        assert result.get("ghost_like_score_adjusted_max") == 0.5
        assert result.get("scattered_light_risk_median") == 0.3
        assert result.get("scattered_light_risk_max") == 0.3
        assert result.get("aperture_sign_consistent_all") is True
        assert result.get("aperture_sign_consistent_any_false") is False

    def test_multiple_sectors_aggregation(self) -> None:
        """Multiple sectors are correctly aggregated with median/max."""
        sectors: list[GhostSectorInput] = [
            {"sector": 1, "ghost_like_score_adjusted": 0.2, "scattered_light_risk": 0.1},
            {"sector": 2, "ghost_like_score_adjusted": 0.5, "scattered_light_risk": 0.4},
            {"sector": 3, "ghost_like_score_adjusted": 0.8, "scattered_light_risk": 0.7},
        ]
        result = build_ghost_summary(sectors)

        # Median of [0.2, 0.5, 0.8] = 0.5
        assert result.get("ghost_like_score_adjusted_median") == 0.5
        assert result.get("ghost_like_score_adjusted_max") == 0.8
        # Median of [0.1, 0.4, 0.7] = 0.4
        assert result.get("scattered_light_risk_median") == 0.4
        assert result.get("scattered_light_risk_max") == 0.7

    def test_non_finite_values_filtered_nan(self) -> None:
        """NaN values are filtered out before aggregation."""
        sectors: list[GhostSectorInput] = [
            {"sector": 1, "ghost_like_score_adjusted": 0.2},
            {"sector": 2, "ghost_like_score_adjusted": float("nan")},
            {"sector": 3, "ghost_like_score_adjusted": 0.8},
        ]
        result = build_ghost_summary(sectors)

        # NaN should be filtered, so median of [0.2, 0.8] = 0.5
        assert result.get("ghost_like_score_adjusted_median") == 0.5
        assert result.get("ghost_like_score_adjusted_max") == 0.8

    def test_non_finite_values_filtered_inf(self) -> None:
        """Inf values are filtered out before aggregation."""
        sectors: list[GhostSectorInput] = [
            {"sector": 1, "scattered_light_risk": 0.3},
            {"sector": 2, "scattered_light_risk": float("inf")},
            {"sector": 3, "scattered_light_risk": float("-inf")},
        ]
        result = build_ghost_summary(sectors)

        # Inf should be filtered, only 0.3 remains
        assert result.get("scattered_light_risk_median") == 0.3
        assert result.get("scattered_light_risk_max") == 0.3

    def test_mix_of_none_and_valid_values(self) -> None:
        """Mix of None and valid values works correctly."""
        sectors: list[GhostSectorInput] = [
            {"sector": 1, "ghost_like_score_adjusted": 0.1, "scattered_light_risk": None},
            {"sector": 2, "ghost_like_score_adjusted": None, "scattered_light_risk": 0.5},
            {"sector": 3, "ghost_like_score_adjusted": 0.3, "scattered_light_risk": 0.7},
        ]
        result = build_ghost_summary(sectors)

        # Ghost scores: [0.1, 0.3], median = 0.2
        assert result.get("ghost_like_score_adjusted_median") == 0.2
        assert result.get("ghost_like_score_adjusted_max") == 0.3
        # Scatter risks: [0.5, 0.7], median = 0.6
        assert result.get("scattered_light_risk_median") == 0.6
        assert result.get("scattered_light_risk_max") == 0.7

    def test_aperture_sign_consistent_mixed(self) -> None:
        """aperture_sign_consistent_all and any_false are computed correctly."""
        sectors: list[GhostSectorInput] = [
            {"sector": 1, "aperture_sign_consistent": True},
            {"sector": 2, "aperture_sign_consistent": False},
            {"sector": 3, "aperture_sign_consistent": True},
        ]
        result = build_ghost_summary(sectors)

        assert result.get("aperture_sign_consistent_all") is False
        assert result.get("aperture_sign_consistent_any_false") is True

    def test_aperture_sign_consistent_all_true(self) -> None:
        """All True sign consistency values."""
        sectors: list[GhostSectorInput] = [
            {"sector": 1, "aperture_sign_consistent": True},
            {"sector": 2, "aperture_sign_consistent": True},
        ]
        result = build_ghost_summary(sectors)

        assert result.get("aperture_sign_consistent_all") is True
        assert result.get("aperture_sign_consistent_any_false") is False


# =============================================================================
# test_build_localization_summary
# =============================================================================


class TestBuildLocalizationSummary:
    """Tests for build_localization_summary function."""

    def test_none_input_returns_empty_summary(self) -> None:
        """None input returns empty LocalizationSummary."""
        result = build_localization_summary(None)
        assert result == {}

    def test_valid_localization_on_target(self) -> None:
        """Valid localization with ON_TARGET verdict."""
        localization: LocalizationInput = {
            "verdict": "ON_TARGET",
            "target_distance_arcsec": 5.0,
            "uncertainty_semimajor_arcsec": 2.0,
            "host_ambiguous_within_1pix": False,
            "warnings": [],
        }
        result = build_localization_summary(localization)

        assert result.get("localization_verdict") == "ON_TARGET"
        assert result.get("localization_target_distance_arcsec") == 5.0
        assert result.get("localization_uncertainty_semimajor_arcsec") == 2.0
        assert result.get("localization_low_confidence") is False
        assert result.get("host_ambiguous_within_1pix") is False

    def test_low_confidence_from_warnings(self) -> None:
        """Low confidence flagged when warnings are present."""
        localization: LocalizationInput = {
            "verdict": "ON_TARGET",
            "uncertainty_semimajor_arcsec": 5.0,  # Under threshold
            "warnings": ["some warning"],
        }
        result = build_localization_summary(localization)

        assert result.get("localization_low_confidence") is True

    def test_low_confidence_from_high_uncertainty(self) -> None:
        """Low confidence flagged when uncertainty > 10 arcsec."""
        localization: LocalizationInput = {
            "verdict": "ON_TARGET",
            "uncertainty_semimajor_arcsec": 15.0,  # Over threshold
            "warnings": [],
        }
        result = build_localization_summary(localization)

        assert result.get("localization_low_confidence") is True

    def test_high_confidence_under_threshold(self) -> None:
        """High confidence when uncertainty <= 10 and no warnings."""
        localization: LocalizationInput = {
            "verdict": "ON_TARGET",
            "uncertainty_semimajor_arcsec": 10.0,  # Exactly at threshold
            "warnings": [],
        }
        result = build_localization_summary(localization)

        assert result.get("localization_low_confidence") is False

    def test_v09_reliability_true(self) -> None:
        """V09 reliable when explicit flag is True."""
        localization: LocalizationInput = {"verdict": "ON_TARGET"}
        v09: V09Metrics = {
            "localization_reliable": True,
            "warnings": [],
        }
        result = build_localization_summary(localization, v09)

        assert result.get("v09_localization_reliable") is True

    def test_v09_reliability_false_explicit(self) -> None:
        """V09 not reliable when explicit flag is False."""
        localization: LocalizationInput = {"verdict": "ON_TARGET"}
        v09: V09Metrics = {
            "localization_reliable": False,
            "warnings": [],
        }
        result = build_localization_summary(localization, v09)

        assert result.get("v09_localization_reliable") is False

    def test_v09_reliability_inferred_true(self) -> None:
        """V09 reliable when inferred from distance < 1 px and no warnings."""
        localization: LocalizationInput = {"verdict": "ON_TARGET"}
        v09: V09Metrics = {
            "distance_to_target_pixels": 0.5,
            "warnings": [],
        }
        result = build_localization_summary(localization, v09)

        assert result.get("v09_localization_reliable") is True

    def test_v09_reliability_inferred_false_has_warnings(self) -> None:
        """V09 not reliable when inferred but warnings present."""
        localization: LocalizationInput = {"verdict": "ON_TARGET"}
        v09: V09Metrics = {
            "distance_to_target_pixels": 0.5,
            "warnings": ["warning"],
        }
        result = build_localization_summary(localization, v09)

        assert result.get("v09_localization_reliable") is False

    def test_v09_reliability_inferred_false_distance_too_large(self) -> None:
        """V09 not reliable when inferred distance >= 1 px."""
        localization: LocalizationInput = {"verdict": "ON_TARGET"}
        v09: V09Metrics = {
            "distance_to_target_pixels": 1.0,
            "warnings": [],
        }
        result = build_localization_summary(localization, v09)

        assert result.get("v09_localization_reliable") is False


# =============================================================================
# test_build_host_plausibility_summary
# =============================================================================


class TestBuildHostPlausibilitySummary:
    """Tests for build_host_plausibility_summary function."""

    def test_none_input_returns_empty_result(self) -> None:
        """None input returns empty HostPlausibilitySummary."""
        result = build_host_plausibility_summary(None)
        assert result == {}

    def test_with_impossible_source_ids(self) -> None:
        """Correctly counts impossible source IDs."""
        host: HostPlausibilityInput = {
            "physically_impossible_source_ids": ["src_1", "src_2", "src_3"],
        }
        result = build_host_plausibility_summary(host)

        assert result.get("host_physically_impossible_count") == 3
        assert result.get("host_physically_impossible_source_ids") == [
            "src_1",
            "src_2",
            "src_3",
        ]

    def test_empty_impossible_source_ids(self) -> None:
        """Empty impossible source IDs list."""
        host: HostPlausibilityInput = {
            "physically_impossible_source_ids": [],
        }
        result = build_host_plausibility_summary(host)

        assert result.get("host_physically_impossible_count") == 0
        assert "host_physically_impossible_source_ids" not in result

    def test_with_multiple_scenarios_picks_best_feasible(self) -> None:
        """Picks best feasible scenario (lowest depth_correction_factor)."""
        scenarios: list[HostScenario] = [
            {
                "source_id": "impossible_src",
                "depth_correction_factor": 0.5,
                "physically_impossible": True,
            },
            {
                "source_id": "feasible_high_dcf",
                "depth_correction_factor": 2.0,
                "flux_fraction": 0.8,
                "true_depth_ppm": 500,
                "physically_impossible": False,
            },
            {
                "source_id": "feasible_low_dcf",
                "depth_correction_factor": 1.2,
                "flux_fraction": 0.9,
                "true_depth_ppm": 300,
                "physically_impossible": False,
            },
        ]
        host: HostPlausibilityInput = {"scenarios": scenarios}
        result = build_host_plausibility_summary(host)

        # Should pick feasible_low_dcf (lowest DCF among non-impossible)
        assert result.get("host_feasible_best_source_id") == "feasible_low_dcf"
        assert result.get("host_feasible_best_flux_fraction") == 0.9
        assert result.get("host_feasible_best_true_depth_ppm") == 300

    def test_all_scenarios_impossible(self) -> None:
        """No feasible best when all scenarios are impossible."""
        scenarios: list[HostScenario] = [
            {
                "source_id": "src_1",
                "depth_correction_factor": 0.5,
                "physically_impossible": True,
            },
            {
                "source_id": "src_2",
                "depth_correction_factor": 0.3,
                "physically_impossible": True,
            },
        ]
        host: HostPlausibilityInput = {"scenarios": scenarios}
        result = build_host_plausibility_summary(host)

        assert "host_feasible_best_source_id" not in result

    def test_with_rationale_and_requires_followup(self) -> None:
        """Rationale and requires_followup are captured."""
        host: HostPlausibilityInput = {
            "requires_resolved_followup": True,
            "rationale": "Multiple plausible hosts within aperture",
        }
        result = build_host_plausibility_summary(host)

        assert result.get("host_requires_resolved_followup") is True
        assert (
            result.get("host_plausibility_rationale") == "Multiple plausible hosts within aperture"
        )


# =============================================================================
# test_compute_missing_families
# =============================================================================


class TestComputeMissingFamilies:
    """Tests for compute_missing_families function."""

    def test_no_tpf_all_tpf_dependent_families_missing(self) -> None:
        """Without TPF, all TPF-dependent families are missing."""
        flags: CheckPresenceFlags = {"has_tpf": False}
        result = compute_missing_families(flags)

        missing = result.get("missing_feature_families") or []
        assert FAMILY_TPF_LOCALIZATION in missing
        assert FAMILY_PIXEL_TIMESERIES in missing
        assert FAMILY_GHOST_RELIABILITY in missing

    def test_has_tpf_no_localization_tpf_localization_missing(self) -> None:
        """With TPF but no localization or diff_image, TPF_LOCALIZATION missing."""
        flags: CheckPresenceFlags = {
            "has_tpf": True,
            "has_localization": False,
            "has_diff_image": False,
            "has_pixel_timeseries": True,
            "has_ghost_summary": True,
        }
        result = compute_missing_families(flags)

        missing = result.get("missing_feature_families") or []
        assert FAMILY_TPF_LOCALIZATION in missing
        assert FAMILY_PIXEL_TIMESERIES not in missing
        assert FAMILY_GHOST_RELIABILITY not in missing

    def test_has_tpf_with_diff_image_localization_not_missing(self) -> None:
        """With TPF and diff_image (no localization), TPF_LOCALIZATION not missing."""
        flags: CheckPresenceFlags = {
            "has_tpf": True,
            "has_localization": False,
            "has_diff_image": True,
            "has_pixel_timeseries": True,
            "has_ghost_summary": True,
        }
        result = compute_missing_families(flags)

        missing = result.get("missing_feature_families") or []
        assert FAMILY_TPF_LOCALIZATION not in missing

    def test_full_coverage_empty_missing_list(self) -> None:
        """Full coverage returns empty missing families list."""
        flags: CheckPresenceFlags = {
            "has_tpf": True,
            "has_localization": True,
            "has_diff_image": False,
            "has_aperture_family": True,
            "has_pixel_timeseries": True,
            "has_ghost_summary": True,
            "has_host_plausibility": True,
        }
        result = compute_missing_families(flags)

        assert result.get("missing_feature_families") == []
        assert result.get("tpf_coverage_ok") is True

    def test_no_host_plausibility(self) -> None:
        """Missing host_plausibility is tracked."""
        flags: CheckPresenceFlags = {
            "has_tpf": True,
            "has_localization": True,
            "has_host_plausibility": False,
        }
        result = compute_missing_families(flags)

        missing = result.get("missing_feature_families") or []
        assert FAMILY_HOST_PLAUSIBILITY in missing

    def test_tpf_coverage_ok_requires_all_components(self) -> None:
        """tpf_coverage_ok requires TPF, localization/diff_image, and aperture."""
        # Missing aperture
        flags: CheckPresenceFlags = {
            "has_tpf": True,
            "has_localization": True,
            "has_aperture_family": False,
        }
        result = compute_missing_families(flags)
        assert result.get("tpf_coverage_ok") is False

        # Missing localization AND diff_image
        flags2: CheckPresenceFlags = {
            "has_tpf": True,
            "has_localization": False,
            "has_diff_image": False,
            "has_aperture_family": True,
        }
        result2 = compute_missing_families(flags2)
        assert result2.get("tpf_coverage_ok") is False

    def test_empty_flags_defaults(self) -> None:
        """Empty flags default to False, all families missing."""
        flags: CheckPresenceFlags = {}
        result = compute_missing_families(flags)

        missing = result.get("missing_feature_families") or []
        # All TPF-dependent families should be missing (has_tpf defaults to False)
        assert FAMILY_TPF_LOCALIZATION in missing
        assert FAMILY_PIXEL_TIMESERIES in missing
        assert FAMILY_GHOST_RELIABILITY in missing
        assert FAMILY_HOST_PLAUSIBILITY in missing


# =============================================================================
# test_build_aggregates
# =============================================================================


class TestBuildAggregates:
    """Integration tests for build_aggregates function."""

    def test_all_inputs_populated(self) -> None:
        """Integration test with all inputs populates all sub-summaries."""
        ghost_sectors: list[GhostSectorInput] = [
            {"sector": 1, "ghost_like_score_adjusted": 0.3, "scattered_light_risk": 0.2},
        ]
        localization: LocalizationInput = {
            "verdict": "ON_TARGET",
            "target_distance_arcsec": 3.0,
            "warnings": [],
        }
        v09: V09Metrics = {
            "localization_reliable": True,
            "warnings": [],
        }
        host_plausibility: HostPlausibilityInput = {
            "requires_resolved_followup": False,
            "physically_impossible_source_ids": [],
        }
        presence_flags: CheckPresenceFlags = {
            "has_tpf": True,
            "has_localization": True,
            "has_aperture_family": True,
            "has_pixel_timeseries": True,
            "has_ghost_summary": True,
            "has_host_plausibility": True,
        }

        result = build_aggregates(
            ghost_sectors=ghost_sectors,
            localization=localization,
            v09=v09,
            host_plausibility=host_plausibility,
            presence_flags=presence_flags,
        )

        # Verify ghost summary
        ghost = result.get("ghost") or {}
        assert ghost.get("ghost_like_score_adjusted_median") == 0.3
        assert ghost.get("scattered_light_risk_max") == 0.2

        # Verify localization summary
        loc = result.get("localization") or {}
        assert loc.get("localization_verdict") == "ON_TARGET"
        assert loc.get("v09_localization_reliable") is True

        # Verify host plausibility summary
        hp = result.get("host_plausibility") or {}
        assert hp.get("host_requires_resolved_followup") is False
        assert hp.get("host_physically_impossible_count") == 0

        # Verify coverage
        cov = result.get("coverage") or {}
        assert cov.get("missing_feature_families") == []
        assert cov.get("tpf_coverage_ok") is True

    def test_minimal_inputs(self) -> None:
        """build_aggregates works with no inputs."""
        result = build_aggregates()

        assert result.get("ghost") == {}
        assert result.get("localization") == {}
        assert result.get("host_plausibility") == {}
        assert result.get("pixel_host") == {}
        # Coverage will have missing families since no presence flags
        cov = result.get("coverage") or {}
        missing = cov.get("missing_feature_families") or []
        assert FAMILY_HOST_PLAUSIBILITY in missing

    def test_partial_inputs(self) -> None:
        """build_aggregates handles partial inputs gracefully."""
        ghost_sectors: list[GhostSectorInput] = [
            {"sector": 5, "ghost_like_score_adjusted": 0.7},
        ]
        result = build_aggregates(ghost_sectors=ghost_sectors)

        ghost = result.get("ghost") or {}
        assert ghost.get("ghost_like_score_adjusted_median") == 0.7
        assert result.get("localization") == {}
        assert result.get("host_plausibility") == {}
