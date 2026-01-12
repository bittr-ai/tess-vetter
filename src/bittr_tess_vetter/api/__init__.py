"""Public API for bittr-tess-vetter.

This module provides the user-facing API for transit candidate vetting.

Types (v2):
- Ephemeris: Transit ephemeris (period, t0, duration)
- LightCurve: Simplified light curve container
- StellarParams: Stellar parameters from TIC
- CheckResult: Vetting check result
- Candidate: Transit candidate container (NEW in v2)
- TPFStamp: Target Pixel File data container (NEW in v2)
- VettingBundleResult: Orchestrator output with provenance (NEW in v2)

Types (v3):
- TransitFitResult: Physical transit model fit result
- TransitTime: Single transit timing measurement
- TTVResult: Transit timing variation analysis summary
- OddEvenResult: Odd/even depth comparison for EB vetting
- ActivityResult: Stellar activity characterization
- Flare: Individual flare detection
- StackedTransit: Stacked transit light curve data
- TrapezoidFit: Trapezoid model fit parameters
- RecoveryResult: Transit recovery result from active star

Main Entry Point (v2):
- vet_candidate: Run complete tiered vetting pipeline

Transit Primitives:
- odd_even_result: Odd/even depth comparison for EB detection

LC-Only Checks (V01-V05):
- odd_even_depth: V01 - Compare depth of odd vs even transits
- secondary_eclipse: V02 - Search for secondary eclipse
- duration_consistency: V03 - Check duration vs stellar density
- depth_stability: V04 - Check depth consistency across transits
- v_shape: V05 - Distinguish U-shaped vs V-shaped transits
- vet_lc_only: Orchestrator for all LC-only checks

Catalog Checks (V06-V07):
- nearby_eb_search: V06 - Search for nearby eclipsing binaries
- exofop_disposition: V07 - Check ExoFOP TOI dispositions
- vet_catalog: Orchestrator for catalog checks

Pixel Checks (V08-V10):
- centroid_shift: V08 - Detect centroid motion during transit
- difference_image_localization: V09 - Locate transit source
- aperture_dependence: V10 - Check depth vs aperture size
- vet_pixel: Orchestrator for pixel checks

Exovetter Checks (V11-V12):
- modshift: V11 - ModShift test for secondary eclipse detection
- sweet: V12 - SWEET test for stellar variability
- vet_exovetter: Orchestrator for exovetter checks

v3 Transit Fitting:
- fit_transit: Fit physical transit model using batman
- quick_estimate: Fast analytic parameter estimation

v3 Timing Analysis:
- measure_transit_times: Measure mid-times for all transits
- analyze_ttvs: Compute O-C residuals and TTV statistics

v3 Activity Characterization:
- characterize_activity: Full stellar activity characterization
- mask_flares: Remove flare events from light curves

v3 Transit Recovery:
- recover_transit: Recover transit signal from active star
- detrend: Detrend light curve while preserving transits
- stack_transits: Phase-fold and stack all transits

Example:
    >>> import numpy as np
    >>> from bittr_tess_vetter.api import (
    ...     LightCurve, Ephemeris, Candidate, vet_candidate
    ... )
    >>>
    >>> # Create light curve from your data
    >>> lc = LightCurve(time=time_array, flux=flux_array, flux_err=flux_err_array)
    >>>
    >>> # Define transit candidate
    >>> eph = Ephemeris(period_days=3.5, t0_btjd=1850.0, duration_hours=2.5)
    >>> candidate = Candidate(ephemeris=eph, depth_ppm=500)
    >>>
    >>> # Run complete vetting pipeline
    >>> result = vet_candidate(lc, candidate)
    >>> for r in result.results:
    ...     print(f"{r.id} {r.name}: passed={r.passed} (confidence={r.confidence:.2f})")
"""

from __future__ import annotations

import ast
import importlib
import importlib.util as _importlib_util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

"""
Implementation note:
`bittr_tess_vetter.api` historically re-exported a large surface area by eagerly importing many
submodules. To reduce import-time cost and circular-import fragility without breaking client code,
we now resolve exports lazily via `__getattr__` (PEP 562).
"""

MLX_AVAILABLE = _importlib_util.find_spec("mlx") is not None

_MLX_GUARDED_EXPORTS: set[str] = {
    "MlxTopKScoreResult",
    "MlxT0RefinementResult",
    "smooth_box_template",
    "score_fixed_period",
    "score_fixed_period_refine_t0",
    "score_top_k_periods",
    "integrated_gradients",
}

__all__ = [
    # Types (v2)
    "Ephemeris",
    "LightCurve",
    "StellarParams",
    "CheckResult",
    "Candidate",
    "TPFStamp",
    "VettingBundleResult",
    # Types (v3)
    "TransitFitResult",
    "TransitTime",
    "TTVResult",
    "OddEvenResult",
    "ActivityResult",
    "Flare",
    "StackedTransit",
    "TrapezoidFit",
    "RecoveryResult",
    # Main orchestrator (v2)
    "vet_candidate",
    # Preferred short aliases (non-breaking)
    "vet",
    "periodogram",
    "localize",
    "aperture_family_depth_curve",
    # Evidence helpers
    "checks_to_evidence_items",
    # Generic evidence contracts
    "EvidenceEnvelope",
    "EvidenceProvenance",
    "compute_evidence_code_hash",
    "load_evidence",
    "save_evidence",
    # Prefilters (PFxx)
    "compute_depth_over_depth_err_snr",
    "compute_phase_coverage",
    # Cadence / aperture helpers (host-facing)
    "default_cadence_mask",
    "create_circular_aperture_mask",
    # Utilities
    "DEFAULT_TOP_K_CAP",
    "DEFAULT_VARIANT_SUMMARIES_CAP",
    "DEFAULT_NEIGHBORS_CAP",
    "DEFAULT_PLOTS_CAP",
    "cap_top_k",
    "cap_neighbors",
    "cap_plots",
    "cap_variant_summaries",
    "ToleranceResult",
    "HARMONIC_RATIOS",
    "check_tolerance",
    # Ephemeris matching
    "EphemerisEntry",
    "EphemerisIndex",
    "EphemerisMatch",
    "EphemerisMatchResult",
    "MatchClass",
    "build_index_from_csv",
    "classify_matches",
    "compute_harmonic_match",
    "compute_match_score",
    "load_index",
    "run_ephemeris_matching",
    "save_index",
    "wrap_t0",
    # Transit primitives
    "odd_even_result",
    # Low-level primitives (host-facing)
    "PerformancePreset",
    "PeriodogramPeak",
    "PeriodogramResult",
    "run_periodogram",
    "auto_periodogram",
    "ls_periodogram",
    "tls_search",
    "tls_search_per_sector",
    "search_planets",
    "refine_period",
    "compute_transit_model",
    "compute_bls_model",
    "detect_sector_gaps",
    "split_by_sectors",
    "merge_candidates",
    # Primitive catalog (host-facing)
    "PrimitiveInfo",
    "PRIMITIVES_CATALOG",
    "list_primitives",
    # Pixel/PRF facade (host-facing)
    "PRFParams",
    "PRFFitResult",
    "BackgroundParams",
    "PRFModel",
    "PRFBackend",
    "HypothesisScore",
    "MultiSectorConsensus",
    "ApertureHypothesisFit",
    "AperturePrediction",
    "ApertureConflict",
    "TransitWindow",
    "PixelTimeseriesFit",
    "TimeseriesEvidence",
    "TimeseriesDiagnostics",
    "build_prf_model",
    "build_prf_model_at_pixels",
    "evaluate_prf_weights",
    "prf_params_to_dict",
    "prf_params_from_dict",
    "fit_result_to_dict",
    "fit_result_from_dict",
    "get_prf_model",
    "score_hypotheses_prf_lite",
    "aggregate_multi_sector",
    "fit_aperture_hypothesis",
    "score_hypotheses_with_prf",
    "predict_depth_vs_aperture",
    "predict_all_hypotheses",
    "propagate_aperture_uncertainty",
    "detect_aperture_conflict",
    "compute_aperture_chi2",
    "extract_transit_windows",
    "fit_transit_amplitude_wls",
    "fit_all_hypotheses_timeseries",
    "aggregate_timeseries_evidence",
    "select_best_hypothesis_timeseries",
    "compute_timeseries_diagnostics",
    "assess_sector_quality",
    "compute_sector_weights",
    "compute_joint_log_likelihood",
    "compute_all_hypotheses_joint",
    "select_best_hypothesis_joint",
    "MARGIN_RESOLVE_THRESHOLD",
    "FLIP_RATE_MIXED_THRESHOLD",
    "FLIP_RATE_UNSTABLE_THRESHOLD",
    "get_in_transit_mask",
    "get_out_of_transit_mask",
    "get_out_of_transit_mask_windowed",
    "get_odd_even_transit_indices",
    "measure_transit_depth",
    "count_transits",
    "SmoothTemplateConfig",
    "SmoothTemplateScoreResult",
    "PhaseShiftNullResult",
    "ConcentrationMetrics",
    "LocalT0SensitivityResult",
    "downsample_evenly",
    "smooth_box_template_numpy",
    "score_fixed_period_numpy",
    "phase_shift_t0s",
    "scores_for_t0s_numpy",
    "compute_phase_shift_null",
    "compute_concentration_metrics",
    "compute_local_t0_sensitivity_numpy",
    "SystematicsProxyResult",
    "compute_systematics_proxy",
    # Alias diagnostics extras
    "PhaseShiftEvent",
    "classify_alias",
    "compute_harmonic_scores",
    "compute_secondary_significance",
    "detect_phase_shift_events",
    # Ghost features
    "GhostFeatures",
    "compute_aperture_contrast",
    "compute_difference_image",
    "compute_edge_gradient",
    "compute_ghost_features",
    "compute_prf_likeness",
    "compute_spatial_uniformity",
    # Negative controls
    "ControlType",
    "generate_control",
    "generate_flux_invert",
    "generate_null_inject",
    "generate_phase_scramble",
    "generate_time_scramble",
    # Reliability curves
    "compute_conditional_rates",
    "compute_reliability_curves",
    "recommend_thresholds",
    # Sector consistency
    "ConsistencyClass",
    "SectorMeasurement",
    "compute_sector_consistency",
    # LC-only checks (V01-V05)
    "odd_even_depth",
    "secondary_eclipse",
    "duration_consistency",
    "depth_stability",
    "v_shape",
    "vet_lc_only",
    # Catalog checks (V06-V07)
    "nearby_eb_search",
    "exofop_disposition",
    "vet_catalog",
    # Pixel checks (V08-V10)
    "centroid_shift",
    "difference_image_localization",
    "aperture_dependence",
    "vet_pixel",
    # Exovetter checks (V11-V12)
    "modshift",
    "sweet",
    "vet_exovetter",
    # v3 modules
    "transit_fit",
    "timing",
    "activity",
    "recovery",
    # v3 transit fitting functions
    "fit_transit",
    "quick_estimate",
    # v3 timing analysis functions
    "measure_transit_times",
    "analyze_ttvs",
    # v3 TTV track search
    "TTVSearchBudget",
    "TTVTrackSearchResult",
    "estimate_search_cost",
    "identify_observing_windows",
    "run_ttv_track_search",
    "run_ttv_track_search_for_candidate",
    "should_run_ttv_search",
    # v3 activity characterization functions
    "characterize_activity",
    "mask_flares",
    # v3 transit recovery functions
    "recover_transit",
    "recover_transit_timeseries",
    "detrend",
    "stack_transits",
    # FPP (TRICERATOPS)
    "calculate_fpp",
    "TriceratopsFppPreset",
    "FAST_PRESET",
    "STANDARD_PRESET",
    # WCS-aware / pixel report tools (v0.2)
    "localize_transit_source",
    "LocalizationVerdict",
    "ReferenceSource",
    "LocalizationResult",
    "compute_aperture_family_depth_curve",
    "ApertureFamilyResult",
    "DEFAULT_RADII_PX",
    "compute_pixel_scale",
    "extract_wcs_from_header",
    "wcs_sanity_check",
    "pixel_to_world",
    "pixel_to_world_batch",
    "world_to_pixel",
    "world_to_pixel_batch",
    "compute_localization_diagnostics",
    "LocalizationDiagnostics",
    "LocalizationImages",
    "TransitParams",
    "generate_pixel_vet_report",
    "PixelVetReport",
    # Optional MLX (guarded)
    "MLX_AVAILABLE",
]

if MLX_AVAILABLE:
    __all__.extend(
        [
            "MlxTopKScoreResult",
            "MlxT0RefinementResult",
            "smooth_box_template",
            "score_fixed_period",
            "score_fixed_period_refine_t0",
            "score_top_k_periods",
            "integrated_gradients",
        ]
    )


if TYPE_CHECKING:
    # The import list below serves two purposes:
    # 1) Keeps type checkers happy for the stable, top-level exports.
    # 2) Acts as the single source of truth for the lazy-export resolver, which parses
    #    these `from ... import ...` statements from the module source.

    # Types (v2)
    # v3 modules
    from bittr_tess_vetter.api import activity, recovery, timing, transit_fit

    # v3 activity characterization
    from bittr_tess_vetter.api.activity import characterize_activity, mask_flares

    # Catalog checks (V06-V07)
    from bittr_tess_vetter.api.catalog import (
        exofop_disposition,
        nearby_eb_search,
        vet_catalog,
    )

    # Exovetter checks (V11-V12)
    from bittr_tess_vetter.api.exovetter import (
        modshift,
        sweet,
        vet_exovetter,
    )

    # FPP (TRICERATOPS) presets
    from bittr_tess_vetter.api.fpp import (
        FAST_PRESET,
        STANDARD_PRESET,
        TriceratopsFppPreset,
        calculate_fpp,
    )

    # TRICERATOPS cache helpers (host-facing)
    from bittr_tess_vetter.api.triceratops_cache import (  # noqa: F401
        CalculateFppInput,
        FppResult,
        estimate_transit_duration,
        load_cached_triceratops_target,
        prefetch_trilegal_csv,
        save_cached_triceratops_target,
    )

    # Light curve cache contract (host-facing)
    from bittr_tess_vetter.api.lightcurve import (  # noqa: F401
        LightCurveData,
        LightCurveRef,
        make_data_ref,
    )

    # Target model contract (host-facing)
    from bittr_tess_vetter.api.target import (  # noqa: F401
        StellarParameters,
        Target,
    )

    # Detection/periodogram model contract (host-facing)
    from bittr_tess_vetter.api.detection import (  # noqa: F401
        Detection,
        PeriodogramPeak,
        PeriodogramResult,
        TransitCandidate,
        VetterCheckResult,
    )

    # LC-only checks (V01-V05)
    from bittr_tess_vetter.api.lc_only import (
        depth_stability,
        duration_consistency,
        odd_even_depth,
        secondary_eclipse,
        v_shape,
        vet_lc_only,
    )

    # Pixel checks (V08-V10)
    from bittr_tess_vetter.api.pixel import (
        aperture_dependence,
        centroid_shift,
        difference_image_localization,
        vet_pixel,
    )

    # v3 transit recovery
    from bittr_tess_vetter.api.recovery import (  # noqa: F401
        PreparedRecoveryInputs,
        RecoveryResult,
        detrend,
        recover_transit,
        prepare_recovery_inputs,
        recover_transit_timeseries,
        stack_transits,
    )

    # v3 timing analysis
    from bittr_tess_vetter.api.timing import analyze_ttvs, measure_transit_times

    # v3 TTV track search (detection aid)
    from bittr_tess_vetter.api.ttv_track_search import (  # noqa: F401
        TTVSearchBudget,
        TTVTrackSearchResult,
        estimate_search_cost,
        identify_observing_windows,
        run_ttv_track_search,
        run_ttv_track_search_for_candidate,
        should_run_ttv_search,
    )

    # Stellar dilution / implied-size physics (metrics-only)
    from bittr_tess_vetter.api.stellar_dilution import (  # noqa: F401
        DilutionScenario,
        HostHypothesis,
        PhysicsFlags,
        build_host_hypotheses_from_profile,
        compute_depth_correction_factor_from_flux_fraction,
        compute_dilution_scenarios,
        compute_flux_fraction_from_mag_list,
        compute_implied_radius,
        compute_target_flux_fraction_from_neighbor_mags,
        evaluate_physics_flags,
    )

    # v3 transit fitting
    from bittr_tess_vetter.api.transit_fit import TransitFitResult, fit_transit, quick_estimate

    # Transit primitives
    from bittr_tess_vetter.api.transit_primitives import odd_even_result

    # Cadence / aperture helpers (host-facing)
    from bittr_tess_vetter.api.aperture import create_circular_aperture_mask
    from bittr_tess_vetter.api.cadence_mask import default_cadence_mask
    from bittr_tess_vetter.api.transit_masks import get_out_of_transit_mask_windowed

    # Periodogram facade (host-facing)
    from bittr_tess_vetter.api.periodogram import (  # noqa: F401
        PerformancePreset,
        PeriodogramPeak,
        PeriodogramResult,
        auto_periodogram,
        compute_transit_model,
        compute_bls_model,
        detect_sector_gaps,
        ls_periodogram,
        merge_candidates,
        refine_period,
        run_periodogram,
        search_planets,
        split_by_sectors,
        tls_search,
        tls_search_per_sector,
    )

    # Transit model facade (host-facing)
    from bittr_tess_vetter.api.transit_model import compute_transit_model  # noqa: F401

    # Light curve stitching (host-facing)
    from bittr_tess_vetter.api.stitch import (  # noqa: F401
        SectorDiagnostics,
        StitchedLC,
        stitch_lightcurves,
    )

    # Primitive catalog (host-facing)
    from bittr_tess_vetter.api.primitives import (  # noqa: F401
        PRIMITIVES_CATALOG,
        PrimitiveInfo,
        list_primitives,
    )

    # Pixel/PRF compute facade (host-facing)
    from bittr_tess_vetter.api.pixel_prf import (  # noqa: F401
        FLIP_RATE_MIXED_THRESHOLD,
        FLIP_RATE_UNSTABLE_THRESHOLD,
        MARGIN_RESOLVE_THRESHOLD,
        ApertureConflict,
        ApertureHypothesisFit,
        AperturePrediction,
        BackgroundParams,
        HypothesisScore,
        MultiSectorConsensus,
        PixelTimeseriesFit,
        PRFBackend,
        PRFFitResult,
        PRFModel,
        PRFParams,
        TimeseriesDiagnostics,
        TimeseriesEvidence,
        TransitWindow,
        aggregate_multi_sector,
        aggregate_timeseries_evidence,
        assess_sector_quality,
        build_prf_model,
        build_prf_model_at_pixels,
        compute_all_hypotheses_joint,
        compute_aperture_chi2,
        compute_joint_log_likelihood,
        compute_sector_weights,
        compute_timeseries_diagnostics,
        detect_aperture_conflict,
        evaluate_prf_weights,
        extract_transit_windows,
        fit_all_hypotheses_timeseries,
        fit_aperture_hypothesis,
        fit_result_from_dict,
        fit_result_to_dict,
        fit_transit_amplitude_wls,
        get_prf_model,
        predict_all_hypotheses,
        predict_depth_vs_aperture,
        propagate_aperture_uncertainty,
        prf_params_from_dict,
        prf_params_to_dict,
        score_hypotheses_prf_lite,
        score_hypotheses_with_prf,
        select_best_hypothesis_joint,
        select_best_hypothesis_timeseries,
    )

    # Low-level primitives (host-facing)
    from bittr_tess_vetter.api.ephemeris_specificity import (  # noqa: F401
        ConcentrationMetrics,
        LocalT0SensitivityResult,
        PhaseShiftNullResult,
        SmoothTemplateConfig,
        SmoothTemplateScoreResult,
        compute_concentration_metrics,
        compute_local_t0_sensitivity_numpy,
        compute_phase_shift_null,
        downsample_evenly,
        phase_shift_t0s,
        score_fixed_period_numpy,
        scores_for_t0s_numpy,
        smooth_box_template_numpy,
    )
    from bittr_tess_vetter.api.ephemeris_refinement import (  # noqa: F401
        EphemerisRefinementCandidate,
        EphemerisRefinementCandidateResult,
        EphemerisRefinementConfig,
        EphemerisRefinementRunResult,
        refine_candidates_numpy,
        refine_one_candidate_numpy,
    )
    from bittr_tess_vetter.api.systematics import SystematicsProxyResult, compute_systematics_proxy
    from bittr_tess_vetter.api.transit_masks import (  # noqa: F401
        count_transits,
        get_in_transit_mask,
        get_odd_even_transit_indices,
        get_out_of_transit_mask,
        measure_transit_depth,
    )
    from bittr_tess_vetter.api.ephemeris_match import (  # noqa: F401
        EphemerisEntry,
        EphemerisIndex,
        EphemerisMatch,
        EphemerisMatchResult,
        MatchClass,
        build_index_from_csv,
        classify_matches,
        compute_harmonic_match,
        compute_match_score,
        load_index,
        run_ephemeris_matching,
        save_index,
        wrap_t0,
    )
    from bittr_tess_vetter.api.alias_diagnostics import (  # noqa: F401
        PhaseShiftEvent,
        classify_alias,
        compute_harmonic_scores,
        compute_secondary_significance,
        detect_phase_shift_events,
    )
    from bittr_tess_vetter.api.ghost_features import (  # noqa: F401
        GhostFeatures,
        compute_aperture_contrast,
        compute_difference_image,
        compute_edge_gradient,
        compute_ghost_features,
        compute_prf_likeness,
        compute_spatial_uniformity,
    )
    from bittr_tess_vetter.api.negative_controls import (  # noqa: F401
        ControlType,
        generate_control,
        generate_flux_invert,
        generate_null_inject,
        generate_phase_scramble,
        generate_time_scramble,
    )
    from bittr_tess_vetter.api.reliability_curves import (  # noqa: F401
        compute_conditional_rates,
        compute_reliability_curves,
        recommend_thresholds,
    )
    from bittr_tess_vetter.api.sector_consistency import (  # noqa: F401
        ConsistencyClass,
        SectorMeasurement,
        compute_sector_consistency,
    )

    # Detrending (host-facing)
    from bittr_tess_vetter.api.detrend import (  # noqa: F401
        WOTAN_AVAILABLE,
        flatten,
        flatten_with_wotan,
        median_detrend,
        normalize_flux,
        sigma_clip,
        wotan_flatten,
    )

    # Sandbox compute primitives (host-facing)
    from bittr_tess_vetter.api.sandbox_primitives import (  # noqa: F401
        AstroPrimitives,
        astro,
        box_model,
        detrend,
        fold,
        periodogram,
    )

    # TPF cache facades (host-facing)
    from bittr_tess_vetter.api.tpf import (  # noqa: F401
        TPFCache,
        TPFData,
        TPFHandler,
        TPFNotFoundError,
        TPFRef,
    )
    from bittr_tess_vetter.api.tpf_fits import (  # noqa: F401
        TPFFitsCache,
        TPFFitsData,
        TPFFitsNotFoundError,
        TPFFitsRef,
    )

    # Types (v3) - re-exported from types.py
    from bittr_tess_vetter.api.types import (
        ActivityResult,
        Candidate,
        CheckResult,
        Ephemeris,
        Flare,
        LightCurve,
        OddEvenResult,
        StackedTransit,
        StellarParams,
        TPFStamp,
        TransitTime,
        TrapezoidFit,
        TTVResult,
        VettingBundleResult,
    )

    # Main orchestrator
    from bittr_tess_vetter.api.vet import vet_candidate

    # Evidence helpers
    from bittr_tess_vetter.api.evidence import checks_to_evidence_items
    from bittr_tess_vetter.api.evidence_contracts import (  # noqa: F401
        EvidenceEnvelope,
        EvidenceProvenance,
        compute_code_hash as compute_evidence_code_hash,
        load_evidence,
        save_evidence,
    )

    # WCS-aware pixel tools (v0.2 supported surface)
    from bittr_tess_vetter.api.aperture_family import (
        ApertureFamilyResult,
        DEFAULT_RADII_PX,
        compute_aperture_family_depth_curve,
    )
    from bittr_tess_vetter.api.localization import (
        LocalizationDiagnostics,
        LocalizationImages,
        TransitParams,
        compute_localization_diagnostics,
    )
    from bittr_tess_vetter.api.report import PixelVetReport, generate_pixel_vet_report
    from bittr_tess_vetter.api.wcs_localization import (
        LocalizationResult,
        LocalizationVerdict,
        ReferenceSource,
        localize_transit_source,
    )
    from bittr_tess_vetter.api.wcs_utils import (
        compute_pixel_scale,
        extract_wcs_from_header,
        pixel_to_world,
        pixel_to_world_batch,
        wcs_sanity_check,
        world_to_pixel,
        world_to_pixel_batch,
    )

    # Prefilters (PFxx)
    from bittr_tess_vetter.api.prefilter import (  # noqa: F401
        compute_depth_over_depth_err_snr,
        compute_phase_coverage,
    )

    # Utilities
    from bittr_tess_vetter.api.caps import (  # noqa: F401
        DEFAULT_NEIGHBORS_CAP,
        DEFAULT_PLOTS_CAP,
        DEFAULT_TOP_K_CAP,
        DEFAULT_VARIANT_SUMMARIES_CAP,
        cap_neighbors,
        cap_plots,
        cap_top_k,
        cap_variant_summaries,
    )
    from bittr_tess_vetter.api.tolerances import (  # noqa: F401
        HARMONIC_RATIOS,
        ToleranceResult,
        check_tolerance,
    )

    # Optional MLX (guarded)
    from bittr_tess_vetter.api.mlx import (  # noqa: F401
        MlxTopKScoreResult,
        MlxT0RefinementResult,
        integrated_gradients,
        score_fixed_period,
        score_fixed_period_refine_t0,
        score_top_k_periods,
        smooth_box_template,
    )


def _iter_stmts_in_order(stmts: list[ast.stmt]) -> list[ast.stmt]:
    ordered: list[ast.stmt] = []
    for stmt in stmts:
        ordered.append(stmt)
        if isinstance(stmt, ast.If):
            ordered.extend(_iter_stmts_in_order(stmt.body))
            ordered.extend(_iter_stmts_in_order(stmt.orelse))
        elif isinstance(stmt, ast.Try):
            ordered.extend(_iter_stmts_in_order(stmt.body))
            for handler in stmt.handlers:
                ordered.extend(_iter_stmts_in_order(handler.body))
            ordered.extend(_iter_stmts_in_order(stmt.orelse))
            ordered.extend(_iter_stmts_in_order(stmt.finalbody))
        elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)):
            ordered.extend(_iter_stmts_in_order(stmt.body))
            ordered.extend(_iter_stmts_in_order(stmt.orelse))
    return ordered


_EXPORT_MAP: dict[str, tuple[str, str]] | None = None


def _get_export_map() -> dict[str, tuple[str, str]]:
    global _EXPORT_MAP
    if _EXPORT_MAP is not None:
        return _EXPORT_MAP

    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    exports: dict[str, tuple[str, str]] = {}
    for stmt in _iter_stmts_in_order(tree.body):
        if not isinstance(stmt, ast.ImportFrom) or stmt.module is None:
            continue
        module = stmt.module
        if stmt.level:
            module = "." * stmt.level + module
        for alias in stmt.names:
            if alias.name == "*":
                continue
            local_name = alias.asname or alias.name
            if local_name.startswith("_"):
                continue
            exports[local_name] = (module, alias.name)

    _EXPORT_MAP = exports
    return exports


_ALIASES: dict[str, str] = {
    # Primary orchestration
    "vet": "vet_candidate",
    # Discovery / ephemeris tools
    "periodogram": "run_periodogram",
    # Pixel localization/reporting
    "localize": "localize_transit_source",
    "aperture_family_depth_curve": "compute_aperture_family_depth_curve",
}


def __getattr__(name: str) -> Any:
    if name == "MLX_AVAILABLE":
        return MLX_AVAILABLE
    if name in _MLX_GUARDED_EXPORTS and not MLX_AVAILABLE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    alias_target = _ALIASES.get(name)
    if alias_target is not None:
        mod = sys.modules[__name__]
        value = getattr(mod, alias_target)
        globals()[name] = value
        return value

    exports = _get_export_map()
    target = exports.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module, attr = target
    # Module re-exports (e.g., `from bittr_tess_vetter.api import transit_fit`).
    if module == "bittr_tess_vetter.api" and attr == name:
        value: Any = importlib.import_module(f"{module}.{name}")
    else:
        mod = importlib.import_module(module)
        value = getattr(mod, attr)

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    try:
        exports = _get_export_map()
    except Exception:
        exports = {}
    return sorted(set(globals().keys()) | set(__all__) | set(exports.keys()) | set(_ALIASES.keys()))
