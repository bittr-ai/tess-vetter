# ruff: noqa: F401

"""bittr-tess-vetter public API.

Golden Path (recommended imports):
    from bittr_tess_vetter.api import (
        # Core types
        LightCurve, Ephemeris, Candidate, CheckResult, VettingBundleResult,
        # Entry points
        vet_candidate, VettingPipeline, run_periodogram,
        # Introspection
        list_checks, describe_checks,
    )

Advanced Usage:
    from bittr_tess_vetter.api.primitives import fold, detrend, check_odd_even_depth
    from bittr_tess_vetter.api.experimental import ...

For custom pipelines:
    from bittr_tess_vetter.api import (
        CheckRegistry, VettingCheck, CheckTier, CheckRequirements,
    )
    from bittr_tess_vetter.validation.register_defaults import register_all_defaults

Core Types:
    - LightCurve: Light curve data container
    - Ephemeris: Transit ephemeris (period, t0, duration)
    - Candidate: Transit candidate container
    - CheckResult: Vetting check result
    - VettingBundleResult: Aggregated pipeline output with provenance

Entry Points:
    - vet_candidate: Run complete vetting pipeline (convenience wrapper)
    - VettingPipeline: Full pipeline with custom check selection
    - run_periodogram: Detect periodic signals in light curves
    - localize_transit_source: WCS-aware transit source localization
    - fit_transit: Fit physical transit model
    - calculate_fpp: Calculate false positive probability

Introspection:
    - list_checks: List available vetting checks
    - describe_checks: Human-readable check descriptions

Registry Types (for extensibility):
    - CheckRegistry: Registry for vetting checks
    - VettingCheck: Protocol for implementing checks
    - CheckTier: Check tier classification
    - CheckRequirements: Data requirements for checks

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
    ...     print(f"{r.id} {r.name}: status={r.status}")
"""

from __future__ import annotations

import ast
import importlib
import importlib.util as _importlib_util
import sys
import types as _types
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

# =============================================================================
# Golden Path Exports (documented, recommended)
# =============================================================================
# These are the primary exports for researcher-facing use.
# All other exports are still accessible via lazy loading but not advertised.

__all__ = [
    # -------------------------------------------------------------------------
    # Core Types
    # -------------------------------------------------------------------------
    "LightCurve",
    "Ephemeris",
    "Candidate",
    "CheckResult",
    "VettingBundleResult",
    "StellarParams",
    "TPFStamp",
    # -------------------------------------------------------------------------
    # Entry Points
    # -------------------------------------------------------------------------
    "vet_candidate",
    "vet_many",
    "VettingPipeline",
    "run_periodogram",
    "localize_transit_source",
    "fit_transit",
    "calculate_fpp",
    # -------------------------------------------------------------------------
    # Introspection
    # -------------------------------------------------------------------------
    "list_checks",
    "describe_checks",
    # -------------------------------------------------------------------------
    # Registry Types (for extensibility)
    # -------------------------------------------------------------------------
    "CheckRegistry",
    "VettingCheck",
    "CheckTier",
    "CheckRequirements",
    "PipelineConfig",
    # -------------------------------------------------------------------------
    # Short Aliases (convenience)
    # -------------------------------------------------------------------------
    "vet",  # -> vet_candidate
    "periodogram",  # -> run_periodogram
    "localize",  # -> localize_transit_source
    # -------------------------------------------------------------------------
    # Optional MLX (guarded)
    # -------------------------------------------------------------------------
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

    # =========================================================================
    # Golden Path: Pipeline and Registry (v0.1.0)
    # =========================================================================
    # =========================================================================
    # Types (v2)
    # =========================================================================
    # v3 modules
    from bittr_tess_vetter.api import activity, recovery, timing, transit_fit

    # v3 activity characterization
    from bittr_tess_vetter.api.activity import characterize_activity, mask_flares
    from bittr_tess_vetter.api.alias_diagnostics import (  # noqa: F401
        PhaseShiftEvent,
        classify_alias,
        compute_harmonic_scores,
        compute_secondary_significance,
        detect_phase_shift_events,
    )

    # Cadence / aperture helpers (host-facing)
    from bittr_tess_vetter.api.aperture import create_circular_aperture_mask

    # WCS-aware pixel tools (v0.2 supported surface)
    from bittr_tess_vetter.api.aperture_family import (
        DEFAULT_RADII_PX,
        ApertureFamilyResult,
        compute_aperture_family_depth_curve,
    )
    from bittr_tess_vetter.api.cadence_mask import default_cadence_mask

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

    # Catalog checks (V06-V07)
    from bittr_tess_vetter.api.catalog import (
        exofop_disposition,
        nearby_eb_search,
        vet_catalog,
    )

    # Detection/periodogram model contract (host-facing)
    from bittr_tess_vetter.api.detection import (  # noqa: F401
        Detection,
        PeriodogramPeak,
        PeriodogramResult,
        TransitCandidate,
        VetterCheckResult,
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
    from bittr_tess_vetter.api.ephemeris_refinement import (  # noqa: F401
        EphemerisRefinementCandidate,
        EphemerisRefinementCandidateResult,
        EphemerisRefinementConfig,
        EphemerisRefinementRunResult,
        refine_candidates_numpy,
        refine_one_candidate_numpy,
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

    # Evidence helpers
    from bittr_tess_vetter.api.evidence import checks_to_evidence_items
    from bittr_tess_vetter.api.evidence_contracts import (  # noqa: F401
        EvidenceEnvelope,
        EvidenceProvenance,
        load_evidence,
        save_evidence,
    )
    from bittr_tess_vetter.api.evidence_contracts import (
        compute_code_hash as compute_evidence_code_hash,
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
    from bittr_tess_vetter.api.ghost_features import (  # noqa: F401
        GhostFeatures,
        compute_aperture_contrast,
        compute_difference_image,
        compute_edge_gradient,
        compute_ghost_features,
        compute_prf_likeness,
        compute_spatial_uniformity,
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

    # Light curve cache contract (host-facing)
    from bittr_tess_vetter.api.lightcurve import (  # noqa: F401
        LightCurveData,
        LightCurveRef,
        make_data_ref,
    )
    from bittr_tess_vetter.api.localization import (
        LocalizationDiagnostics,
        LocalizationImages,
        TransitParams,
        compute_localization_diagnostics,
    )

    # Optional MLX (guarded)
    from bittr_tess_vetter.api.mlx import (  # noqa: F401
        MlxT0RefinementResult,
        MlxTopKScoreResult,
        integrated_gradients,
        score_fixed_period,
        score_fixed_period_refine_t0,
        score_top_k_periods,
        smooth_box_template,
    )
    from bittr_tess_vetter.api.negative_controls import (  # noqa: F401
        ControlType,
        generate_control,
        generate_flux_invert,
        generate_null_inject,
        generate_phase_scramble,
        generate_time_scramble,
    )

    # Periodogram facade (host-facing)
    from bittr_tess_vetter.api.periodogram import (  # noqa: F401
        PerformancePreset,
        # PeriodogramPeak and PeriodogramResult imported from detection above
        auto_periodogram,
        compute_bls_model,
        compute_transit_model,
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
    from bittr_tess_vetter.api.pipeline import (  # noqa: F401
        PipelineConfig,
        VettingPipeline,
        describe_checks,
        list_checks,
    )

    # Pixel checks (V08-V10)
    from bittr_tess_vetter.api.pixel import (
        aperture_dependence,
        centroid_shift,
        difference_image_localization,
        vet_pixel,
    )
    from bittr_tess_vetter.api.pixel_localize import (
        localize_transit_host_multi_sector,
        localize_transit_host_single_sector,
        localize_transit_host_single_sector_with_baseline_check,
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
        prf_params_from_dict,
        prf_params_to_dict,
        propagate_aperture_uncertainty,
        score_hypotheses_prf_lite,
        score_hypotheses_with_prf,
        select_best_hypothesis_joint,
        select_best_hypothesis_timeseries,
    )

    # Prefilters (PFxx)
    from bittr_tess_vetter.api.prefilter import (  # noqa: F401
        compute_depth_over_depth_err_snr,
        compute_phase_coverage,
    )

    # Primitive catalog (host-facing)
    from bittr_tess_vetter.api.primitives import (  # noqa: F401
        PRIMITIVES_CATALOG,
        PrimitiveInfo,
        list_primitives,
    )

    # v3 transit recovery
    from bittr_tess_vetter.api.recovery import (  # noqa: F401
        PreparedRecoveryInputs,
        RecoveryResult,
        detrend,
        prepare_recovery_inputs,
        recover_transit,
        recover_transit_timeseries,
        stack_transits,
    )
    from bittr_tess_vetter.api.reliability_curves import (  # noqa: F401
        compute_conditional_rates,
        compute_reliability_curves,
        recommend_thresholds,
    )
    from bittr_tess_vetter.api.report import PixelVetReport, generate_pixel_vet_report

    # Sandbox compute primitives (host-facing)
    from bittr_tess_vetter.api.sandbox_primitives import (  # noqa: F401
        AstroPrimitives,
        astro,
        box_model,
        # detrend imported from recovery above
        fold,
        periodogram,
    )
    from bittr_tess_vetter.api.sandbox_primitives import (
        detrend as sandbox_detrend,  # noqa: F401
    )
    from bittr_tess_vetter.api.sector_consistency import (  # noqa: F401
        ConsistencyClass,
        SectorMeasurement,
        compute_sector_consistency,
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

    # Light curve stitching (host-facing)
    from bittr_tess_vetter.api.stitch import (  # noqa: F401
        SectorDiagnostics,
        StitchedLC,
        stitch_lightcurves,
    )
    from bittr_tess_vetter.api.systematics import SystematicsProxyResult, compute_systematics_proxy

    # Target model contract (host-facing)
    from bittr_tess_vetter.api.target import (  # noqa: F401
        StellarParameters,
        Target,
    )

    # v3 timing analysis
    from bittr_tess_vetter.api.timing import analyze_ttvs, measure_transit_times
    from bittr_tess_vetter.api.tolerances import (  # noqa: F401
        HARMONIC_RATIOS,
        ToleranceResult,
        check_tolerance,
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

    # v3 transit fitting
    from bittr_tess_vetter.api.transit_fit import TransitFitResult, fit_transit, quick_estimate
    from bittr_tess_vetter.api.transit_masks import (  # noqa: F401
        count_transits,
        get_in_transit_mask,
        get_odd_even_transit_indices,
        get_out_of_transit_mask,
        get_out_of_transit_mask_windowed,
        measure_transit_depth,
    )

    # Transit model facade (host-facing)
    # compute_transit_model is available from periodogram import above (re-exported)
    # Transit primitives
    from bittr_tess_vetter.api.transit_primitives import odd_even_result

    # TRICERATOPS cache helpers (host-facing)
    from bittr_tess_vetter.api.triceratops_cache import (  # noqa: F401
        CalculateFppInput,
        FppResult,
        estimate_transit_duration,
        load_cached_triceratops_target,
        prefetch_trilegal_csv,
        save_cached_triceratops_target,
    )

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
    from bittr_tess_vetter.api.vet import vet_candidate, vet_many
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
    from bittr_tess_vetter.validation.registry import (  # noqa: F401
        CheckRegistry,
        CheckRequirements,
        CheckTier,
        VettingCheck,
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


class _APIModule(_types.ModuleType):
    """Module wrapper that makes top-level aliases stable.

    Python sets package attributes for imported submodules (e.g. importing
    `bittr_tess_vetter.api.vet` sets `bittr_tess_vetter.api.vet` to a module),
    which can collide with our preferred callable aliases (`vet`, `periodogram`, …).

    This wrapper forces alias names to resolve to their target callables even if a
    submodule of the same name has been imported.
    """

    def __getattribute__(self, name: str) -> Any:  # noqa: D401
        if name in _ALIASES:
            return getattr(self, _ALIASES[name])
        return super().__getattribute__(name)


sys.modules[__name__].__class__ = _APIModule
