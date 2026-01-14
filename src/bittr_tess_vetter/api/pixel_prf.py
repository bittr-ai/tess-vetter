"""Pixel/PRF compute facade (host-facing).

This module exposes a stable import surface for pixel-level PRF/PSF and host
hypothesis compute routines. Host applications should import from here rather
than `bittr_tess_vetter.compute.*` directly.

All exports are pure compute (array-in/array-out). No I/O and no network.
"""

from __future__ import annotations

from bittr_tess_vetter.api.references import (
    BRYSON_2010,
    BRYSON_2013,
    HIGGINS_BELL_2022,
    TORRES_2011,
    cite,
    cites,
)

# Aperture prediction + conflict detection
from bittr_tess_vetter.compute.aperture_prediction import (  # noqa: F401
    ApertureConflict,
    AperturePrediction,
    compute_aperture_chi2,
    detect_aperture_conflict,
    predict_all_hypotheses,
    predict_depth_vs_aperture,
    propagate_aperture_uncertainty,
)

# Joint likelihood (multi-sector inference)
from bittr_tess_vetter.compute.joint_likelihood import (  # noqa: F401
    assess_sector_quality,
    compute_all_hypotheses_joint,
    compute_joint_log_likelihood,
    compute_sector_weights,
    select_best_hypothesis_joint,
)

# Hypothesis scoring
from bittr_tess_vetter.compute.pixel_host_hypotheses import (  # noqa: F401
    FLIP_RATE_MIXED_THRESHOLD,
    FLIP_RATE_UNSTABLE_THRESHOLD,
    MARGIN_RESOLVE_THRESHOLD,
    ApertureHypothesisFit,
    HypothesisScore,
    MultiSectorConsensus,
    aggregate_multi_sector,
    fit_aperture_hypothesis,
    score_hypotheses_prf_lite,
)
from bittr_tess_vetter.compute.pixel_hypothesis_prf import (  # noqa: F401
    PRFBackend,
    score_hypotheses_with_prf,
)

# PRF-lite
from bittr_tess_vetter.compute.pixel_prf_lite import (  # noqa: F401
    build_prf_model,
    build_prf_model_at_pixels,
    evaluate_prf_weights,
)

# Time-series inference on pixels
from bittr_tess_vetter.compute.pixel_timeseries import (  # noqa: F401
    DEFAULT_BASELINE_ORDER,
    DEFAULT_MARGIN_THRESHOLD,
    DEFAULT_MIN_IN_TRANSIT,
    DEFAULT_WINDOW_MARGIN,
    PixelTimeseriesFit,
    TimeseriesDiagnostics,
    TimeseriesEvidence,
    TransitWindow,
    aggregate_timeseries_evidence,
    compute_timeseries_diagnostics,
    extract_transit_windows,
    fit_all_hypotheses_timeseries,
    fit_transit_amplitude_wls,
    select_best_hypothesis_timeseries,
)
from bittr_tess_vetter.compute.prf_psf import (  # noqa: F401
    AVAILABLE_BACKENDS,
    ParametricPSF,
    PRFModel,
    get_prf_model,
)

# PRF schemas + PSF model
from bittr_tess_vetter.compute.prf_schemas import (  # noqa: F401
    BackgroundParams,
    PRFFitResult,
    PRFParams,
    fit_result_from_dict,
    fit_result_to_dict,
    prf_params_from_dict,
    prf_params_from_json,
    prf_params_to_dict,
    prf_params_to_json,
)

# Attach citations to imported callables (no wrapping; adds __references__ metadata).
build_prf_model = cites(cite(BRYSON_2010, "PRF model construction"))(build_prf_model)
build_prf_model_at_pixels = cites(cite(BRYSON_2010, "PRF evaluation on pixel grids"))(  # type: ignore[assignment]
    build_prf_model_at_pixels
)
evaluate_prf_weights = cites(cite(BRYSON_2010, "PRF weight evaluation"))(evaluate_prf_weights)

score_hypotheses_prf_lite = cites(
    cite(BRYSON_2010, "PRF-based modeling and centroid inference lineage"),
    cite(BRYSON_2013, "Pixel-level diagnostics for background false positives"),
    cite(HIGGINS_BELL_2022, "TESS-specific source localization in crowded fields"),
)(score_hypotheses_prf_lite)

score_hypotheses_with_prf = cites(
    cite(BRYSON_2010, "PRF-based modeling and centroid inference lineage"),
    cite(BRYSON_2013, "Pixel-level diagnostics for background false positives"),
    cite(HIGGINS_BELL_2022, "TESS-specific source localization in crowded fields"),
)(score_hypotheses_with_prf)

fit_aperture_hypothesis = cites(
    cite(BRYSON_2013, "Aperture/difference-image evidence for blend rejection"),
    cite(TORRES_2011, "Blend scenario modeling and rejection"),
)(fit_aperture_hypothesis)

predict_depth_vs_aperture = cites(
    cite(BRYSON_2013, "Aperture-dependent depth behavior in blended signals"),
    cite(TORRES_2011, "Blend scenario modeling and rejection"),
)(predict_depth_vs_aperture)

detect_aperture_conflict = cites(
    cite(BRYSON_2013, "Aperture/difference-image evidence for blend rejection"),
    cite(TORRES_2011, "Blend scenario modeling and rejection"),
)(detect_aperture_conflict)

__all__ = [
    # PRF-lite
    "build_prf_model",
    "build_prf_model_at_pixels",
    "evaluate_prf_weights",
    # PRF schemas + PSF model
    "PRFParams",
    "PRFFitResult",
    "BackgroundParams",
    "prf_params_to_dict",
    "prf_params_from_dict",
    "fit_result_to_dict",
    "fit_result_from_dict",
    "prf_params_to_json",
    "prf_params_from_json",
    "ParametricPSF",
    "PRFModel",
    "get_prf_model",
    "AVAILABLE_BACKENDS",
    # Hypothesis scoring
    "HypothesisScore",
    "MultiSectorConsensus",
    "ApertureHypothesisFit",
    "score_hypotheses_prf_lite",
    "aggregate_multi_sector",
    "fit_aperture_hypothesis",
    "PRFBackend",
    "score_hypotheses_with_prf",
    # Thresholds
    "MARGIN_RESOLVE_THRESHOLD",
    "FLIP_RATE_MIXED_THRESHOLD",
    "FLIP_RATE_UNSTABLE_THRESHOLD",
    # Aperture prediction + conflicts
    "AperturePrediction",
    "ApertureConflict",
    "predict_depth_vs_aperture",
    "predict_all_hypotheses",
    "propagate_aperture_uncertainty",
    "detect_aperture_conflict",
    "compute_aperture_chi2",
    # Pixel time-series inference
    "DEFAULT_WINDOW_MARGIN",
    "DEFAULT_MIN_IN_TRANSIT",
    "DEFAULT_BASELINE_ORDER",
    "DEFAULT_MARGIN_THRESHOLD",
    "TransitWindow",
    "PixelTimeseriesFit",
    "TimeseriesEvidence",
    "TimeseriesDiagnostics",
    "extract_transit_windows",
    "fit_transit_amplitude_wls",
    "fit_all_hypotheses_timeseries",
    "aggregate_timeseries_evidence",
    "select_best_hypothesis_timeseries",
    "compute_timeseries_diagnostics",
    # Joint likelihood
    "assess_sector_quality",
    "compute_sector_weights",
    "compute_joint_log_likelihood",
    "compute_all_hypotheses_joint",
    "select_best_hypothesis_joint",
]
