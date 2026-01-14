"""Transit timing, vetting, and physical model fitting tools.

This module provides tools for:
- Measuring individual transit times (TTV analysis)
- Odd/even depth comparison for eclipsing binary vetting
- Physical transit model fitting with limb darkening (batman)

Exports:
- Timing primitives: measure_single_transit, measure_all_transit_times,
                     compute_ttv_statistics
- Vetting primitives: split_odd_even, compare_odd_even_depths,
                      compute_odd_even_result
- Model fitting: fit_transit_model, quick_estimate, get_ld_coefficients,
                 compute_batman_model, compute_derived_parameters
- Results: TransitTime, TTVResult, OddEvenResult, TransitFitResult, ParameterEstimate
"""

from __future__ import annotations

from bittr_tess_vetter.transit.batman_model import (
    ParameterEstimate,
    TransitFitResult,
    compute_batman_model,
    compute_derived_parameters,
    fit_transit_model,
    get_ld_coefficients,
    quick_estimate,
)
from bittr_tess_vetter.transit.result import (
    OddEvenResult,
    TransitTime,
    TTVResult,
)
from bittr_tess_vetter.transit.timing import (
    compute_ttv_statistics,
    measure_all_transit_times,
    measure_single_transit,
)
from bittr_tess_vetter.transit.ttv_track_search import (
    TTVSearchBudget,
    TTVTrackCandidate,
    TTVTrackHypothesis,
    TTVTrackSearchResult,
    estimate_search_cost,
    identify_observing_windows,
    run_ttv_track_search,
    score_periodic_model,
    score_track_hypothesis,
    should_run_ttv_search,
)
from bittr_tess_vetter.transit.vetting import (
    compare_odd_even_depths,
    compute_odd_even_result,
    split_odd_even,
)

__all__ = [
    # Timing primitives
    "measure_single_transit",
    "measure_all_transit_times",
    "compute_ttv_statistics",
    # Vetting primitives
    "split_odd_even",
    "compare_odd_even_depths",
    "compute_odd_even_result",
    # Model fitting (batman)
    "fit_transit_model",
    "quick_estimate",
    "get_ld_coefficients",
    "compute_batman_model",
    "compute_derived_parameters",
    # Result types
    "TransitTime",
    "TTVResult",
    "OddEvenResult",
    "TransitFitResult",
    "ParameterEstimate",
    # TTV track search (experimental detection aid)
    "TTVSearchBudget",
    "TTVTrackCandidate",
    "TTVTrackHypothesis",
    "TTVTrackSearchResult",
    "estimate_search_cost",
    "identify_observing_windows",
    "run_ttv_track_search",
    "score_periodic_model",
    "score_track_hypothesis",
    "should_run_ttv_search",
]
