API Reference
=============

This page documents the public API of bittr-tess-vetter.

The recommended import pattern is:

.. code-block:: python

   import bittr_tess_vetter.api as btv

Core Types
----------

Data containers and type definitions used throughout the API.

.. currentmodule:: bittr_tess_vetter.api

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   Ephemeris
   LightCurve
   StellarParams
   CheckResult
   Candidate
   TPFStamp
   VettingBundleResult

Transit Fitting Types
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   TransitFitResult
   TransitTime
   TTVResult
   OddEvenResult
   TrapezoidFit
   StackedTransit
   RecoveryResult

Activity Types
^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   ActivityResult
   Flare

Main Entry Point
----------------

The primary orchestration function for running the complete vetting pipeline.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   vet_candidate

Aliases:

- ``btv.vet`` is an alias for ``vet_candidate``

Periodogram and Detection
-------------------------

Functions for transit detection and periodogram analysis.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   run_periodogram
   auto_periodogram
   ls_periodogram
   tls_search
   tls_search_per_sector
   search_planets
   refine_period
   compute_transit_model
   compute_bls_model
   detect_sector_gaps
   split_by_sectors
   merge_candidates

Types:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   PerformancePreset
   PeriodogramPeak
   PeriodogramResult

Aliases:

- ``btv.periodogram`` is an alias for ``run_periodogram``

Vetting Checks
--------------

LC-Only Checks (V01-V05)
^^^^^^^^^^^^^^^^^^^^^^^^

Light curve-only vetting checks that do not require external data.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   odd_even_depth
   secondary_eclipse
   duration_consistency
   depth_stability
   v_shape
   vet_lc_only

Catalog Checks (V06-V07)
^^^^^^^^^^^^^^^^^^^^^^^^

Checks that query external catalogs (require ``network=True``).

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   nearby_eb_search
   exofop_disposition
   vet_catalog

Pixel Checks (V08-V10)
^^^^^^^^^^^^^^^^^^^^^^

Pixel-level vetting checks.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   centroid_shift
   difference_image_localization
   aperture_dependence
   vet_pixel

Exovetter Checks (V11-V12)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Integration with the Exovetter package.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   modshift
   sweet
   vet_exovetter

Pixel Localization
------------------

WCS-aware localization and pixel analysis tools.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   localize_transit_source
   localize_transit_host_single_sector
   localize_transit_host_single_sector_with_baseline_check
   localize_transit_host_multi_sector
   compute_aperture_family_depth_curve
   generate_pixel_vet_report

Types:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   LocalizationResult
   LocalizationVerdict
   ReferenceSource
   LocalizationDiagnostics
   LocalizationImages
   TransitParams
   ApertureFamilyResult
   PixelVetReport

Aliases:

- ``btv.localize`` is an alias for ``localize_transit_source``
- ``btv.aperture_family_depth_curve`` is an alias for ``compute_aperture_family_depth_curve``

Transit Fitting (v3)
--------------------

Physical transit model fitting using batman.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   fit_transit
   quick_estimate

Timing Analysis (v3)
--------------------

Transit timing measurements and TTV analysis.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   measure_transit_times
   analyze_ttvs

TTV Track Search
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   run_ttv_track_search
   run_ttv_track_search_for_candidate
   estimate_search_cost
   identify_observing_windows
   should_run_ttv_search

Types:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   TTVSearchBudget
   TTVTrackSearchResult

Activity Characterization (v3)
------------------------------

Stellar activity analysis and flare detection.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   characterize_activity
   mask_flares

Transit Recovery (v3)
---------------------

Signal recovery for active stars.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   recover_transit
   recover_transit_timeseries
   detrend
   stack_transits

PRF and Pixel Modeling
----------------------

Point Response Function modeling for pixel-level analysis.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   build_prf_model
   build_prf_model_at_pixels
   evaluate_prf_weights
   get_prf_model
   score_hypotheses_prf_lite
   score_hypotheses_with_prf
   fit_aperture_hypothesis
   aggregate_multi_sector

Types:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   PRFParams
   PRFFitResult
   BackgroundParams
   PRFModel
   PRFBackend
   HypothesisScore
   MultiSectorConsensus
   ApertureHypothesisFit
   AperturePrediction
   ApertureConflict

False Positive Probability
--------------------------

FPP estimation using TRICERATOPS+.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   calculate_fpp

Types:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   TriceratopsFppPreset
   FAST_PRESET
   STANDARD_PRESET

Ephemeris Matching
------------------

Tools for matching ephemerides across catalogs.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   run_ephemeris_matching
   build_index_from_csv
   compute_match_score
   compute_harmonic_match
   classify_matches
   load_index
   save_index
   wrap_t0

Types:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   EphemerisEntry
   EphemerisIndex
   EphemerisMatch
   EphemerisMatchResult
   MatchClass

Transit Masks and Utilities
---------------------------

Helper functions for transit analysis.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   get_in_transit_mask
   get_out_of_transit_mask
   get_out_of_transit_mask_windowed
   get_odd_even_transit_indices
   measure_transit_depth
   count_transits
   odd_even_result

Prefilters
^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   compute_depth_over_depth_err_snr
   compute_phase_coverage

Cadence and Aperture
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   default_cadence_mask
   create_circular_aperture_mask

Evidence and Provenance
-----------------------

Tools for evidence serialization and provenance tracking.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   checks_to_evidence_items
   EvidenceEnvelope
   EvidenceProvenance
   compute_evidence_code_hash
   load_evidence
   save_evidence

Utilities
---------

General utility functions and constants.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   cap_top_k
   cap_neighbors
   cap_plots
   cap_variant_summaries
   check_tolerance

Types:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   ToleranceResult

Constants:

- ``DEFAULT_TOP_K_CAP``
- ``DEFAULT_VARIANT_SUMMARIES_CAP``
- ``DEFAULT_NEIGHBORS_CAP``
- ``DEFAULT_PLOTS_CAP``
- ``HARMONIC_RATIOS``
- ``MLX_AVAILABLE``

MLX Acceleration (Optional)
---------------------------

GPU-accelerated functions for Apple Silicon (requires MLX).

These functions are only available when ``MLX_AVAILABLE`` is ``True``.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   smooth_box_template
   score_fixed_period
   score_fixed_period_refine_t0
   score_top_k_periods
   integrated_gradients

Types:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   MlxTopKScoreResult
   MlxT0RefinementResult
