API Reference
=============

This page documents the public API of tess-vetter.

The recommended import pattern is:

.. code-block:: python

   import tess_vetter.api as btv

Core Types
----------

Data containers and type definitions used throughout the API.

.. currentmodule:: tess_vetter.api

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

Reporting Helpers
-----------------

Helpers for turning `VettingBundleResult` outputs into researcher-friendly tables,
summaries, and markdown reports (policy-free).

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   VettingTableOptions
   format_check_result
   format_vetting_table
   summarize_bundle
   render_validation_report_markdown

Presets:

- ``preset="default"``: runs the default check set (15 checks total; see below).
- ``preset="extended"``: runs the default set plus additional metrics-only diagnostics (V16-V21).

Datasets (Optional IO Convenience)
----------------------------------

Helpers for loading local tutorial-style datasets into in-memory API types.
These functions never use the network.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   LocalDataset
   load_local_dataset
   load_tutorial_target

Workflow Helpers (Policy-Free)
------------------------------

Thin orchestration helpers that compose existing APIs without adding thresholds
or verdict policy.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   VettingSession
   run_check
   run_checks
   WorkflowResult
   run_candidate_workflow
   PerSectorVettingResult
   per_sector_vet

Export Helpers (Policy-Free)
----------------------------

Helpers for exporting results into shareable formats (JSON/CSV/Markdown).

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   ExportFormat
   export_bundle

FPP Workflow Helpers (Policy-Free)
----------------------------------

Helpers that reduce glue around TRICERATOPS(+), without embedding thresholds.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   hydrate_cache_from_dataset
   load_contrast_curve_exofop_tbl

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

LC-Only Checks (V01-V05, V13, V15)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Additional LC-only checks are available via the pipeline registry:

- ``V13``: ``data_gaps`` (missing-cadence fraction near each transit epoch)
- ``V15``: ``transit_asymmetry`` (left/right asymmetry in transit window)

Use :func:`~tess_vetter.api.vet_candidate` with ``checks=["V13", "V15"]`` or
discover check IDs via :func:`~tess_vetter.api.list_checks`.

Notes on V13 (data gaps):

- ``missing_frac_max`` can be ``1.0`` when the ephemeris predicts transits during large
  gaps (e.g., between sectors). For interpretation, prefer the coverage-aware fields:
  ``missing_frac_max_in_coverage``, ``missing_frac_median_in_coverage``,
  ``n_epochs_evaluated_in_coverage``, and ``n_epochs_missing_ge_0p25_in_coverage``.

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

Notes on V09 (difference image localization):

- For bright/saturated targets, naive difference images can be **edge-dominated** or show
  non-physical depth maps. The V09 result includes **diagnostic metrics** to help you assess
  whether the localization is reliable:
  - ``localization_reliable`` (bool)
  - ``max_depth_pixel_edge_distance`` / ``target_pixel_edge_distance`` (pixels from the stamp edge)
  - ``target_depth_ppm_abs`` / ``max_depth_ppm_abs`` and ``concentration_ratio_abs``
  - flags such as ``DIFFIMG_MAX_AT_EDGE`` and ``DIFFIMG_UNRELIABLE``

These are intentionally policy-free: callers can decide how to gate or visualize the results.

Exovetter Checks (V11-V12)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Integration with the Exovetter package.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   modshift
   sweet
   vet_exovetter

Notes on V11 (ModShift):

- Exovetter may use a sign convention where dips are negative. The V11 wrapper reports
  magnitude-based ``primary_signal``/``secondary_signal`` (plus ``*_signed`` debug fields) and
  computes ratios (e.g. ``secondary_primary_ratio``) using magnitudes for interpretability.

Sector Metrics (metrics-only)
-----------------------------

Helpers for per-sector/per-chunk diagnostics when you have stitched or labeled
multi-sector time series. These functions report quantitative metrics only; they
do not apply pass/fail thresholds.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   SectorEphemerisMetrics
   compute_sector_ephemeris_metrics
   compute_sector_ephemeris_metrics_from_stitched

Extended Metrics Checks (V16-V21)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optional, metrics-only diagnostic checks. These are not part of the default preset.
Enable them with :func:`~tess_vetter.api.vet_candidate` using ``preset="extended"``.

- ``V16``: model competition (transit vs alternatives)
- ``V17``: ephemeris reliability regime (quality diagnostics for fixed ephemeris scoring)
- ``V18``: ephemeris sensitivity sweep (how score/depth respond to small ephemeris perturbations)
- ``V19``: alias/harmonic diagnostics (period/alias plausibility signals)
- ``V20``: ghost/scattered-light features (requires pixel-level inputs; may be skipped)
- ``V21``: sector-to-sector consistency (host-provided metrics; may be skipped)

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

Notes:

- Use ``replicates`` to get lower-variance FPP/NFPP plus ``fpp_summary`` / ``nfpp_summary``
  percentiles (recommended when making validation-grade claims).

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
