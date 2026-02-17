Quickstart
==========

This guide will help you get started with bittr-tess-vetter for TESS transit
detection and vetting.

Import Convention
-----------------

The recommended import alias follows patterns from astropy (``import astropy.units as u``):

.. code-block:: python

   import bittr_tess_vetter.api as btv

Core Types
----------

The API provides several core data types:

- :class:`~bittr_tess_vetter.api.LightCurve`: Time-series photometry container
- :class:`~bittr_tess_vetter.api.Ephemeris`: Transit ephemeris (period, t0, duration)
- :class:`~bittr_tess_vetter.api.Candidate`: Transit candidate container
- :class:`~bittr_tess_vetter.api.CheckResult`: Individual vetting check result
- :class:`~bittr_tess_vetter.api.VettingBundleResult`: Complete vetting pipeline output

Basic Transit Detection
-----------------------

Run a periodogram search to find transit signals:

.. code-block:: python

   import numpy as np
   import bittr_tess_vetter.api as btv

   # Your light curve data
   time = np.array([...])      # Time in BTJD
   flux = np.array([...])      # Normalized flux
   flux_err = np.array([...])  # Flux uncertainties

   # Run periodogram (TLS by default)
   result = btv.run_periodogram(time=time, flux=flux, flux_err=flux_err)

   # Access results
   print(f"Best period: {result.best_period_days:.4f} days")
   print(f"Best t0: {result.best_t0_btjd:.4f} BTJD")
   print(f"Best depth: {result.best_depth_ppm:.1f} ppm")

Transit Candidate Vetting
-------------------------

Once you have a candidate, run the vetting pipeline:

.. code-block:: python

   import bittr_tess_vetter.api as btv

   # Create light curve object
   lc = btv.LightCurve(time=time, flux=flux, flux_err=flux_err)

   # Define candidate ephemeris
   ephemeris = btv.Ephemeris(
       period_days=3.5,
       t0_btjd=1850.0,
       duration_hours=2.5,
   )

   # Create candidate
   candidate = btv.Candidate(ephemeris=ephemeris, depth_ppm=500)

   # Run vetting (metrics-only mode, no network)
   bundle = btv.vet_candidate(
       lc,
       candidate,
       network=False,
   )

   # Review results
   for check in bundle.results:
       print(f"{check.id} {check.name}: status={check.status} flags={check.flags}")

CLI Vetting Help and Semantics
------------------------------

The ``btv vet`` help text points to this page and the API reference:

.. code-block:: bash

   uv run python -m bittr_tess_vetter.cli.enrich_cli vet --help

Use :doc:`verification/confidence_semantics` as the canonical reference for:

- how to interpret ``CheckResult.status`` and ``CheckResult.confidence``
- process-level ``btv vet`` exit codes (``0`` through ``5``)

Useful machine-readable fields in ``btv vet`` JSON output:

- ``summary.n_flagged`` and ``summary.n_network_errors`` for triage-ready counts
- ``provenance.sectors_requested``, ``provenance.sectors_used``,
  and ``provenance.discovered_sectors`` for sector provenance
- ``provenance.detrend.depth_ppm`` / ``provenance.detrend.depth_err_ppm``
  when ``--detrend`` is enabled (recommended depth handoff for ``btv fpp --depth-ppm``)

Canonical CLI Verdict Contract (machine consumers)
--------------------------------------------------

For machine consumers, the canonical verdict fields across diagnostic CLI commands are:

- top-level: ``verdict`` and ``verdict_source``
- nested: ``result.verdict`` and ``result.verdict_source``

Backward compatibility: legacy command-specific fields are still emitted and remain supported.
Use the canonical fields for new integrations.

Recent contract additions (machine consumers)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``btv fpp`` now emits ``schema_version=cli.fpp.v3`` and
  ``provenance.runtime.degenerate_guard`` (max-points quality-guard attempts/provenance).
- Pipeline evidence output now emits ``schema_version=pipeline.evidence_table.v2`` and
  ``detrend_invariance_policy_*`` fields in each row.
- ``btv resolve-neighbors`` now emits ``multiplicity_risk`` (RUWE/NSS/duplicated-source risk rollup).
- ``btv localize-host`` and ``btv dilution`` now emit ``reliability_summary`` in both top-level and
  ``result`` payloads; when a reference-sources file includes ``multiplicity_risk``, it is threaded through.

Per-command legacy-to-canonical mapping:

.. list-table::
   :header-rows: 1

   * - Command
     - Legacy verdict source fields
     - Canonical fields
   * - ``btv activity``
     - ``activity.interpretation_label`` then ``activity.recommendation`` then ``activity.variability_class``
     - ``verdict`` / ``verdict_source`` and ``result.verdict`` / ``result.verdict_source``
   * - ``btv rv-feasibility``
     - ``result.rv_feasibility.verdict``
     - ``verdict`` / ``verdict_source`` and ``result.verdict`` / ``result.verdict_source``
   * - ``btv dilution``
     - ``physics_flags.requires_resolved_followup`` then ``physics_flags.planet_radius_inconsistent``
     - ``verdict`` / ``verdict_source`` and ``result.verdict`` / ``result.verdict_source``
   * - ``btv timing``
     - ``next_actions.primary_recommendation`` then ``next_actions[0].code``
     - ``verdict`` / ``verdict_source`` and ``result.verdict`` / ``result.verdict_source``
   * - ``btv systematics-proxy``
     - ``systematics_proxy.interpretation_label`` then score-banded ``systematics_proxy.score``
     - ``verdict`` / ``verdict_source`` and ``result.verdict`` / ``result.verdict_source``
   * - ``btv localize-host``
     - ``result.consensus.action_hint`` then ``result.consensus.verdict``
     - ``verdict`` / ``verdict_source`` and ``result.verdict`` / ``result.verdict_source``
   * - ``btv model-compete``
     - ``result.interpretation_label`` then ``result.model_competition.interpretation_label``
     - ``verdict`` / ``verdict_source`` and ``result.verdict`` / ``result.verdict_source``
   * - ``btv ephemeris-reliability``
     - ``result.interpretation_label`` then ``result.label`` then ``result.reliability_label``
     - ``verdict`` / ``verdict_source`` and ``result.verdict`` / ``result.verdict_source``

Vetting Check Categories
------------------------

The vetting pipeline includes several categories of checks. By default,
:func:`~bittr_tess_vetter.api.vet_candidate` runs a 15-check "default" preset.
You can opt into additional diagnostics by passing ``preset="extended"``.

LC-Only Checks (V01-V05, V13, V15)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These checks use only the light curve data:

- **V01 odd_even_depth**: Compare depth of odd vs even transits
- **V02 secondary_eclipse**: Search for secondary eclipse
- **V03 duration_consistency**: Check duration vs stellar density
- **V04 depth_stability**: Check depth consistency across transits
- **V05 v_shape**: Distinguish U-shaped vs V-shaped transits
- **V13 data_gaps**: Estimate missing-cadence fraction near each transit epoch
- **V15 transit_asymmetry**: Left/right asymmetry in the transit window (ramp/step proxy)

.. note::

   V13 reports both a global missingness summary (which can include predicted epochs with
   no time coverage) and coverage-aware fields (``*_in_coverage``). For interpretation,
   prefer the coverage-aware metrics to avoid inter-sector gap confusion.

Catalog Checks (V06-V07)
^^^^^^^^^^^^^^^^^^^^^^^^

These checks query external catalogs (requires ``network=True``):

- **V06 nearby_eb_search**: Search for nearby eclipsing binaries
- **V07 exofop_disposition**: Check ExoFOP TOI dispositions

Pixel Checks (V08-V10)
^^^^^^^^^^^^^^^^^^^^^^

These checks analyze pixel-level data:

- **V08 centroid_shift**: Detect centroid motion during transit
- **V09 difference_image_localization**: Locate transit source (see note below)
- **V10 aperture_dependence**: Check depth vs aperture size

.. note::

   V09 is a diagnostic localization proxy and can be unreliable for bright/saturated targets.
   Use the reported metrics (e.g. ``localization_reliable``,
   ``max_depth_pixel_edge_distance``) and flags (e.g. ``DIFFIMG_UNRELIABLE``) to decide whether
   to trust the localization for a given target.

False Positive Probability (TRICERATOPS+)
-----------------------------------------

For statistical validation-style workflows, compute FPP/NFPP via :func:`~bittr_tess_vetter.api.calculate_fpp`.
For stability, use ``replicates`` and review the returned ``fpp_summary`` / ``nfpp_summary``.

Exovetter Checks (V11-V12)
^^^^^^^^^^^^^^^^^^^^^^^^^^

These checks integrate with `exovetter <https://github.com/spacetelescope/exovetter>`_
(requires the ``exovetter`` extra):

- **V11 modshift**: Modshift transit/secondary significance checks
- **V12 sweet**: SWEET test for periodic out-of-transit variability

.. note::

   ModShift signal metrics may be reported with different sign conventions depending on the
   underlying implementation. The API reports magnitude-based ``primary_signal`` and
   ``secondary_signal`` and also exposes signed values via ``*_signed`` fields.

Extended Metrics Checks (V16-V21, opt-in)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These checks are metrics-only diagnostics (they do not apply any subjective thresholds).
Enable them with ``preset="extended"``:

.. code-block:: python

   bundle = btv.vet_candidate(lc, candidate, preset="extended", network=False)

- **V16 model_competition**: Transit vs alternative model comparisons
- **V17 ephemeris_reliability_regime**: Score reliability diagnostics
- **V18 ephemeris_sensitivity_sweep**: Sensitivity of metrics to ephemeris perturbations
- **V19 alias_diagnostics**: Alias/harmonic plausibility diagnostics
- **V20 ghost_features**: Ghost/scattered-light features (may be skipped without pixel inputs)
- **V21 sector_consistency**: Sector-to-sector consistency (may be skipped unless provided)

If V16 indicates model-competition concern, you can pass a prior vet summary into
``btv detrend-grid`` to preserve that context in grid output:

.. code-block:: bash

   uv run python -m bittr_tess_vetter.cli.enrich_cli detrend-grid \
     --tic-id 123 --period-days 10 --t0-btjd 2000 --duration-hours 2 \
     --vet-summary-path path/to/vet.json

Using Aliases
-------------

The API provides convenient short aliases for common operations:

.. code-block:: python

   import bittr_tess_vetter.api as btv

   # btv.vet is an alias for btv.vet_candidate
   bundle = btv.vet(lc, candidate)

   # btv.periodogram is an alias for btv.run_periodogram
   result = btv.periodogram(time=time, flux=flux, flux_err=flux_err)

   # btv.localize is an alias for btv.localize_transit_source
   localization = btv.localize(...)

Network Behavior
----------------

Catalog-backed checks are always opt-in. You must pass ``network=True``
(and provide metadata like RA/Dec and TIC ID) to enable external queries;
otherwise those checks return skipped results.

.. code-block:: python

   # Without network - catalog checks are skipped
   bundle = btv.vet_candidate(lc, candidate, network=False)

   # With network - requires additional metadata
   bundle = btv.vet_candidate(
       lc,
       candidate,
       network=True,
       ra_deg=123.456,
       dec_deg=-12.345,
       tic_id=123456789,
   )

Advanced: Transit Recovery
--------------------------

For active stars with significant variability, use the recovery module:

.. code-block:: python

   import bittr_tess_vetter.api as btv

   # Detrend the light curve
   detrended = btv.detrend(time, flux, flux_err)

   # Stack transits for better SNR
   stacked = btv.stack_transits(time, flux, flux_err, ephemeris)

   # Recover transit signal
   result = btv.recover_transit(time, flux, flux_err, ephemeris)

Next Steps
----------

- See the :doc:`api` reference for complete API documentation
- See :doc:`verification/confidence_semantics` for confidence and CLI exit-code contracts
- Explore the ``working_docs/`` directory for internal design notes
