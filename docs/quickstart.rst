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

Vetting Check Categories
------------------------

The vetting pipeline includes several categories of checks:

LC-Only Checks (V01-V05)
^^^^^^^^^^^^^^^^^^^^^^^^

These checks use only the light curve data:

- **V01 odd_even_depth**: Compare depth of odd vs even transits
- **V02 secondary_eclipse**: Search for secondary eclipse
- **V03 duration_consistency**: Check duration vs stellar density
- **V04 depth_stability**: Check depth consistency across transits
- **V05 v_shape**: Distinguish U-shaped vs V-shaped transits

Catalog Checks (V06-V07)
^^^^^^^^^^^^^^^^^^^^^^^^

These checks query external catalogs (requires ``network=True``):

- **V06 nearby_eb_search**: Search for nearby eclipsing binaries
- **V07 exofop_disposition**: Check ExoFOP TOI dispositions

Pixel Checks (V08-V10)
^^^^^^^^^^^^^^^^^^^^^^

These checks analyze pixel-level data:

- **V08 centroid_shift**: Detect centroid motion during transit
- **V09 difference_image_localization**: Locate transit source
- **V10 aperture_dependence**: Check depth vs aperture size

Exovetter Checks (V11-V12)
^^^^^^^^^^^^^^^^^^^^^^^^^^

These checks integrate with `exovetter <https://github.com/spacetelescope/exovetter>`_
(requires the ``exovetter`` extra):

- **V11 modshift**: Modshift transit/secondary significance checks
- **V12 sweet**: SWEET test for periodic out-of-transit variability

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
- Explore the ``working_docs/`` directory for internal design notes
