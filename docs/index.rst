bittr-tess-vetter
=================

Domain library for TESS transit detection and vetting (array-in/array-out).

This package is intentionally "domain-only": array-in/array-out astronomy algorithms
without any platform-specific tooling (stores, manifests, agent frameworks, etc.).

Features
--------

- **Transit detection**: TLS/LS periodograms, multi-planet search, candidate merging
- **Vetting pipeline**: Tiered checks (LC-only, catalog, pixel, Exovetter)
- **Pixel diagnostics**: Centroid shift, difference images, WCS-aware localization
- **Transit recovery**: Detrend + stack + trapezoid fitting for active stars
- **FPP estimation**: TRICERATOPS+ support (optional)

Quick Example
-------------

.. code-block:: python

   import numpy as np
   import bittr_tess_vetter.api as btv

   # Create light curve
   lc = btv.LightCurve(time=time, flux=flux, flux_err=flux_err)

   # Define candidate
   candidate = btv.Candidate(
       ephemeris=btv.Ephemeris(period_days=3.5, t0_btjd=1850.0, duration_hours=2.5),
       depth_ppm=500,
   )

   # Run vetting pipeline (metrics-only results)
   bundle = btv.vet_candidate(lc, candidate, network=False)

   for r in bundle.results:
       print(f"{r.id} {r.name}: status={r.status} flags={r.flags}")

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   stability
   tutorials
   tutorials/blend_localization

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
