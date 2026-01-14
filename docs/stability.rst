API Stability
=============

This document describes the stability guarantees for bittr-tess-vetter's public API.

Stability Tiers
---------------

**Stable** - Will not change in backwards-incompatible ways without a major version bump:

* Core types: ``Candidate``, ``Ephemeris``, ``CheckResult``, ``LightCurve``, ``VettingBundleResult``
* Main entry point: ``vet_candidate()``
* Check IDs: V01-V12

**Provisional** - API may evolve based on user feedback:

* Periodogram wrappers: ``run_tls()``, ``run_bls()``
* Pixel analysis: ``analyze_centroids()``, ``localize_source()``
* FPP calculation: ``calculate_fpp()``
* Transit fitting: ``fit_transit()``

**Internal** - Not covered by stability guarantees:

* Anything not exported in ``bittr_tess_vetter.api.__all__``
* Modules under ``platform/``
* Private functions (``_prefixed``)

Deprecation Policy
------------------

Deprecated features will:

1. Emit ``DeprecationWarning`` for at least one minor release
2. Be documented in the changelog
3. Be removed in the next major release
