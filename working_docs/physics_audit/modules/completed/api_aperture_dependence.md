# Module Review: `api/aperture.py` / `pixel/aperture.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

Aperture-dependence (depth vs aperture size) is one of the fastest pixel-level blend diagnostics.
It’s used as early evidence when full WCS localization or aperture-family curves aren’t available.

## Scope (functions)

- `bittr_tess_vetter.pixel.aperture.compute_aperture_dependence`
- `bittr_tess_vetter.api.aperture.compute_aperture_dependence` (wrapper/export)

## Audit checklist (to fill)

### Units + conventions

- [x] `TransitParams.period` in days; `t0` in same units as `time`; `duration` in days; `depth` is fractional
- [x] Depth outputs are in ppm and use `(baseline - in_transit)/baseline`
- [x] Aperture radii are in pixels and center convention is `(row, col)`

### Data hygiene / edge cases

- [x] Cadences with non-finite time or all-NaN frames are dropped before masks/measurements
- [x] Depth measurement is robust to NaNs (nansum/nanmedian)
- [x] If insufficient in/oot data, raises a clear `ValueError` (no crash)

### Tests

- [x] On-target transit → depth stable across apertures
- [x] Off-target contaminant → depth changes with aperture radius
- [x] NaN cadences ignored → similar results to clean baseline

## Notes (final)

- This is the “proxy” aperture-dependence diagnostic (distinct from the WCS-aware `aperture_family` curve). It includes background subtraction using an annulus when possible and a local-baseline per-epoch depth estimator (with a sector-wide fallback).
- The function now drops unusable cadences (non-finite time or all-NaN frames) before computing transit epochs, backgrounds, and aperture sums; drop counts are recorded in `result.notes` and flagged via `dropped_invalid_cadences`.
