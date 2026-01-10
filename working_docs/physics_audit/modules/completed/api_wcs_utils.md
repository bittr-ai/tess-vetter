# Module Review: `api/wcs_utils.py` / `pixel/wcs_utils.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

`wcs_utils` is shared plumbing for all WCS-aware pixel work:
- localization centroid → sky position
- reference-source distance ranking (Gaia neighbors, target)
- pixel scale conversions (arcsec ↔ pixels)

Small convention mistakes (row/col ordering, origin) or numerical edge cases can silently corrupt
all downstream localization verdicts.

## Scope (functions)

- `world_to_pixel`, `pixel_to_world` (+ batch variants)
- `compute_angular_distance`
- `compute_pixel_scale`
- `get_reference_source_pixel_positions`, `compute_source_distances`
- `wcs_sanity_check`

## Audit checklist (to fill)

### Units + conventions

- [x] RA/Dec are in **degrees**
- [x] `compute_angular_distance` returns **arcseconds**
- [x] Pixel coordinates use `(row, col)` while astropy WCS uses `(x, y) = (col, row)`
- [x] `origin` is passed through explicitly and defaults to 0-indexed

### Numerical robustness

- [x] `compute_angular_distance` is stable for very small/very large separations (clips rounding drift)
- [x] Pixel scale uses WCS when available and falls back to 21 arcsec/pixel if not

### Tests

- [x] Round-trip: world→pixel→world is consistent for sample WCS
- [x] Angular distance: 0 for identical coords; ~1 arcsec for 1 arcsec offsets
- [x] Pixel scale: ~21 arcsec/pixel for TESS-like WCS

## Notes (final)

- `compute_angular_distance` now clips the Haversine intermediate `a` to `[0, 1]` to prevent rare `arcsin(sqrt(a))` domain issues from floating-point drift.
