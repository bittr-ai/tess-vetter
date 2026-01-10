# Module Review: `api/centroid.py` / `pixel/centroid.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

Centroid-shift checks are a primary false-positive diagnostic (background EB / blend).
If masking, coordinate conventions, or uncertainty estimation are wrong, you can get systematic false “centroid shift”
or miss real off-target events.

## Scope (functions)

- `bittr_tess_vetter.api.centroid.*` (public wrapper)
- `bittr_tess_vetter.pixel.centroid.*` (centroid computation + shift + significance)

## Audit checklist (to fill)

### Units + conventions

- [x] Inputs: `period_days` (days), `t0_btjd` (days), `duration_hours` (hours)
- [x] Output centroid coordinates use `(x, y) = (col, row)` consistently (documented in `pixel/centroid.py`)
- [x] `centroid_shift_arcsec = centroid_shift_pixels * pixel_scale_arcsec`

### Data hygiene / edge cases

- [x] Cadences are filtered for finite time + finite pixels before computing centroids (quality flags are unavailable at this layer)
- [x] In/out-of-transit windows are phase-based and robust to sector gaps
- [x] NaNs do not poison centroid estimates (frame-level NaNs treated as 0 weight; stacked-frame path uses nanmean)
- [x] If insufficient in/oot points, returns metrics with warnings (no crash)

### Statistics / uncertainty

- [x] Significance definition is clear (sigma relative to bootstrap/analytic uncertainty)
- [x] Bootstrap/permutation paths are deterministic with seed and respect cadence masking
- [x] Confidence intervals correspond to percentiles of bootstrap/permutation distributions

### Tests

- [x] On-target transit (single star) → small shift
- [x] Off-target injected (two stars) → larger shift
- [x] NaN cadences are ignored (results stable vs clean baseline)

## Notes (final)

- This module’s centroid coordinates are `(x, y) = (col, row)`; WCS-aware localization uses `(row, col)`. The shift *magnitude* is invariant, but any downstream code printing/reporting centroids should label axes explicitly.
- `compute_centroid_shift` now applies a conservative cadence mask (finite time + at least one finite pixel per cadence) so `n_in_transit` / `n_out_of_transit` counts reflect *usable* cadences and NaN-only frames do not inflate confidence.
