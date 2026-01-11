# Module Review: `api/aperture_family.py` / `pixel/aperture_family.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

Aperture-family depth curves are a primary blend diagnostic:
- if depth increases with aperture radius, it suggests the transit signal is off-target (contamination)
- if depth is flat, it supports an on-target transit interpretation

Small mistakes in masking (quality/NaNs), phase windowing, or depth definition will cause systematic false blend flags.

## Scope (functions)

- `compute_aperture_family_depth_curve` (depth vs aperture radius; slope + blend_indicator)

## Audit checklist (to fill)

### Units + conventions

- [x] Inputs: `period` (days), `t0` (BTJD days), `duration_hours` (hours)
- [x] Depth is reported in ppm and is defined as `(baseline - in_transit) / baseline`

### Data hygiene / edge cases

- [x] Cadences are filtered for `quality==0` and finite values before computing masks/depths
- [x] In/out-of-transit selection is phase-based (robust to sector gaps)
- [x] If insufficient in/oot points, returns `blend_indicator="unstable"` and warnings

### Tests

- [x] Single-star transit affecting all pixels → depth approximately constant vs radius (`blend_indicator` not increasing)
- [x] Blended transit on off-center source → depth increases with radius (increasing when significant)
- [x] Quality-flagged cadences are ignored (depth remains finite when enough good points exist)
- [x] Slope significance threshold is explicitly tested (`tests/pixel/test_aperture_family.py`, `SLOPE_SIGNIFICANCE_THRESHOLD`)

## Notes (final)

- Cadence filtering: `compute_aperture_family_depth_curve` now drops `quality!=0`, non-finite times, and cadences with no finite pixels in the stamp (via a robust `nanmedian` check). This keeps phase masks and depth medians from being poisoned by known-bad cadences.
- Per-cadence aperture sum: `_extract_aperture_lightcurve` uses `np.nansum` so isolated NaN pixels do not propagate into the summed aperture LC.
- Behavior under missing data: if too few in/oot points are available for a given aperture, that depth is returned as NaN; if fewer than 3 apertures yield finite depths, the module returns `blend_indicator="unstable"` and includes warnings.
