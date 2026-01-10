# Module Review: `api/localization.py` / `pixel/localization.py` / `pixel/difference.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

When WCS/FITS-based localization isn’t available, the pipeline falls back to “proxy” pixel diagnostics:
- difference image brightest pixel
- OOT-derived target proxies (OOT brightest pixel, OOT centroid)

These are often used early to determine whether a signal is plausibly on-target. If cadence masking or NaN handling
is wrong, proxy localization can become unstable and misleading.

## Scope (functions)

- `bittr_tess_vetter.pixel.difference.compute_difference_image`
- `bittr_tess_vetter.pixel.localization.compute_localization_diagnostics`

## Audit checklist (to fill)

### Units + conventions

- [x] Transit ephemeris uses `period` (days), `t0` (BTJD days), `duration` (days) consistently (`TransitParams` from `pixel/aperture.py`)
- [x] Difference image sign is `diff = median(oot) - median(in)` → positive at transit source
- [x] Pixel coordinates are consistent (`(row, col)`)

### Data hygiene / edge cases

- [x] Cadences with non-finite time or all-NaN frames are dropped before masks/medians
- [x] Medians are NaN-robust (`nanmedian`)
- [x] If insufficient in/oot data remains, raises a clear `ValueError` (no crash)

### Tests

- [x] Single-star transit → diff brightest near star
- [x] Off-target injected → diff brightest offset (and scores reflect)
- [x] NaN cadences are ignored (results stable vs clean baseline)

## Notes (final)

- This “proxy” path is intentionally WCS-free; it assumes the target is at the stamp center, but also reports OOT-derived target proxies (OOT brightest pixel and OOT centroid) to reduce false “off-target” conclusions when the target is off-center in the cutout.
- Both proxy modules now drop unusable cadences (non-finite time or all-NaN frames) and compute `nanmedian` images so flagged cadences do not poison medians.
