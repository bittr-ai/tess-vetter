# Module Review: `api/wcs_localization.py` / `pixel/wcs_localization.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

WCS difference-image localization is one of the first “is the signal on-target?” checks an agent uses.
If cadence filtering, mask windows, or centroid math is wrong, you get systematic OFF_TARGET/AMBIGUOUS
false positives (or missed blends).

## Scope (functions)

- `compute_difference_image_centroid` (difference image = oot median − in median; centroid/gaussian-fit)
- `bootstrap_centroid_uncertainty` (bootstrap distribution → ellipse)
- `localize_transit_source` (end-to-end: centroid → WCS → bootstrap → distances → verdict)

## Audit checklist (to fill)

### Units + conventions

- [x] Inputs: `period` (days), `t0` (BTJD days), `duration_hours` (hours)
- [x] Difference image sign: `diff = median(oot) - median(in)` produces **positive** signal at transit source
- [x] Pixel coordinates are `(row, col)`; WCS conversions use the same origin convention

### Data hygiene / edge cases

- [x] Cadences are filtered for `quality==0` and finite values before masks/medians
- [x] Medians are NaN-robust (`nanmedian`) to avoid poisoning diff images
- [x] Bootstrap operates on the same filtered cadence set as the centroid
- [x] Diagnostics report cadence counts used/dropped
- [x] Diagnostics include difference-image significance proxies (`diff_peak_snr`, `peak_pixel_depth_sigma`, `baseline_mode`)

### Tests

- [x] Single-star on-target transit → centroid near target
- [x] Off-target transit (secondary) → centroid offset appropriately
- [x] Quality-flagged/NaN cadences are ignored (centroid remains finite and stable)

## Notes (final)

- Cadence hygiene is centralized via `pixel/cadence_mask.default_cadence_mask`:
  - all downstream masks/medians operate on `quality==0` cadences with finite time and at least one finite pixel.
- Difference-image construction uses `nanmedian` so isolated NaNs do not poison the per-pixel medians.
- `localize_transit_source` reports `n_cadences_total/used/dropped` and emits a warning when cadences are dropped.
- Difference-image “quality” is now measurable (metrics-only):
  - `diff_peak_snr`: robust Z-score of the brightest diff-image pixel using MAD-derived sigma (not a calibrated p-value).
  - `peak_pixel_depth_sigma`: median(oot) − median(in) at the peak pixel, with a robust MAD-based error estimate.
  - `baseline_mode`: `"local"` if `oot_window_mult` is finite, else `"global"`.
  These exist specifically to prevent overconfident OFF_TARGET claims when the diff image is baseline-sensitive or low-SNR.
