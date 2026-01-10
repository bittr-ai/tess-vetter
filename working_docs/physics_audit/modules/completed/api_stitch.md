# Module Review: `api/stitch.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

Stitching/normalization errors silently contaminate **all downstream physics** (depths, SNR, odd/even, periodograms).

## Scope (functions)

- `_compute_mad`
- `_summarize_quality_flags`
- `_validate_lightcurve_dict`
- `_compute_normalization_factor_v1`
- `stitch_lightcurves`
- `stitch_lightcurve_data`

## Audit checklist (to fill)

### Units + conventions

- [x] `time` is days; `cadence_seconds` is seconds; no implicit unit mixing
- [x] Flux normalization factor definition is explicit (median of `quality==0` cadences when possible; else median of finite flux)

### Statistical robustness

- [x] Normalization resistant to outliers and in-transit points (median-based; no OOT-only heuristic)
- [x] Sector-to-sector scaling stable under variable baseline (per-sector normalization to median)

### Numerical stability / edge cases

- [x] Handles NaNs (falls back without divide-by-NaN)
- [x] Handles empty sectors, all-flagged sectors (falls back to finite-flux median normalization)
- [x] Handles cadence differences (20s vs 120s) without mis-inference (uses within-sector deltas; returns dominant cadence)

### Tests

- [x] Existing tests for stitch invariants
- [x] Add synthetic: two sectors with known scaling, gaps, NaNs

## Notes (initial pass)

- Cadence inference is now based on **within-sector adjacent deltas only** (ignores cross-sector gaps), which prevents large inter-sector gaps from corrupting `cadence_seconds`.
  - Code: `src/bittr_tess_vetter/api/stitch.py` `_infer_cadence_seconds()`
  - Test: `tests/test_api/test_stitch_api.py::test_stitch_lightcurve_data_cadence_ignores_cross_sector_gaps`
