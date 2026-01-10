# Module Review: `api/lightcurve.py` (+ `domain/lightcurve.py`)

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

The light curve container contract is upstream of every downstream physics check:
- `valid_mask` semantics determine what “data exists” for every statistic and mask.
- dtype/shape immutability prevents subtle cache corruption.
- metadata like `duration_days` and `cadence_seconds` is used for sanity checks, window sizing, and reporting.

## Scope (functions / types)

- `bittr_tess_vetter.api.lightcurve.LightCurveRef`
- `bittr_tess_vetter.api.lightcurve.make_data_ref` (re-export)
- `bittr_tess_vetter.domain.lightcurve.LightCurveData`
- `bittr_tess_vetter.domain.lightcurve.make_data_ref`
- Construction points that must honor the contract:
  - `bittr_tess_vetter.api.types.LightCurve.to_internal` (dtype + finite-mask normalization)
  - `bittr_tess_vetter.api.stitch.stitch_lightcurve_data` (valid_mask + cadence inference)
  - `bittr_tess_vetter.io.mast_client` (download → normalization → cadence inference)

## Audit checklist (to fill)

### Units + conventions

- [x] `time` arrays are BTJD days (same scale as `t0_btjd`); no implicit conversion in the container
- [x] `cadence_seconds` is seconds (informational; stitched series may be mixed-cadence)
- [x] Flux is expected normalized near 1.0 for downstream depth metrics; IO normalizes, API does not silently renormalize

### Container invariants (safety)

- [x] `LightCurveData` enforces dtype + equal-length arrays for `time/flux/flux_err/quality/valid_mask`
- [x] Arrays are made immutable (read-only) after construction to prevent cached mutation
- [x] `LightCurveRef` is frozen + forbids extras (stable API response shape)

### Numerical stability / edge cases

- [x] Derived stats (`duration_days`, `median_flux`, `flux_std`) use only valid samples and are robust to non-finite values
- [x] Empty and all-invalid light curves return sane metadata (`n_points==0`, `duration_days==0`, medians NaN)
- [x] `quality_flags_present` is stable for empty and non-empty arrays

### Contract consistency across constructors

- [x] `api.types.LightCurve.to_internal()` always excludes non-finite time/flux/flux_err from `valid_mask`
- [x] `api.stitch.stitch_lightcurve_data()` constructs `valid_mask` consistent with the above (quality==0 AND finite time/flux/flux_err)
- [x] `io.mast_client` cadence inference ignores non-finite `dt` and falls back to a default when needed

### Tests

- [x] `tests/test_api/test_lightcurve_api.py` covers dtype validation, immutability, ref freezing, empty/all-invalid edge cases
- [x] `tests/test_api/test_stitch_api.py` covers cadence inference (ignores cross-sector gaps) and stitched contract invariants

## Notes (final)

- This module is intentionally “boring”: it defines a strict ndarray container and a stable metadata-only reference (`LightCurveRef`) to keep tool outputs lightweight and deterministic.
- The most important physics safety property is **what counts as valid data**: `valid_mask` must exclude non-finite samples and typically excludes quality-flagged cadences (`quality==0`) for TESS light curves.
