# Module Review: `api/pixel_prf.py` (+ PRF/likelihood internals)

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

This is the highest-leverage “who hosts the transit?” computation:
- it combines pixel-level evidence (PRF fits, aperture depth curves, hypothesis competition)
- it can strongly downweight or upweight planet probability downstream

Small unit/normalization mistakes here can create confident-but-wrong host assignments.

This library must remain **metrics-only**: it should emit likelihoods/scores/margins and diagnostic flags, not
guardrail policy.

## Scope (files / entrypoints)

API layer:
- `src/bittr_tess_vetter/api/pixel_prf.py`
- `src/bittr_tess_vetter/api/pixel_host_hypotheses.py` (multi-sector consensus semantics)

Key internals commonly exercised:
- `src/bittr_tess_vetter/compute/pixel_prf_lite.py`
- `src/bittr_tess_vetter/compute/prf_schemas.py`
- `src/bittr_tess_vetter/compute/prf_psf.py`
- `src/bittr_tess_vetter/compute/pixel_host_hypotheses.py`
- `src/bittr_tess_vetter/compute/pixel_hypothesis_prf.py`
- `src/bittr_tess_vetter/compute/aperture_prediction.py`
- `src/bittr_tess_vetter/compute/pixel_timeseries.py`
- `src/bittr_tess_vetter/compute/joint_inference_schemas.py`
- `src/bittr_tess_vetter/compute/joint_likelihood.py`

## Audit checklist (to fill)

### Units + conventions

- [x] Pixel coordinates convention consistent (row/col vs x/y) across PRF model and WCS utils
- [x] Flux units: PRF fit uses raw pixel flux (arbitrary units) and treats transit signature consistently
- [x] Time conventions: BTJD used consistently when building transit windows / epoch masks

### Likelihood / scoring correctness

- [x] Likelihood definition is explicit (SSE for PRF-lite; Gaussian logL for parametric backend) and matches code
- [x] Any priors (e.g., background, jitter) are stated and default values are safe
- [x] Model comparison score/margin is stable to masking and NaNs

### Numerical stability

- [x] Handles saturated pixels / clipped plateaus without exploding likelihoods (non-finite masking; no special saturation model)
- [x] Handles missing/NaN pixels and quality-flagged cadences deterministically
- [x] Multi-sector aggregation uses consistent weighting (vote + margin tie-break; explicit flip-rate thresholds)

### Failure regimes

- [x] Identify when PRF-lite is expected to fail (crowding, saturated stars, off-center cutouts)
- [x] Ensure the API emits diagnostic flags/metrics rather than making policy calls

### Tests

- [x] Deterministic synthetic test: on-target injection → best hypothesis is target
- [x] Deterministic synthetic test: off-target injection → best hypothesis is contaminant
- [x] Sensitivity test: scaling transit depth increases the margin in expected direction
- [x] Robustness test: masking pixels/NaNs doesn’t flip best hypothesis (unless evidence becomes ambiguous)

## Notes (initial)

- We already removed policy-ish outputs in tess-vetter (e.g., no `recommended_verdict`, no boolean `localization_inconsistent`).
  This audit should confirm PRF modules follow the same principle: emit margins/flip-rates/diagnostics only.

## Notes (audit)

### Coordinate conventions

- All PRF/PSF entry points use `row, col` (0-indexed; fractional allowed) and treat arrays as `(n_rows, n_cols)`:
  - `compute/pixel_prf_lite.build_prf_model(center_row, center_col, shape, ...)`
  - `compute/prf_psf.ParametricPSF.evaluate(center_row, center_col, shape, ...)`
- This matches the WCS-localization/pixel stack convention (`pixel/wcs_localization.py` also returns centroids as `(row, col)`).

### Difference-image sign convention (important)

- The intended convention is: `diff_image = median(out_of_transit) - median(in_transit)`, so **transit dimming is positive**.
- PRF scoring functions do not require the sign for ranking (they use SSE / likelihood), but `fit_amplitude` is convention-dependent and is
  consumed by downstream “quality weighting” logic (e.g., rough SNR estimates in `compute/joint_likelihood.py` use `abs(amplitude)`).
- Tests were updated to match the positive-dimming convention.

### Scoring / likelihood definitions

- PRF-lite scoring (`compute/pixel_host_hypotheses.score_hypotheses_prf_lite`):
  - Fits `diff_image ≈ amplitude * PRF(center=row/col) + background` by linear least squares.
  - Uses `fit_loss = sum(residual^2)` over finite pixels as the ranking statistic.
- Parametric PRF scoring (`compute/pixel_hypothesis_prf.score_hypotheses_with_prf`):
  - Optionally fits background gradient `(b0, bx, by)` and reports `log_likelihood` under a Gaussian assumption.
  - Variance can be scalar or per-pixel; a small floor avoids divide-by-zero.

### Multi-sector aggregation

- `compute/pixel_host_hypotheses.aggregate_multi_sector` is a **vote** over `best_source_id`, with ties broken by total winning-margin.
  - Flip-rate is computed only over “confident” sectors (`margin >= margin_threshold`).
  - Outputs are diagnostics (`consensus_best_source_id`, `consensus_margin`, `disagreement_flag`, `flip_rate`, counts), not policy.

### Failure regimes (expected)

- PRF-lite (single-Gaussian) can fail or become ambiguous in:
  - crowded/overlapping sources (degenerate fits),
  - saturated/clipped stamps (model mismatch),
  - strongly off-center targets or truncated cutouts,
  - heavy scattered light / structured backgrounds (background model too simple).
- In these regimes, the correct behavior is **high loss / low margin / mixed flip-rate**, not a baked-in “verdict”.

### Tests (coverage pointers)

- API facade import surface: `tests/test_api/test_pixel_prf_api.py`
- PRF-lite primitives: `tests/test_compute/test_pixel_prf_lite.py`
- PRF-lite hypothesis scoring + aggregation + aperture-fit: `tests/test_compute/test_pixel_host_hypotheses.py`
  - Includes on-target, off-target, ambiguous, multi-sector flip-rate, NaN robustness, and margin scaling tests.
- Parametric PRF scoring: `tests/test_compute/test_pixel_hypothesis_prf.py`
- Joint likelihood + sector weights: `tests/test_compute/test_joint_likelihood.py`
