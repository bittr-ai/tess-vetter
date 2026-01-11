# Module Review: `compute/bls_like_search.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is high leverage

Even though TLS is the primary search method, this BLS-like implementation is used as a fallback/comparison and it encodes core ephemeris conventions (period, `t0`, duration) that can silently diverge from the rest of the pipeline if not audited.

## File: `compute/bls_like_search.py`

### Data models

- `BlsLikeCandidate`:
  - `period_days`: days
  - `t0_btjd`: BTJD days
  - `duration_hours`: hours
  - `score`: unitless detection score
- `BlsLikeSearchResult`:
  - Same best-ephemeris fields + `runtime_seconds` + `notes` for provenance/debugging.

### Helper: `_rolling_mean_circular(x, window)`

- Circular rolling mean used to “smooth” phase-binned flux over an assumed duration.
- Validates `window >= 1`; implements wrap by concatenating `x[:window-1]`.

### Helper: `_phase_bin_means(time_btjd, flux, period, nbins)`

- Phase convention:
  - Computes phase as `(time_btjd % period) / period` (range `[0,1)`), then bins into `nbins`.
  - This assumes `time_btjd` is BTJD-like (order ~1e3 days), where modulus is numerically safe.
- Outputs:
  - `means`: per-bin mean flux with `NaN` for empty bins
  - `counts`: per-bin sample counts

### Helper: `_bls_score_from_binned_flux(binned_flux, binned_counts, duration_bins)`

- Computes a cheap detection statistic from binned flux:
  - Uses `overall = nanmedian(binned_flux)` as baseline.
  - Computes a circular rolling mean over `duration_bins` and finds the minimum window.
  - Estimates `depth = overall - min_window_mean`.
  - Uses `scatter = nanstd(valid_bins)` and returns `score = depth / scatter`.
- Guardrails:
  - Requires enough populated bins; otherwise returns `(-inf, 0)` to signal “not searchable”.

### Main: `bls_like_search_numpy(...)`

- Search loop:
  1) For each `period` in `period_grid`, compute phase-binned means/counts.
  2) For each duration in `duration_hours_grid`:
     - Convert `duration_hours → duration_days`, then `duration_bins ~ duration_days/period * nbins`.
     - Use `_bls_score_from_binned_flux` to get an approximate transit-phase location.
     - Convert the minimum bin index → `t0_guess` in BTJD by aligning to `median(time_btjd)`.
     - Locally refine `t0` by scanning `±local_refine_width_phase * period`.
        - Template is a box in **time**: in-transit if `min(phase, 1-phase) * period <= duration_days/2`.
        - Performs an (optional) inverse-variance weighted least-squares estimate for depth with:
          - `y = 1 - flux` (assumes flux is normalized baseline ~1).
          - `depth_hat = argmin ||y - depth*template||` and `depth_sigma = sqrt(1/denom)`.
          - Score `z = depth_hat/depth_sigma` (unitless).
- Units + conventions:
  - `time_btjd`: days (BTJD expected), `period_grid`: days, `duration_hours_grid`: hours.
  - Output `best_t0_btjd` is a mid-transit time in the same day-scale as `time_btjd`.
- Assumptions:
  - Flux is approximately normalized to a baseline near 1.0 (since depth is computed from `1 - flux`).
  - Inputs are pre-cleaned (finite arrays; no explicit NaN filtering inside the search loop).

### Main: `bls_like_search_numpy_top_k(...)`

- Computes a single best candidate per period (best local `t0`/duration/score), then sorts and returns top-K.
- Guardrail: `top_k >= 1`.

## Tests

- `tests/test_compute/test_bls_like_search.py` provides deterministic unit/integration coverage:
  - Circular rolling mean correctness.
  - Phase binning behavior, including NaN empty bins.
  - Score monotonicity with transit depth.
  - End-to-end recovery of injected box transits and determinism.

## Fixes / follow-ups

No physics correctness issues identified. The two most important conventions to keep consistent across callers:
- `time_btjd` should be BTJD-like (not full JD) to avoid modulus precision issues.
- Flux should be normalized (baseline ~1.0) for depth/score interpretation.

