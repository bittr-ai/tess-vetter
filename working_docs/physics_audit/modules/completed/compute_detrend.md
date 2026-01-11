# Module Review: `compute/detrend.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is high leverage

Detrending is a primary “physics risk surface”: an overly aggressive trend model can erase, distort, or invert transits. This module defines the default normalization conventions and the transit-aware detrending option (wotan) used by recovery workflows.

## File: `compute/detrend.py`

### Function: `median_detrend(flux, window=101)`

- Semantics:
  - Multiplicative detrend: `flux / baseline`.
  - `window` is in **points** and must be a positive odd integer.
- NaN handling:
  - Interpolates across NaN gaps before filtering, then restores NaNs in output.
- Numerical guardrail:
  - Baseline values with `|baseline| < 1e-10` are set to `NaN` (warning) to avoid unstable division.

### Function: `normalize_flux(flux, flux_err)`

- Semantics:
  - Normalizes both `flux` and `flux_err` by `nanmedian(flux)` so that median flux is ~1.0 (TESS convention).
- Guardrails:
  - Requires same shapes.
  - If median is NaN or near zero (`< 1e-10`), returns copies without normalization (warning when near-zero but finite).

### Function: `sigma_clip(flux, sigma=5.0)`

- Semantics:
  - Returns a boolean mask of non-outlier points using a robust scatter estimate (MAD×1.4826).
- Guardrails:
  - NaNs are always invalid.
  - If robust scatter is zero/NaN, all non-NaN points are treated as valid.

### Function: `flatten(time, flux, window_length=0.5)`

- Units:
  - `time`: days (BTJD expected).
  - `window_length`: days.
- Method:
  - Converts `window_length` (days) → `window_points` using median cadence (`median(diff(time))`).
  - Uses `median_detrend` with that window (ensures odd and ≥3).
- Guardrails:
  - Requires time sorted ascending; rejects negative `diff(time)`.

### Function: `wotan_flatten(...)`

- Role:
  - Transit-aware detrending via `wotan.flatten`, with `transit_mask` excluding in-transit points from the trend fit.
- Units:
  - `time`: days; `window_length`: days; `break_tolerance`: days; `edge_cutoff`: days; `kernel_size`: days (GP method).
- Mask semantics:
  - `transit_mask=True` means “exclude from fit” (passed to wotan as `mask=`).
- Edge behavior:
  - NaN trend values (often at edges) are replaced with 1.0 and the flattened flux is recomputed as `flux / trend`.

### Function: `flatten_with_wotan(...)`

- Semantics:
  - Attempts `wotan_flatten` first; falls back to `flatten` on failure/unavailability (optionally raising instead).
  - When falling back, warns that `transit_mask` is ignored (since basic median filter is not transit-aware).

## Fixes / follow-ups

No physics correctness issues identified in this module. Key assumptions to keep explicit:
- `time` must be in the same day-scale as any ephemeris-derived masks (BTJD for TESS),
- `window_length` should exceed transit duration to avoid depth suppression.

