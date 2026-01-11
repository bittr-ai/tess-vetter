# Module Review: `compute/primitives.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is high leverage

This module is the “sandbox-injected” compute surface. It is intentionally small and dependency-restricted, but it is used as a building block in multiple higher-level workflows. Any unit/sign/mask mistake here becomes a widely replicated, silent physics error.

## File: `compute/primitives.py`

### Purity contract

- Intended for sandbox injection via `additional_scope`.
- Explicit constraints (must remain true):
  - no I/O / no network
  - no heavy astro deps (astropy/lightkurve/batman)
  - numpy/scipy only

### Function: `periodogram(time, flux, periods)`

- Units + conventions:
  - `time`: days (BTJD expected for TESS).
  - `flux`: normalized flux with baseline ~1.0 (mean-subtracted internally).
  - `periods`: days.
- Method:
  - Uses SciPy Lomb–Scargle (`signal.lombscargle`) with `normalize=True`.
  - Converts periods → angular frequencies via `2π / period`.
- Guardrails:
  - Requires at least 3 finite points.
  - Rejects non-positive periods and extremely small periods (`< 1e-10` days) to avoid overflow.
  - If the mean-subtracted flux has zero variance, returns an all-zero power spectrum (avoids divide-by-zero normalization warnings).

### Function: `fold(time, flux, period, t0)`

- Units:
  - `time`: days, `period`: days, `t0`: days.
- Output:
  - phase in `[0, 1)` with phase 0 at `t0`, sorted by phase for plotting.
- Guardrails:
  - requires `period > 0` and aligned input shapes.

### Function: `detrend(flux, window=101)`

- Semantics:
  - Multiplicative median-filter detrend: `flux / trend`.
- Window semantics:
  - Window size is in **points** (not days).
  - If window is even, it is incremented to the next odd value (symmetric filtering).
- NaN handling:
  - Fills NaNs with `nanmedian(flux)` for the trend estimate, then restores NaNs in output.
- Numerical guardrail:
  - Trend values with `|trend| < 1e-10` are treated as invalid and yield `NaN` in output (with a warning).

### Function: `box_model(phase, depth, duration)`

- Units + conventions:
  - `phase`: `[0, 1)` cycles with transit centered at phase 0.0 (wrap-aware).
  - `depth`: fractional (not ppm).
  - `duration`: fraction of orbit in `(0, 0.5)`.
- Output:
  - box transit model with flux `1 - depth` in transit.

### Namespacing: `astro` and `AstroPrimitives`

- `astro` is a `SimpleNamespace` to satisfy sandbox type validation while keeping a stable `astro.<fn>` API.
- `AstroPrimitives` is kept for backwards compatibility and type hints.

## Fixes / follow-ups

No physics correctness issues identified in this module.
