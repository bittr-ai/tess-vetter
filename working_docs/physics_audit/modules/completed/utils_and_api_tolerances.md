# Module Review: `utils/tolerances.py` + `api/tolerances.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is high leverage

Tolerance logic quietly controls whether two ephemerides/metrics are considered “consistent”. Unit mistakes here (days vs phase fraction, ppm vs fraction) can silently widen or narrow acceptance bands and lead to false matches or missed matches.

## File: `utils/tolerances.py`

### Model: `ToleranceResult`

- `delta`: signed numeric difference (`replayed - original`).
- `relative_error`: meaning depends on check type:
  - period/depth: relative error (dimensionless)
  - t0: phase fraction of the reference period (dimensionless)
- `tolerance_used`: human-readable description of the rule applied.

### Period tolerance: `_check_period_tolerance`

- Units: days.
- Primary rule: relative tolerance on period (`|ΔP|/|P| <= relative`).
- Optional harmonics: checks `{1/2, 1/3, 2, 3} × P` within the same relative tolerance.

### Epoch tolerance: `_check_t0_tolerance`

- Units:
  - `t0` values are in BTJD days.
  - tolerance is a **phase fraction** of an external `reference_period` (days).
- Wrap handling:
  - If `|Δt0|` is large, also checks modulo-period wrap: `min(Δ mod P, P - (Δ mod P)) / P`.
- Important requirement:
  - Callers must supply `reference_period` in the `t0_btjd` tolerance config (since `check_tolerance` only receives t0 values).
  - Negative `reference_period` is normalized via `abs()` for safety.

### Depth tolerance: `_check_depth_tolerance`

- Relative tolerance on `depth` (caller must ensure consistent units: ppm-to-ppm or fraction-to-fraction).
- If `original == 0`, falls back to absolute comparison against the same numeric threshold.

### Default tolerance: `_check_default_tolerance`

- Absolute tolerance fallback when no specialized rule exists.

### Dispatcher: `check_tolerance(name, original, replayed, tolerances)`

- Uses per-parameter config if present, else falls back to `tolerances["default"]`, else a hardcoded absolute tolerance of `0.001`.

## File: `api/tolerances.py`

- Public API re-export surface: delegates to `utils/tolerances.py` (no behavior changes).

## Fixes / follow-ups

- Clarified and hardened `t0_btjd` tolerance handling: `reference_period` is treated as an absolute (positive) period for phase computation.

