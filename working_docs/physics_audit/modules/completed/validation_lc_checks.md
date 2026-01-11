# Module Review: `validation/lc_checks.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

These are the canonical “LC-only” vetting computations (V01–V10). They run constantly and any unit mistake
(period/t0/duration) or baseline/windowing bug propagates into most downstream interpretation.

This module must remain **metrics-only**: it emits measurements + warnings; host owns any guardrail policy.

## Scope (functions)

Tier 1 (LC-only):
- `check_odd_even_depth` (V01)
- `check_secondary_eclipse` (V02)
- `check_duration_consistency` (V03)
- `check_depth_stability` (V04)
- `check_v_shape` (V05)

## Audit checklist (to fill)

### Units + conventions

- [x] Inputs are consistently: `period` (days), `t0` (BTJD days), `duration_hours` (hours)
- [x] Phase convention: `phase = ((time - t0)/period) % 1` with `phase_dist = min(phase, 1-phase)` centers the transit at phase 0
- [x] Depth outputs are dual-labeled where needed:
  - fractional keys preserved in some checks (legacy)
  - ppm keys added consistently (`*_ppm`)
  - all checks remain metrics-only (`passed=None`, `_metrics_only=True`)

### Data hygiene / edge cases

- [x] All LC-only checks operate on `time/flux = lightcurve.*[lightcurve.valid_mask]`
- [x] Minimum-transit / minimum-points gating emits `warnings` and/or `status` with reduced `confidence` (not policy)
- [x] “duration too long relative to period” is emitted as a warning and results in a low-confidence metrics-only return

### Physics correctness

- [x] V01: per-epoch depth uses local baselines; aggregation uses median depth per parity and sigma ~ median(epoch_sigma)/sqrt(n_transits)
- [x] V02: secondary window defaults to 0.35–0.65 (center 0.5, half-width 0.15); baseline windows are adjacent and explicitly avoid phase near transit (0–0.1, 0.9–1.0)
- [x] V03: expected duration uses solar scaling `13h * (P/yr)^(1/3)` and applies density correction `T ∝ ρ^{-1/3}` when stellar params available
- [x] V04: per-epoch depth uses local baselines similar to V01; computes reduced chi2 from per-epoch depths vs uncertainties (with optional red-noise inflation)
- [x] V05: trapezoid model uses a symmetric trapezoid parameterized by `tflat_ttotal_ratio`; depth is estimated by linear LS for each candidate ratio and the best ratio is chosen by min(chi2)
  - Bootstrap uncertainty resamples in-transit points with replacement (seeded RNG) and reports CI-derived half-width as `tflat_ttotal_ratio_err`
  - Edge depth diagnostics (`depth_bottom`, `depth_edge`, `shape_ratio`) are computed from median flux in phase sub-windows and are baseline-normalized

### Statistics / uncertainty

- [x] Confidence is primarily data-quantity/coverage driven (N transits, N in-transit points, warnings), not a guardrail verdict
- [x] MAD / red-noise inflation is used only to inflate uncertainty estimates (no embedded pass/fail thresholds)

### Tests

- [x] Existing tests cover V01/V02/V04/V05 metrics-only behavior on synthetic data (`tests/validation/test_odd_even_depth.py`, `tests/validation/test_secondary_and_depth_stability.py`, `tests/validation/test_v_shape.py`)
- [x] Added “gap / baseline sensitivity” regression test for V01 (long gap handled without crashing and still yields usable transit counts)

## Notes (initial)

- V03 is now metrics-only (`passed=None`) and reports `duration_ratio` + `density_corrected` + stellar params when available.
- V05 explicitly returns `status="invalid_duration"` when `duration_days >= 0.5 * period` (metrics-only).
