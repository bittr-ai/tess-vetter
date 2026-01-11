# Module Review: `validation/exovetter_checks.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

These checks (V11 ModShift, V12 SWEET) are used early in `run_vetting_pipeline` and can create extremely strong
false-positive signals if units are wrong (BTJD vs BJD), if a folded light curve is passed by mistake, or if raw
exovetter outputs are misinterpreted.

This module must remain **metrics-only**: it reports measurements but does not decide PASS/WARN/REJECT.

## Scope (functions)

- `_is_likely_folded` (heuristic guard to avoid calling exovetter on folded inputs)
- `_inputs_summary` (baseline/cadence metadata)
- `run_modshift` (V11)
- `run_sweet` (V12)

## Audit checklist (to fill)

### Units + conventions

- [x] `candidate.period` is in days; passed to `Tce(period=... * u.day)`
- [x] `candidate.t0` is BTJD days; passed to `Tce(epoch=... * u.day, epoch_offset=exo_const.btjd)`
- [x] `candidate.depth` is fractional; passed to `Tce(depth=depth_frac * 1e6 * ppm)`
- [x] `candidate.duration_hours` is hours; passed to `Tce(duration=... * u.hour)`
- [x] `_LightkurveLike.time_format == "btjd"` (consistent with BTJD epoch_offset)

### Output semantics

- [x] Always returns `VetterCheckResult(passed=None)` and sets `details["_metrics_only"]=True`
- [x] Any derived ratios (e.g. `secondary_primary_ratio`) are computed from raw ModShift signals without thresholds

### Data hygiene / edge cases

- [x] Folded-input detection (`_is_likely_folded`) blocks ModShift with `status="invalid"` + warning
- [x] Import failures/execution failures return `status="error"` with `EXOVETTER_*` warnings (no crash)
- [x] Confirm `_is_likely_folded` has basic unit test coverage and matches intended usage (it is a heuristic, not a validator)

### Physics correctness

- [x] ModShift primary/secondary/tertiary signals are taken directly from exovetter metrics (`pri/sec/ter`)
- [x] `Fred` and `false_alarm_threshold` are passed through from exovetter (`Fred`, `false_alarm_threshold`)
- [x] `secondary_primary_ratio` is defined as `sec/pri` with a `pri==0 → 0.0` fallback (host should treat `pri==0` as low-quality)
- [x] `run_sweet` is a pass-through: host must interpret `details["raw_metrics"]` and tolerate exovetter version drift

### Statistics / uncertainty

- [x] ModShift metrics are treated as dimensionless “signal” outputs from exovetter; this wrapper does not attempt to reinterpret units
- [x] `confidence` is a lightweight data-availability heuristic:
  - V11 uses `n_transits_expected/5` (capped to 1.0)
  - V12 uses `n_points/200` (capped to 1.0)
  This is explicitly not a physics confidence score; host should avoid mapping it directly to PASS/WARN.

### Tests

- [x] Unit test: folded input triggers `status="invalid"` for V11 (`tests/validation/test_exovetter_checks.py`)
- [x] Unit test: metrics dict is coerced to JSON scalars (no numpy scalars leak) (`tests/validation/test_exovetter_checks.py`, `_as_jsonable_metrics`)

## Notes (initial)

- `run_modshift` detects likely-folded input using baseline heuristics before importing/executing exovetter. This is
  critical because exovetter expects a time-series light curve, not a folded phase curve.
- Both `run_modshift` and `run_sweet` use a minimal `_LightkurveLike` wrapper and set a default `flux_err` when none
  is provided. That default can affect absolute “sigma”-like metrics; any host policy should treat these outputs as
  approximate unless flux errors are supplied by the upstream light curve pipeline.
