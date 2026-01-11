# Module Review: `api/lightcurve.py` (+ `domain/lightcurve.py`, `io/mast_client.py`)

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

`LightCurveData` is the core container that essentially every computation consumes. If its units, masking,
or normalization semantics drift, every downstream “physics” calculation can silently become wrong.

This module should remain a **data/contract layer** (no pass/fail policy).

## Scope (files / entrypoints)

API layer:
- `src/bittr_tess_vetter/api/lightcurve.py` (`LightCurveRef`, re-export `LightCurveData`, `make_data_ref`)

Core contract:
- `src/bittr_tess_vetter/domain/lightcurve.py` (`LightCurveData`, `make_data_ref`)

Primary producer:
- `src/bittr_tess_vetter/io/mast_client.py` (`MASTClient.download_lightcurve` builds `LightCurveData`)

## Audit checklist

### Units + conventions

- [x] `time` is BTJD days (float64) everywhere; no JD/BJD mix (by contract; producer uses `lc.time.value`)
- [x] `flux` is normalized (median ~1) and `flux_err` follows the same normalization
- [x] `cadence_seconds` is derived from time deltas in **seconds** and is robust to gaps/outliers (median positive finite dt)
- [x] `valid_mask` semantics are consistent: True means usable and implies finite time/flux/flux_err

### Masking + robustness

- [x] `valid_mask` construction excludes quality-flagged points deterministically (`(quality & mask) == 0`)
- [x] Derived stats (`duration_days`, `median_flux`, `flux_std`, `gap_fraction`) behave sensibly for edge cases
- [x] Arrays are immutable in the container to avoid cache mutation bugs

### API surface correctness

- [x] `make_data_ref` format is stable and used consistently across cache layers (`lc:{tic_id}:{sector}:{flux_type}`)
- [x] `LightCurveRef.from_data()` is deterministic and contains no raw arrays
- [x] No “policy-ish” outputs (PASS/WARN/REJECT) appear in this layer

### Tests

- [x] Contract tests cover dtype/shape validation and immutability (`tests/test_api/test_lightcurve_api.py`)
- [x] Edge cases: empty LC and all-invalid mask (`tests/test_api/test_lightcurve_api.py`)
- [x] Producer test: `download_lightcurve()` normalization keeps median ~1 and `cadence_seconds` in a reasonable range (`tests/io/test_mast_client.py`)

## Notes (audit)

### Current contract (as implemented)

- `domain/lightcurve.LightCurveData` enforces:
  - dtypes: `time/flux/flux_err` float64, `quality` int32, `valid_mask` bool
  - shape alignment across all arrays
  - immutability (`arr.flags.writeable = False`) for safe caching/reuse
- Derived properties:
  - `duration_days`: uses `valid_mask & isfinite(time)` and returns 0 for empty/all-invalid
  - `median_flux`/`flux_std`: uses `valid_mask & isfinite(flux)`; returns NaN for all-invalid
  - `gap_fraction`: `1 - n_valid/n_points` (0 for empty)

### Producer behavior (MASTClient)

- `valid_mask` is built as: finite time/flux/flux_err AND `(quality & mask) == 0`.
- If `normalize=True`, `flux` and `flux_err` are divided by the median of `flux_raw[valid_mask]` (only if that median is finite and > 0).
- `cadence_seconds` is derived from the median of positive finite `dt = diff(time[valid_mask])` (falls back to 120s if unavailable).

### Potential footguns / follow-ups

- The normalization semantics are “median of valid raw flux”. This is good for stability, but callers must remember that
  absolute flux units are lost; pixel-level tools should use TPF flux, not LC flux.
- `quality_flags_present` currently returns unique values from `quality` (not only flagged ones). That’s fine as “what values exist”,
  but if someone interprets it as “flags used”, they may want to drop `0`.
