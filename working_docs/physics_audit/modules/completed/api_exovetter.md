# Module Review: `api/exovetter.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is high leverage

This module is the public API surface for exovetter-derived diagnostics (V11/V12). It is a boundary between:
- facade types (`LightCurve`, `Candidate`, `CheckResult`), and
- the internal exovetter wrappers (`validation/exovetter_checks.py`), which implement the actual ModShift/SWEET computations.

The key physics risks at this layer are unit/field conversion mistakes (depth fraction vs ppm, days vs hours) and inconsistent “skipped/error/metrics-only” semantics that change downstream interpretation.

## File: `api/exovetter.py`

### Inputs / unit conventions

- `LightCurve.time`: days (BTJD expected for TESS).
- `Candidate.ephemeris.period_days`: days
- `Candidate.ephemeris.t0_btjd`: BTJD days (same time base as `LightCurve.time`)
- `Candidate.ephemeris.duration_hours`: hours
- `Candidate.depth`: **fractional depth** (0–1); derived from either:
  - `Candidate.depth_fraction`, or
  - `Candidate.depth_ppm / 1e6`

### Helper: `_candidate_to_internal(candidate)`

- Converts facade `Candidate` → internal `TransitCandidate`.
- Requires depth (fraction). If missing, raises `ValueError`.
- Sets `snr=0.0` as a placeholder (exovetter checks do not use SNR).

### Helper: `_convert_result(result)`

- Converts internal `VetterCheckResult` (pydantic) → facade `CheckResult` (dataclass).
- Preserves `passed=None` semantics for metrics-only checks and copies `details` dict.

### Helper: `_make_skipped_result(...)`

- Returns a `CheckResult` with:
  - `passed=None`, `confidence=0.0`
  - `details={"status":"skipped","reason":...}`
- This is used when the caller disables a check.

### Function: `modshift(...)` (V11)

- If disabled:
  - returns a skipped result via `_make_skipped_result("V11", ...)`.
- If depth missing:
  - returns `passed=None` with `details={"status":"error","reason":"Candidate depth is required ..."}` and `confidence=0.20`.
- Otherwise:
  - converts inputs and calls `validation.exovetter_checks.run_modshift(...)`.
  - returns metrics-only `CheckResult` (`passed=None`) with details including `_metrics_only=True`.

### Function: `sweet(...)` (V12)

Same wrapper semantics as `modshift`, calling `validation.exovetter_checks.run_sweet(...)`.

### Function: `vet_exovetter(...)`

- Orchestrates V11 then V12.
- Always returns a two-element list (one per check), with “skipped” results when disabled by `enabled`.
- `config` is accepted but currently unused beyond per-check routing (reserved; thresholds/policy belong in host apps).

## Cross-references

- Core exovetter computations: `working_docs/physics_audit/modules/completed/validation_exovetter_checks.md`

## Fixes / follow-ups

No physics correctness issues identified in this wrapper.

