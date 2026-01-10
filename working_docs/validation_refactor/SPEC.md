# Validation Refactor Spec: Metrics-Only `bittr-tess-vetter` + Host Guardrails

## Goal

Make `bittr-tess-vetter` a **pure computation / metrics** library:
- no PASS/WARN/REJECT aggregation
- no disposition classification
- no threshold-derived booleans (e.g., `suspicious`, `significant_secondary`, `classification`)
- no “interpretation” strings that embed policy

All policy/guardrail decisions move to the host application (currently `astro-arc-tess`).

## Non-goals

- Preserve backward compatibility with previous return shapes (single client).
- Redesign the host’s overall guardrail framework; only extend it to cover metrics that no longer include booleans.

## Definitions

- **Metric**: numeric/statistical output (e.g., depth ppm, sigma, rel diff, coverage fractions).
- **Policy output**: any PASS/WARN/REJECT, “validated planet” style claims, or threshold-derived boolean classification (or strings derived from those).

## Current Problems

1) `bittr_tess_vetter.validation.base` contains policy helpers:
   - `compute_verdict`, `compute_disposition`, `generate_summary`, `aggregate_results`
   - an `AggregationConfig` whose purpose is policy orchestration

2) Multiple validation functions embed policy via booleans:
   - V01 odd/even: `details["suspicious"]`
   - V02 secondary: `details["significant_secondary"]`
   - V05 v-shape: `details["classification"]` and legacy shape labels implying policy
   - `transit/OddEvenResult.is_suspicious` and `to_dict()["interpretation"]`

3) Several class-based “checks” return `passed=True/False` with thresholds in docstrings and configs (e.g. `validation/checks_basic.py`, `validation/checks_pixel.py`, `validation/exovetter_checks.py`).

## Target Architecture

### In `bittr-tess-vetter`

**A) Keep**
- Array/mask primitives and measurement functions (phase folding, in/oot masks, depth measurement).
- Functions that compute per-check **metrics** and emit `VetterCheckResult` with:
  - `passed=None`
  - `details["_metrics_only"]=True`
  - numeric diagnostics + warnings

**B) Remove**
- Verdict/disposition/summary aggregation from the library entirely.
- Threshold booleans and “interpretation” strings.
- Any requirement that checks return `passed=True/False` based on thresholds.

**C) Naming**
- The package name `validation` can remain, but it must strictly mean “metrics for vetting checks”, not “final validation outcomes”.

### In the host (`astro-arc-tess`)

Move/implement all interpretation as guardrails operating on `Vxx.metrics`:
- V01 odd/even EB indicator (already exists in host guardrails)
- V05 transit-shape guardrail (already exists in host guardrails)
- V02 LC-secondary eclipse guardrail (needs to be added)
- Any other thresholds that were previously embedded in tess-vetter must be defined and versioned in the host’s guardrails config.

## Detailed Change List (tess-vetter)

### 1) Remove aggregation API surface

- Delete `src/bittr_tess_vetter/api/aggregation.py` (already done locally; ensure committed).
- Remove any exports/re-exports of aggregation helpers from `src/bittr_tess_vetter/api/__init__.py`.
- Remove any tests that assert PASS/WARN/REJECT behavior (see `tests/validation/test_aggregation_metrics_only.py`).

### 2) Remove policy types from the domain model

Update `src/bittr_tess_vetter/domain/detection.py`:
- Remove `Verdict`, `Disposition`, and `ValidationResult`.
- Remove `Detection.validation` and any helpers like `is_planet_candidate` that imply disposition.

Update `src/bittr_tess_vetter/api/detection.py`:
- Stop re-exporting `Verdict`, `Disposition`, `ValidationResult`.

Update `src/bittr_tess_vetter/domain/__init__.py` and `src/bittr_tess_vetter/api/__init__.py` accordingly.

### 3) Make all vetting check results metrics-only

Policy requirement:
- For *all* `VetterCheckResult` instances produced by this library’s vetting checks, set `passed=None` and include `details["_metrics_only"]=True`.

Files:
- `src/bittr_tess_vetter/validation/lc_checks.py`
  - Remove `details["suspicious"]` (V01).
  - Remove `details["significant_secondary"]` (V02).
  - Remove `details["classification"]` and legacy “shape” labels (V05).
  - Remove docstrings describing PASS/FAIL decision rules.
  - Keep: `delta_sigma`, `rel_diff`, depth ppm + errors, phase coverage, warnings, and any counts.

- `src/bittr_tess_vetter/transit/result.py` and `src/bittr_tess_vetter/transit/vetting.py`
  - Remove `OddEvenResult.is_suspicious` and any “interpretation” field in `to_dict()`.
  - Keep only metrics: depth_odd/even/diff, relative diff %, sigma, n_odd/n_even.

- Class-based checks (`src/bittr_tess_vetter/validation/checks_basic.py`, `checks_pixel.py`, `exovetter_checks.py`, etc.)
  - Ensure they return `passed=None` (metrics-only) and do not describe PASS/FAIL policy in docstrings.
  - Keep thresholds only if they are strictly *algorithm parameters* (not decision rules). Prefer removing threshold constants from these classes entirely.

### 4) Remove “deprecated/back-compat” language

Since there is a single client, remove “DEPRECATED”, “back-compat”, and similar migration messaging from code/docstrings in this library.

## Detailed Change List (host / astro-arc-tess)

### 1) Ensure host guardrails cover removed booleans

**V01 odd/even**
- Host already has an odd/even guardrail function that consumes sigma + rel diff.
- Ensure host uses:
  - `delta_sigma` and `rel_diff` (or `relative_depth_diff_percent` if available from some surfaces)
  - power-gating via `n_odd_transits`, `n_even_transits`

**V05 transit shape**
- Host already has a transit-shape guardrail.
- Ensure it consumes `tflat_ttotal_ratio` (primary), and does not rely on a categorical `classification` from tess-vetter.

**V02 LC-only secondary eclipse (NEW)**
- Add a guardrail function that interprets the metrics produced by tess-vetter V02:
  - `secondary_depth_ppm`
  - `secondary_depth_sigma`
  - `secondary_phase_coverage`
  - `n_secondary_events_effective`
  - `n_secondary_points` / `n_baseline_points`
- Add thresholds/config (in host guardrails config) for WARN/BLOCK tiers.
  - Initial parity thresholds can mirror the previous bittr “significant_secondary” computation:
    - sigma threshold (e.g., 3.0)
    - depth threshold (e.g., 0.005 fractional = 5000 ppm)
  - Host decides final policy language + corroboration logic.

### 2) Update host pipeline wiring

In the host vetting pipeline, evaluate the new V02 guardrail using V02 metrics.

## Test Plan

### tess-vetter (focused)

- Update/remove:
  - `tests/validation/test_aggregation_metrics_only.py` (aggregation no longer exists).
  - Assertions that check `suspicious`, `significant_secondary`, `classification`, `OddEvenResult.is_suspicious`, or `interpretation`.
- Replace with assertions on raw metrics:
  - V01 returns `delta_sigma`, `rel_diff`, counts; `passed is None`.
  - V02 returns secondary depth ppm/sigma, coverage, counts; `passed is None`.
  - V05 returns `tflat_ttotal_ratio` (and error), depth ppm, coverage; `passed is None`.

### host (smoke)

- Run the host tool(s) that consume these metrics and ensure:
  - guardrails still populate for V01/V05 as before (based on metrics).
  - new V02 guardrail triggers appropriately on synthetic/known EB-like cases.

## Acceptance Criteria

1) `bittr-tess-vetter` contains **no** PASS/WARN/REJECT aggregation helpers and no verdict/disposition types.
2) `bittr-tess-vetter` check outputs are metrics-only (`passed=None`, `_metrics_only=True`) and contain no threshold-derived booleans.
3) The host guardrails implement the policy interpretation for V01/V02/V05 based on metrics.
4) Focused unit tests pass in both repos for the touched surfaces.

