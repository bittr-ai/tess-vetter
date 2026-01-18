# Spec: Refactor Primitives Before Validation Workflow

## Purpose

Before implementing an end-to-end reusable “TOI validation pipeline”, we should extract the notebook-only glue into **reusable, testable API helpers** that fit the current `bittr_tess_vetter.api.*` architecture:

- `api/pipeline.py` remains the **check runner** (registry + requirement gating).
- The new work should live in `api/*workflow*` style helpers (composition + structured outputs).

This refactor-first step reduces complexity, prevents duplicated logic, and makes the eventual pipeline implementation largely “orchestration only”.

## Goals

- Extract the **per-sector transit-masked detrending + normalization** logic into a reusable API helper.
- Extract the **depth estimation used in tutorial workflows** into a reusable helper (baseline + detrended).
- Provide a cache hydration helper for TRICERATOPS that **preserves quality flags** when available.
- Define a small **artifact writing contract** (manifest/packet/logs) without entangling the check runner.
- Keep all outputs **metrics-first** and **policy-free**.

## Non-goals

- No changes to check definitions/algorithms beyond moving orchestration glue.
- No subjective “PASS/FAIL” thresholds added in these helpers.
- No requirement to support “download from MAST” in this refactor step (offline/local dataset compatibility is enough).

## Current pain points (from tutorial 10)

1. **Detrending is notebook-specific**
   - Transit-masked WOTAN detrending logic lives in the notebook.
   - There’s no reusable object capturing per-sector detrend diagnostics and the stitched detrended product.

2. **Quality flags are inconsistently propagated**
   - Stitching now preserves quality flags in tutorial 10, but other helpers (e.g. TRICERATOPS cache hydration) may not.

3. **Depth selection is inconsistent across stages**
   - Baseline depth, detrended depth, and the “depth used for FPP” must be explicitly computed and recorded.

4. **Artifact/report writing is not a library primitive**
   - Notebook writes “packet JSON” ad hoc; a pipeline should have a stable artifact layout.

## Proposed refactor deliverables

### 1) `api/detrend_per_sector.py` (new module)

**API**:

```python
result = btv.detrend_per_sector_transit_masked(
    lc_by_sector: dict[int, LightCurve],
    ephemeris: Ephemeris,
    *,
    method: str = "wotan_biweight",
    window_days: float = 0.5,
    mask_duration_multiplier: float = 1.5,
    normalize: bool = True,
    finite_filter: bool = True,
)
```

**Output type**: `PerSectorDetrendResult` (new dataclass in this module)

Fields (minimum):
- `schema_version: int`
- `detrended_by_sector: dict[int, LightCurve]` (quality preserved/filtered consistently)
- `per_sector_diagnostics: list[dict[str, Any]]` (n_points, time range, flux stats, warnings)
- `method: str`, `window_days: float`, `mask_duration_multiplier: float`
- `warnings: list[str]`

Notes:
- Must be **pure** (no disk writes).
- Must not require network.
- Must preserve `quality` where possible:
  - if original LC has `quality`, carry it through finite filtering.
  - if absent, treat as zeros (but record that in provenance/diagnostics).

### 2) `api/depth_estimation.py` (new module)

**API**:

```python
depth = btv.estimate_transit_depth_ppm(
    lc: LightCurve,
    ephemeris: Ephemeris,
    *,
    buffer_factor: float = 3.0,
)
```

**Output**: `DepthEstimate` (dataclass)
- `depth_ppm_hat: float`
- `depth_ppm_err: float`
- `n_points_total: int`
- `n_points_in: int`
- `n_points_oot: int`
- `provenance: dict[str, Any]` (method + params)

Rationale:
- The pipeline needs a consistent, reusable “depth estimate” function to avoid duplicated logic.

### 3) `api/fpp_helpers.py` enhancement (new helper)

Add:

```python
cache = btv.hydrate_cache_from_lc_by_sector(
    lc_by_sector: dict[int, LightCurve],
    tic_id: int,
    *,
    flux_type: str = "pdcsap",
    cache_dir: str | Path | None = None,
    cadence_seconds: float = 120.0,
    sectors: list[int] | None = None,
)
```

Behavior:
- Reuses `LightCurve` arrays; preserves `quality` if present.
- Applies finite filtering and sets `valid_mask`.

Keep existing `hydrate_cache_from_dataset(...)` for backwards compatibility.

### 4) `api/artifacts.py` (new module)

Define a minimal artifact contract for the eventual pipeline. This module should:
- create deterministic folder layouts
- write JSON packets (with stable key ordering)
- write logs passed in as strings

**API**:

```python
paths = btv.write_validation_artifacts(
    output_dir: str | Path,
    *,
    packet: dict[str, Any],
    manifest: dict[str, Any],
    logs: dict[str, str] | None = None,
)
```

Constraints:
- No implicit global state; everything passed in.
- Avoid embedding absolute paths unless explicitly requested (support both relative + absolute in the manifest).

### 5) Testing additions

Add targeted tests:
- `detrend_per_sector_transit_masked`:
  - finite filtering does not crash on NaNs/infs
  - preserves `quality` length alignment
  - deterministic given fixed inputs
- `estimate_transit_depth_ppm`:
  - consistent output for a tiny synthetic transit
- `hydrate_cache_from_lc_by_sector`:
  - writes per-sector keys and preserves `quality`
- `write_validation_artifacts`:
  - produces expected files with JSON-serializable payloads

## Integration points (existing architecture)

- `api/workflow.py` can be upgraded to optionally use:
  - `estimate_transit_depth_ppm`
  - `detrend_per_sector_transit_masked` (for “robustness rerun”)
- `api/per_sector.py` continues to own per-sector vetting orchestration, but may reuse the new detrend helper when needed.

## Success criteria

- Tutorial 10 can be rewritten to call the new helpers, with reduced bespoke code.
- The new helpers do not add policy; they only emit metrics + diagnostics.
- Unit tests cover the new helpers and pass in CI.

## Rollout / sequencing

1. Add `api/detrend_per_sector.py` + tests.
2. Add `api/depth_estimation.py` + tests.
3. Add `hydrate_cache_from_lc_by_sector` to `api/fpp_helpers.py` + tests.
4. Add `api/artifacts.py` + tests.
5. Refactor tutorial 10 to use these helpers (separate commit).
6. Then implement the pipeline orchestrator (next spec).

