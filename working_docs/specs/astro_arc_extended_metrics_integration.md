# Spec: Wire `bittr-tess-vetter` Extended Metrics Into `astro-arc-tess`

Status: draft  
Owner: `bittr-tess-vetter` + `astro-arc-tess`  
Motivation driver: make the newly-added “extended metrics” checks (V16–V21) available to `astro-arc-tess`’s evidence-first `run_vetting_pipeline` tool, without embedding subjective thresholds or hard dispositions in either library.

## 1) Problem Statement

`astro-arc-tess` currently delegates a subset of vetting checks to `bittr-tess-vetter` inside:
- `astro_arc.validation.pipeline.run_vetting_pipeline` (builds `checks=list(enabled)` and calls `bittr_tess_vetter.api.vet.vet_candidate`)

`bittr-tess-vetter` now supports an opt-in extended check set (V16–V21) via:
- `register_extended_defaults()`
- `vet_candidate(..., preset="extended")`

But these extended checks are not yet executed in `astro-arc-tess` because:
- `astro-arc-tess` currently passes an explicit `checks=[...]` list limited to V01–V15 (+ exovetter) and never includes V16+.
- The `preset="extended"` registration must be chosen in a way that is compatible with explicit check selection (i.e., the registry must include V16+ even when `checks` is provided).

## 2) Goals / Non-Goals

### Goals
- Make V16–V21 available in `astro-arc-tess`’s `run_vetting_pipeline` output as additional **EvidenceItem** entries (metrics + flags).
- Preserve **metrics-only** semantics end-to-end:
  - no “PASS/WARN/REJECT” in `bittr-tess-vetter`
  - no hard-coded thresholds in `astro-arc-tess` (guardrails are allowed, but must be explicit and configurable)
- Keep runtime predictable:
  - allow callers to enable/disable extended checks
  - skip gracefully when required inputs aren’t present (e.g., no TPF aperture mask)

### Non-Goals
- Not attempting to reproduce LEO/Robovetter policy thresholds (FA/FP classification).
- Not requiring multi-sector per-sector depth fits in v1 (V21 should be opt-in via context).

## 3) Current Architecture (Relevant Pieces)

### 3.1 Where `astro-arc-tess` calls `bittr-tess-vetter`

In `astro_arc.validation.pipeline.run_vetting_pipeline`:
- constructs `enabled` check IDs via `_build_enabled_checks(...)`
- calls `vet_candidate(..., checks=list(enabled), ...)`
- converts each `CheckResult` into `EvidenceItem` via `_evidence_from_bittr_check`

### 3.2 Check titles

`astro_arc.validation.pipeline._CHECK_TITLES` currently maps only V01–V15 (+ V11b). New IDs will still render (fallback uses check name), but titles should be added for consistency.

## 4) Proposed Integration

### 4.1 Add an “extended metrics” switch to `run_vetting_pipeline`

In `astro-arc-tess`:
- Add a boolean input to:
  - `astro_arc.validation.vetting_input.RunVettingPipelineInput`
  - the MCP tool signature in `astro_arc.mcp_tools.run_vetting_pipeline`

Proposed name:
- `bittr_extended_metrics: bool = False`

Semantics:
- `False`: current behavior (only V01–V15 subset chosen by `_build_enabled_checks`)
- `True`: include V16–V21 when inputs permit (see 4.3), and call `bittr` with `preset="extended"`

### 4.2 Ensure registry contains extended checks even when `checks` is provided

`astro-arc-tess` uses explicit `checks=[...]` to avoid running checks that are invalid for folded/recovered inputs.

Therefore, `bittr-tess-vetter.api.vet.vet_candidate` should treat `preset` as controlling the registry registration regardless of whether `checks` is passed.

Required behavior (in `bittr-tess-vetter`):
- If `preset == "extended"`: register `register_extended_defaults(registry)` (which includes default + V16–V21)
- Else: register `register_all_defaults(registry)`
- Then `VettingPipeline(checks=checks, ...)` selects which ones actually run

This avoids “unknown check ID” errors when `astro-arc-tess` requests V16+ explicitly.

### 4.3 Extend `_build_enabled_checks` to include V16–V21 (metrics-only)

Add to `astro_arc.validation.pipeline._build_enabled_checks`:

When `lightcurve_kind == "time_series"` and `bittr_extended_metrics=True`:
- Always add:
  - `V16` (model competition)
  - `V17` (ephemeris reliability regime)
  - `V18` (ephemeris sensitivity sweep)
  - `V19` (alias diagnostics)

When pixel data is present:
- Add `V20` only if the provided TPF stamp includes an aperture mask (see 4.4).
  - Otherwise allow `V20` to be skipped with a clear reason.

When `context["sector_measurements"]` is provided:
- Add `V21` (sector consistency).
  - Otherwise skip.

Notes:
- V16–V19 are “LC-only” in terms of required arrays, but they are **not** appropriate for folded-only inputs, so keep them time-series only in v1.
- V21 is explicitly host-provided; do not attempt to derive sector measurements in the `bittr` library.

### 4.4 Provide optional aperture mask to enable `V20` (ghost features)

`bittr-tess-vetter`’s V20 implementation requires:
- `tpf.flux`, `tpf.time`, and `tpf.aperture_mask`

Today, `astro-arc-tess` constructs a minimal `TPFStamp(time, flux)` (no aperture mask).

Two options:
1) **v1 (simple):** do not enable V20 from `astro-arc-tess`; let it remain skipped.
2) **v1.1 (recommended):** when `tpf_fits_ref(s)` are available, attach the SPOC aperture mask (or a chosen analysis aperture) into the `TPFStamp` passed to `bittr` so V20 can run.

This spec proposes v1.1; v1 can ship without it.

### 4.5 Evidence output format

No changes required: `_evidence_from_bittr_check` already:
- uses `result.details` and adds `engine="bittr_tess_vetter"`
- flags “check_unknown” when metrics-only

Add titles in `_CHECK_TITLES`:
- `V16`: “Model competition (transit vs alternatives)”
- `V17`: “Ephemeris reliability regime”
- `V18`: “Ephemeris sensitivity sweep”
- `V19`: “Alias/harmonic diagnostics”
- `V20`: “Ghost/scattered-light features (pixel)”
- `V21`: “Sector-to-sector consistency”

## 5) Data Contracts

### 5.1 `bittr` preset

`astro-arc-tess` calls:
```python
bundle = vet_candidate(
  lc, cand,
  ...,
  preset="extended" if bittr_extended_metrics else "default",
  checks=list(enabled),
  context={...},
)
```

### 5.2 `V21` sector measurements

If/when `astro-arc-tess` wants V21:
- Provide `context["sector_measurements"] = [...]` with dict entries:
  - `sector: int`
  - `depth_ppm: float`
  - `depth_err_ppm: float`
  - optional: `duration_hours`, `duration_err_hours`, `n_transits`, `shape_metric`, `quality_weight`

## 6) Guardrails (Optional, Host Policy)

This spec does **not** require new guardrails, but suggests two low-risk additions:
- If V16 strongly prefers a non-transit model (winner != `transit_only`), emit a guardrail like `MODEL_COMPETITION_NON_TRANSIT` (severity warn).
- If V18 indicates high sensitivity (large spreads), emit `EPHEMERIS_SENSITIVE` (severity warn).

These should be implemented purely in `astro-arc-tess` (host policy), not in `bittr-tess-vetter`.

## 7) Testing Plan

### 7.1 `bittr-tess-vetter`
- Unit test: `vet_candidate(..., preset="extended", checks=["V16"])` does not error and returns a `CheckResult` for V16.
- Unit test: `vet_candidate(..., preset="extended", checks=["V20"])` returns `skipped` when aperture mask is missing (no crash).

### 7.2 `astro-arc-tess`
- Integration test: `run_vetting_pipeline` with `bittr_extended_metrics=True` includes EvidenceItems for V16–V19.
- If v1.1 implemented: with a FITS-backed TPF carrying aperture mask, V20 is present and `status="ok"`.

## 8) Rollout

1) Land `bittr` preset registration behavior (4.2).
2) Land `astro-arc-tess` input flag + enabled-check expansion + titles.
3) Optionally land V20 aperture mask plumbing.

