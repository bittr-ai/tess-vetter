# Spec: Extended Metrics Checks Wired Into `vet_candidate`

Status: draft  
Owner: `bittr-tess-vetter`  
Motivation driver: expand the “golden path” (`vet_candidate`) to optionally include additional *metrics-only* vetting diagnostics already present in the codebase (model competition, ephemeris reliability/sensitivity, alias diagnostics, ghost features, sector consistency), without introducing hard verdict thresholds in library code.

## 1) Problem Statement

`bittr-tess-vetter` already contains advanced diagnostics modules exposed through `bittr_tess_vetter.api.*`, but the default `vet_candidate()` run (via `register_all_defaults()`) does not include them. As a result:
- users see “15 checks” as the effective full pipeline, even though richer diagnostics exist;
- notebooks/tutorials must hand-wire extra steps ad hoc;
- host apps that want “Robovetter-like completeness” must reinvent wiring logic.

We need a first-class, *opt-in* extended check set that remains faithful to the project’s “metrics-first, policy-in-host” design.

## 2) Goals / Non-Goals

### Goals
- Add an **extended** vetting check set that can be enabled from the main API without breaking the default meaning of “full vetting”.
- Each added check must be:
  - **metrics-only**: compute features + return flags/notes; do not embed pass/fail thresholds as a “verdict”.
  - **deterministic and reproducible** given the same inputs (modulo explicitly stochastic inputs).
  - **consistent with result schema**: `status` ∈ {`ok`, `skipped`, `error`}.
- Preserve current ergonomics: `vet_candidate(...)` stays the single-call entry point.

### Non-Goals (v1)
- No attempt to reproduce a curated Kepler/TESS policy layer (e.g., FA/FP/PC classification).
- No “auto-download everything” requirements (checks may depend on optional inputs; missing inputs should `skipped`).
- No redefinition of existing check IDs or semantics.

## 3) Current Pattern (Baseline)

Key components:
- `src/bittr_tess_vetter/api/vet.py:vet_candidate` orchestrates:
  - public API types → internal types
  - `VettingPipeline(checks=..., registry=...)`
  - `register_all_defaults(registry)` to populate the default registry
- Result contract is standardized via `src/bittr_tess_vetter/validation/result_schema.py`:
  - “ok” means “computed successfully”, not “passed a threshold”
  - metrics are JSON-serializable scalars in `result.metrics`
  - policy is explicitly out of scope (“host decides”)

This spec extends that pattern by adding *new checks* + *new registration function(s)*, not by adding post-hoc policy logic.

## 4) Proposed API Changes

### 4.1 `vet_candidate` optional preset

Add a new kwarg:
```python
def vet_candidate(..., preset: Literal["default", "extended"] = "default", ...) -> VettingBundleResult
```

Behavior:
- `preset="default"`: current behavior; uses `register_all_defaults`.
- `preset="extended"`: uses a new registry registration function (see below) that includes all default checks plus new extended metrics checks.

Notes:
- This is a convenience layer; callers can still supply explicit `checks=[...]`.
- Default remains stable: no behavior change unless explicitly requested.

### 4.2 New registration helper

Add:
- `register_extended_defaults(registry: CheckRegistry) -> None`

Implementation:
- call existing `register_all_defaults(registry)`
- then register additional checks defined in this spec.

## 5) Proposed New Checks (Metrics-Only)

All checks below must follow the wrapper pattern:
- wrapper lives in `src/bittr_tess_vetter/validation/checks_*_wrapped.py`
- wrapper calls compute implementation from `src/bittr_tess_vetter/compute/*` or `src/bittr_tess_vetter/validation/*`
- wrapper returns `ok_result/skipped_result/error_result`

### V16: `model_competition` (Transit vs Alternatives)

Purpose:
- Provide AIC/BIC-based model comparison to assess whether a transit-only model is preferred over plausible alternatives (e.g., sinusoid/systematics/EB-like), and whether the period is near known artifact aliases.

Inputs:
- requires LC arrays + ephemeris; may optionally use `context` for additional priors.

Uses:
- `bittr_tess_vetter.api.model_competition.run_model_competition` (already implemented).

Output metrics (example; keep stable names once shipped):
- `best_model`: str (e.g., `"transit"|"transit+sinusoid"|"eb_like"|...`)
- `aic_transit`, `aic_transit_sinusoid`, `aic_eb_like` (float)
- `delta_aic_transit_minus_best` (float; 0 if transit is best)
- `bic_transit_minus_best` (float)
- `period_alias_class`: str (e.g., `"none"|"near_known_artifact"|"harmonic_suspect"`)
- `artifact_prior` (float)

Flags (non-policy):
- `MODEL_PREFERS_NON_TRANSIT` (if best_model != transit)
- `PERIOD_NEAR_KNOWN_ARTIFACT`

### V17: `ephemeris_reliability_regime`

Purpose:
- Compute reliability-regime diagnostics: phase-shift null, period neighborhood confusability, and few-point dominance metrics.

Inputs:
- LC arrays + ephemeris.

Uses:
- `bittr_tess_vetter.api.ephemeris_reliability.compute_reliability_regime_numpy`.

Output metrics:
- `phase_shift_null_p_value`
- `period_peak_to_next_ratio`
- `max_ablation_score_drop`
- `n_in_transit_points` (if available)

Flags:
- `EPHEMERIS_NOT_SPECIFIC` (when diagnostics indicate confusability; purely descriptive)

### V18: `ephemeris_sensitivity_sweep`

Purpose:
- Quantify how stable the candidate’s score/depth is under small perturbations of `t0` (and optionally mild detrend variants), to detect “knife-edge” ephemerides.

Inputs:
- LC arrays + ephemeris.

Uses:
- `bittr_tess_vetter.api.ephemeris_sensitivity_sweep.run_ephemeris_sensitivity_sweep_numpy` (or current public entry point; align to actual function names when implementing).

Output metrics:
- `score_spread_iqr_over_median`
- `depth_spread_iqr_over_median`
- `best_variant_detrender` (if variants used)

Flags:
- `EPHEMERIS_HIGH_SENSITIVITY`

### V19: `alias_diagnostics`

Purpose:
- Compute harmonic/alias indicators (e.g., evidence for P/2, 2P, etc.) and secondary-phase significance summaries.

Inputs:
- LC arrays + ephemeris.

Uses:
- `bittr_tess_vetter.api.alias_diagnostics.*` helpers.

Output metrics:
- `best_harmonic`: str
- `best_harmonic_score`: float
- `secondary_significance_sigma`: float

Flags:
- `ALIAS_CANDIDATE_PRESENT`

### V20: `ghost_features` (Pixel/Systematics Diagnostics)

Purpose:
- Quantify ghost/scattered-light-like signatures in the pixel data and difference images.

Inputs:
- requires `tpf` (or a FITS-backed equivalent if/when supported).

Uses:
- `bittr_tess_vetter.api.ghost_features.compute_ghost_features` (already implemented).

Output metrics (examples; align to actual GhostFeatures fields):
- `spatial_uniformity`
- `edge_gradient`
- `prf_likeness`
- `aperture_contrast`

Flags:
- `GHOST_FEATURES_PRESENT` (descriptive)

### V21: `sector_consistency` (Host-Provided Measurements)

Purpose:
- Classify whether per-sector transit measurements are mutually consistent.

Inputs:
- **not derivable from stitched LC alone** without additional bookkeeping. This check must consume a `context` payload:
  - `context["sector_measurements"] = [{"sector": int, "depth_ppm": float, "depth_err_ppm": float}, ...]`

Uses:
- `bittr_tess_vetter.validation.sector_consistency.compute_sector_consistency`.

Output metrics:
- `consistency_class`: str (`"EXPECTED_SCATTER"|"INCONSISTENT"|...`)
- `chi2_p_value`: float
- `n_sectors_used`: int
- `outlier_sectors`: str (comma-separated) or `raw` if structured is needed

Flags:
- `SECTOR_INCONSISTENT`
- `SKIPPED:NO_SECTOR_MEASUREMENTS` when missing

## 6) Wiring Details

### 6.1 Where checks live

Add wrapper modules consistent with existing structure:
- `src/bittr_tess_vetter/validation/checks_model_competition_wrapped.py`
- `src/bittr_tess_vetter/validation/checks_ephemeris_diagnostics_wrapped.py`
- `src/bittr_tess_vetter/validation/checks_ghost_features_wrapped.py`
- `src/bittr_tess_vetter/validation/checks_sector_consistency_wrapped.py`

### 6.2 Registry additions

Extend `src/bittr_tess_vetter/validation/register_defaults.py`:
- keep `register_all_defaults()` unchanged
- add `register_extended_defaults()` which calls it and then registers V16–V21.

### 6.3 `vet_candidate` selection logic

Update `src/bittr_tess_vetter/api/vet.py`:
- accept `preset="default"|"extended"`
- if `checks` is explicitly provided, it wins (no preset registration changes needed beyond ensuring the check exists in the registry)
- if `checks` is `None`, choose registry registration based on `preset`

## 7) Metrics vs Thresholds (Hard Rule)

Checks may emit:
- numerical metrics
- descriptive flags
- notes explaining limitations/skips

Checks must not:
- decide “planet vs false positive”
- encode mission-specific operating points as hard cutoffs in library code
- label `status="error"` for “failed a threshold”; `error` is reserved for computation failure

If a check needs a numerical threshold to generate a *flag*, that threshold must be:
- explicitly described as a “flagging heuristic”, and
- configurable via `context` or a check config object, and
- defaulted conservatively.

Prefer: compute continuous metrics, let hosts threshold.

## 8) Compatibility / Rollout

- Additive change: no behavior changes when `preset` is omitted.
- New check IDs V16–V21 are stable once shipped; avoid renaming.
- Tutorials can switch to `preset="extended"` to include richer diagnostics without bespoke wiring.

## 9) Open Questions

- Check ID numbering: confirm V16–V21 do not conflict with planned IDs.
- Sector-consistency measurement contract: choose canonical keys (`depth_ppm`, `depth_err_ppm`, optional `duration_hours`).
- Whether to include these in `register_all_defaults` in a future major release.

