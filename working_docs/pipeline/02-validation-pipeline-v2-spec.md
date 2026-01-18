# Spec (v2): Reusable Validation Workflow Pipeline

## Summary

Introduce a reusable, composable “validation workflow pipeline” that reproduces the tutorial-style end-to-end analysis (baseline vetting → robustness rerun → AO-assisted FPP) as a single programmatic entrypoint.

This pipeline is **not** a replacement for `api/pipeline.py` (check runner). It is a **workflow orchestrator** implemented in `bittr_tess_vetter.api.*` that composes existing APIs and the refactor primitives from `01-refactor-first-spec.md`.

## Goals

- One-call workflow for researchers and downstream clients (CLI/services/notebooks).
- Outputs are:
  - **metrics-first**
  - **policy-free** (no subjective thresholds baked in)
  - **fully traceable** (artifact paths + config + seeds + logs)
- Supports:
  - local dataset mode (tutorial datasets)
  - cache-only mode (host provides LCs/TPFs)
  - optional network-assisted stages (catalog checks, FPP)
- Produces a single “validation packet” JSON + optional markdown/plots via an artifact writer.

## Non-goals

- Not a discovery search (TLS/BLS search) unless added later as an optional stage.
- Not a “planet validator verdict engine” — any “VALIDATED” label is a separate optional policy layer.

## Architectural placement

Add a new module:

- `src/bittr_tess_vetter/api/validation_workflow.py`

and supporting types:
- `ValidationWorkflowConfig`
- `ValidationWorkflowResult`

This module composes:
- `api/datasets.py` (offline datasets)
- `api/workflow.py` and/or `api/vet.py`
- `api/per_sector.py`
- `api/detrend_per_sector.py` (new)
- `api/depth_estimation.py` (new)
- `api/fpp.py` + `api/fpp_helpers.py`
- `api/artifacts.py` (new)
- `api/vetting_report.py` / `api/report.py` (formatting)

## Public API

```python
result = btv.run_validation_workflow(
    target=ValidationTarget(...),
    ephemeris=Ephemeris(...),
    stellar=StellarParams(...),
    config=ValidationWorkflowConfig(...),
)
```

### Types

```python
@dataclass(frozen=True)
class ValidationTarget:
    tic_id: int | None
    toi: str | None
    name: str | None
    ra_deg: float | None
    dec_deg: float | None

@dataclass(frozen=True)
class ValidationWorkflowConfig:
    schema_version: int = 1

    # Phase planning / execution
    #
    # The workflow is intentionally split into phases so callers can run:
    # - triage only (cheap checks, fast feedback)
    # - validation grade (robustness + FPP with replicates)
    #
    # Phases are deterministic and composable; each phase writes structured outputs
    # into the final packet whether or not later phases run.
    phases: tuple[Literal["phase1", "phase2", "phase3"], ...] = ("phase1", "phase2", "phase3")

    # Optional: stop early based on a policy function that consumes metrics/flags
    # (not subjective prose). This is *off* by default to preserve auditability.
    early_exit_enabled: bool = False
    early_exit_after_phase: Literal["phase1", "phase2"] = "phase1"
    early_exit_policy: Literal[
        "none",
        "strict_fp_triage",
        "strict_data_quality",
    ] = "none"

    # Inputs / sources
    mode: Literal["dataset", "inputs"]
    dataset_path: str | None = None
    lc_by_sector: dict[int, LightCurve] | None = None
    tpf_by_sector: dict[int, TPFStamp] | None = None

    # Vetting execution
    preset_baseline: Literal["default", "extended"] = "default"
    baseline_checks: list[str] | None = None
    network: bool = False

    # Robustness rerun
    detrend_enabled: bool = True
    detrend_method: str = "wotan_biweight"
    detrend_window_days: float = 0.5
    detrend_mask_duration_multiplier: float = 1.5
    detrend_window_sweep_days: list[float] | None = None
    rerun_sensitive_checks: list[str] = ("V12", "V16", "V17", "V19")

    # FPP
    fpp_enabled: bool = True
    fpp_depth_source: Literal["baseline", "detrended"] = "detrended"
    fpp_cache_dir: str | None = None
    fpp_preset: Literal["fast", "standard"] = "fast"
    fpp_overrides: dict[str, Any] | None = None
    fpp_replicates: int = 5
    fpp_seed: int | None = None
    fpp_timeout_seconds: float | None = None
    contrast_curve_path: str | None = None
    contrast_curve_filter: str | None = None

    # Outputs / artifacts
    output_dir: str | None = None
    write_artifacts: bool = True
    write_markdown_report: bool = False
    render_plots: bool = False
```

### Result

```python
@dataclass(frozen=True)
class ValidationWorkflowResult:
    schema_version: int
    target: dict[str, Any]
    ephemeris: dict[str, Any]
    stellar: dict[str, Any] | None

    baseline: dict[str, Any]  # stitched baseline + depth estimate
    baseline_bundle: VettingBundleResult
    per_sector: PerSectorVettingResult | None

    detrended: dict[str, Any] | None  # detrended stitched + depth estimate + sweep
    rerun: dict[str, Any] | None  # sensitive check rerun outputs + deltas

    fpp: dict[str, Any] | None  # fpp/nfpp + summaries + logs path

    packet: dict[str, Any]  # consolidated JSON-serializable packet
    artifact_paths: dict[str, str] | None
    provenance: dict[str, Any]
    warnings: list[str]
```

## Phase model (two-level / three-level workflow)

### Phase 1 — Triage / baseline evidence (cheap)

Purpose:
- Establish that there is a coherent transit-like signal and capture basic false-positive indicators.
- Avoid expensive steps (robustness sweep, FPP) unless requested.

Typical content:
- stitch baseline LC, baseline depth estimate
- baseline vetting bundle (`preset="default"` by default)
- optional per-sector vetting summary (still cheap if LCs already present)

### Phase 2 — Robustness / systematics follow-up (moderate)

Purpose:
- Address common “yellow flags” (rotation/systematics/stitched-baseline bias) by rerunning sensitive diagnostics
  on a per-sector detrended product.

Typical content:
- transit-masked per-sector detrend + normalize
- detrended stitched LC + detrended depth estimate
- rerun sensitive checks (default: V12/V16/V17/V19)
- optional detrend-window sweep table (metrics-only)

### Phase 3 — External statistical validation (expensive / network)

Purpose:
- Compute TRICERATOPS(+)-style FPP/NFPP with AO/contrast curve constraints, using replicates for stability.

Typical content:
- cache hydration from per-sector LCs
- contrast curve parsing (optional but recommended)
- FPP replicates + summaries + persisted logs

## Early-exit policy hook (optional)

The pipeline can optionally stop after Phase 1 or Phase 2 based on a **policy function**
that consumes only structured metrics/flags (no subjective text) and returns a machine-readable decision.

### Policy contract

```python
@dataclass(frozen=True)
class EarlyExitDecision:
    stop: bool
    severity: Literal["hard_stop", "soft_stop"]
    reasons: list[str]        # stable string identifiers
    evidence: dict[str, Any]  # metrics/flags used to decide (JSON-serializable)
```

### Built-in policies (initial)

These policies must use **metrics and flags** already produced by checks and pipeline stages; they do not invent new thresholds silently.

- `none`: never early-exit
- `strict_data_quality`:
  - stops when Phase 1 indicates “analysis is not meaningful” due to insufficient data coverage
  - examples (illustrative, final thresholds should be explicit constants and documented):
    - too few in-transit points
    - depth estimate undefined / non-finite
    - LC entirely empty after quality masking
- `strict_fp_triage`:
  - stops when Phase 1 shows strong FP signatures
  - examples (illustrative; again thresholds are explicit + documented):
    - large odd-even significance
    - significant secondary eclipse at phase ~0.5
    - strong V-shaped signature + other EB markers
    - (optionally) strong centroid shift consistent with off-target if pixel localization is reliable

### Execution semantics

- When `early_exit_enabled=True`:
  - run the requested phase(s) up to `early_exit_after_phase`
  - compute `EarlyExitDecision`
  - if `stop=True`, record:
    - `warnings += ["EARLY_EXIT:<policy>:<severity>"]`
    - `packet["early_exit"] = decision.to_dict()`
  - do **not** run later phases

Default should remain `early_exit_enabled=False` for audit-grade validation runs.

## Workflow stages (deterministic, resumable)

### Stage 0 — Resolve inputs

- Validate `ephemeris` present and plausible (period > 0, duration > 0).
- Validate that `tic_id` is present if:
  - running FPP (strongly recommended)
  - producing cache keys

### Stage 1 — Load data

If `mode="dataset"`:
- `load_local_dataset(dataset_path)` → `lc_by_sector`, `tpf_by_sector` (optional)

If `mode="inputs"`:
- require `lc_by_sector` and optionally `tpf_by_sector`

### Stage 2 — Baseline stitch + depth estimate

- Stitch `lc_by_sector` using `stitch_lightcurves` (quality propagated).
- Estimate depth using `estimate_transit_depth_ppm`:
  - record `depth_ppm_hat` and `depth_ppm_err`
- Create `Candidate` with explicit depth.

### Stage 3 — Baseline vetting

Run:
- `vet_candidate(stitched_lc, candidate, ...)` → `baseline_bundle`
- Optionally run per-sector vetting:
  - `per_sector_vet(lc_by_sector, candidate, tpf_by_sector=..., ...)`

Phase mapping:
- Phase 1 ends after this stage.

### Stage 4 — Robustness rerun (per-sector detrend)

If `detrend_enabled`:
- `detrend_per_sector_transit_masked(...)` → `detrended_by_sector`
- Stitch detrended sectors
- Depth estimate on detrended stitched LC
- Run sensitive check reruns on detrended stitched LC:
  - `vet_candidate(..., preset="extended", checks=rerun_sensitive_checks)` (or sessions if available)
- Optional window sweep:
  - repeat detrend+stitch+depth+runs for each `detrend_window_sweep_days` and return a table of metrics

Outputs:
- pre/post deltas for sensitive check metrics (no pass/fail)

Phase mapping:
- Phase 2 ends after this stage.

### Stage 5 — FPP (AO-assisted)

If `fpp_enabled`:
- Choose depth from `fpp_depth_source`
- Build `PersistentCache` using `hydrate_cache_from_lc_by_sector` (quality preserved)
- Parse contrast curve if provided
- Call `calculate_fpp(..., replicates, seed, overrides, timeout_seconds)`
- Persist raw logs through the artifact writer

### Stage 6 — Packet + artifacts

Always produce an in-memory `packet`:
- target identifiers
- ephemeris
- depth estimates (baseline + detrended)
- baseline bundle summary + per-sector summary (optional)
- robustness rerun metrics + deltas + window sweep table
- FPP result (if run), including summary percentiles
- provenance: package version, git SHA (if available), seeds, config echo, timestamps

If `write_artifacts` and `output_dir` provided:
- Write packet + manifest + logs (and optional report/plots)

## Artifact contract (via `api/artifacts.py`)

Recommended output layout:

```
output_dir/
  manifest.json
  validation_packet.json
  logs/
    fpp_triceratops_stdout.log
    workflow.log
  plots/ (optional)
  data/  (optional: stitched arrays)
```

## Error handling / reporting

- Stages should not throw on expected missing inputs:
  - pixel checks: mark skipped if no TPF
  - catalog checks: skipped if `network=False`
  - FPP: return structured `error` dict if timeouts occur; include replicate error list
- `ValidationWorkflowResult.warnings` aggregates:
  - skipped stages
  - suspicious diagnostics (as strings) without converting them to a verdict

## Determinism rules

- All randomized subcomponents accept explicit seeds and record them:
  - `fpp_seed`, `base_seed` + replicate seeds
  - check runner `random_seed` if used by any checks (wired via pipeline config)
- Log capture (stdout/stderr) is persisted.
- Outputs are JSON-serializable.

## Documentation updates

Add an “End-to-end validation workflow” section in docs:
- `docs/quickstart.rst`: show a minimal `run_validation_workflow` example.
- `docs/api.rst`: document `run_validation_workflow`, config/result types.

## Testing strategy

1. Offline deterministic integration test using tutorial dataset:
   - runs the workflow with `mode="dataset"`, `network=False`, `fpp_enabled=False`
   - asserts packet keys exist and check IDs present
2. FPP stage unit test:
   - mock TRICERATOPS handler to avoid network dependency
   - assert replicate summary keys in output (`fpp_summary`)
3. Robustness stage test:
   - synthetic LC with injected transit + sinusoid
   - detrending reduces sinusoid metrics while preserving depth within tolerance

## Migration plan (from tutorial 10)

- Phase 1: refactor primitives (this is the first spec).
- Phase 2: implement `run_validation_workflow` and update tutorial 10 to:
  - call the workflow once
  - then optionally drill into per-check results (still educational)
- Phase 3: add CLI wrapper (optional).
