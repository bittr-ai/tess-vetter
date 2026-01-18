# TOI Validation Pipeline (Reusable) — Spec

## Context

We currently have an “audit-grade” notebook workflow (e.g. `docs/tutorials/10-toi-5807-check-by-check.ipynb`) that:

- Loads a target dataset (or cached/online products).
- Runs metrics-first vetting checks (LC, catalog, pixel, exovetter).
- Performs robustness follow-ups (per-sector transit-masked detrend + rerun sensitive checks).
- Computes AO-assisted TRICERATOPS FPP/NFPP with replicates.
- Writes a consolidated packet for traceability.

This spec defines a **reusable, programmatic pipeline** that reproduces that workflow for **any target**, as a stable API surface for other clients (CLI, notebooks, services).

## Goals

- **One call** to run an end-to-end validation workflow with:
  - deterministic configuration + provenance
  - explicit artifacts (inputs/outputs/logs) written to disk
  - **metrics-only outputs** (no hard-coded “PASS/FAIL thresholds” inside the pipeline)
- A **structured result**:
  - check results (per check and grouped by phase)
  - robustness comparisons (pre/post detrend deltas)
  - FPP/NFPP replicate summaries
  - warnings/flags and coverage limitations
- Works in three modes:
  1. **Offline from local dataset** (tutorial-style)
  2. **Online-assisted** (download MAST/TPF/Gaia/ExoFOP as needed)
  3. **Hybrid** (use cached data; download only missing)

## Non-goals

- Not a planet validation “authority”: the pipeline outputs **metrics + evidence**; policy is applied externally.
- Not a full “discovery” pipeline (period search) unless explicitly added as an optional stage.
- Not a replacement for DV products or bespoke PRF-fitting for saturated targets (but can integrate later).

## Design principles

- **Metrics-first**: pipeline does not encode subjective thresholds; any dispositions are computed in an explicit, optional policy layer.
- **Deterministic by default**: stable seeds, stable formatting, bounded timeouts, stable artifact paths.
- **Traceable**: every derived result points to the exact inputs and parameters used; raw logs are persisted.
- **Composable**: each stage can be run independently and returns a structured payload.
- **Resumable**: if artifacts exist (cached), the pipeline can skip/reuse intermediate products.

## Public API (proposal)

### Primary entrypoint

```python
result = btv.run_validation_pipeline(
    target=ValidationTarget(tic_id=..., toi="5807.01", name=None, ra_deg=None, dec_deg=None),
    ephemeris=Ephemeris(...),
    stellar=StellarParams(...),
    config=ValidationPipelineConfig(...),
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
class ValidationPipelineConfig:
    # Runtime/IO
    output_dir: str  # root artifact dir
    cache_dir: str | None
    network: bool
    timeout_seconds_total: float | None

    # Data acquisition
    mode: Literal["dataset", "cache", "download", "hybrid"]
    dataset_path: str | None  # used when mode="dataset" or "hybrid"
    sectors: list[int] | None
    flux_type: Literal["pdcsap", "sap"]  # for downloaded/cached LCs
    tpf_sectors: list[int] | None

    # Check execution
    checks_default: list[str]  # e.g. default preset IDs
    checks_extended: list[str]  # opt-in
    include_pixel_checks: bool
    include_catalog_checks: bool
    include_exovetter_checks: bool

    # Robustness
    detrend_enabled: bool
    detrend_method: str  # e.g. "wotan_biweight"
    detrend_window_days: float
    detrend_transit_mask_mult: float
    detrend_window_sweep_days: list[float] | None

    # FPP
    fpp_enabled: bool
    fpp_preset: Literal["fast", "standard"]
    fpp_overrides: dict[str, Any] | None
    fpp_replicates: int
    fpp_seed: int | None
    fpp_timeout_seconds: float | None
    contrast_curve_path: str | None
    contrast_curve_filter: str | None

    # Output shaping
    write_packet_json: bool
    write_markdown_report: bool
    render_plots: bool
    plot_types: list[str] | None
```

### Returned result

```python
@dataclass(frozen=True)
class ValidationPipelineResult:
    target: dict[str, Any]
    ephemeris: dict[str, Any]
    stellar: dict[str, Any]

    artifacts: dict[str, str]  # paths
    provenance: dict[str, Any]  # versions, seeds, hashes
    warnings: list[str]

    # Phase outputs
    baseline: dict[str, Any]  # stitched baseline LC, depth estimates, etc
    checks: dict[str, Any]  # per-check results
    robustness: dict[str, Any]  # per-sector detrend outputs + deltas
    fpp: dict[str, Any]  # fpp/nfpp + summary + logs
```

## Pipeline stages

### Stage A — Input resolution

Inputs:
- target identifiers (TIC/TOI/name/coords)
- ephemeris + stellar params

Outputs:
- resolved identifiers (TIC required for TESS products)
- config sanity: ensure BTJD consistency, positive durations, etc

### Stage B — Data acquisition

Supported sources:
- **LocalDataset** (tutorial folder structure, AO curves, cached LCs/TPFs)
- Cached LC/TPF in `cache_dir`
- Online download (MAST for LC/TPF; Gaia/ExoFOP if enabled)

Outputs:
- per-sector LC objects (including real `quality` flags)
- per-sector TPF refs (optional)
- a stitched LC (baseline)

### Stage C — Baseline depth estimate

Goal: produce a single, explicit `Candidate(depth_ppm=...)` used by downstream checks.

Outputs:
- baseline depth estimate + uncertainty (metrics-only)
- candidate object
- coverage metadata (time span, n_points, sector list)

### Stage D — Core checks (default preset)

Runs:
- LC-only checks (V01–V05, V13, V15)
- Catalog checks (V06–V07) if `network=True`
- Pixel checks (V08–V10) if TPFs present
- Exovetter checks (V11, V11b, V12) if installed/enabled

Outputs:
- list/dict of `CheckResult` plus consolidated counts
- per-sector summaries where applicable

### Stage E — Robustness follow-up (per-sector detrend)

Triggered by config (always-on for pipeline) and/or by caller policy.

Procedure:
- for each sector LC:
  - finite filtering
  - transit-masked detrend (WOTAN or configured)
  - normalize sector
- stitch detrended sectors
- re-estimate depth on detrended stitched LC
- rerun sensitive checks on detrended LC (default: V12, V16, V17, V19)
- optional window-length sweep returning metrics per window (no thresholds)

Outputs:
- detrended LC + depth estimate
- pre/post deltas for sensitive check metrics
- sweep results table (if enabled)

### Stage F — FPP/NFPP (TRICERATOPS+)

Inputs:
- cache hydrated with per-sector LCs
- AO contrast curve (optional but recommended)
- **depth_ppm used** is explicitly recorded (baseline vs detrended)

Procedure:
- `calculate_fpp(..., replicates=N, seed=...)`
- persist raw TRICERATOPS stdout/stderr logs
- report `fpp_summary`/`nfpp_summary` when replicates > 1

Outputs:
- structured FPP result + replicate summary + log path

### Stage G — Artifact packaging / report

Artifacts:
- `validation_packet.json` (single file with key metrics + paths)
- optional `report.md` (human-readable)
- optional plots (phase-fold, odd/even overlay, per-sector fold, pixel summary)

## Artifacts & directory layout

All outputs live under `output_dir`:

```
output_dir/
  manifest.json
  validation_packet.json
  logs/
    fpp_triceratops_stdout.log
    pipeline.log
  data/
    stitched_baseline.parquet|npz
    stitched_detrended.parquet|npz
  plots/                       (optional)
    phase_fold.png
    odd_even.png
    per_sector_phase_fold.png
    pixel_summary.png
```

The `manifest.json` records:
- config used
- timestamps
- versions (package versions + git SHA if available)
- seeds and timeouts
- hash of key inputs (dataset path contents hash optional)

## Determinism & reproducibility requirements

- Every stochastic component must accept an explicit seed:
  - FPP replicates use `base_seed` and increment
  - any randomized diagnostics must accept seed
- Capture and persist external logs (TRICERATOPS stdout/stderr).
- Avoid embedding system-dependent paths in “headline” results; store absolute paths in artifacts, but also store relative paths for portability.
- All outputs should be JSON-serializable, stable key ordering where printed.

## Error handling

No stage should silently fail:
- Each stage returns either `ok` with outputs or `error` with:
  - `error_type`
  - `stage`
  - `message`
  - partial outputs (when safe)
- Network-required stages should degrade gracefully when `network=False`.
- If pixel data is missing, pixel checks are explicitly marked `skipped` with reasons.
- FPP timeouts should be returned as structured errors with `replicate_errors`.

## Extensibility hooks

- `stage_hooks`: optional callbacks invoked before/after each stage to allow clients to:
  - add custom checks
  - override artifact writing
  - stream metrics into other systems
- `extra_checks`: list of additional check IDs to run (registered via existing registry).

## CLI (optional, but recommended)

Provide a CLI wrapper for batch/automation:

```
btv validate --tic 188646744 --period 14.2423724 --t0 3540.26317 --duration 4.046 \
  --output-dir runs/toi5807 --network --fpp --ao docs/tutorials/data/tic188646744/PHARO_Kcont_plot.tbl
```

CLI should:
- print a compact summary (counts + key metrics)
- write artifacts to `output_dir`
- exit non-zero only on pipeline errors (not “planet vs not”)

## Testing strategy

- Unit tests:
  - stage outputs are JSON-serializable
  - determinism: same seed => same `fpp_summary` structure keys (values can vary if external services vary; gate via mocks)
  - V13 coverage-aware fields present and consistent
- Integration tests:
  - local dataset pipeline run (no network) produces packet + manifest
  - “network disabled” => catalog/FPP stages skip or return structured error

## Rollout plan

1. Introduce new API entrypoint behind `btv.run_validation_pipeline` (policy-free).
2. Add tutorial 10 to call the pipeline and then *optionally* show “check-by-check” drilldowns.
3. Add CLI wrapper.
4. Add optional policy layer (separate module) for `VALIDATED`/`LIKELY` labels, if desired.

