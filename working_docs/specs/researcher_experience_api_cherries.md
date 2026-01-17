# Spec: Researcher Experience “Cherry-on-Top” API Helpers (Reusable, Client-Safe)

Owner: bittr-tess-vetter  
Status: Draft  
Scope: Public API additions that reduce notebook glue and improve reproducibility, without adding subjective thresholds.

## 1. Background

The project already provides strong algorithmic primitives, but tutorials currently require substantial “plumbing”:

- manual CSV/NPZ loading of tutorial artifacts
- repeated print/table code (partially solved via `format_vetting_table`, `summarize_bundle`, `render_validation_report_markdown`)
- ad hoc stitching / per-sector reruns
- repeated plotting patterns
- manual export of results to shareable formats

This spec proposes a small set of researcher-focused API helpers that are:

- **metrics-first** (no new pass/fail thresholds),
- **composable** (helpers call existing primitives),
- **optional-deps friendly** (matplotlib/lightkurve/etc. remain optional),
- **reproducible** (explicit provenance + deterministic outputs when seeds are set).

## 1.1 Architectural Contract (Staff-level)

These additions must be **reusable by multiple clients** (notebooks, CLIs, pipelines, MCP tools) and remain architecturally clean:

- **Pure computations stay in** `compute/` and `validation/` (array-in/array-out).
- The **public API layer** (`bittr_tess_vetter.api.*`) may add:
  - stable façades for computations,
  - thin orchestration that *only composes existing APIs*,
  - IO/dataset convenience behind explicit opt-in functions and optional dependencies.
- The API layer must not encode *policy* (no “validated” verdicts) and must avoid hard-wiring tutorial assumptions.

## 2. Goals

1. Reduce tutorial and user-notebook boilerplate by 5–10× for common workflows.
2. Provide “one call” orchestration for end-to-end runs while preserving a transparent, inspectable output.
3. Standardize local dataset loading (tutorial bundles and user-provided folders).
4. Standardize plots and exports so researchers can generate shareable artifacts quickly.
5. Provide first-class per-sector (or per-chunk) vetting convenience, since sector consistency is central to credibility.

## 3. Non-goals

- Adding new scientific checks or changing existing check semantics.
- Changing default pipeline behavior (`preset="default"` remains baseline).
- Making any network-heavy dependency mandatory.
- Implementing “final verdict” policy logic (PASS/WARN/REJECT) inside the library.

## 4. Proposed API Additions

### 4.1 `btv.run_candidate_workflow(...) -> WorkflowResult` (Orchestration, No IO)

**Intent:** One entry point that performs the common end-to-end workflow while remaining transparent.

**Key constraint:** This function accepts *data objects* and *configuration* only. It does not download data.

**Signature (proposed):**
```python
def run_candidate_workflow(
    *,
    # Inputs (choose one)
    lc: LightCurve | None = None,
    lc_by_sector: dict[int, LightCurve] | None = None,
    dataset: LocalDataset | None = None,
    # Candidate context
    candidate: Candidate,
    stellar: StellarParams | None = None,
    # Optional TPF inputs
    tpf_by_sector: dict[int, TPFStamp] | None = None,
    # Behavior toggles
    preset: str = "default",
    checks: list[str] | None = None,
    network: bool = False,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    # Optional add-ons
    run_fpp: bool = False,
    contrast_curve: ContrastCurve | None = None,
    run_pixel_report: bool = False,
    # Reproducibility / config
    random_seed: int | None = None,
    extra_params: dict[str, Any] | None = None,
) -> WorkflowResult: ...
```

**Behavior:**
- If `dataset` is provided, uses its LC/TPF.
- Else if `lc_by_sector` is provided, optionally stitches for “global” checks and/or runs per-sector bundles.
- Else if `lc` is provided, runs a single bundle.
- Optional multi-sector pixel vetting:
  - runs V08–V10 per sector when `tpf_by_sector` is present and returns a per-sector table plus a summary.
- Optional FPP:
  - calls `calculate_fpp` when `run_fpp=True` (requires network + deps).

**Output:**
- `WorkflowResult` dataclass with:
  - `bundle` (primary `VettingBundleResult`)
  - `bundle_per_sector` (optional list/dict)
  - `sector_ephemeris_metrics` (optional; from `compute_sector_ephemeris_metrics`)
  - `fpp_result` (optional)
  - `pixel_report` (optional)
  - `provenance` (inputs, versions, enabled components)

**Notes:**
- Keep it policy-free: workflow returns metrics; host decides thresholds.
- `WorkflowResult.provenance` must include a schema version (e.g., `"schema_version": 1`) to support evolution.

### 4.2 Dataset Loading (IO convenience; separate module)

**Intent:** Make user-provided folders easy to load without copy/paste, and provide a thin wrapper for bundled tutorial datasets.

**Design split:**
- `load_local_dataset(...)` is the **general-purpose**, reusable feature.
- `load_tutorial_target(...)` is a **repo convenience wrapper** implemented in terms of `load_local_dataset(...)`.

#### `load_tutorial_target`
```python
def load_tutorial_target(name: str) -> LocalDataset
```

- Example: `load_tutorial_target("tic188646744")`
- Loads from `docs/tutorials/data/<name>/` by convention.
- Returns a `LocalDataset` with:
  - `lc_by_sector` parsed from `sector*_pdcsap.csv`
  - `tpf_by_sector` from `sector*_tpf.npz` when present
  - `artifacts` (e.g., AO contrast curve tables) if present

#### `load_local_dataset`
```python
def load_local_dataset(path: str | Path, *, pattern_overrides: dict[str, str] | None = None) -> LocalDataset
```

- Same `LocalDataset` output, but for arbitrary user folders.
- Supports CSV/NPZ formats used by tutorials.

**Constraints:**
- No pandas dependency required (use `csv` module).
- Clear error messages for missing files.
- `LocalDataset` should be JSON-serializable (for pipelines), or provide `to_dict()`.

**Module boundary:** `bittr_tess_vetter.api.datasets` (or `api/local_dataset.py`).

### 4.3 Plot Helpers (`btv.plot_*`) (Optional UI)

**Intent:** Standardize common plots so researchers don’t reimplement each time.

Proposed minimal set (matplotlib optional):

- `plot_phase_fold(lc, candidate, *, bin_minutes=..., xlim_hours=...) -> Figure`
- `plot_transit_windows(lc, candidate, *, n_windows=...) -> Figure`
- `plot_odd_even(bundle) -> Figure` (uses V01 metrics/raw when available)
- `plot_secondary(bundle) -> Figure` (uses V02 outputs when available)
- `plot_pixel_summary(bundle_or_pixel_report) -> Figure`

**Optional-deps handling:**
- If matplotlib missing, raise `ImportError` with install hint.
- Keep plotting logic shallow; no “smart” interpretation or thresholds.

**Module boundary:** `bittr_tess_vetter.api.plots` (imports matplotlib lazily).

### 4.4 Export Helpers (`btv.export_bundle(...)`) (Reproducibility / interchange)

**Intent:** Make results easy to share and archive.

```python
def export_bundle(
    bundle: VettingBundleResult,
    *,
    format: Literal["json", "csv", "md"],
    path: str | Path | None = None,
    include_raw: bool = False,
) -> str | None
```

- `format="json"` returns serialized JSON or writes to `path`.
- `format="csv"` exports a flat table (one row per check) including selected metrics/flags.
- `format="md"` exports `render_validation_report_markdown(...)` output.

**Stability requirement (CSV):**
- Define a stable column contract (e.g., `id,name,status,confidence,flags_json,metrics_json,notes_json`).
- Avoid exploding arbitrary metric keys into columns by default; provide an optional `metric_keys=[...]` to expand selected keys.

**Module boundary:** `bittr_tess_vetter.api.export`.

### 4.5 Per-sector Convenience (`btv.per_sector_vet(...)`) (Common research loop)

**Intent:** First-class “run the same vetting per sector/chunk” pattern.

```python
def per_sector_vet(
    lc_by_sector: dict[int, LightCurve],
    candidate: Candidate,
    *,
    stellar: StellarParams | None = None,
    tpf_by_sector: dict[int, TPFStamp] | None = None,
    preset: str = "default",
    checks: list[str] | None = None,
    network: bool = False,
    tic_id: int | None = None,
) -> PerSectorVettingResult
```

`PerSectorVettingResult` includes:
- `bundles_by_sector`
- `summary_table` (list[dict] or pandas-friendly records)
- `sector_ephemeris_metrics` (optional)

**Module boundary:** `bittr_tess_vetter.api.per_sector`.

## 5. UX / Design Principles

- **Policy-free outputs**: no new boolean “validated” flags.
- **Deterministic**: all helpers accept `random_seed` and thread it through where relevant.
- **Stable import paths**: everything is reachable from `bittr_tess_vetter.api`.
- **Machine-friendly**: every helper returns structured data suitable for tables/JSON.
- **Optional deps**: plotting and network steps must degrade gracefully.
- **Layering**: IO convenience must be isolated; orchestration must not silently download.

## 6. Testing Plan

- Unit tests for:
  - `load_local_dataset` parsing CSV/NPZ (use tiny fixture files)
  - `export_bundle` JSON/CSV/MD correctness (schema + smoke)
  - `per_sector_vet` output structure and deterministic ordering
  - `run_candidate_workflow` minimal LC-only run (no network) smoke
- Integration smoke tests can reuse the existing tutorial data directory when available.

## 7. Documentation Plan

- Add a “Researcher UX” section in `docs/quickstart.rst` showing:
  - `load_tutorial_target` + `run_candidate_workflow` + `render_validation_report_markdown`
- Update tutorial notebooks to replace ad hoc print blocks with:
  - `format_vetting_table`
  - `export_bundle(..., format=\"md\")` (once implemented)

## 8. Rollout / Compatibility

- All additions are additive and non-breaking.
- Prefer `btv.*` exports for stable imports; keep internal modules private.
- Maintain current tutorial code until new helpers are battle-tested; then migrate notebooks.

## 9. Packaging / Dependency Strategy

- `bittr-tess-vetter` core installs must not require `matplotlib`, `lightkurve`, or `pandas`.
- `load_local_dataset` must work without pandas.
- Plot helpers import matplotlib lazily and error with install hints.
- Any “download” helper (if added later) must live in `platform/` (or explicitly named `api.download_*`) and require opt-in extras.

## 10. Versioning & Stability

For each new structured return type (`WorkflowResult`, `LocalDataset`, `PerSectorVettingResult`):

- include `schema_version: int` in serialized forms,
- document which fields are stable vs provisional,
- avoid renaming keys without a major version bump, or provide compatibility shims.
