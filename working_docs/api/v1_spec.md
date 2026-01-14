# bittr-tess-vetter: v1 Spec (Incremental Extraction Plan)

**Date:** 2026-01-08  
**Status:** DRAFT (v1)  
**Repo:** `bittr-tess-vetter` (new)  
**Python package:** `bittr_tess_vetter`  
**Distribution name:** `bittr-tess-vetter`  
**License:** MIT

## 0) Summary

Create a new repo (`bittr-tess-vetter`) that contains *domain logic only* (TESS transit detection + vetting + pixel-level diagnostics + phase-folding + detrending + optional MLX acceleration). Keep MCP/tooling, caching/persistence, evidence/manifest/provenance, and agent/guardrail orchestration in this repo (`astro-arc-tess`).

Extraction is incremental: for each module moved, `astro-arc-tess` swaps its implementation to call into `bittr_tess_vetter` while preserving behavior (within stated tolerances) and preserving existing MCP tool schemas.

## 1) Goals / Non-goals

### Goals
- **Incremental migration**: move code in small, testable PRs while keeping `astro-arc-tess` behavior stable.
- **Hard boundary**: `bittr-tess-vetter` contains no MCP server, no persistent store, and no tool manifests/evidence packets. It is a standalone library with no external bittr-* dependencies.
- **Stable Python API**: provide a clear set of library entry points used by `astro-arc-tess` tool handlers.
- **Determinism & reproducibility hooks**: library functions accept explicit seeds and return sufficient metadata to support manifests in `astro-arc-tess`.
- **Optional accelerators**: MLX, batman, triceratops, numpyro are optional extras, not hard deps.
- **Citation-first clarity**: domain code carries in-code citations so users can see what is standard vs novel at the point of use.

### Non-goals (v1)
- Replace `astro-arc-tess` MCP tool surface or tool I/O schemas.
- Re-design evidence/manifest/validity systems (remain in `astro-arc-tess` + `bittr-*` contracts).
- Rewrite algorithms for performance (migration first; perf later).
- Solve all network/caching concerns (those stay in `astro-arc-tess`).

## 2) Boundary Rules (this is what “domain-only” means)

### `bittr-tess-vetter` MUST NOT
- Import `bittr_reason_core`, `bittr_evidence`, `bittr_validity`, or any MCP/server/runtime layer.
- Read/write a persistent store, compute blob refs, or emit tool/pipeline manifests.
- Depend on a session cache; no implicit global caches.
- Perform network calls by default (catalog access must be optional and explicit).

### `bittr-tess-vetter` MAY
- Depend on standard scientific stack: `numpy`, `scipy`, `astropy` (optionally), `pydantic`.
- Provide *optional* subpackages/extras for:
  - `mlx` acceleration
  - `batman-package` model fitting
  - `exovetter` wrappers
  - `triceratops` FPP
  - `numpyro/jax` Phase 5 inference
- Expose clean, typed data models and “compute” entry points that operate on arrays and plain models.

### `astro-arc-tess` remains responsible for
- MCP registration (`astro_arc.server`, `astro_arc.mcp_tools/*`)
- cache + persistence (`astro_arc.cache/*`, `astro_arc.io/*`, `astro_arc_tess.storage/*`)
- evidence + manifests + refs (`astro_arc.evidence/*`, `astro_arc.tools/*`, `astro_arc.reporting/*`)
- policy/guardrails (`astro_arc.validation.guardrails`, `bittr-validity` integration)

## 3) Current Repo Map (where the boundary is today)

In `astro-arc-tess`, the cleanest “platform vs domain” boundary lines up as:

- **Platform (stays here)**:
  - `src/astro_arc/server.py`
  - `src/astro_arc/mcp_tools/*`
  - `src/astro_arc/tools/*` (replay/ablation/claim-level/dispatch hooks)
  - `src/astro_arc/evidence/*` and anything producing `man_tool:*`, `ev:*`
  - `src/astro_arc_tess/storage/persistent_store.py`
  - guardrail aggregation (`src/astro_arc/validation/guardrails.py`)

- **Domain (migrate to `bittr-tess-vetter`)**:
  - `src/astro_arc/compute/*` (periodograms, detrending, models, MLX algorithms)
  - `src/astro_arc/transit/*`
  - `src/astro_arc/pixel/*`
  - `src/astro_arc/search/*`
  - `src/astro_arc/validation/*` (checks + core vetting computations, but *not* guardrail aggregation)
  - `src/astro_arc/stellar/*` (physics, dilution; provenance wrappers likely stay here)
  - `src/astro_arc/recovery/*`
  - `src/astro_arc/activity/*`
  - `src/astro_arc/phase4/*`, `src/astro_arc/phase5/*` (later phases; heavier deps)

## 4) Proposed `bittr-tess-vetter` Package Layout (v1)

```
bittr-tess-vetter/
  pyproject.toml
  README.md
  src/bittr_tess_vetter/
    __init__.py
    types/
      __init__.py
      lightcurve.py      # LightCurveData + helpers (arrays)
      candidates.py      # TransitCandidate + related small models
      results.py         # Standard result types (PeriodogramResult, VettingReport, etc.)
    utils/
      __init__.py
      tolerances.py      # ToleranceResult + helpers (no bittr deps)
    compute/
      __init__.py
      detrend.py
      transit.py         # box model + masks + depth metrics
      bls_like_search.py
      periodogram.py     # TLS/LS/auto search (no caches)
      mlx_detection.py   # optional (extra: mlx)
    pixel/
      __init__.py
      aperture.py
      centroid.py
      difference.py
      localization.py
      wcs_utils.py       # optional (extra: pixel/astropy)
      wcs_localization.py
    vetting/
      __init__.py
      base.py            # shared masks/metrics
      lc_checks.py
      checks_basic.py
      checks_secondary.py
      checks_pixel.py
      pipeline.py        # returns VettingReport (not EvidencePacket)
      exovetter_checks.py  # optional (extra: exovetter)
      triceratops_fpp.py   # optional (extra: triceratops)
    transit/
      __init__.py
      timing.py
      vetting.py
      batman_model.py    # optional (extra: fit)
    recovery/
      __init__.py
      primitives.py
      result.py
    activity/
      __init__.py
      primitives.py
      result.py
    phase4/              # deferred to later milestones
    phase5/              # deferred to later milestones
  tests/
    ... (ported unit tests by module)
```

## 5) Library API (what `astro-arc-tess` should call)

v1 API is “array-in, result-out”. No refs. No stores.

### 5.1 Types
- `LightCurveData` (numpy arrays + metadata; no MCP refs)
  - `time: float64[]`, `flux: float64[]`, `flux_err: float64[]`, `valid_mask: bool[]`, plus optional `tic_id`, `sector`, `cadence_seconds`
- `TransitCandidate`
  - `period_days`, `t0_btjd`, `duration_hours`, `depth_fraction`, optional `snr`
- `PeriodogramResult`
  - method, peaks, best_* fields (matching current `astro_arc.domain.detection.PeriodogramResult` shape)
- `VettingReport` (new)
  - `candidate: TransitCandidate`
  - `evidence_items: list[EvidenceItemLike]` where each item is `{id, title, metrics, flags}`
  - `metadata: dict` (inputs, assumptions, optional warnings)

### 5.2 Compute entry points
- `bittr_tess_vetter.compute.periodogram.auto_periodogram(time, flux, flux_err, *, min_period, max_period, preset, method, ...) -> PeriodogramResult`
- `bittr_tess_vetter.compute.transit.compute_box_model(time, *, period_days, t0_btjd, duration_hours, depth_fraction) -> float64[]`
- `bittr_tess_vetter.vetting.pipeline.run_vetting(time, flux, flux_err, *, candidate, stellar=None, tpf=None, options=...) -> VettingReport`
- `bittr_tess_vetter.pixel.*` functions for difference imaging, centroid shift, aperture dependence, WCS localization (if enabled)
- `bittr_tess_vetter.recovery.*` for detrending + stacking + trapezoid fitting used by `recover_transit`

### 5.3 Optional/extra APIs (behind extras)
- `bittr_tess_vetter.compute.mlx_detection.*` (`extra: mlx`)
- `bittr_tess_vetter.transit.batman_model.*` (`extra: fit`)
- `bittr_tess_vetter.vetting.exovetter_checks.*` (`extra: exovetter`)
- `bittr_tess_vetter.vetting.triceratops_fpp.*` (`extra: triceratops`)
- Phase 5 inference (`extra: phase5`)

## 6) How `astro-arc-tess` Adapts (adapter points)

The MCP tools should remain stable and simply translate:

- **Cache → arrays**: tool handlers (`astro_arc.mcp_tools.*`) read cached `LightCurveData` / TPF FITS and pass arrays into `bittr_tess_vetter`.
- **Library results → evidence/manifests/validity**:
  - Convert `VettingReport.evidence_items` into `astro_arc.domain.evidence.EvidenceItem` + `EvidencePacket`.
  - Run `astro_arc.validation.guardrails.*` on top of the library’s returned metrics/context.
  - Persist blobs/manifests in `PersistentStore` as today.

Concrete “touch points” in this repo:
- `src/astro_arc/mcp_tools/run_periodogram.py` calls `astro_arc.compute.periodogram.auto_periodogram` → later swap to `bittr_tess_vetter.compute.periodogram.auto_periodogram`
- `src/astro_arc/mcp_tools/compute_transit_model.py` calls `astro_arc.compute.compute_bls_model` → swap to library box model
- `src/astro_arc/mcp_tools/recover_transit.py` calls `astro_arc.recovery.*` → swap to `bittr_tess_vetter.recovery.*`
- `src/astro_arc/validation/pipeline.py` currently returns `EvidencePacket` → replace with thin wrapper:
  - `bittr_tess_vetter.vetting.pipeline.run_vetting` returns `VettingReport`
  - wrapper converts to `EvidencePacket` for compatibility

## 7) Incremental Extraction Order (PR-sized)

Each step should be small enough to review, and should have a “prove behavior” checklist.

### Step 1: Foundational types (no tools touched)
- Move/copy: `LightCurveData`, `TransitCandidate`, `PeriodogramResult` equivalents.
- Keep in `astro-arc-tess`: MCP-facing `LightCurveRef` (metadata-only).
- Prove: unit tests for dtype/shape validation + model validation.

### Step 2: Pure transit primitives
- Move/copy: `src/astro_arc/compute/transit.py` + minimal helpers it needs.
- Adapt: `astro_arc.compute` wrapper re-exports the new implementation.
- Prove: numeric equivalence on existing tests that touch `compute_bls_model`/masks.

### Step 3: Detrending primitives (CPU)
- Move/copy: `src/astro_arc/compute/detrend.py` (keep `wotan` optional).
- Prove: deterministic outputs on fixture light curves (within tolerances).

### Step 4: Periodograms
- Move/copy: `src/astro_arc/compute/periodogram.py` and any helper modules it requires.
- Keep TLS as optional if you want a minimal core; otherwise keep as base dep (matches today).
- Prove: peak ordering + best-period stability on fixtures; ensure tolerances are recorded upstream.

### Step 5: Pixel fundamentals (non-WCS)
- Move/copy: `src/astro_arc/pixel/difference.py`, `centroid.py`, `aperture.py`, `localization.py`.
- Prove: existing pixel unit tests ported; then `astro-arc-tess` tests pass via wrapper.

### Step 6: Vetting checks (non-network)
- Move/copy: `src/astro_arc/validation/base.py`, `lc_checks.py`, `checks_basic.py`, `checks_secondary.py`, `checks_pixel.py`.
- Change API: checks return plain `VetterCheckResultLike` / `EvidenceItemLike`, not `EvidencePacket`.
- Prove: `run_vetting_pipeline` evidence items match (id/metrics/flags) modulo serialization differences.

### Step 7: Vetting pipeline core (library)
- Implement: `bittr_tess_vetter.vetting.pipeline.run_vetting(...) -> VettingReport`.
- Adapt: `astro_arc.validation.pipeline.run_vetting_pipeline` becomes a wrapper producing `EvidencePacket`.
- Prove: `tests/validation/test_evidence_vetting_pipeline.py` equivalence + pixel evidence tests.

### Step 8: Recovery + activity
- Move/copy: `src/astro_arc/recovery/*`, `src/astro_arc/activity/*`.
- Adapt: `recover_transit` tool continues to orchestrate cache and return MCP-friendly outputs.
- Prove: recovery primitive tests; avoid brittle floating diffs by using tolerances.

### Step 9+: Network clients & Phase 4/5 (optional, later)
- Catalog/MAST clients can be included as extras; keep them out of the minimal core.
- Phase 4 and Phase 5 are separate milestones due to heavy deps and evolving schema.

## 8) “Prove Behavior” Definition of Done (per step)

For every extracted unit:
- `astro-arc-tess` continues to pass its existing tests for that subsystem (or tighter: at least the affected test modules).
- `bittr-tess-vetter` has at least the corresponding unit tests ported/copied (to avoid regressions when the repos diverge).
- A “compat shim” remains in `astro-arc-tess` so MCP tools and internal imports don’t churn.
- Any intentional behavior change is:
  - explicitly documented
  - guarded by a version string or feature flag

## 9) Dependency / Extras Plan (recommended)

Suggested extras in `bittr-tess-vetter`:
- `mlx`: `mlx>=0.28.0`
- `pixel`: `astropy` (+ any WCS helpers you use)
- `fit`: `batman-package`, `emcee`, `ldtk`, `wotan`
- `validation`: `exovetter`, `triceratops`
- `phase5`: `numpyro`, `jax`, `jaxlib`
- `dev`: `pytest`, `ruff`, `mypy`

The default install should be small and non-networking.

## 10) Versioning

- Semantic-ish: `0.y.z` while APIs are settling.
- Bump **minor** when:
  - function signatures change
  - return schemas change
  - numeric behavior changes beyond defined tolerances
- Bump **patch** for:
  - bugfixes that keep result compatibility
  - performance improvements without output changes

## 11) First Migration Target (recommended)

Start by extracting modules that:
- have no dependency on `bittr_reason_core` / evidence/validity contracts
- operate purely on arrays
- already have good unit test coverage here

Concretely, best initial candidates from this repo:
- `src/astro_arc/pixel/localization.py`
- `src/astro_arc/compute/transit.py`
- `src/astro_arc/recovery/primitives.py`

Those give fast wins and prove the “library + wrappers” pattern before touching the core vetting pipeline.

## 12) Citation-First Policy (required)

`bittr-tess-vetter` should make the “what’s standard vs what’s novel” distinction obvious *in the code*, not just in papers/blog posts.

### 12.1 Per-module docstring structure

Every non-trivial algorithm module (e.g. `compute/periodogram.py`, `vetting/*`, `pixel/*`, `recovery/*`) should include a short, consistent header in the module docstring:

- **What this implements**: one paragraph.
- **Novelty**: `standard` | `adapted` | `novel` (plus 1–3 bullets explaining why).
- **References**: a small list of primary citations (DOI/arXiv/ADS bibcode) that map to the implementation.

### 12.2 Reference format

Prefer machine-stable identifiers that are easy to look up:
- DOI when available
- arXiv ID when not
- ADS bibcode for astronomy-native citations

If the module is a wrapper around a well-known library/algorithm, cite the original method paper *and* the library when appropriate (e.g., TLS method paper; batman paper; exovetter references).

### 12.3 Central reference index

In the new repo, add one central place for longer-form citations so modules can stay concise:
- `REFERENCES.md` (human readable) and optionally `references.bib` (BibTeX)

Modules should keep a minimal “References” list inline and may also link to a stable key in the central index (e.g., `[TLS2019]`, `[BATMAN2015]`).

### 12.4 Contribution requirement

PR checklist item for `bittr-tess-vetter`:
- If a PR adds/changes an algorithm, it must update module **Novelty** + **References** (and central index if needed).

### 12.5 Repo-level citation metadata

Add repo-level citation metadata (`CITATION.cff`) so downstream users can cite the software cleanly, in addition to in-code references.
