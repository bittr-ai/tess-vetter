codex

# Architecture Review (v6): Workflow-Building UX for Researchers

Date: 2026-01-14  
Scope: Assess `bittr-tess-vetter` as a library for astrophysicists to build custom workflows/pipelines, and propose the top next priorities (post v0.1.0 pipeline refactor + adapter decoupling).

## Executive Summary

The v0.1.0 refactor made the library dramatically more “workflowable”:
- curated golden-path API surface (`bittr_tess_vetter.api.__all__` is small),
- structured `CheckResult`/`VettingBundleResult` schemas,
- `VettingPipeline`, `CheckRegistry`, and check introspection.

The next big UX step is to move from “you can build workflows” to “it’s effortless to build workflows”. That requires:
- first-class **batch execution** + table exports,
- stronger **typing and config schemas** (remove `Any`/ad-hoc dicts),
- consistent **optional-dependency gating** and “skipped” semantics,
- **documentation alignment** with the new v0.1.0 result shapes.

## What’s Working Well (Keep)

- `VettingPipeline.run(...)` already encodes a workflow primitive: “requirements → skip → run → aggregate”.
- `list_checks()` exposes requirements/citations (critical for workflow transparency).
- Result schemas forbid extra fields (good for downstream strictness).
- Adapter decoupling (astro-arc-tess) means you can evolve the library without tool breakage risk.

## Key Workflow Pain Points (Observed)

1) **Batch workflows are not first-class**
   - There is no `run_many()`/`vet_many()` API and no standard “results table” output.
   - Researchers will immediately want to run N candidates across a sector set and rank/filter.

2) **Optional dependency gating is incomplete**
   - `CheckRequirements.optional_deps` exists, but `VettingPipeline` doesn’t enforce it.
   - This will cause runtime exceptions in the middle of pipelines instead of clean `skipped` results with an install hint.

3) **Type ergonomics are still rough at the boundaries**
   - `CheckInputs` uses `Any` for `lc` and `candidate`.
   - `VettingPipeline.run()` type hints currently reference internal domain types, while the golden path encourages facade types.
   - For workflow authors, this increases confusion and reduces IDE help.

4) **Config ergonomics are underpowered**
   - There’s a `CheckConfig.extra_params: dict[str, Any]`, but no per-check typed config schema.
   - Workflows need validated configs (especially for reproducibility and collaboration).

5) **Docs drift / tutorial drift risk**
   - `docs/quickstart.rst` and README examples still show legacy fields like `passed=...` in output formatting.
   - Researchers will copy/paste docs; drift becomes a support sink and reduces confidence.

## Top Next Priorities (Recommended)

### P0 — Workflow primitives that eliminate boilerplate

**P0.0 Docs alignment (stop drift)**
- Update README + `docs/quickstart.rst` examples to match v0.1.0 outputs:
  - Use `status/metrics/flags` (not legacy `passed`).
  - Show “golden path” imports and `VettingPipeline` for custom workflows.

Why it’s top priority:
Researchers copy/paste docs. Drift is a trust-killer and becomes a long-term support sink.

**P0.1 Batch API + table export (start with the simple case)**
- Add `vet_many(lc, candidates, ...)` or `VettingPipeline.run_many(lc, candidates, ...)` supporting:
  - **one LC shared across many candidates** (multi-planet/system candidates, alternate ephemerides, etc.)
  - outputs:
    - `list[VettingBundleResult]` (full details)
    - a compact summary “table” (list of dicts) with stable columns:
      `candidate_index`, `period_days`, `t0_btjd`, `duration_hours`, `depth_ppm`,
      `n_ok`, `n_skipped`, `n_error`, `flags_top`, `runtime_ms`
- Explicitly defer “many LCs / many candidates” to a later tranche (it introduces indexing + memory management complexity).

Why it’s top priority:
It’s the workflow most researchers actually run (survey → rank → deep dive). Without it, every user writes bespoke glue.

**P0.2 Enforce optional deps → structured skip**
- In `VettingPipeline._check_requirements_met`, add enforcement for `requirements.optional_deps`:
  - If missing, return `skipped` with flags like `SKIPPED:EXTRA_MISSING:triceratops` and a clear note/install hint.
- Standardize how checks declare required extras (string names should match install extras).

Why:
Workflows must be robust across environments (laptops, clusters). Mid-run hard errors kill throughput and trust.

### P1 — Make workflows pleasant to write (typing + config)

**P1.1 Typed inputs at the workflow boundary**
- Replace `Any` in `CheckInputs` with concrete types (or Protocols) for:
  - `LightCurveData` / `LightCurve` (decide one canonical representation for pipeline execution)
  - `TransitCandidate` / `Candidate`
- Provide one canonical conversion path:
  - either pipeline accepts facade types and converts internally, or
  - pipeline is “internal-type-only” but you provide `pipeline.run_public(...)`.

Why:
Researchers rely on IDEs and notebooks; strong typing is a huge UX multiplier.

**P1.2 Per-check config models**
- Introduce `BaseCheckConfig` (Pydantic) and one config model per check with defaults.
- Update `list_checks()` to include config schema metadata (at least default values + fields).

Why:
Workflows become reproducible artifacts when configs are validated and serializable.

### P2 — Ecosystem features (bigger, but powerful)

**P2.1 Plugin checks via entry points**
- Support registering third-party checks via Python entry points.
- Keep core registry, but add `discover_checks()` for installed plugins.

Why:
This is how you turn the library into a platform without central maintenance.

**P2.2 Dataset abstractions**
- Introduce a small “dataset” model (even if minimal):
  - `LightCurveDataset` (time/flux/err + masks + metadata + provenance)
  - optional multi-sector container.

Why:
Many workflow bugs come from repeated ad-hoc handling of masks, sectors, cadence, and normalization.

## Recommended “Next 48 Hours” Plan (High ROI)

1) Fix docs drift (README + `docs/quickstart.rst`) to use the v0.1.0 result schema (`status/metrics/flags`).
2) Implement P0.2 optional-dep gating (small, high leverage).
3) Implement P1.1 typing cleanup so `VettingPipeline.run()` matches the golden-path types (no `Any` for lc/candidate).
4) Implement P0.1 batch API (simple case: one LC + many candidates) + summary-table output.
5) Add workflow-pattern tests for `vet_many`:
   - mixed ok/skip/error batch behavior
   - deterministic ordering
   - schema invariants on summary rows (stable columns)

## Success Metrics

You’ll know the API is “workflow-easy” when:
- A researcher can run 1k candidates and get a sortable table in <20 lines of code.
- Missing optional deps yield clean `skipped` results rather than exceptions.
- `list_checks()` is sufficient to auto-generate a workflow UI/config file.
