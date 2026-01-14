# Spec (Aggressive): v0.1.0 “Amazing First Release” Refactor — bittr-tess-vetter

**Date:** 2026-01-14  
**Status:** Draft (aggressive, pre-release)  
**Audience:** maintainers (final pre-release API shaping)  
**Key premise:** This is **pre-public-release**, so we can break internal/backward compatibility now to land the *best* v0.1.0 contract.  
**External constraint:** `astro-arc-tess` now uses an adapter layer (`astro_arc.bittr.*`), so we can change `bittr-tess-vetter` aggressively as long as we update the adapter in lockstep before tagging.

---

## 0) Goals (What “Amazing” Means)

1. **Pipeline-ready outputs by default**
   - Every check result has a stable, machine-readable schema.
   - Batch processing is first-class (not “just call vet_candidate in a loop”).
2. **Extensible vetting architecture**
   - Adding checks does not require editing a monolithic orchestrator.
   - Check requirements and citations are introspectable.
3. **A small, obvious “golden path” API**
   - New researchers can learn the library in minutes.
   - Advanced primitives remain available, but are clearly separated.
4. **Determinism + reproducibility hooks**
   - Any stochastic component has `random_seed` and/or bounded budgets.
   - Outputs include minimal provenance.
5. **No silent footguns**
   - No stub check implementations exported as if real.
   - “Skipped” is explicit, structured, and discoverable (not hidden in warnings).

---

## 1) Non-goals

- No policy/guardrails layer (keep “metrics-only”; policy belongs elsewhere).
- No new physics algorithms (only reorganize/normalize + remove contradictions).
- No attempt to maintain backwards compatibility with *pre-release* API shape.

---

## 2) Breaking Changes (Explicitly Allowed in This Tranche)

These are allowed now because you haven’t published yet:

1. **Change public result types** (especially `CheckResult` and bundle outputs).
2. **Move/rename exports** within `bittr_tess_vetter.api` and submodules.
3. **Remove stub/deferred checks** and any public exports that point to them.
4. **Change how skipping is represented** (stop relying on `warnings.warn`).
5. **Reshape the `bittr_tess_vetter.api` root export surface** to a curated “golden path”.

After v0.1.0, this becomes much harder—so we do it now.

---

## 3) Proposed v0.1.0 Public Contract

### 3.1 Golden-path namespace (`bittr_tess_vetter.api`)

`bittr_tess_vetter.api` should become the “researcher-facing façade”, intentionally small and stable:

**Core types**
- `LightCurve`, `Ephemeris`, `Candidate`, `TPFStamp`
- `CheckResult`, `VettingBundleResult` (new stable schemas; see below)

**Core entry points**
- `run_periodogram(...)`
- `vet_candidate(...)`
- `localize_transit_source(...)` (or `pixel_localize(...)` — pick one canonical name)
- `recover_transit(...)`
- `fit_transit(...)` (if supported in v0.1.0)
- `calculate_fpp(...)` / `calculate_fpp_handler(...)` (optional extra)

**Batch/pipeline**
- `VettingPipeline` (first-class; see §4)
- `list_checks()` / `describe_checks()` (introspection)

### 3.2 Advanced namespaces

Move everything else out of the root:

- `bittr_tess_vetter.api.primitives.*` (supported building blocks)
- `bittr_tess_vetter.api.experimental.*` (explicitly unstable / “use at your own risk”)

This is mainly a documentation + import-surface decision; the code can stay where it is initially, but the *recommended import paths* should match this structure.

---

## 4) Core Refactor: Results + Pipeline Framework

### 4.1 Replace “details dict” as the primary interface

Today, `CheckResult.details: dict[str, Any]` is the de facto output contract. That’s not pipeline-grade.

**New v0.1.0 contract**: structured, typed results.

#### `CheckStatus`
`Literal["ok", "skipped", "error"]`

#### `CheckResult` (public)
Fields:
- `id: str` (e.g. `"V01"`)
- `name: str`
- `status: CheckStatus`
- `confidence: float | None` (optional/nullable when skipped)
- `metrics: dict[str, float | int | str | bool | None]`
- `flags: list[str]`
- `notes: list[str]`
- `provenance: dict[str, float | int | str | bool | None]` (minimal; no blobs)
- `raw: dict[str, Any] | None` (optional; for legacy/unstructured extra data)

Rules:
- `metrics` must be JSON-serializable scalars only.
- `flags` are stable machine-readable identifiers (no prose).
- `notes` are human-readable and can change; downstream should not parse them.

#### `VettingBundleResult` (public)
Fields:
- `results: list[CheckResult]`
- `warnings: list[str]` (human-readable summaries)
- `provenance: dict[str, Any]` (inputs used, versions, timing, enabled checks)
- `inputs_summary: dict[str, Any]` (what data was provided: stellar/tpf/network/etc.)

### 4.2 Centralize constructors and normalization

Add `bittr_tess_vetter.validation.result_schema` with:
- `ok_result(id, name, *, metrics, flags=None, notes=None, confidence=None, provenance=None, raw=None)`
- `skipped_result(id, name, *, reason_flag, notes=None, provenance=None, raw=None)`
- `error_result(id, name, *, error, flags=None, notes=None, provenance=None, raw=None)`

All checks must use these helpers.

### 4.3 Introduce `VettingCheck` + `CheckRegistry`

Add `bittr_tess_vetter.validation.registry`:

- `CheckTier`: `LC_ONLY`, `CATALOG`, `PIXEL`, `EXOVETTER`, `AUX` (optional)
- `CheckRequirements`:
  - `needs_tpf: bool`
  - `needs_network: bool`
  - `needs_ra_dec: bool`
  - `needs_tic_id: bool`
  - `needs_stellar: bool` (soft requirement allowed)
  - `optional_deps: list[str]` (e.g. `["mlx"]`, `["triceratops"]`)

- `VettingCheck` protocol:
  - `id`, `name`, `tier`, `requirements`, `citations`
  - `run(inputs, config) -> CheckResult`

- `CheckRegistry` with `register`, `get`, `list`, `list_by_tier`.
- `register_default_checks()` registers V01–V12 built-ins.

### 4.4 Add `VettingPipeline`

Add `bittr_tess_vetter.api.pipeline` (or `api/vetting_pipeline.py`):

- `VettingPipeline(checks: list[str] | None = None, *, registry=None, default_config=None, emit_warnings=False)`
- `run(lc, candidate, *, stellar=None, tpf=None, network=False, ra_deg=None, dec_deg=None, tic_id=None, context=None) -> VettingBundleResult`
- `describe(...) -> dict` (what will run, what will be skipped, why)

`vet_candidate()` becomes a thin wrapper:
- `return VettingPipeline().run(...)`

---

## 5) Remove Contradictions: One Source of Truth for Checks

### 5.1 Delete stubs / “deferred” check exports

Do not ship “deferred: no catalog cache” results as if they are checks.

Actions:
- Remove V06/V07 stub functions from `validation/lc_checks.py`.
- Ensure catalog checks live in one place (prefer `validation/checks_catalog.py`).
- Ensure pixel checks live in one place (prefer `validation/checks_pixel.py`).
- Update `validation/__init__.py` to export only real implementations.

### 5.2 Align docstrings and module boundaries

Ensure “where a check lives” matches how it’s executed:
- LC-only checks in `validation/lc_checks.py`
- Catalog checks in `validation/checks_catalog.py`
- Pixel checks in `validation/checks_pixel.py`
- Exovetter in `validation/exovetter_checks.py`

---

## 6) Optional Dependencies: Crisp, Uniform Behavior

Add a library-side exception:
- `bittr_tess_vetter.errors.MissingOptionalDependency(extra: str, install_hint: str)`

Rules:
- Core API functions that require extras should raise `MissingOptionalDependency`.
- `VettingPipeline` should convert missing optional deps into:
  - `CheckResult(status="skipped", flags=[f"EXTRA_MISSING:{extra}"])`
  - plus a human `warning` entry.

---

## 7) Tests (Must-have for v0.1.0)

### 7.1 Result schema contract tests
- Every check returns `CheckResult` with valid `status` and JSON-serializable `metrics`.
- `vet_candidate()` always returns `VettingBundleResult` with `inputs_summary`.

### 7.2 Registry tests
- `register_default_checks()` registers V01–V12 and they are runnable/skippable per requirements.

### 7.3 No-stubs tests
- Ensure no exported check reports `stub/deferred` for V06/V07 in normal operation when network is enabled (or, if network is disabled, it must be `skipped` with structured reason—not “deferred”).

### 7.4 Adapter compatibility (lockstep safety)
- Add a small “adapter contract” test in bittr that asserts the adapter-used symbols exist (or use the astro-arc-tess test suite as the contract by running it in CI before tagging).

---

## 8) Execution Plan (Aggressive, One-Day)

Phase 1: Land new types + constructors
- Implement new `CheckResult` / `VettingBundleResult` (public)
- Implement `validation.result_schema` helpers
- Update LC-only checks first (V01–V05) to use helpers

Phase 2: Registry + pipeline
- Add registry + default registration
- Implement `VettingPipeline` and migrate `vet_candidate()` to it

Phase 3: Catalog/pixel/exovetter normalization
- Convert remaining checks to new results
- Remove stubs and unify exports

Phase 4: API surface shaping
- Curate `bittr_tess_vetter.api` exports to the golden path
- Move other exports under `api.primitives` / `api.experimental` (aliases allowed during pre-release)

Phase 5: Update astro-arc-tess adapter in lockstep
- Adjust `astro_arc.bittr.*` to new bittr API contract
- Run adapter tests + E2E tests in astro-arc-tess

Phase 6: Tag and publish
- Build docs
- Build wheel/sdist
- Tag `v0.1.0` once both repos are green

---

## 9) Release Notes (What We’ll Claim)

For v0.1.0, the story to the community is:
- “A pipeline-ready, metrics-first vetting library with stable result schemas.”
- “Extensible check registry + pipeline runner.”
- “Clear separation of golden-path API vs advanced primitives.”

