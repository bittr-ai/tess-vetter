# Architecture Review: bittr-tess-vetter

Scope: repo-level architecture review of the `bittr-tess-vetter` Python package (domain library).

This review focuses on layering, boundaries, cross-cutting concerns (network, caching, determinism),
and prioritizes follow-up work. It is not a physics audit of individual algorithms (see
`working_docs/physics_audit/` for that).

Last updated: 2026-01-11

## Executive Summary

`bittr-tess-vetter` is structured as a “domain-only” library with a stable, host-facing facade
(`src/bittr_tess_vetter/api/`) wrapping internal modules (`compute/`, `validation/`, `pixel/`,
`transit/`, `recovery/`, `activity/`). The codebase is unusually strong on:

- A stable public API surface, with wrapper modules and explicit types (`api/types.py`)
- Explicit provenance patterns (citations via `api/references.py`, evidence conversion helpers)
- Tiered vetting orchestration (`api/vet.py`) with clear data gating (LC-only vs catalog vs pixel)
- Extensive unit + API tests (`tests/`) across most subsystems

Main architectural risks are (a) import-time / surface-area complexity of `api/__init__.py`,
(b) drift between “domain-only” aspiration vs in-repo caching/I/O/catalog installation utilities,
and (c) “policy vs metrics” separation being mostly enforced by convention rather than types.

## Current Architecture (as implemented)

### Layers / Packages

1. **Public facade (`api/`)**
   - Intended stable import surface for hosts (`from bittr_tess_vetter.api import ...`).
   - Provides wrapper functions that:
     - Convert facade types to internal types
     - Validate shapes/units in a host-friendly way
     - Apply gating (e.g., `network=False` => return skipped results)
   - Central orchestration: `api/vet.py` (tiered pipeline runner).

2. **Domain models (`domain/`)**
   - Core “validated models” (mostly Pydantic) used internally and in compute/validation outputs.
   - Example: `domain/detection.py` defines `TransitCandidate`, `VetterCheckResult`, etc.

3. **Core algorithms (`compute/`, `transit/`, `recovery/`, `activity/`, `pixel/`)**
   - Implements array-based computations; typically expects normalized arrays and explicit ephemerides.
   - Optional accelerator path: MLX (`compute/mlx_detection.py`), gated by availability.
   - Pixel/wcs tooling is more “data-product-like” but still array-in/array-out.

4. **Vetting checks (`validation/`)**
   - Implements check logic (LC-only, catalog-backed, pixel-backed, exovetter-style).
   - Outputs `VetterCheckResult` (metrics-first, host applies policy).

5. **I/O and external data (`io/`, `catalogs/`, `network/`, `ext/`)**
   - `io/`: caching and MAST client helpers.
   - `catalogs/`: external catalog clients + disk-backed snapshot store.
   - `network/timeout.py`: best-effort timeouts (SIGALRM-based on Unix).
   - `ext/`: vendored TRICERATOPS+ tree for FPP.

### Primary Execution Flows

**A) Detection**
- Light curve ingestion / normalization: `api/types.LightCurve.to_internal()`
- Period search: `api/periodogram.py` → `compute/periodogram.py` and TLS/BLS/LS tooling
- Candidate construction: facade `Candidate` + internal `TransitCandidate`

**B) Vetting**
- Entry point: `api/vet.vet_candidate()`
  - Always runs LC-only checks (V01–V05)
  - Runs catalog checks (V06–V07) only when `network=True` *and* required metadata present
  - Runs pixel checks (V08–V10) only when TPF provided
  - Attempts exovetter checks (V11–V12), which handle missing dependency internally
- Output: `VettingBundleResult` + check-level details + provenance metadata
- Evidence conversion: `api/evidence.checks_to_evidence_items()`

**C) “Non-periodic” timing tools**
- “Measure and analyze TTVs”: `api/timing.py` (template/trapezoid per-transit fits + O-C stats)
- “TTV track search” (detection aid): `api/ttv_track_search.py` → `transit/ttv_track_search.py`
  - Useful after you already have an approximate ephemeris but multi-window timing shifts smear stacking.

## Cross-Cutting Design Themes

### 1) Policy vs Metrics

The codebase is explicit about “metrics-only” (host applies dispositions), but the enforcement is
mostly by convention:

- `domain/VetterCheckResult.passed` is documented as metrics-only (typically `None`).
- Facade wrappers force metrics-only in some places (e.g. `api/lc_only._apply_policy_mode`), but
  the “policy_mode” abstraction currently supports only `"metrics_only"` and raises otherwise.

### 2) Provenance / Reproducibility

Strengths:
- Strong citation story (`api/references.py` registry + `@cites` decorator)
- Evidence-friendly serialization (`api/evidence.py`)

Potential gaps:
- Some algorithms have determinism controls (e.g. TTV track search random seed), but this pattern
  is not uniformly surfaced across all stochastic or sampling-based components.

### 3) Optional Dependencies and Heavy Imports

- MLX is handled correctly as optional: availability is probed and APIs are guarded.
- TRICERATOPS+ is vendored but still brings a large dependency surface (extras).
- `api/__init__.py` re-exports a very large surface area and imports many modules eagerly; this
  risks slow import, accidental heavy dependency import, and increased circular-import fragility.

### 4) “Domain-only” vs “I/O-adjacent”

README positions this as “domain-only”, yet the repository includes:
- session and persistent caches (`io/cache.py`, `tests/io/...`)
- catalog snapshot installation (`catalogs/store.py`) including network fetch

This is not inherently wrong, but it does create an implicit second axis:
“pure algorithms” vs “data access + reproducibility tooling”. The boundary is currently by
directory convention rather than an explicit package-level separation.

### 5) Platform Assumptions

- `io/cache.py` uses `fcntl` and so is Unix-centric.
- `network/timeout.py` uses SIGALRM; on platforms without it, timeouts become best-effort no-ops.

## Testing Posture (observed)

Strength:
- Broad unit coverage across `api/`, `compute/`, `validation/`, `pixel/`, and integration tests.
- There are explicit tests for the public API export surface (`tests/test_api/test_api_top_level_exports.py`),
  which is important given the large `api/__init__.py`.

Risk:
- With such a large import surface, tests may catch regressions but maintaining the export list
  can become a high-friction chore.

## Key Findings (with priorities)

### P0 (High impact / near-term)

1) **Reduce import-time coupling in `api/__init__.py`**
   - Problem: `api/__init__.py` eagerly imports/re-exports a large fraction of the codebase.
   - Impact: slower imports for host apps, higher circular-import risk, harder optional-dep hygiene.
   - Suggested direction:
     - Split into a minimal `api/__init__.py` (core types + a few orchestrators) and an
       explicit “kitchen sink” module (e.g. `api/all.py`) for power users; or
     - Introduce lazy exports (module-level `__getattr__` pattern) for heavy submodules.

2) **Make “metrics-only” a type-level invariant (or centralize policy)**
   - Problem: “policy_mode” exists but is only `"metrics_only"`; enforcement is scattered.
   - Impact: hosts may misinterpret `passed` semantics; future expansion risks inconsistent behavior.
   - Suggested direction:
     - Consider removing `policy_mode` until more modes exist, or
     - Encode metrics-only in the type (e.g. `passed: None` in the internal check result type) and
       provide a separate policy layer (host-facing) for PASS/WARN/REJECT derivations.

3) **Clarify/standardize boundaries for “I/O-adjacent” helpers**
   - Problem: caching + catalog installation are valuable, but they blur the “domain-only” framing.
   - Impact: harder to reason about what’s safe in restricted environments; risk of CWD-based caches.
   - Suggested direction:
     - Ensure any disk-writing default is explicit and documented (avoid CWD surprises),
       and keep all network/disk entry points clearly gated.

### P1 (Medium impact / follow-up)

4) **Unify facade/internal model duplication**
   - Problem: facade types are dataclasses; internal types are Pydantic models; conversions are repeated.
   - Impact: more boilerplate, more places for unit/shape mismatch bugs.
   - Suggested direction:
     - Define explicit conversion helpers in one place (per type), or
     - Consider a single “public dataclass + internal validator function” approach for the hottest paths.

5) **Determinism/provenance normalization across stochastic tools**
   - Problem: some tools accept seeds/budgets, others don’t.
   - Impact: harder to reproduce results across runs for pipelines.
   - Suggested direction:
     - Standardize a small pattern: `random_seed`, `budget`, `runtime_seconds`, `provenance` for
       any search/sampling method.

6) **Doc drift cleanup**
   - Example: some internal docs/comments describe catalog checks as “deferred”, while
     `validation/checks_catalog.py` and `api/catalog.py` implement them.
   - Impact: new contributors lose time; review/audit results become harder to trust.

### P2 (Lower urgency / longer-term)

7) **Explicit plugin/registry for checks**
   - Problem: check orchestration is currently a set of manual tiered calls.
   - Impact: adding/removing checks requires touching multiple modules.
   - Suggested direction:
     - Introduce a “check registry” with metadata (tier, requirements, default enabled, citations)
       to reduce boilerplate and ease extension.

8) **Packaging ergonomics**
   - Problem: `bittr-reason-core` is configured as a local editable dep in `pyproject.toml` via uv sources.
   - Impact: contributors outside the mono-repo layout may hit install friction.
   - Suggested direction:
     - Document expected workspace layout and/or provide a fallback (PyPI version) path.

## Suggested Next Actions (practical sequence)

1) Decide whether the canonical host import should be “minimal facade” or “kitchen sink”.
2) If minimal: refactor `api/__init__.py` to reduce eager imports; keep backwards compatibility via
   `api/all.py` and/or documented migration.
3) Make metrics-only semantics unambiguous (type or documentation), then update wrappers accordingly.
4) Tighten and document cache/network boundaries (default paths, platform limitations).

