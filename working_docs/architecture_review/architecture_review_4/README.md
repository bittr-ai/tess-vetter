codex

# Architecture Review (v4, API/Code-Focused): bittr-tess-vetter

Date: 2026-01-14  
Scope: “release-candidate audit” focused on **API and code architecture changes** that would materially improve adoption by astronomy/astrophysics researchers and enable third-party pipeline building.  
Assumption: All Tranche 3 hygiene items from `architecture_review_3/ORCHESTRATION_PLAN*.md` are complete, so this review avoids re-litigating them (docs build artifacts, Windows `fcntl` import guard, README `[dev]` drift, version alignment, etc.).

## Executive Summary (Critical)

You are in a strong “usable today” state, but the **API architecture will become your primary scaling bottleneck** as soon as outside researchers start building pipelines on top of it.

The two biggest structural issues are:

1) **A very large, flat public surface** (`bittr_tess_vetter.api.__all__ == 229`) with a complex lazy-export mechanism (~956 LOC). This is hard to learn, hard to govern, and hard to evolve without breaking users.

2) **Vetting check architecture is not yet a pipeline framework**: checks are mostly function-based, the orchestrator is hard-coded, and outputs are “dict-of-any” details without a stable schema. This makes custom checks and downstream automation fragile.

This document proposes concrete (sometimes large) changes, prioritizing what will reduce long-term maintenance burden and maximize scientific usability.

## P0 (High Impact) — API Must Become Governable

### P0.1 — Split “Supported API” vs “Everything Exported”

Problem:
- `bittr_tess_vetter.api` exposes 229 symbols, many of which are low-level primitives.
- New users do not know what is “the library” vs “internal helper”.
- Maintainers must treat almost everything as stable to avoid breaking imports, which slows evolution.

Recommendation (non-breaking path):
- Define an explicit **supported subset** and make it prominent in docs.
- Keep the full surface importable for power users, but clearly label it “advanced/unstable”.

Concrete design:
- `bittr_tess_vetter.api` (stable): the 10–20 most important entry points + core types:
  - `LightCurve`, `Ephemeris`, `Candidate`, `TPFStamp`
  - `run_periodogram`, `vet_candidate`, `pixel_localize`/`localize_transit_source`, `recover_transit`, `fit_transit`, `calculate_fpp`
  - `checks_to_evidence_items` (if you want evidence-first integration)
- `bittr_tess_vetter.api.primitives` (stable-ish): “building blocks” meant for pipelines.
- `bittr_tess_vetter.api.experimental` (explicitly unstable): everything else currently re-exported only for internal host needs.

Enforcement mechanism:
- Add a tiny “contract test” asserting the stable surface and its doc coverage.
- Add `DeprecationWarning` (or documentation-only deprecation first) for symbols planned to move tiers.

### P0.2 — Replace “dict[str, Any] details” with a Stable Result Schema

Problem:
- `CheckResult.details` is a free-form dict with check-specific keys.
- Pipeline builders need stable machine-readable structure for aggregation, UI, and persistence.

Recommendation:
- Standardize `CheckResult` / `VetterCheckResult` into a schema like:
  - `status`: `"ok" | "skipped" | "error"`
  - `metrics`: `dict[str, float | int | str | bool | None]` (JSON-serializable)
  - `flags`: `list[str]` (machine-readable tags)
  - `artifacts`: optional small arrays/paths/refs (or keep out of core results)
  - `notes`: `list[str]` (human-readable)
  - `provenance`: optional `dict` (version, inputs used, timing)

Migration strategy:
- Keep `details` for one minor series, but populate it with the above nested keys while continuing to include legacy keys.
- Provide helper accessors:
  - `result.metrics`, `result.flags`, `result.status` (properties mapping into `details` for backwards compatibility).

This is the single most important change for “pipelines can be built on it”.

### P0.3 — Eliminate Duplicated/Stub Check Implementations

Problem:
- `src/bittr_tess_vetter/validation/lc_checks.py` contains “deferred/stub” implementations for V06/V07 (and some pixel placeholders), while real implementations exist in `validation/checks_catalog.py`, `validation/checks_pixel.py`, and API wrappers use those.
- `bittr_tess_vetter.validation.__init__` re-exports the stub functions (`check_nearby_eb_search`, `check_known_fp_match`, `run_all_checks`), which creates a trap for contributors and downstream users.

Recommendation (clean architecture):
- Make `validation/lc_checks.py` **LC-only only** (V01–V05) and remove V06–V10 stubs from it.
- Move check orchestration into one place (see registry below) or update `run_all_checks()` to call the real catalog/pixel implementations.
- Update `validation/__init__.py` to export the correct implementations and stop exporting stubs.

This reduces internal contradictions and makes future refactors safer.

## P1 (High Leverage) — Make Vetting a Pipeline Framework

### P1.1 — Introduce a First-Class Check Protocol + Registry

Problem:
- `vet_candidate()` hardcodes tier logic and check IDs.
- Adding a new check requires editing multiple files and deciding where to wire it.

Recommendation:
- Define a `VettingCheck` protocol/class with:
  - `id`, `name`, `tier`, `requires` (lc/stellar/tpf/network/coords/tic_id), `citations`
  - `run(inputs, config) -> CheckResult`
- Build a `CheckRegistry` that:
  - registers built-ins at import time (or via a `register_default_checks()` call)
  - supports third-party registration (entry points later if desired)

Then `vet_candidate()` becomes:
- resolve inputs → determine runnable checks → run in order → return bundle.

This is a “big change”, but it pays off immediately:
- easier to add/remove checks
- easier to run subsets
- easier to batch-run pipelines
- easier to introspect what a pipeline will do (critical for reproducibility)

### P1.2 — Add a Pipeline Object (Optional but Powerful)

Add an object-oriented option:
- `pipeline = VettingPipeline(checks=[...], default_config=..., concurrency=...)`
- `bundle = pipeline.run(lc, candidate, stellar=..., tpf=..., network=..., ...)`

This solves common research workflows:
- “run the same check set for 10k candidates”
- “only LC-only checks in prefilter stage”
- “custom check set for a paper”

### P1.3 — Replace “warnings.warn” with Structured Warnings by Default

`warnings.warn` is noisy in notebooks and batch pipelines.

Recommendation:
- Default: record skip reasons in `VettingBundleResult.warnings` (and in per-check `status="skipped"`).
- Optional: `emit_warnings=True` to also raise Python warnings for interactive usage.

## P2 (Strategic) — Installation and Performance as API Constraints

Even if you keep Tranche 3 changes, two larger “API adjacent” design decisions matter for researchers:

### P2.1 — Minimal Core vs Heavy Features

If you want wide adoption, define a minimal installation that supports:
- LC-only vetting + basic detrend + (optional) BLS-lite

Then push heavier stacks behind extras:
- TLS/numba, MCMC stacks (emcee/arviz), TRICERATOPS, etc.

This is not only packaging: it should shape the API:
- functions that require extras should raise a crisp `MissingDependencyError` with install hint.

### P2.2 — Import-Time / First-Use Latency Budget

The current approach (very large `api/__init__.py`, extensive re-exports) tends to increase first-use latency and complicate optional dependency boundaries.

Recommendation:
- Keep lazy exports, but shrink and automate the mapping:
  - generate export maps from a single source-of-truth list (or from module-level `__all__`)
  - avoid hand-maintaining a 900+ line `__init__.py` long-term

## Suggested “v0.2+” Roadmap (API Breaking Changes Allowed Later)

If you are willing to do large changes, the cleanest staged plan is:

1) v0.1.x: introduce stable result schema keys (`status/metrics/flags`) while keeping existing `details` keys.
2) v0.2: add `VettingCheck` registry + optional `VettingPipeline`; start migrating `vet_candidate` to use the registry internally.
3) v0.3: move most “primitives” out of `bittr_tess_vetter.api` root and into `api.primitives`; keep lazy aliases with deprecation warnings.
4) v1.0: formalize stability tiers, remove deprecated root exports, finalize “supported API” contract.

## Go/No-Go (From an API/Code Perspective)

If Tranche 3 is done, you can ship v0.1.0. But if your goal is “very easy to use library where pipelines can be built on it”, the items above are the difference between:
- a capable internal library, and
- an ecosystem-friendly platform others can safely extend.

