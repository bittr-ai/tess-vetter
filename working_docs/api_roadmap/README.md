# API Roadmap (bittr-tess-vetter)

Goal: converge on a **robust, curated, stable API surface** that is easy for clients to use,
without embedding interpretation/policy in `bittr-tess-vetter` (metrics-only library).

Last updated: 2026-01-11

## Target End State

- **One recommended import surface** for clients:
  - `bittr_tess_vetter.api.facade` (or similarly named module).
- `bittr_tess_vetter.api.__init__` remains a **compatibility aggregator**:
  - continues to re-export a broad surface (for back-compat),
  - but the facade is the only thing recommended in docs and examples.
- **No interpretation/policy** in this package:
  - results remain metrics-only (`passed=None`, `_metrics_only=True`),
  - disposition/triage/guardrails live in `astro-arc-tess/src/astro_arc/validation` (tess-validate layer).

## Phase 0 — Decide the Facade Surface (design)

Define a small set of entrypoints/types that cover most client use:

- Primary orchestration:
  - `vet` (alias of `vet_candidate`)
- Core types:
  - `LightCurve`, `Ephemeris`, `Candidate`, `TPFStamp`
- Primary “domain checks” entrypoints:
  - `vet_lc_only`, `vet_catalog`, `vet_pixel`, `vet_exovetter`
- Common primitives:
  - `run_periodogram` / `auto_periodogram`
  - `fit_transit`
  - `measure_transit_times`, `analyze_ttvs`

Deliverable:
- `api/facade.py` exports only this curated surface.
- One short table documenting what’s in/out of the facade.

## Phase 1 — Additive Aliases (fast client ergonomics win)

Add a small set of **short, stable aliases** in `api/__init__.py` (non-breaking):

- Examples:
  - `vet = vet_candidate`
  - `localize = localize_transit_source`
  - `aperture_family_depth_curve = compute_aperture_family_depth_curve`

Guidelines:
- Keep aliases minimal and curated (top ~10–20).
- Do not remove existing exports.
- Prefer aliases that are unambiguous and map 1:1 to existing functions.

Deliverable:
- Aliases + a small API export test ensuring they remain importable.

## Phase 2 — Introduce `api.facade` (robustness step)

Create `src/bittr_tess_vetter/api/facade.py`:

- The only module referenced in new docs/examples.
- Keeps naming consistent and discoverable.
- Avoids exposing internal “grab bag” exports to new clients.

Deliverable:
- `api/facade.py`
- A dedicated test: `from bittr_tess_vetter.api.facade import ...` imports cleanly.

## Phase 3 — Client Migration

- Update downstream clients to:
  - import from `bittr_tess_vetter.api.facade`, or
  - use the new aliases in `bittr_tess_vetter.api` if they want to stay on top-level imports.

Deliverable:
- PRs in downstream repos.
- “preferred imports” examples in the host repos.

## Phase 4 — Deprecation (optional, later)

Once clients have migrated:

- Add narrow `FutureWarning` deprecations for a small set of legacy names that are:
  - confusing,
  - redundant,
  - or overly verbose.
- Maintain a migration mapping table in docs.

Deliverable:
- Deprecation warnings + migration table.

## Phase 5 — Major Version Cleanup (breaking, intentional)

In a major release:

- Remove deprecated names from `api/__init__.py` (or move them to `api.compat`).
- Keep `api.facade` stable.

Deliverable:
- Major-version changelog section + upgrade guide.

## Notes / Constraints

- `api/__init__.py` currently uses **lazy exports** (PEP 562) to keep import-time cost down.
  Any new facade/aliases should preserve that property (avoid eager heavy imports).
- Do not introduce interpretation/policy here; keep this package metrics-only.

