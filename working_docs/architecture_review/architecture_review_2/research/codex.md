codex

# Architecture / Open-Source Readiness Review (2026-01-14)

Scope: codebase review of `bittr-tess-vetter` for open-sourcing to astronomy/astrophysics researchers, with emphasis on library ergonomics, boundaries (domain vs platform), reproducibility, and contributor experience. This is not a physics/algorithm audit.

## Executive Summary

The repo is close to being a strong open-source research library: the public API is thoughtfully organized (`bittr_tess_vetter.api`), there is extensive test coverage, strong type hints, and a rare “citation-first” discipline that researchers will appreciate.

The main blockers to “open source ready for outsiders” are not architecture fundamentals; they are *polish and governance* issues: docs + examples, consistency of repo claims vs implementation, green CI (tests/lint), and clarifying “domain-only” boundaries and optional dependencies.

## What’s Strong (Keep)

- **Clear intent and layering**: `api/` as the stable surface; domain modules (`compute/`, `validation/`, `pixel/`, `transit/`, `recovery/`, `activity/`) are reasonably well separated; platform I/O lives under `platform/`.
- **Research provenance culture**: `api/references.py` + `@cites` is unusually good for a library that wants adoption in academia.
- **Metrics-first outputs**: returning check metrics without policy judgments is a good fit for pipelines and downstream decision systems.
- **Test depth**: `pytest` run shows very high pass rate and lots of targeted unit tests across subsystems.

## Current Health Check Results (Local)

- `uv run pytest`: **1 failure**, **1617 passed**, **47 skipped**
  - Failure: `tests/pixel/test_tpf_fits.py::TestTPFFitsRefSerialization::test_from_string_invalid[...]`
  - Root cause: error message mismatch (`Invalid TPF FITS reference format` expected; code raises `Invalid exptime_seconds: ...`).
- `uv run ruff check .`: **115 errors** (imports unsorted, unused variables, a few style warnings). README currently recommends this command; as-is it fails.
- `uv run mypy src`: `mypy` was not available in the current environment because the `uv` dev dependency group does not include it (see packaging notes below).

## Key Findings (Prioritized)

### P0 — Open-source “table stakes” (blockers)

1) **Docs for outsiders are missing**
   - There is good internal documentation (`working_docs/`), but there is no public-facing docs site and no “researcher workflow” narrative (common tasks, gotchas, minimal examples).
   - Recommendation: add a docs build (MkDocs or Sphinx), and publish on ReadTheDocs / GitHub Pages.

2) **Green CI is not guaranteed**
   - `pytest` is almost green (1 failing test), but `ruff check .` fails with many errors.
   - Recommendation: decide what “CI-required” means (e.g., `ruff check src tests` with consistent rules, or `ruff check src` only) and make it green before announcing.

3) **Repository messaging drift vs implementation**
   - README claims `src/bittr_tess_vetter/io/` and `src/bittr_tess_vetter/catalogs/` exist; the actual code uses `src/bittr_tess_vetter/platform/io/` and `platform/catalogs/`.
   - README references a `bittr-reason-core` local dependency, but `pyproject.toml` / `uv.lock` do not contain it.
   - Some docstrings still reference `astro_arc.*` paths (e.g., `src/bittr_tess_vetter/platform/catalogs/store.py`).
   - Recommendation: fix drift so new users don’t lose trust in the docs early.

4) **License + community meta files likely incomplete**
   - `pyproject.toml` declares MIT, but there is no obvious top-level `LICENSE` file.
   - Recommendation: add `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and `CITATION.cff` before publishing.

### P1 — API + boundary ergonomics (high leverage for researchers)

5) **`api/catalog.py` vs `api/catalogs.py` naming collision is real**
   - `api/catalog.py` = vetting checks (V06–V07); `api/catalogs.py` = host-facing catalog clients / snapshot store.
   - This is easy to mis-import and increases support load.
   - Recommendation: keep both for compatibility, but add stronger top-level docstrings + recommended imports (and consider a future rename with aliases).

6) **“Domain-only” claim conflicts with shipped platform helpers**
   - The project includes network clients, caching, and disk-backed catalog snapshots under `platform/`. That’s fine, but it’s not “pure domain-only” in the sense many researchers expect.
   - Recommendation (non-breaking): document the split explicitly and treat `platform/*` as opt-in (extras/import gating), or consider splitting into two distributions later (`bittr-tess-vetter` + `bittr-tess-vetter-platform`).

7) **Default cache path is CWD-relative**
   - `platform/io/cache.py` defaults to `Path.cwd() / ".bittr-tess-vetter" / ...` if env vars aren’t set.
   - For library use, CWD-relative writes surprise users (e.g., Jupyter, SLURM job dirs, read-only working dirs).
   - Recommendation: default to an OS-appropriate user cache dir (e.g., `platformdirs`) and keep env var overrides.

8) **Cross-platform behavior should be explicit**
   - Cache locking uses `fcntl` (Unix-specific).
   - Network timeouts use SIGALRM (best-effort/no-op on some platforms).
   - Recommendation: document platform support (macOS/Linux first-class, Windows “best effort”), and ensure graceful degradation is tested.

### P2 — Library governance + maintainability

9) **Huge public surface area needs “export governance”**
   - Lazy exports are a good fix for import-time cost, but the surface area is large and easy to churn accidentally.
   - Recommendation: add a small set of “API contract” tests (critical exports + aliases + invariants like metrics-only).

10) **Optional dependency strategy could be clearer**
   - Base dependencies include heavy stacks (TLS/numba, emcee/arviz/ldtk). This may be OK, but many research users prefer a minimal core plus extras.
   - Recommendation: define a “minimal install path” (core CPU vetting) and move heavier features into extras where feasible.

## Suggestions for Open-Source User Experience

- **“Start here” docs**: 2–3 short recipes:
  1) vet a candidate from arrays (no network, no TPF),
  2) enable catalog checks (network on, TIC/coords),
  3) run pixel localization (TPF provided) and interpret host ambiguity.
- **Notebook examples** (even if not shipped in the wheel): one reproducible end-to-end analysis with small cached fixtures.
- **Explicit stability policy**: what is stable in `bittr_tess_vetter.api` (and what is experimental), plus deprecation policy.
- **Reproducibility knobs**: standardize `random_seed`, `budget`/`timeout_seconds`, and return provenance fields for anything stochastic or heuristic.

## Concrete Action List (Minimal Path to “Ready”)

1) Make CI green:
   - Fix the single failing `pytest` case.
   - Decide on `ruff` scope/rules and make it pass.
2) Align docs with reality:
   - Update README code map (platform paths).
   - Remove/clarify the `bittr-reason-core` note.
   - Fix `astro_arc.*` docstring remnants.
3) Add open-source meta:
   - `LICENSE` (MIT text), `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`.
4) Add lightweight docs site + 1–2 examples:
   - Publish basic API reference + quickstarts.

## Notes / Small Paper Cuts Observed

- Several docstrings/examples hardcode `/tmp/...` paths (fine for quick examples, but consider `tempfile` patterns or explicitly mark as “example only”).
- `pyproject.toml` has both `[project.optional-dependencies].dev` and `[dependency-groups].dev`; `uv sync` may not install `mypy` unless you ensure the dev group includes it (or update docs to use `--extra dev`).

