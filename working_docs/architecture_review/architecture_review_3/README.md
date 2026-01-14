# Architecture Review (v3, Critical): bittr-tess-vetter

Date: 2026-01-14  
Reviewer: codex (critical pass)

Scope: repo-level architecture review for long-term open-source maintainability and “pipeline builder” ergonomics. Not a physics audit.

## Executive Summary

The project is in a strong place (tests/lint/mypy green; docs/CI/release automation present), but there are several *high-impact architecture hygiene issues* that will create friction for external adopters and contributors if not addressed.

Most of the risks are “sharp edges” rather than algorithmic correctness:

- The **package boundary story (“domain-only”) is not actually enforceable** with the current module placement (I/O + caching exists outside `platform/`, and some domain modules do network work).
- The repo currently **commits Sphinx build artifacts** (`docs/_build/**`), which is a major open-source hygiene problem.
- The repo’s **install story is internally inconsistent** (README references extras that don’t exist; base dependencies are heavy for “minimal” use).
- Claimed **Windows support is overstated** due to import-time Unix-only dependencies (`fcntl`).

These are solvable, but they should be treated as *release-quality issues* because they affect first impressions and downstream maintainability.

## Current State (Observed)

Local health checks in this working tree:
- `uv run ruff check .`: passes
- `uv run mypy src/bittr_tess_vetter`: passes (164 files)
- `uv run pytest`: passes (with a single RuntimeWarning in `compute/primitives.py`)

Public API surface:
- `bittr_tess_vetter.api.__all__`: 229 exports
- Lazy export machinery in `src/bittr_tess_vetter/api/__init__.py` is ~956 LOC and is a long-term maintenance hotspot.

## P0 Findings (High Impact / Should Fix Before Broader Adoption)

### P0.1 — Sphinx build artifacts are committed

`docs/_build/**` is checked into git (HTML + doctrees). This is a serious repo hygiene issue:
- bloats clone size
- creates noisy diffs and merge conflicts
- risks stale docs being served/linked

Evidence:
- `git ls-files docs/_build` returns many files.

Recommendation:
- Add `docs/_build/` to `.gitignore` and remove the tracked artifacts.
- Decide whether `docs/_autosummary/` is generated-at-build (recommended) or committed (acceptable but heavy); document that choice.

### P0.2 — “Domain-only” boundary is leaky / inconsistent

The README claims:
- Pure domain logic is in `api/`, `compute/`, `validation/`, `transit/`, `recovery/`, `activity/`
- “Opt-in infrastructure” is in `platform/`

But:
- Pixel “data product” modules include filesystem caches outside `platform/` (e.g. `src/bittr_tess_vetter/pixel/tpf.py` and `tpf_fits.py`).
- FPP support includes network primitives outside `platform/` (e.g. `src/bittr_tess_vetter/validation/triceratops_fpp.py` uses `urllib`).

This is not just semantics: it makes it harder for downstream apps to reason about “safe imports” and to use the library in restricted environments (clusters, sandboxed notebooks, corporate networks).

Recommendation (pick one direction and enforce it):
- **Option A (strict boundary):** move all disk/network code (including pixel caches and FPP network access) under `platform/` and keep `compute/validation/pixel/transit/...` as array-in/array-out.
- **Option B (honest boundary):** stop calling the package “domain-only” and instead document it as “domain-first, with optional platform helpers shipped in-tree”.

### P0.3 — Windows support claim is not accurate (import-time `fcntl`)

`src/bittr_tess_vetter/platform/io/cache.py` imports `fcntl` at module import time. On Windows this raises `ImportError` immediately; there is no “graceful fallback” unless guarded.

Recommendation:
- Use a try/except import for `fcntl` and implement a Windows-safe lock/no-lock path.
- Add a small CI job (or at least documentation) validating import behavior on Windows.

### P0.4 — Install docs drift: README references extras that do not exist

`pyproject.toml` defines extras: `wotan`, `ldtk`, `triceratops`, `docs`, `all`. There is **no** `dev` extra.

But README still advertises:
- `python -m pip install -e ".[dev]"`

Recommendation:
- Fix README install instructions (and keep `uv` and `pip` paths consistent).

### P0.5 — Versioning is not aligned with the release process

The release workflow triggers on tags `v*`, but package metadata is currently `0.0.1`:
- `pyproject.toml` → `version = "0.0.1"`
- `src/bittr_tess_vetter/__init__.py` → `__version__ = "0.0.1"`

If you tag `v0.1.0` without bumping the project version, you’ll publish a wheel claiming `0.0.1`. This will confuse users and break reproducibility.

Recommendation:
- Single-source the version (either via hatch dynamic versioning or via `importlib.metadata.version()`), and ensure tag/version match.

## P1 Findings (Medium Impact / Next Iteration)

### P1.1 — Base dependencies are heavy for a “minimal” research library

Base dependencies include `numba`, `emcee`, and `arviz`. Many users want:
- “Just vet a candidate” without MCMC stacks
- “Just do CPU vetting” without TLS/numba

Recommendation:
- Define an explicit **minimal core** and push heavyweight capabilities into extras, e.g.:
  - `detection-tls` (TLS + numba),
  - `fit` (emcee/arviz/batman),
  - `platform` (requests + catalogs + caches),
  - `triceratops` (current bundle).

### P1.2 — Runtime dependency on `setuptools` is unusual

`setuptools` is a build-time tool; shipping it as a runtime dependency is typically unnecessary and can cause conflicts in constrained environments.

If the intent is “security patching”, that belongs in build tooling / CI images, not in end-user runtime deps.

Recommendation:
- Remove `setuptools` from `[project.dependencies]` unless there is a concrete runtime import path requiring it.

### P1.3 — Public API surface is large and “flat”

`bittr_tess_vetter.api` exports 229 symbols, and the file implementing the lazy export machinery is ~956 LOC.

Risks:
- Steep learning curve for new users
- High likelihood of accidental API churn
- Difficult for maintainers to reason about what is stable vs incidental

Recommendation:
- Introduce **stability tiers** (e.g. “stable”, “provisional”, “internal”) and reflect that in docs.
- Consider a smaller “core facade” module (e.g. `bittr_tess_vetter.api.core`) and keep the full surface available but clearly marked.

### P1.4 — Check orchestration is not extensible for third-party pipelines

`vet_candidate()` hardcodes check IDs (V01–V12) and tier logic. This is fine for a reference pipeline, but pipeline builders will want:
- custom checks,
- custom ordering,
- custom gating rules,
- metadata about check requirements and citations.

Recommendation:
- Add a lightweight check registry (protocol + metadata) so new checks can be registered without editing the central orchestrator.

### P1.5 — ReadTheDocs config is redundant/conflicting

`.readthedocs.yaml` uses both:
- `build.jobs.post_install` with `uv pip install --system -e ".[docs]"`
- `python.install` also installing with `extra_requirements: [docs]`

This can lead to double-installs and confusing behavior.

Recommendation:
- Choose one installation mechanism and delete the other.

## P2 Findings (Nice-to-Have, but Worth Tracking)

- Standardize reproducibility knobs across all “search/sampling” APIs (`random_seed`, `budget`, `timeout_seconds`) and ensure provenance records these consistently.
- Audit docstring examples for hardcoded `/tmp/...` paths (fine as examples, but better to show `tempfile`).
- Consider splitting “platform” into a separate distribution later if adoption suggests it (keeps minimal core clean).

## Suggested Next Actions (If You Want the Highest ROI)

1) Repo hygiene fixes: stop committing `docs/_build/**`, update `.gitignore`, remove empty cache dirs from the repo root if they’re not required.
2) Fix README install commands (extras and minimal install story).
3) Align tag/version and single-source the version.
4) Decide and enforce the boundary rule for network/disk code (strict vs “honest”).
5) Define a minimal core install + extras, so researchers can adopt without pulling MCMC stacks by default.

