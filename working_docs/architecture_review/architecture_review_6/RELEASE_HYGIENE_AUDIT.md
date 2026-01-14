# Release Hygiene Audit (v0.1.0)

**Date**: 2026-01-14  
**Author**: codex  
**Scope**: Open-source readiness cleanup (packaging, metadata, licensing, docs build sanity).  
**Non-goals**: New features, API redesign, performance tuning.

## Executive Summary

The codebase is in good shape for an initial public release. The largest concrete “release hygiene” issue found was **sdist bloat/leakage** (local caches and internal working documents were being shipped). That has been fixed by making the sdist manifest explicit.

Key status:
- **Packaging**: wheel looked clean; **sdist was not** → fixed.
- **Metadata**: added missing `classifiers`, `urls`, and `license-files` → fixed.
- **Licensing/Citation**: core BSD-3-Clause is consistent across files; vendored code has an in-tree license file.
- **Docs**: Sphinx HTML build succeeds; tutorials appear aligned with the post-v0.1.0 API.

## Findings

### 1) Packaging: sdist contents were unsafe / noisy (fixed)

**Impact**: Shipping `.uv-cache/`, `working_docs/`, and `.github/` in the sdist is undesirable for:
- size + install-time overhead
- accidental leakage of internal notes
- accidental inclusion of local state (caches)

**Fix applied**: `pyproject.toml` now defines an explicit sdist manifest:
- `only-include = ["src", "tests", "docs", …project metadata…]`
- `exclude = [".uv-cache", "working_docs", ".github", "uv.lock", caches, build outputs, …]`

This keeps the sdist “source-distribution shaped” and avoids surprising payloads.

### 2) Project metadata: missing classifiers/urls/license-files (fixed)

**Impact**: On PyPI/metadata consumers, the project looked incomplete:
- no OSI classifier for license
- no audience/topic classifiers
- no project URLs
- no license-files declaration

**Fix applied**: `pyproject.toml` now includes:
- `license-files = ["LICENSE"]`
- `classifiers = [...]` including BSD + astronomy topic + python versions
- `urls = { Repository=..., Issues=... }`

### 3) Licensing + citations: consistent (good)

Checked artifacts:
- `LICENSE` is BSD-3-Clause and matches `pyproject.toml` / `CITATION.cff`.
- `CITATION.cff` exists and includes required keys for common citation tooling.
- `THIRD_PARTY_NOTICES.md` exists.
- Vendored TRICERATOPS+ code includes `src/bittr_tess_vetter/ext/triceratops_plus_vendor/LICENSE`.

Notes:
- Optional extras include GPL-licensed dependencies (e.g. `pytransit`). This is documented inline in `pyproject.toml` and should also be called out in docs (already partially done).

### 4) Docs + tutorials: build and drift checks (good)

Docs:
- `uv run sphinx-build -b html docs docs/_build/html` succeeds.

Tutorial drift:
- Quick scan for legacy API strings (e.g. `policy_mode`, `all_passed`) in `docs/tutorials/` found none.

### 5) CI + templates: present (good)

`.github/` contains:
- `ci.yml` and `release.yml`
- issue/PR templates
- `dependabot.yml`

These should stay in the repo (but are now excluded from the **sdist**).

## Concrete Cleanup Changes Made

- `pyproject.toml`
  - Added `[tool.hatch.build.targets.sdist]` with strict `only-include` + `exclude`.
  - Added `license-files`, `classifiers`, and `urls` under `[project]`.

## Remaining Cleanup Priorities (No New Features)

P0 (before publishing to PyPI):
- Verify `uv run hatch build -t sdist -t wheel` output contents match expectations (spot-check root files, ensure no caches).
- Decide whether shipping `tests/` and `docs/` in the sdist is desired. Current config includes both; if you want “minimal sdist”, remove them from `only-include`.

P1 (nice-to-have, low risk):
- Add a small “Release checklist” section to `README.md` (supported Python versions, optional extras matrix, citation pointer, license note re: GPL extras).
- Ensure `.uv-cache/` remains globally ignored for contributors (it is not currently showing up in `git status`, but repository-level ignore can be added if desired).

## Commands Used (for reproducibility)

- Build artifacts:
  - `uv run hatch build -t sdist -t wheel`
- Inspect sdist for unwanted payload:
  - `python -c "…tarfile listing…"`
- Docs build:
  - `uv run sphinx-build -b html docs docs/_build/html`
- Sanity checks:
  - `uv run ruff check .`
  - `uv run pytest -q`

