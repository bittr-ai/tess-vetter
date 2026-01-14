# Facade Pattern Remnants Analysis

**Date:** 2026-01-14
**Scope:** Identify and catalog remnants from the removed `api.facade` module

## Executive Summary

The `api/facade.py` module was removed on 2026-01-11 (commit `7172275`). However, the term "facade" persists throughout the codebase in three contexts:

1. **Legitimate usage:** API modules that are genuinely thin facades over internal implementations (intentional, correct terminology)
2. **Stale documentation:** Planning docs that reference the now-removed `api.facade` module
3. **Orphaned artifacts:** Stale `.pyc` cache files and a confusingly-named test file

---

## 1. Legitimate "Facade" Usage (No Action Needed)

These references use "facade" correctly to describe the API layer's role as a thin wrapper:

| File | Line | Context |
|------|------|---------|
| `src/bittr_tess_vetter/api/mlx.py` | 3, 9 | "thin facade over compute.mlx_detection" |
| `src/bittr_tess_vetter/api/lc_only.py` | 4, 79, 85 | "converting between facade types and internal types" |
| `src/bittr_tess_vetter/api/pixel.py` | 4, 60, 66 | "converting between facade types and internal types" |
| `src/bittr_tess_vetter/api/tpf.py` | 1 | "TPF cache API facade (host-facing)" |
| `src/bittr_tess_vetter/api/catalog.py` | 4, 46, 52, 106, 109 | facade type conversion |
| `src/bittr_tess_vetter/api/exovetter.py` | 4, 36, 42, 56, 59, 147, 211 | facade type conversion |
| `src/bittr_tess_vetter/api/transit_primitives.py` | 4, 67 | facade type conversion |
| `src/bittr_tess_vetter/api/types.py` | 3 | "user-facing types for the API facade" |
| `src/bittr_tess_vetter/api/__init__.py` | 208, 377, 534, 553, 570, 722 | "facade (host-facing)" comments |
| Various `api/*.py` files | docstrings | "stable facade", "host-facing facade" |
| `README.md` | 85 | "stable host-facing facade (recommended import surface)" |

**Recommendation:** Keep these. The terminology is accurate and helpful.

---

## 2. Stale Documentation (Cleanup Needed)

### High Priority: `working_docs/api_roadmap/README.md`

This file describes a **planned but now-abandoned** `api.facade` module. The entire document is outdated:

| Line | Problematic Content |
|------|---------------------|
| 11 | `bittr_tess_vetter.api.facade (or similarly named module)` |
| 14 | "the facade is the only thing recommended in docs and examples" |
| 19 | "Phase 0 - Decide the Facade Surface (design)" |
| 35 | "`api/facade.py` exports only this curated surface" |
| 55-65 | "Phase 2 - Introduce `api.facade`" with detailed implementation plan |
| 70 | "import from `bittr_tess_vetter.api.facade`" |
| 95 | "Keep `api.facade` stable" |

**Recommendation:** Either:
- Delete this file entirely (the roadmap was executed differently), or
- Update it to reflect actual decisions (Phase 1 aliases done; `api.facade` rejected in favor of full surface)

### Medium Priority: `working_docs/api/api_facade_spec.md`

This is an **outdated design spec** for the facade approach. References:
- Line 87: "Implement `bittr_tess_vetter.api` facade"
- Throughout: Describes a narrower curated API that wasn't implemented

**Recommendation:** Archive or delete. The actual implementation uses the full `api/__init__.py` surface.

### Low Priority: Already-Noted Issues

`working_docs/architecture_review/followup_2026-01-11.md` (line 33) already documents:
> "The recently removed `api.facade` left behind a test file name that implies a facade still exists."

This is a meta-note about cleanup needed, not a problem itself.

---

## 3. Orphaned Artifacts (Cleanup Needed)

### Stale `.pyc` Cache Files

```
src/bittr_tess_vetter/api/__pycache__/facade.cpython-312.pyc
tests/test_api/__pycache__/test_facade_api.cpython-312-pytest-9.0.2.pyc
```

**Recommendation:** Run `git clean -fdx __pycache__` or delete manually. These are harmless but confusing.

### Confusingly-Named Test File

**File:** `tests/test_api/test_triceratops_cache_facade.py`

This file tests `api/triceratops_cache.py` (which still exists and uses "facade" legitimately). The test function is named `test_triceratops_cache_facade_exports()`.

**Assessment:** This is actually fine - the `triceratops_cache` module docstring says "TRICERATOPS cache + helper API facade (host-facing)". The naming is consistent.

### Renamed Test: `test_api_aliases.py`

The commit `7172275` modified `tests/test_api/test_facade_api.py` but the file no longer exists on disk. It appears to have been renamed to `test_api_aliases.py` (both files have identical structure).

However, `test_api_aliases.py` still contains:
```python
def test_facade_imports() -> None:
    # Canonical surface is `bittr_tess_vetter.api` (full surface is first-class).
```

**Recommendation:** Rename `test_facade_imports()` to `test_api_surface_imports()` for clarity.

---

## 4. Git History Context

### What Was the Facade?

From commit `f36440a` ("feat: add curated API facade and short aliases"):
- `api/facade.py` was introduced as a curated subset of the full API surface
- Intent: provide a smaller, more discoverable entry point for new users

### Why Was It Removed?

From commit `7172275` ("chore: remove api facade module"):
- The team decided the full `api/__init__.py` surface should be first-class
- The facade created confusion about which imports were "correct"
- Maintaining two parallel surfaces was unnecessary overhead

The follow-up doc (`followup_2026-01-11.md`) confirms:
> "Single first-class public surface: `bittr_tess_vetter.api`"

---

## 5. Cleanup Recommendations

### Immediate (P0)

| Action | File | Details |
|--------|------|---------|
| Delete or update | `working_docs/api_roadmap/README.md` | References `api.facade` throughout |
| Delete or archive | `working_docs/api/api_facade_spec.md` | Outdated design spec |
| Rename function | `tests/test_api/test_api_aliases.py:20` | `test_facade_imports` -> `test_api_surface_imports` |
| Clean cache | `*/__pycache__/facade*.pyc` | Stale bytecode |

### Optional (P1)

| Action | File | Details |
|--------|------|---------|
| Consider | `tests/test_api/test_pixel_prf_api.py:9` | Function named `test_pixel_prf_facade_imports` |
| Consider | `tests/test_api/test_triceratops_cache_facade.py` | File uses "facade" in name (but module uses term too) |

---

## 6. Summary Statistics

| Category | Count | Action |
|----------|-------|--------|
| Legitimate "facade" usage in code | ~30 references | Keep |
| Stale documentation | 2 files | Delete/update |
| Stale cache files | 2 files | Clean |
| Confusing test names | 1-2 functions | Rename |
