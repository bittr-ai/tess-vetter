# Follow-on Spec: Finish v4 Cleanup (Remove Stubs + Remove Default warnings.warn)

**Date:** 2026-01-14  
**Status:** Draft  
**Scope:** Small-but-important cleanup to fully satisfy the remaining items from `architecture_review_4/README.md` after the v0.1.0 pipeline refactor branch landed.

---

## 0) Goals

1) **Eliminate stub/deferred check implementations** that are still present and discoverable via imports.  
2) **Stop emitting `warnings.warn` by default** from API entry points; represent skips/errors structurally in results.  
3) Keep behavior deterministic and **maintain the new pipeline + schema contract**.

---

## 1) Background / Current State

After the v0.1.0 pipeline refactor:
- Checks V01–V12 are executed via wrapped implementations and registry/pipeline.
- A stable result schema exists (`CheckResult` with `status/metrics/flags/...`).

Remaining legacy artifacts:
- `src/bittr_tess_vetter/validation/lc_checks.py` still contains “deferred/stub” implementations for V06–V10 and docstrings stating “deferred to v2”.
- `warnings.warn(...)` is still used in:
  - `src/bittr_tess_vetter/api/vet.py`
  - `src/bittr_tess_vetter/api/lc_only.py`
  - `src/bittr_tess_vetter/api/experimental.py`

These are footguns for researchers:
- Stubs appear callable/importable and can silently be used by mistake.
- Warnings are noisy in notebooks and batch pipelines, and are not machine-actionable.

---

## 2) Proposed Changes

### 2.1 Remove or quarantine `validation/lc_checks.py` stubs (P0.3 completion)

Target: make it impossible (or at least obviously discouraged) to reach stubbed “deferred” checks via normal imports.

**Approach A (preferred): delete stubs**
- In `validation/lc_checks.py`, remove:
  - `check_nearby_eb_search` (V06 stub)
  - `check_known_fp_match` (V07 stub)
  - V08–V10 stub implementations
  - Any “deferred to v2” orchestration paths that reference these
- Keep LC-only checks V01–V05 only.

**Approach B (acceptable): quarantine stubs**
- Move the stubbed content into `validation/_legacy_stubs.py` (leading underscore).
- Do not re-export it in `validation/__init__.py` or `api.*`.
- Add a loud module docstring: “legacy; not part of v0.1.0 pipeline”.

Acceptance criteria:
- `rg -n \"status\":\\s*\"deferred\" src/bittr_tess_vetter/validation` returns no matches **outside** explicit legacy module (if Approach B).
- No public-facing `validation.__all__` symbols refer to stub/deferred checks.

### 2.2 Stop default `warnings.warn` in API surface (P1.3 completion)

Replace warning emission with structured information:

- For skips due to missing inputs:
  - return `CheckResult(status=\"skipped\", flags=[\"SKIPPED:NETWORK_DISABLED\"])`, etc.
  - add a human-readable entry to `VettingBundleResult.warnings`

Introduce an opt-in knob:
- `emit_warnings: bool = False` on:
  - `vet_candidate`
  - `vet_lc_only` (if it currently warns)
  - `VettingPipeline` (constructor default false)

Behavior:
- If `emit_warnings=False` (default): do not call `warnings.warn`.
- If `emit_warnings=True`: mirror `bundle.warnings` entries via Python warnings (UserWarning).

Acceptance criteria:
- `rg -n \"warnings\\.warn\\(\" src/bittr_tess_vetter/api` returns either zero matches or only within explicitly documented experimental APIs.
- Running a typical `vet_candidate(..., network=False, tpf=None)` produces structured skip results and populates `bundle.warnings` without emitting Python warnings by default.

### 2.3 Clean up wrapper “deferred/stub” detection paths

The wrapped catalog/pixel/LC wrappers contain code paths like:
- `if details.get(\"deferred\") or details.get(\"stub\")`

Once stubs are removed, these paths should be removed or kept only for defensive parsing of legacy cached results.

Decision:
- If you expect no legacy-stub results to enter the pipeline anymore, delete this logic.
- If you want defensive robustness, keep but mark as “legacy normalization”, and translate it into:
  - `status=\"skipped\"`, flags like `SKIPPED:LEGACY_STUB_RESULT`

---

## 3) Files to Change (Expected)

- `src/bittr_tess_vetter/validation/lc_checks.py` (remove stub sections)
- `src/bittr_tess_vetter/validation/__init__.py` (ensure exports align with real checks)
- `src/bittr_tess_vetter/api/vet.py` (remove default warnings; add emit_warnings)
- `src/bittr_tess_vetter/api/lc_only.py` (same)
- `src/bittr_tess_vetter/api/experimental.py` (either remove warnings or explicitly justify)
- Potentially:
  - `src/bittr_tess_vetter/validation/checks_*_wrapped.py` (remove legacy-stub parsing)
  - docs: `docs/quickstart.rst` and/or API docs to mention `emit_warnings`

---

## 4) Tests

Add/update tests to lock in the new behavior:

1) **No stubs exported**
- Import from `bittr_tess_vetter.validation` and assert stub symbols are absent (or point to real impls).

2) **No warnings by default**
- Use `pytest.warns(None)` (or warnings capture) around `vet_candidate` in skip scenarios and assert no warnings emitted.
- Assert `bundle.warnings` contains user-actionable messages instead.

3) **emit_warnings=True emits**
- With `emit_warnings=True`, assert warnings are emitted and match bundle warnings.

---

## 5) Rollout

- Land as a small follow-up PR on top of the v0.1.0 pipeline refactor branch.
- After merge, re-run:
  - `uv run pytest`
  - `uv run ruff check .`
  - `uv run mypy src/bittr_tess_vetter`

