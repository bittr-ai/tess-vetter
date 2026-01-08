# bittr-tess-vetter API Surface Additions (v0.2) — Spec

**Date:** 2026-01-08  
**Status:** DRAFT  
**Scope:** `bittr_tess_vetter.api` façade and submodules (`bittr_tess_vetter.api.*`)

## 0) Context / Current State (as implemented)

The repository already contains a fairly broad `bittr_tess_vetter.api` package:

- Tiered orchestrator: `bittr_tess_vetter.api.vet.vet_candidate()`
- LC-only wrappers: `bittr_tess_vetter.api.lc_only` (V01–V05)
- Pixel wrappers: `bittr_tess_vetter.api.pixel` (V08–V10; non-WCS)
- Domain/prefilter/systematics primitives under `bittr_tess_vetter.api.*`
- WCS-aware modules *exist* under `bittr_tess_vetter.api`:
  - `bittr_tess_vetter.api.wcs_localization`
  - `bittr_tess_vetter.api.wcs_utils`
  - `bittr_tess_vetter.api.aperture_family`
  - `bittr_tess_vetter.api.localization`
  - `bittr_tess_vetter.api.report`
  but are **not re-exported** from `bittr_tess_vetter.api` root (`api/__init__.py`).

Separately, the core check implementations increasingly support **metrics-only mode**
(`passed=None`, `details["_metrics_only"]=True`) so that host apps (e.g. `astro-arc-tess`)
can apply guardrails/policy externally. The public `CheckResult` dataclass now reflects
this with `passed: bool | None`.

## 1) Goals (v0.2)

1. **Make `bittr_tess_vetter.api` a complete “supported import surface”** for host apps:
   - no deep imports from `bittr_tess_vetter.pixel.*` / `bittr_tess_vetter.validation.*`
   - expose WCS-aware and pixel-report utilities on the supported surface
2. **Standardize metrics-only semantics** across the API (`passed=None`, metrics in `details`).
3. **Provide aggregation helpers that are safe with `passed=None`** (unknowns).

## 2) Non-goals (v0.2)

- No guardrail policy or `bittr_validity` integration inside this repo.
- No MCP/tool schemas; no persistence/caching; no network by default.
- No requirement to change underlying algorithms.

## 3) Proposed “Tools” to Add to the Public API Surface

This section defines what should be callable/importable by host applications as
part of the supported API surface.

### 3.1 Re-export WCS-aware pixel tools at the API root

Expose these symbols from `bittr_tess_vetter.api` (root `__init__.py`), not only from
submodules, so host apps can depend on stable import paths:

- From `bittr_tess_vetter.api.wcs_localization` (delegates to pixel WCS localization):
  - `localize_transit_source(...)`
  - `LocalizationVerdict` (enum)
  - `ReferenceSource` (type/model)
  - `LocalizationResult` (type/model)
- From `bittr_tess_vetter.api.aperture_family`:
  - `compute_aperture_family_depth_curve(...)`
  - `ApertureFamilyResult`
  - `DEFAULT_RADII_PX`
- From `bittr_tess_vetter.api.wcs_utils`:
  - `pixel_to_world`, `world_to_pixel` (+ batch variants)
  - `extract_wcs_from_header`, `wcs_sanity_check`, `compute_pixel_scale`
- From `bittr_tess_vetter.api.localization`:
  - `compute_localization_diagnostics(...)` (difference images + summary)
  - `LocalizationDiagnostics`, `LocalizationImages`, `TransitParams`
- From `bittr_tess_vetter.api.report`:
  - `generate_pixel_vet_report(...)`
  - `PixelVetReport`

Rationale: `astro-arc-tess` already consumes WCS-aware localization and aperture-family
diagnostics. Keeping those behind deep imports defeats the purpose of the façade.

### 3.2 Add aggregation helpers to `bittr_tess_vetter.api`

Host apps often want a quick “rollup” of check results even when operating in
metrics-only mode. Provide a supported aggregation utility that:

- treats `passed=None` as **unknown** (not fail)
- returns WARN when unknowns exist (unless caller opts out)

**Proposed API:**

```python
from dataclasses import dataclass
from enum import Enum
from typing import Any

class UnknownPolicy(str, Enum):
    WARN = "WARN"      # default: any unknown check -> WARN verdict
    IGNORE = "IGNORE"  # unknown checks do not affect verdict/disposition

@dataclass(frozen=True)
class AggregateResult:
    verdict: str          # "PASS" | "WARN" | "REJECT"
    disposition: str      # "PLANET" | "UNCERTAIN" | "FALSE_POSITIVE"
    n_passed: int
    n_failed: int
    n_unknown: int
    failed_ids: list[str]
    unknown_ids: list[str]
    summary: str

def aggregate_checks(
    checks: list["CheckResult"],
    *,
    unknown_policy: UnknownPolicy = UnknownPolicy.WARN,
) -> AggregateResult: ...
```

Notes:
- This should be a thin wrapper around `bittr_tess_vetter.validation.base` logic.
- Do **not** force callers to import internal `VetterCheckResult`/`ValidationResult`.

### 3.3 Policy semantics: metrics-only only

This repository intentionally does **not** implement a legacy boolean PASS/FAIL policy
layer. Checks are metrics-only so that host apps (e.g. `astro-arc-tess`) can apply
guardrails/policy externally.

If a `policy_mode` parameter exists on wrappers/orchestrators, it must:
- default to `"metrics_only"`
- reject any other value (raise `ValueError`)

**Proposed API (examples):**

```python
def vet_lc_only(
    lc: LightCurve,
    ephemeris: Ephemeris,
    *,
    stellar: StellarParams | None = None,
    enabled: set[str] | None = None,
    config: dict[str, dict[str, Any]] | None = None,
    policy_mode: str = "metrics_only",
) -> list[CheckResult]: ...

def vet_candidate(
    lc: LightCurve,
    candidate: Candidate,
    *,
    policy_mode: str = "metrics_only",
    config: dict[str, dict[str, Any]] | None = None,
    ...
) -> VettingBundleResult: ...
```

Rationale:
- `astro-arc-tess` wants metrics-only results to feed guardrails.
- This library stays focused on computing evidence/metrics, not policy.

### 3.4 Add evidence-friendly conversion helpers (optional but recommended)

Host apps frequently convert `CheckResult` into evidence-like structures
(`id/title/metrics/flags`). Provide a helper to avoid repeated boilerplate and
ensure consistent naming:

```python
def checks_to_evidence_items(checks: list[CheckResult]) -> list[dict[str, Any]]: ...
```

Constraints:
- output must be JSON-serializable
- no “validity/guardrails” claims; keep it informational

## 4) Compatibility / Versioning

### 4.1 `CheckResult.passed` is nullable

In v0.2, the public contract is:
- `passed: bool | None` on `bittr_tess_vetter.api.types.CheckResult`
- callers should check `details.get("_metrics_only")` and/or `passed is None`

### 4.2 Stable import policy

Supported import surface (v0.2):
- `from bittr_tess_vetter import api` and `from bittr_tess_vetter.api import ...`
- `from bittr_tess_vetter.api.<submodule> import ...` for the listed façade modules

Not supported:
- importing from `bittr_tess_vetter.validation.*`, `bittr_tess_vetter.pixel.*`, etc.

## 5) Acceptance Criteria

1. All proposed tools are importable from `bittr_tess_vetter.api` root.
2. `vet_candidate()` and `vet_lc_only()` accept `policy_mode` and `config`, and thread them through.
3. Aggregation helper returns consistent outputs and never treats unknowns as failures by default.
4. Unit tests cover:
   - metrics-only behavior (`passed is None`, `_metrics_only=True`)
   - aggregation behavior with unknowns

## 6) Example Usage (host app)

```python
from bittr_tess_vetter.api import (
    LightCurve, Ephemeris, Candidate,
    vet_candidate, aggregate_checks,
    localize_transit_source, compute_aperture_family_depth_curve,
)

lc = LightCurve(time=t, flux=f, flux_err=fe)
eph = Ephemeris(period_days=5.0, t0_btjd=1325.0, duration_hours=3.0)
candidate = Candidate(ephemeris=eph, depth_ppm=1500)

bundle = vet_candidate(lc, candidate, policy_mode="metrics_only")
agg = aggregate_checks(bundle.results)
```
