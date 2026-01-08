# bittr-tess-vetter API Façade (v0.1) — Spec

## Purpose
Define a small, stable, researcher-friendly public API for `bittr-tess-vetter` that:
- Makes it easy to build custom vetting pipelines (pick-and-choose primitives).
- Keeps `astro-arc-tess` integration simple (shims call façade, not deep modules).
- Preserves internal refactor freedom while migration is ongoing.

This is intentionally **thin** at first: mostly re-exports + typed dataclasses + a couple of orchestrators.

## Scope (v0.1)
### In scope
- A new module: `bittr_tess_vetter.api` (and small submodules as needed).
- Stable dataclasses for common inputs/outputs.
- Stable function names for the already-migrated “LC-only” checks and odd/even primitive.
- A small “LC-only orchestrator” that composes LC-only checks into a list of results.
- A citations-in-code convention for every check/orchestrator.

### Not in scope (yet)
- Any I/O (no downloading, no caching, no TESS/TPF fetching).
- Pixel-level orchestration (TPF localization, aperture family) beyond re-exporting existing primitives.
- TLS / `wotan` detrending integration decisions.
- `astro-arc-tess` evidence/guardrails/disposition logic (host-app responsibility).

## Design Principles
- **Pure functions by default**: accept arrays/dataclasses; return dataclasses.
- **Typed units**: fields and docstrings state units (days vs hours, fractional depth vs ppm).
- **Stable imports**: users should import from `bittr_tess_vetter.api` (not deep modules).
- **Citations in code**: every check has explicit references (paper/software) to distinguish standard vs novel parts.
- **Provisional stability**: `api` is “stable-ish”; anything else may move.

## Public API Surface
### Package layout
- `bittr_tess_vetter/api/__init__.py`
- `bittr_tess_vetter/api/types.py`
- `bittr_tess_vetter/api/lc_only.py`
- `bittr_tess_vetter/api/transit_primitives.py`

### Types (`bittr_tess_vetter.api.types`)
Proposed dataclasses (names final, fields stable once released):
- `Ephemeris(period_days: float, t0_btjd: float, duration_hours: float)`
- `LightCurve(time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray | None = None, quality: np.ndarray | None = None, valid_mask: np.ndarray | None = None)`
  - Must accept host-app `LightCurveData` easily (conversion helper ok).
- `StellarParams(teff_k: float | None, logg: float | None, radius_rsun: float | None, mass_msun: float | None, ...)` (thin wrapper or alias to existing domain type)
- `CheckResult(id: str, name: str, passed: bool, confidence: float, details: dict[str, Any])`
  - Keep `details` JSON-serializable.

### Transit primitives (`bittr_tess_vetter.api.transit_primitives`)
- `odd_even_result(time, flux, flux_err, ephemeris: Ephemeris, *, relative_threshold_percent: float = 10.0) -> OddEvenResult`
  - Thin wrapper around `bittr_tess_vetter.transit.vetting.compute_odd_even_result`.

### LC-only checks (`bittr_tess_vetter.api.lc_only`)
Stable wrappers (thin, mostly re-export):
- `odd_even_depth(lc: LightCurve, ephemeris: Ephemeris) -> CheckResult`  (V01-style)
- `secondary_eclipse(lc: LightCurve, ephemeris: Ephemeris) -> CheckResult` (V02-style)
- `duration_consistency(ephemeris: Ephemeris, stellar: StellarParams | None) -> CheckResult` (V03-style)
- `depth_stability(lc: LightCurve, ephemeris: Ephemeris) -> CheckResult` (V04-style)
- `v_shape(lc: LightCurve, ephemeris: Ephemeris) -> CheckResult` (V05-style)

### LC-only orchestrator (`bittr_tess_vetter.api.lc_only`)
- `vet_lc_only(lc: LightCurve, ephemeris: Ephemeris, *, stellar: StellarParams | None = None, enabled: set[str] | None = None) -> list[CheckResult]`
  - Default: returns V01–V05 in canonical order.
  - `enabled` can filter by check id (e.g. `{"V01","V03"}`).

## Citations-in-Code Convention
Each public check function and orchestrator must include:
- Docstring section `References:` with:
  - Primary paper(s) for the method (ADS bibcode / DOI / arXiv preferred).
  - Software citations where relevant (e.g., exovetter, TRICERATOPS, wotan).
- A machine-readable constant:
  - `REFERENCES = [{"id": "...", "type": "doi|arxiv|ads|url", "note": "..."}]`

Goal: readers can quickly identify what is standard and what is novel just by opening the module.

## Stability & Versioning
- `bittr_tess_vetter.api` is the only “supported import surface” for external users.
- Backwards-incompatible changes:
  - Require a minor version bump at minimum and a changelog entry.
- `bittr_tess_vetter.validation.*`, `compute.*`, etc. remain internal during migration.

## Integration Plan (astro-arc-tess)
1. Keep current `astro-arc-tess` shims, but update them to call `bittr_tess_vetter.api.*` once it exists.
2. Migrate more logic into `bittr-tess-vetter` as-is.
3. Once migration stabilizes, remove shims by updating `astro-arc-tess` call sites to import from the façade directly (or keep shims as deprecated re-exports if desired).

## Deliverables (Agent Workstream)
- Implement `bittr_tess_vetter.api` façade and type helpers.
- Add minimal unit tests in `bittr-tess-vetter` verifying:
  - Types validate/normalize shapes.
  - Each façade wrapper calls underlying primitive and returns stable `CheckResult`.
  - `vet_lc_only()` ordering and filtering.
- Add one short example in `bittr-tess-vetter` `README` demonstrating:
  - Load arrays (user-provided), create `LightCurve` + `Ephemeris`, call `vet_lc_only`.

