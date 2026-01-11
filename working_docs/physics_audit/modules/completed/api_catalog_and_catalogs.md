# Module Review: `api/catalog.py` + `api/catalogs.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is high leverage

Catalog checks are “context evidence”: they don’t measure the transit directly, but they can strongly influence downstream interpretation (known EB nearby, pre-existing TOI dispositions, known planets, crowding priors). The biggest correctness risks here are:
- coordinate units (degrees vs radians, arcsec vs pixels),
- skip/error semantics when `network=False` or metadata is missing,
- accidental mixing of “policy” (pass/fail) into metrics-only evidence.

## File: `api/catalog.py` (V06–V07 wrappers)

### Inputs / units

- `nearby_eb_search`:
  - `ra_deg`, `dec_deg`: degrees
  - `search_radius_arcsec`: arcseconds (default 42″ = 2 TESS pixels)
  - `candidate_period_days`: days (passed through from ephemeris)
- `exofop_disposition`:
  - `tic_id`: int
  - optional `toi`: float (e.g., 123.01)

### Network gating semantics

- If `network=False`, V06/V07 return a **skipped** `CheckResult`:
  - `passed=None`, `confidence=0.0`
  - `details.status="skipped"`, `reason="network_disabled"`
- `vet_catalog(...)` also guards required metadata regardless of network:
  - V06 requires `ra_deg` and `dec_deg` (otherwise `reason="missing_metadata"` with `missing=[...]`)
  - V07 requires `tic_id` (otherwise `reason="missing_metadata"`)

### Metrics-only semantics

- Underlying implementations (`validation.checks_catalog`) return `passed=None` with `_metrics_only=True`.
- The API wrapper preserves this and does not impose pass/fail policy.

## File: `api/catalogs.py` (host-facing catalog clients)

- This module is a re-export surface over `bittr_tess_vetter.catalogs` and ExoFOP helpers.
- It introduces no physics logic; it exists to give hosts a stable import surface.

## Cross-references

- Underlying check implementations: `working_docs/physics_audit/modules/completed/validation_checks_pixel_and_catalog.md`
- Catalog and crossmatch primitives: `working_docs/physics_audit/modules/completed/catalogs_spatial_and_crossmatch.md`

## Fixes / follow-ups

No physics correctness issues identified in these API wrappers.

