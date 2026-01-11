# Module Review: `validation/checks_pixel.py` + `validation/checks_catalog.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

These modules generate pixel- and catalog-based evidence that strongly affects host interpretation (blend risk, on/off-target priors, “known EB nearby” flags). They also establish missing-data semantics (`error` vs `skipped` vs “low-confidence ok”) that downstream policy uses.

## Scope (functions)

Pixel (V08–V10 wrappers + helpers):
- `check_centroid_shift_with_tpf` (V08 wrapper)
- `compute_pixel_level_depths_ppm` (per-pixel depth map)
- `compute_pixel_depth_map_metrics` (+ `PixelDepthMapMetrics`)
- `check_pixel_level_lc_with_tpf` (V09 wrapper)
- `check_aperture_dependence_with_tpf` (V10 wrapper)

Catalog (V06–V07 raw network lookups):
- `run_nearby_eb_search` (V06)
- `run_exofop_toi_lookup` (V07)

## Cross-references

- `working_docs/physics_audit/modules/completed/api_centroid_shift.md` (centroid units + cadence hygiene patterns)
- `working_docs/physics_audit/modules/completed/api_wcs_localization.md` (difference-image sign + cadence hygiene)
- `working_docs/physics_audit/modules/completed/api_aperture_family.md` (depth definition + missing-data behavior)
- `working_docs/physics_audit/modules/completed/api_aperture_dependence.md` (aperture units + missing-data behavior)
- `working_docs/physics_audit/modules/completed/validation_lc_checks.md` (metrics-only conventions + `status/warnings/confidence` patterns)

## Audit checklist

### Units + conventions

- [x] Pixel wrappers use candidate ephemeris with: `period` (days), `t0` (BTJD days), `duration_hours` (hours).
- [x] `compute_pixel_level_depths_ppm` depth definition is fractional dimming and converts to ppm:
  - `depth_frac = (out_median - in_median) / out_median`
  - `depth_ppm = depth_frac * 1e6`
- [x] Catalog inputs: `(ra_deg, dec_deg)` in degrees and `search_radius_arcsec` in arcsec with explicit conversion to degrees for TAP.
- [x] Results are metrics-only (`passed=None`) and mark `details["_metrics_only"]=True`.

### Missing-data policy / “deferred vs skipped vs error”

- [x] V09/V10 now return `status="insufficient_data"` (metrics-only) for predictable in/out mask failure (`ValueError`) instead of raising or returning `status="error"`.
- [x] V06/V07 treat network/HTTP errors as `status="error"` with `confidence=0.0`.
- [x] “ok path” results now include `details["status"]="ok"` for V08/V09/V10.

### Consistency with `validation/lc_checks.py` semantics

- [x] `validation/lc_checks.py` runs V08–V10 via these wrappers when TPF inputs are present.
- [x] `validation/lc_checks.py` still stubs V06–V07 as deferred; public API (`api/catalog.py`) runs V06–V07 via `validation/checks_catalog.py`.
- [ ] Follow-up (optional): decide whether to keep `validation/lc_checks.py` strictly offline (stubs) or plumb a `network=` flag through to share the same implementation.

### Crossmatch / geometry math

- [x] V06 parses VizieR’s VOTable XML (with a legacy fallback) and returns per-match `ra_deg/dec_deg` and `sep_arcsec` (simple spherical separation; no proper motion).
- [ ] Documented assumption: coordinates are not propagated for proper motion/epoch (likely acceptable at ~42″ search radius, but it is still an assumption).

## Function notes

### `compute_pixel_level_depths_ppm`

- Cadence hygiene: drops non-finite times and all-NaN TPF cadences before building masks (prevents silent “confidence inflation” from unusable cadences).
- Throws `ValueError` when in/oot masks select zero cadences; V09 wrapper maps this to `status="insufficient_data"`.

### `run_nearby_eb_search` (V06)

- Parses returned VOTable (XML) when possible; uses a fallback split parser for non-XML responses to preserve behavior.
- Returns raw matches with basic alias deltas (`1x/2x/0.5x`) when a candidate period is provided.

### `run_exofop_toi_lookup` (V07)

- Supports optional `toi` filter to disambiguate multiple TOI rows for a single TIC.

## Tests

- Added:
  - `tests/validation/test_checks_pixel_wrappers.py`
  - `tests/validation/test_checks_catalog.py`

