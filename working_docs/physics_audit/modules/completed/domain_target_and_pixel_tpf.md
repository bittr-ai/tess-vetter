# Module Review: `domain/target.py` + `pixel/tpf.py` + `pixel/tpf_fits.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is high leverage

- `domain/target.py` defines the canonical host-star parameter contract (units and derived quantities like stellar density). Many guardrails and derived planet quantities depend on this being correct.
- `pixel/tpf.py` and `pixel/tpf_fits.py` define pixel data conventions (shape, time base, quality semantics, WCS preservation). Pixel-level vetting (localization, centroid shifts, blend indicators) is only as reliable as these data conventions.

## File: `domain/target.py`

### Model: `StellarParameters`

- Units + conventions:
  - `teff`: K (non-negative)
  - `logg`: log10(cm/s^2) (no sign restriction)
  - `radius`: R_sun (non-negative)
  - `mass`: M_sun (non-negative)
  - `luminosity`: L_sun (non-negative)
  - `contamination`: unitless “contratio” (non-negative; can exceed 1 per TIC edge cases)
  - `metallicity`: [Fe/H]
- Derived quantity:
  - `stellar_density_solar()` returns `mass / radius^3` in solar density units if available and `radius > 0`.
- Notes / minor code hygiene:
  - There is a formatting issue: `from pydantic import Field` is imported after the `FrozenModel` class definition with no blank line; this is style-only, not a physics bug.

### Model: `Target`

- Units + conventions:
  - `ra/dec`: degrees (ICRS implied by TIC conventions)
  - `pmra/pmdec`: mas/yr
  - `distance_pc`: parsecs
  - `gaia_dr3_id`: Gaia source_id (int)
- Constructor: `from_tic_response`
  - Maps TIC keys: `Teff`, `logg`, `rad`, `mass`, `Tmag`, `contratio`, `lum`, `MH`, `ra`, `dec`, `pmRA`, `pmDEC`, `d`, `GAIA`, `TWOMASS`
  - This is consistent with earlier audit notes in `api_io_and_target.md`.

## File: `pixel/tpf.py`

### Types: `TPFRef`, `TPFData`

- Conventions:
  - `TPFData.time`: documented as BTJD days; stored as array (shape `(n_cadences,)`)
  - `TPFData.flux`: flux cube shape `(n_cadences, n_rows, n_cols)` (tests use `(100, 11, 11)`)
- `TPFRef` schema:
  - `tpf:<tic_id>:<sector>:<camera>:<ccd>`
  - validates camera/ccd are in `1..4` and positive TIC/sector.

### Cache: `TPFCache`

- Storage:
  - uses `np.savez` to persist arrays (preserves dtype/shape).
- Physics risk:
  - This cache is “data plumbing”; the main risk is silently swapping axes or time base. The documented convention is cadence axis 0 and BTJD time; tests cover round-trip behavior.

### Tests

- Existing tests:
  - `tests/pixel/test_tpf.py` provides extensive coverage of parsing/validation, cache round-trip and error semantics.

## File: `pixel/tpf_fits.py`

### Types: `TPFFitsRef`, `TPFFitsData`

- Conventions:
  - `TPFFitsData.time`: documented as BTJD timestamps (days).
  - `TPFFitsData.flux`: `(n_cadences, n_rows, n_cols)` float array.
  - `flux_err`: optional cube aligned with flux.
  - `quality`: per-cadence quality flags (int).
  - `aperture_mask`: `(n_rows, n_cols)` integer mask (SPOC aperture).
  - `wcs`: astropy `WCS` constructed from FITS header.
  - `author` is normalized to lowercase and restricted to a known set.

### Cache: `TPFFitsCache`

- Preserves:
  - FITS bytes (full WCS), and a JSON sidecar with selected header keys (`SIDECAR_HEADER_KEYS`) including time system keys and `TUNIT*` columns where present.
- Integrity:
  - WCS checksum is derived from `wcs.to_header_string()` hashed (best-effort).

### Tests

- Existing tests:
  - `tests/pixel/test_tpf_fits.py` covers ref parsing/validation, cache round-trip preserving dtype/shape/WCS, and sidecar creation.
  - `tests/pixel/test_wcs_localization.py` uses synthetic `TPFFitsData` to validate pixel-localization physics; this indirectly depends on correct conventions here.

## Fixes / follow-ups

No physics correctness issues identified in these modules.

Optional follow-ups:
- Style cleanup in `domain/target.py` import ordering (no behavior change).
- Consider explicitly documenting in `TPFData`/`TPFFitsData` whether times are expected to be BTJD=TDB (as in TESS products) vs generic “days”.

