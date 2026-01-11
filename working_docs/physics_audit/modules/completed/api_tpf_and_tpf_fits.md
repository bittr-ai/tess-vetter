# Module Review: `api/tpf.py` + `api/tpf_fits.py` (facades over `pixel/tpf*.py`)

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

TPF ingestion + representation sets the ground truth for all pixel-level diagnostics (difference images, centroid shifts,
localization). If time bases, pixel conventions, or masks are silently wrong at the data container layer, *every* pixel
check inherits the error.

## Scope

Public API wrappers (assigned):

- `src/bittr_tess_vetter/api/tpf.py`
- `src/bittr_tess_vetter/api/tpf_fits.py`

Underlying implementations re-exported by those facades (audited here for physics conventions):

- `src/bittr_tess_vetter/pixel/tpf.py`
- `src/bittr_tess_vetter/pixel/tpf_fits.py`

## Audit checklist (minimum)

### Coordinate conventions (row/col vs x/y, pixel centers, WCS)

- [x] Pixel data arrays are consistently shaped as `(n_cadences, n_rows, n_cols)` throughout `pixel/tpf.py` and
      `pixel/tpf_fits.py`.
- [x] `TPFFitsCache.put()` writes the `FLUX` table column with `dim="(n_cols,n_rows)"` and flattens the incoming
      `(n_rows,n_cols)` image in row-major order; `TPFFitsCache.get()` rehydrates the same convention so in-memory arrays
      remain `(row, col)`-indexed.
- [x] WCS is stored as an `astropy.wcs.WCS` created from the data HDU header (`WCS(data_hdu.header)`).
- [x] Cross-module convention alignment:
  - `pixel/wcs_utils.py` explicitly converts between `(row, col)` and astropy’s `(x, y)=(col, row)`.
  - `pixel/wcs_localization.py` consumes `TPFFitsData.flux` as `(cadence,row,col)` and returns a centroid in `(row,col)`
    coordinates (then uses `wcs_utils` for world conversions).

Notes / guardrails:
- This layer does not define a global `origin` for WCS transforms; downstream callers must continue to pass `origin`
  explicitly (as already done in `pixel/wcs_utils.py`).

### Time base (BTJD vs BJD_TDB vs indices)

- [x] `TPFData.time` and `TPFFitsData.time` are documented as BTJD days (“TIME” column from the TPF FITS).
- [x] No conversion is applied in cache I/O; this module is a faithful transport layer for time arrays.

Follow-up recommendation (documentation-only or sidecar metadata):
- Include FITS header time-system keys (e.g., `TIMESYS`, `TIMEUNIT`, `BJDREFI/BJDREFF` where present) in the preserved
  header subset / sidecar so BTJD assumptions are externally auditable.

### Flux units (e-/s vs counts; background subtraction)

- [x] `TPFFitsCache.get()` reads `FLUX` (and optionally `FLUX_ERR`) as float64 without unit conversion.
- [x] No background subtraction is applied here (consistent with “cache/transport” responsibility).

Known ambiguity:
- Flux units can differ by author/product; this layer does not preserve `TUNIT*` metadata. Downstream physics code should
  continue to treat flux values as relative measurements unless units are explicitly asserted by the caller.

### Quality flags (masked/ignored; preservation)

- [x] `TPFFitsData.quality` is preserved from the FITS `QUALITY` column (int32) and round-trips via `TPFFitsCache.put()`.
- [x] Downstream cadence filtering is centralized in `pixel/cadence_mask.default_cadence_mask(time, flux, quality)` and
  uses `quality==0` plus finite checks (see `api_wcs_localization.md`, `api_aperture_family.md`).
- [x] The simpler `TPFData` (non-FITS cache) does not carry `quality` flags; consumers must not assume quality masking is
  available when only `TPFData` is provided.

### Aperture masks (indexing; alignment with flux cube)

- [x] `TPFFitsData.aperture_mask` is a 2D array shaped `(n_rows,n_cols)` aligned to `flux.shape[1:]`.
- [x] If no APERTURE extension is present, `TPFFitsCache.get()` uses a default “all ones” mask with matching shape.

Known ambiguity:
- For SPOC products the APERTURE extension is a bitmask (not a boolean). This layer treats `mask > 0` as “in aperture” for
  the `aperture_npixels` metric and otherwise preserves the raw integer mask for downstream interpretation.

## Review template highlights (key functions)

## Function

- Name: `TPFFitsCache.get`
- Location: `src/bittr_tess_vetter/pixel/tpf_fits.py`
- Public API? yes (via `bittr_tess_vetter.api.tpf_fits`)
- Called early by agents? yes (TPF FITS/WCS is used by localization + aperture-family checks)

## Inputs / outputs

- Units + conventions:
  - `time`: days, assumed BTJD (transported directly from FITS “TIME” column)
  - `flux`, `flux_err`: numeric arrays, shape `(n_cadences,n_rows,n_cols)` (units depend on author/product)
  - `quality`: integer flags (TESS-style); downstream assumes `0` means “good”
  - `aperture_mask`: 2D integer mask, preserved if present
  - `wcs`: astropy WCS extracted from the data HDU header
- Output semantics:
  - Returns `None` on cache miss or corruption (warnings logged)

## Numerical stability

- NaN handling: this layer does not NaN-clean; downstream cadence/pixel filters handle finiteness.
- Empty-mask handling: not applicable at this layer.

## Tests

- Existing tests covering this:
  - `tests/pixel/test_tpf_fits.py` (ref parsing, cache round-trip, sidecar creation, WCS checksum)
  - `tests/pixel/test_tpf.py` (npz cache round-trip for array-only TPF)
  - Downstream integration tests indirectly validate convention compatibility:
    - `tests/pixel/test_wcs_localization.py`
    - `tests/pixel/test_aperture_family.py`

## Fixes / changes (if any)

- Proposed fix: (optional) expand `SIDECAR_HEADER_KEYS` to include time-system/unit keywords so BTJD assumptions can be
  verified from cache artifacts alone.
- Backwards-compat impact: additive sidecar metadata only (safe).
