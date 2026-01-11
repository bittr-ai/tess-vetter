# Module Review: `api/io.py` + `api/target.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

These modules define the host-facing contract for:
- how light curves are discovered/downloaded/cached and quality-masked (`api/io.py`), and
- how targets + stellar parameters are represented (`api/target.py`).

Even though they are mostly wrappers, mistakes here (quality-mask defaults, time/flux conventions in returned payloads, parameter units) can silently poison downstream detection/vetting.

## Scope (exports)

### `src/bittr_tess_vetter/api/io.py`

Thin re-export layer over `bittr_tess_vetter.io`:
- `MASTClient`, `MASTClientError`
- `SearchResult`, `ResolvedTarget`, `NameResolutionError`, `TargetNotFoundError`
- `LightCurveNotFoundError`
- `SessionCache`, `PersistentCache`
- Quality mask contract: `QUALITY_FLAG_BITS`, `DEFAULT_QUALITY_MASK`
- Download progress contract: `DownloadProgress`, `DownloadPhase`, `ProgressCallback`

### `src/bittr_tess_vetter/api/target.py`

Thin re-export layer over `bittr_tess_vetter.domain.target`:
- `Target`
- `StellarParameters`

## Audit checklist (filled)

### Units + conventions

- [x] These modules do not introduce new physics computations; they export contracts.
- [x] Quality-flag semantics are centralized in `bittr_tess_vetter.io.mast_client` and surfaced via `DEFAULT_QUALITY_MASK` and `QUALITY_FLAG_BITS`.
- [x] Underlying `MASTClient.download_lightcurve()` returns `LightCurveData.time` as days consistent with the `LightCurveData` contract (“BTJD timestamps”).
- [x] Underlying `Target.from_tic_response()` maps TIC fields into `StellarParameters` with non-negativity validation on `teff/radius/mass/luminosity/contamination`.

### Data hygiene / failure modes

- [x] API exports explicitly include error types (`NameResolutionError`, `TargetNotFoundError`, `LightCurveNotFoundError`, `MASTClientError`) so host apps can distinguish “not found” vs “network/client” failures.
- [x] Cache abstractions are part of the public contract (`SessionCache`, `PersistentCache`).

### Tests

- [~] No direct tests for these API wrapper modules (expected: they are re-export layers).
- [~] Behavioral tests should live against `bittr_tess_vetter.io` and `bittr_tess_vetter.domain.target`.

## Fixes / changes (if any)

- None proposed here; the correctness risk is primarily in the underlying modules (`io` and `domain/target`), not these facades.
