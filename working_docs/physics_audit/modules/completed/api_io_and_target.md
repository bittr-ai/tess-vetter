# Module Review: `api/io.py` + `api/target.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

These modules define the host-facing contract for:
- how light curves are discovered/downloaded/cached and quality-masked (`api/io.py`), and
- how targets + stellar parameters are represented (`api/target.py`).

Even though they are mostly wrappers, mistakes here (quality-mask defaults, time/flux conventions in returned payloads, parameter units) can silently poison downstream detection/vetting.

## Scope (exports + underlying implementations)

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

### Underlying modules audited for physics context

- `src/bittr_tess_vetter/io/mast_client.py`
  - Defines quality-mask semantics and product selection rules
  - Implements `resolve_target`, `get_target_info`, `search_lightcurve`, `download_lightcurve`
- `src/bittr_tess_vetter/io/cache.py`
  - Defines cache key/value persistence and eviction (provenance + trust boundary)
- `src/bittr_tess_vetter/domain/target.py`
  - Defines stellar parameter units and validation used in downstream consistency checks

## Audit checklist (filled)

### Units + conventions

- [x] These modules do not introduce new physics computations; they export contracts.
- [x] Quality-flag semantics are centralized in `bittr_tess_vetter.io.mast_client` and surfaced via `DEFAULT_QUALITY_MASK` and `QUALITY_FLAG_BITS`.
- [x] Underlying `MASTClient.download_lightcurve()` returns `LightCurveData.time` as days consistent with the `LightCurveData` contract (“BTJD timestamps”).
- [x] Underlying `Target.from_tic_response()` maps TIC fields into `StellarParameters` with non-negativity validation on `teff/radius/mass/luminosity/contamination`.

### Data hygiene / failure modes

- [x] API exports explicitly include error types (`NameResolutionError`, `TargetNotFoundError`, `LightCurveNotFoundError`, `MASTClientError`) so host apps can distinguish “not found” vs “network/client” failures.
- [x] Cache abstractions are part of the public contract (`SessionCache`, `PersistentCache`).

### Time systems (BTJD/BJD) and metadata propagation

- [x] LC `time` is taken from lightkurve `lc.time.value` (numeric days); expected to be BTJD for TESS.
- [⚠️] No explicit validation that the time scale is BTJD (relies on upstream conventions); no time-scale metadata is persisted alongside the arrays.

### Sector/cadence selection rules; exposure time handling (20s vs 120s)

- [x] `download_lightcurve` requires `sector` explicitly (no implicit “best sector” selection).
- [x] Optional `exptime` filters search results using ~1s tolerance.
- [x] If multiple products exist and `exptime` is omitted, product choice is deterministic (prefers exptime near 120s; stable tie-break).
- [x] `cadence_seconds` is inferred from the returned time array (median of positive finite deltas), not from `exptime`.

### Flux type semantics (`sap` vs `pdcsap`) and normalization assumptions

- [x] `flux_type` is constrained to `{"pdcsap","sap"}` and selects `pdcsap_flux` vs `sap_flux` (with fallback to `lc.flux`).
- [x] Normalization (when enabled) divides by median of `flux_raw[valid_mask]` (median ~1.0), consistent with the `LightCurveData` contract.
- [x] If `flux_err` is unavailable, a representative per-point uncertainty is estimated from flux scatter and recorded in provenance.
- [x] Light-curve provenance (selected author/exptime, requested exptime, applied `quality_mask`, normalization flag) is stored in `LightCurveData.provenance`.

### Target resolution: ambiguity, radius thresholds, tie-breaking rules

- [x] Fast paths parse `TIC <id>` and bare numeric ids (positive only).
- [x] Name/coordinate resolution uses TIC cone match with default `radius_arcsec=10`.
- [x] Tie-break is closest `dstArcSec`; ambiguous cones attach a `ResolvedTarget.warning` string.
- [⚠️] Default 10" radius is potentially risky in crowded fields; warning is informational only (no “must confirm” guardrail).

### Data provenance + caching keys

- [x] `SessionCache` namespaces by `session_id` and LRU-evicts in-memory.
- [x] `PersistentCache` stores arbitrary values on disk keyed by an opaque string and writes light metadata (including `tic_id/sector` if present).
- [⚠️] `PersistentCache` uses pickle: this is a local-trust boundary (safe only if cache dir contents are trusted).

### Tests

- [~] No direct tests for these API wrapper modules (expected: they are re-export layers).
- [~] Behavioral tests should live against `bittr_tess_vetter.io` and `bittr_tess_vetter.domain.target`.

## Fixes / changes (if any)

- No must-fix physics bug identified in the wrapper modules themselves.
- Follow-ups (remaining):
  - Persist a stable product identifier (e.g., MAST data URI / obsid) if available from lightkurve rows.
  - Persist/validate explicit time-scale metadata (BTJD vs BJD_TDB) for auditability.
