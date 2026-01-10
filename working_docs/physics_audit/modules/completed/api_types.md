# Module Review: `api/types.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Scope

- `Ephemeris` (units + validation)
- `LightCurve` (dtype normalization → `domain.lightcurve.LightCurveData`)
- `CheckResult` (metrics-only semantics)
- `Candidate`, `TPFStamp`, `VettingBundleResult` (contract sanity)

## Audit checklist (to fill)

### Units + conventions

- [x] `Ephemeris.period_days` days, `t0_btjd` BTJD days, `duration_hours` hours (validated >0 where applicable)
- [x] `LightCurve.time` documented as BTJD days; `to_internal()` performs dtype normalization only (no unit conversion)
- [x] `LightCurve.flux` assumed normalized (~1.0); the API does not renormalize (callers should normalize upstream)

### Numerical stability

- [x] `to_internal()` validates consistent array lengths (time/flux/err/quality/mask) and raises on mismatch
- [x] Dtype coercions are explicit (time/flux/err float64; quality int32; valid_mask bool) and safe for expected ranges

### Policy semantics

- [x] `CheckResult.passed=None` is the stable “metrics-only” signal; wrappers also attach `details['_metrics_only']=True`

### Tests

- [x] Existing dtype/default coverage in `tests/test_api/test_types.py`
- [x] Added: mismatched lengths raise; NaNs/Infs are excluded via `valid_mask &= finite_mask` (`tests/test_api/test_types.py`)
