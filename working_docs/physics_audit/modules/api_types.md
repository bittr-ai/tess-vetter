# Module Review: `api/types.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Scope

- `Ephemeris` (units + validation)
- `LightCurve` (dtype normalization → `domain.lightcurve.LightCurveData`)
- `CheckResult` (metrics-only semantics)
- `Candidate`, `TPFStamp`, `VettingBundleResult` (contract sanity)

## Audit checklist (to fill)

### Units + conventions

- [ ] `Ephemeris.period_days` days, `t0_btjd` BTJD days, `duration_hours` hours (validated)
- [ ] `LightCurve.time` documented as BTJD days; no silent conversions
- [ ] `LightCurve.flux` assumed normalized (~1.0); document where normalization is required vs optional

### Numerical stability

- [ ] Ensure `to_internal()` validates consistent array lengths (time/flux/err/quality/mask)
- [ ] Ensure dtype coercions don’t silently overflow (quality int32 ok; time/flux float64 ok)

### Policy semantics

- [ ] Confirm `CheckResult.passed=None` is consistently used for metrics-only checks (and not misinterpreted as fail)

### Tests

- [ ] Identify existing tests that cover dtype + length validation
- [ ] Add tests if missing: mismatched lengths should raise (or document why not)

