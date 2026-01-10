# Module Review: `validation/base.py` (transit masks + depth primitives)

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

`validation/base.py` defines shared conventions for:
- in-transit / out-of-transit masks
- depth measurement primitives used across many checks (odd/even, secondary, systematics proxy, etc.)

If these are inconsistent or numerically unstable, every downstream metric is biased.

## Scope (functions)

- `get_in_transit_mask`
- `get_out_of_transit_mask`
- `measure_transit_depth` (and any shared depth/uncertainty helpers)

## Audit checklist (to fill)

### Units + conventions

- [x] `period_days` / `t0_btjd` are days; `duration_hours` is hours
- [x] In-transit is centered at phase 0; phase convention matches compute/transit (phase in [-0.5, 0.5))
- [x] OOT mask uses a buffer policy that is explicit and consistent across callers

### Numerical robustness

- [x] Handles non-finite time/flux gracefully (masks exclude non-finite time; depth ignores non-finite flux)
- [x] Avoids divide-by-zero when baseline ~0 (depth helper returns (0,1) when insufficient finite points)
- [x] Works with irregular cadence / gaps (phase-based, no uniform sampling assumption)

### Tests

- [x] Mask sanity: expected number of in-transit points for known cadence/ephemeris
- [x] Depth sanity on synthetic injection: recovered depth within tolerance
- [x] OOT buffer works: excludes ingress/egress region as intended

## Notes (final)

- Masks now explicitly exclude non-finite time points so invalid timestamps are never counted as OOT by complement logic.
- `measure_transit_depth` filters non-finite flux before computing medians and uncertainties to prevent NaNs from silently corrupting depth errors.
