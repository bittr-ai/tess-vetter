# Module Review: `api/timing.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

Transit timing is an early “candidate deepening” step: after an ephemeris is known,
agents will often check for TTVs and outlier events. Timing code is sensitive to:
- units (days vs hours vs seconds)
- window sizing (duration-based extraction)
- masking/NaNs and gapped time series

If timing is wrong, it produces convincing but incorrect TTV signals.

## Scope (functions)

- `measure_transit_times` (wrapper around `transit.timing.measure_all_transit_times`)
- `analyze_ttvs` (wrapper around `transit.timing.compute_ttv_statistics`)

## Audit checklist (to fill)

### Units + conventions

- [x] `period_days` (days), `t0_btjd` (BTJD days), `duration_hours` (hours)
- [x] `TransitTime.tc` and `tc_err` are in **days**; derived O-C is in **seconds**
- [x] Default windowing is a multiple of duration (2×) and is symmetric around t_center

### Data hygiene / edge cases

- [x] Uses `LightCurve.to_internal()` and respects `valid_mask` (NaNs excluded)
- [x] Skips transits with insufficient window points or non-convergence
- [x] O-C periodicity metric is 0 for small N and for near-constant residuals

### Tests

- [x] Synthetic: box transit → per-epoch tc recovered within tolerance
- [x] Synthetic: perfect linear ephemeris → RMS small, periodicity ~0
- [x] Synthetic: inject NaNs → still measures (via valid_mask)
