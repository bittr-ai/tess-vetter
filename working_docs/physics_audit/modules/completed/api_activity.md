# Module Review: `api/activity.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

Activity characterization is used early to decide:
- whether to trust a TLS search,
- whether to detrend/mask flares first,
- whether to use the recovery pipeline.

If rotation/flare detection is brittle (NaNs, gaps, unsorted time), it drives bad downstream decisions.

## Scope (functions)

- `characterize_activity` (rotation period + variability + flare detection)
- `mask_flares` (interpolate over flare windows)

## Audit checklist (to fill)

### Units + conventions

- [x] Time is BTJD days, rotation periods are days
- [x] Variability metrics are in ppm where labeled

### Data hygiene / edge cases

- [x] Uses `LightCurve.to_internal()` and respects `valid_mask`
- [x] Robust to NaNs/Infs and large gaps (cadence inference ignores gaps)
- [x] Does not assume time is sorted for baseline/period constraints

### Tests

- [x] Synthetic sinusoid: recovers rotation period within tolerance
- [x] Synthetic flare spikes: flare detector finds events, mask_flares removes them
- [x] Unsorted + NaNs: characterize_activity still returns finite outputs
