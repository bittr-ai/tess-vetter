# Physics Conventions (Contract)

This file defines the expected conventions for physics-related functions. If a function uses different conventions, it must:
1) clearly document the deviation, and
2) validate inputs to prevent silent mistakes.

## Time + ephemeris

- `time` arrays: days (TESS BTJD for TESS light curves, unless explicitly documented otherwise).
- `t0` / `t0_btjd`: days (same scale as `time`).
- `period_days`: days.
- `duration_hours`: hours (convert internally using `/ 24.0`).

## Flux + depth

- Flux is expected to be **normalized near 1.0** when used for simple depth metrics.
- Depth conventions:
  - `depth_ppm`: parts-per-million, `ppm = depth_fraction * 1e6`.
  - `depth` (fractional): unitless, typically `~1e-4` for 100 ppm.
- Any function accepting both must be explicit and validate ranges.

## Phase folding

- Phase definition must be explicit:
  - recommended: phase in `[-0.5, 0.5]` with transit centered near 0.

## Masks

- In-transit mask definitions must be explicit about buffer multipliers and edge handling.
- All mask helpers must handle NaNs and empty selections without crashing.

## Stellar parameters

- `radius_rsun`, `mass_msun`, `teff_k`, `logg_cgs` if present: document source and expected ranges.
- Density:
  - if computed from transit (a/Rs), state assumptions (circular orbit, small planet, etc.).

## Numerical stability

- Functions must define behavior for:
  - `n_transits < 2`, few points in-transit, gapped data, irregular cadence, NaNs.

