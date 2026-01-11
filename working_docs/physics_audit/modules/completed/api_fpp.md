# Module Review: `api/fpp.py` (+ `validation/triceratops_fpp.py`)

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

FPP is one of the strongest “downstream” decision inputs. If units, windowing, downsampling, or priors are
misapplied, the numerical output can look authoritative while being wrong.

This library must stay **metrics-only**: it can compute an FPP, but any “validated/not validated” policy belongs
elsewhere.

## Scope (functions / types)

- `calculate_fpp` (API entrypoint; preset selection + parameter normalization)
- `ExternalLightCurve` (time convention + band labels)
- `TriceratopsFppPreset` (`FAST_PRESET`, `STANDARD_PRESET`)
- Implementation: `validation/triceratops_fpp.calculate_fpp_handler`

## Audit checklist (to fill)

### Units + conventions

- [x] Inputs: `period` (days), `t0` (BTJD days), `depth_ppm` (ppm), `duration_hours` (hours)
- [x] `ExternalLightCurve.time_from_midtransit_days` is relative time (days); the implementation writes these to temp files for TRICERATOPS+
- [x] `ExternalLightCurve.flux` is expected normalized near 1.0 (caller responsibility; no silent renormalization here)

### Presets / parameterization

- [x] `FAST_PRESET`/`STANDARD_PRESET` provide explicit runtime/fidelity tradeoff knobs (windowing/downsample/noise floor)
- [x] `overrides` merges deterministically and is normalized to correct types in the API wrapper
- [x] `timeout_seconds` is passed through; implementation enforces a deadline and passes remaining time to `calc_probs`

### Data hygiene

- [x] `max_points` downsampling is deterministic (index grid via linspace/unique; no RNG)
- [x] Windowing (`window_duration_mult`) uses `dur_days` consistently; minimum half-window is clamped to 0.25 days
- [x] Noise floor logic (`min_flux_err`, `use_empirical_noise_floor`) is explicit conditioning to avoid degenerate NaN posteriors for very bright targets

### Physics correctness

- [x] `depth_ppm` is converted to fractional (`depth_ppm/1e6`) before TRICERATOPS depth computations
- [x] Gaia neighbor search is handled inside TRICERATOPS+ target construction; `contrast_curve` is accepted but not yet integrated (must not be assumed active)
- [x] If `duration_hours` is None, the handler estimates a duration (using stellar radius/mass when available)

### Tests

- [x] Unit test: preset merge behavior (overrides take effect) (`tests/test_api/test_fpp_api.py`)
- [x] Unit test: external LC schema creation works and is passed through (`tests/test_api/test_fpp_api.py`)
- [x] Unit test: degenerate FPP outputs are detected/annotated (`tests/validation/test_triceratops_fpp_replicates.py`)

## Notes (initial)

- `api/fpp.py` delegates almost entirely to `validation/triceratops_fpp.calculate_fpp_handler`; this audit is mainly about
  unit conventions and whether presets/conditioning are applied consistently and transparently.
