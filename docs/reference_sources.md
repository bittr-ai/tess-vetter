# `reference_sources.v1` Schema

This document defines the standardized `reference_sources.v1` JSON payload used for neighbor-resolution outputs and downstream host-localization/dilution workflows.

Implementation references:
- `src/tess_vetter/cli/reference_sources.py`
- `src/tess_vetter/cli/resolve_neighbors_cli.py`
- `src/tess_vetter/cli/localize_host_cli.py`
- `src/tess_vetter/cli/dilution_cli.py`

## Top-level contract

Required fields:
- `schema_version` (string): must be exactly `"reference_sources.v1"`.
- `reference_sources` (array, non-empty): list of source objects.

Optional fields:
- `multiplicity_risk` (object): risk rollup emitted by `btv resolve-neighbors`; consumed by `btv localize-host` and `btv dilution` when present.
- `host_ambiguous` (boolean): optional policy hint used by `btv dilution`.
- `target` (object): producer metadata (for example TIC/TOI and resolved coordinates).
- `provenance` (object): producer metadata (for example Gaia/coordinate lookup details).
- `verdict`, `verdict_source`, `result` (object): canonical verdict envelope fields when produced by `btv resolve-neighbors`.

Unknown top-level fields are allowed by current consumers unless they conflict with required type checks.

## `reference_sources[]` object contract

Validated required fields (shared validator path):
- `name` (string, non-empty)
- `ra` (finite number, degrees)
- `dec` (finite number, degrees)

Validated optional fields (shared validator path):
- `source_id` (string, non-empty if provided)
- `meta` (object)

Common optional fields emitted by `btv resolve-neighbors`:
- `tic_id` (integer, usually for target row)
- `role` (string, commonly `"target"` or `"companion"`)
- `separation_arcsec` (number, commonly for companion rows)
- `g_mag` (number or null)
- `radius_rsun` (number or null)
- `is_target` (boolean; accepted by `btv dilution` parser)

## Additional dilution-specific expectations

When the same file is passed to `btv dilution --reference-sources-file`, the dilution parser applies companion-host inference rules:
- At most one target source.
- Target can be inferred by any of:
  - `is_target: true`
  - `role` in `{"target","primary","host"}`
  - presence of `tic_id`
  - `source_id` prefixed with `tic:`
- For non-target companion rows:
  - `source_id` must parse to an integer ID (raw integer, numeric string, or `<prefix>:<numeric_id>`).
  - `separation_arcsec` is required (directly or from `meta.separation_arcsec`).
- `g_mag` may come from `g_mag` or fallback `meta.phot_g_mean_mag`.

## Example payload

```json
{
  "schema_version": "reference_sources.v1",
  "reference_sources": [
    {
      "name": "Target TIC 123456789",
      "source_id": "tic:123456789",
      "tic_id": 123456789,
      "role": "target",
      "ra": 123.456,
      "dec": -12.345,
      "g_mag": 11.2,
      "separation_arcsec": 0.0,
      "meta": {
        "source": "gaia_dr3_primary"
      }
    },
    {
      "name": "Gaia 987654321",
      "source_id": "gaia:987654321",
      "role": "companion",
      "ra": 123.457,
      "dec": -12.346,
      "separation_arcsec": 3.1,
      "g_mag": 14.7,
      "meta": {
        "source": "gaia_dr3_neighbor",
        "separation_arcsec": 3.1
      }
    }
  ],
  "multiplicity_risk": {
    "risk_level": "moderate"
  }
}
```

## CLI commands that emit or consume `reference_sources.v1`

Emit:
- `btv resolve-neighbors ... -o outputs/reference_sources.json`

Consume:
- `btv localize-host ... --reference-sources-file outputs/reference_sources.json`
- `btv dilution ... --reference-sources-file outputs/reference_sources.json`

Common chain:

```bash
btv resolve-neighbors TOI-123.01 --network-ok -o outputs/reference_sources.json
btv localize-host TOI-123.01 --network-ok --reference-sources-file outputs/reference_sources.json -o outputs/localize_host.json
btv dilution TOI-123.01 --network-ok --reference-sources-file outputs/reference_sources.json -o outputs/dilution.json
```
