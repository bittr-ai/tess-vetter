# CLI Contracts

This page defines machine-consumer JSON contract conventions for `btv` commands and catalogs key `schema_version` values.

Implementation references:
- `src/bittr_tess_vetter/cli/common_cli.py`
- command modules under `src/bittr_tess_vetter/cli/`
- `src/bittr_tess_vetter/api/report_vet_reuse.py`
- `src/bittr_tess_vetter/pipeline_composition/executor.py`

## JSON envelope conventions

For JSON-producing commands, the standard envelope is:
- `schema_version`: schema tag string for that command payload.
- `result`: nested object for stable programmatic consumption.
- `verdict` and `verdict_source`: canonical summary judgment fields for verdict-bearing commands.
- `provenance`: optional machine-readable execution/input provenance.

Not every command emits all fields. `schema_version` is the required discriminator for contract routing.

## Canonical verdict contract

For verdict-bearing commands, canonical fields are emitted in both places:
- top-level: `verdict`, `verdict_source`
- nested: `result.verdict`, `result.verdict_source`

`verdict_source` is a JSONPath-like pointer to the legacy/native field used to derive the canonical verdict.

Canonical pattern:

```json
{
  "schema_version": "cli.<command>.vN",
  "verdict": "<string-or-null>",
  "verdict_source": "$.<path>",
  "result": {
    "verdict": "<string-or-null>",
    "verdict_source": "$.<path>"
  }
}
```

Backward compatibility note:
- command-specific legacy verdict fields still exist for compatibility.
- new consumers should prefer canonical verdict fields above.

## `schema_version` catalog (key commands)

### Core diagnostics and reports
- `btv vet`: `cli.vet.v2`
- `btv vet --plot-data-out`: `cli.vet.plot_data.v1` (plot-data sidecar file)
- `btv fpp`: `cli.fpp.v3`
- `btv fpp prepare`: `cli.fpp.prepare.v1`
- `btv report`: `cli.report.v3`
- `btv measure-sectors`: `cli.measure_sectors.v1`
- `btv detrend-grid`: `cli.detrend_grid.v1`
- `btv model-compete`: `cli.model_compete.v1`
- `btv ephemeris-reliability`: `cli.ephemeris_reliability.v1`
- `btv timing`: `cli.timing.v1`
- `btv activity`: `cli.activity.v1`
- `btv systematics-proxy`: `cli.systematics_proxy.v1`
- `btv rv-feasibility`: `cli.rv_feasibility.v1`
- `btv followup`: `cli.followup.v1`

### Localization / dilution / neighbors
- `btv resolve-neighbors`: `reference_sources.v1`
- `btv localize`: `cli.localize.v1`
- `btv localize-host`: `cli.localize_host.v1`
- `btv dilution`: `cli.dilution.v1`
- `btv resolve-stellar`: `cli.resolve-stellar.v1`
- `btv contrast-curves`: `cli.contrast_curves.v2`
- `btv contrast-curves summarize`: `cli.contrast_curve_summary.v1`

### Data acquisition and discovery
- `btv periodogram`: `cli.periodogram.v1`
- `btv transit-fit`: `cli.fit.v1`
- `btv fetch`: `cli.fetch.v1`
- `btv fetch cache-sectors`: `cli.cache_sectors.v1`
- `btv toi-query`: `cli.toi_query.v1`

### Pipeline outputs relevant to release contracts
- Pipeline evidence table JSON: `pipeline.evidence_table.v5`

## Consumer guidance

- Route by exact `schema_version` string first.
- For verdict-bearing payloads, read canonical fields first and use legacy fields only as fallback.
- For `reference_sources.v1`, see the dedicated schema page: `reference_sources`.
