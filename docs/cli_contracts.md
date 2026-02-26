# CLI Contracts

This page defines machine-consumer JSON contract conventions for `btv` commands and catalogs key `schema_version` values.

Implementation references:
- `src/tess_vetter/cli/common_cli.py`
- command modules under `src/tess_vetter/cli/`
- `src/tess_vetter/api/report_vet_reuse.py`
- `src/tess_vetter/pipeline_composition/executor.py`

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
- `btv vet --split-plot-data`: `cli.vet.plot_data.v1` (writes `<out>.plot_data.json` sidecar; default on)
- `btv fpp`: `cli.fpp.v3`
- `btv fpp --prepare-manifest <manifest.json>`: `cli.fpp.v3` (prepared-manifest compute mode; same runtime path as `btv fpp-run`)
- `btv fpp-prepare`: `cli.fpp.prepare.v1`
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
- `btv contrast-curve-summary`: `cli.contrast_curve_summary.v1`

### Data acquisition and discovery
- `btv periodogram`: `cli.periodogram.v1`
- `btv fit`: `cli.fit.v1`
- `btv fetch`: `cli.fetch.v1`
- `btv cache-sectors`: `cli.cache_sectors.v1`
- `btv toi-query`: `cli.toi_query.v1`

## Timing and Ephemeris fields

### `btv ephemeris-reliability` (`cli.ephemeris_reliability.v1`)
- Additive canonical scalar fields are emitted in both locations:
- top-level: `schedulability_scalar`
- nested: `result.schedulability_scalar`
- Source: `result.schedulability_summary.scalar` (from schedulability summary computation).
- Nullability: may be `null` when summary scalar cannot be derived.

### `btv timing` (`cli.timing.v1`)
- Additive canonical scalar fields are emitted in both locations:
- top-level: `schedulability_scalar`
- nested: `result.schedulability_scalar`
- Source: ephemeris reliability regime + schedulability summary computation evaluated on the timing candidate/series.
- Nullability: may be `null` when schedulability computation is unavailable.

### Pipeline outputs relevant to release contracts
- Pipeline evidence table JSON: `pipeline.evidence_table.v5`
- Pipeline run manifest JSON: `pipeline.run_manifest.v1`
- Per-TOI pipeline result JSON: `pipeline.result.v1`

## Pipeline-Run Contract Notes

For `btv pipeline run`, contract consumers should treat these behaviors as stable:

- Manifest options include `resume`:
  - `run_manifest.json` records `options.resume` exactly from CLI (`--resume` => `true`).
- Resume step semantics:
  - When `--resume` reuses completed checkpoints, per-step rows in `pipeline_result.json` retain `status: "ok"` and set `skipped_resume: true`.
  - Checkpoint marker files remain under `<out_dir>/<toi>/checkpoints/*.done.json` and are used as the resume source of truth with matching input fingerprints.
- Partial-failure semantics with `--continue-on-error`:
  - CLI summary line reports counts as `ok`, `partial`, `failed`.
  - A TOI with at least one failed step and continued execution is labeled `status: "partial"` in both manifest results and per-TOI `pipeline_result.json`.
  - Failed steps are labeled `status: "failed"` in `pipeline_result.json.steps`, while subsequent steps may still be `status: "ok"`.
  - For multi-TOI runs, counts and per-TOI statuses are independent; mixed outcomes are expected (for example, `n_tois=2 ok=1 partial=1 failed=0`).
  - In mixed-outcome runs resumed with `--resume`, previously `ok` TOIs can be fully checkpoint-reused (`skipped_resume: true` on all successful steps) while previously `partial` TOIs rerun only steps without reusable success markers.
  - A resumed mixed-outcome run can converge to all-`ok` when prior failed steps succeed; this is reflected in manifest counts and per-TOI `status` updates.

### Evidence table schema highlights (`pipeline.evidence_table.v5`)
- Per-row key `stellar_contamination_risk_scalar` is included in JSON rows and CSV columns.
- Source mapping:
  - preferred: `summary.stellar_contamination_risk_scalar`
  - fallback: `summary.stellar_contamination_summary.risk_scalar`

## Consumer guidance

- Route by exact `schema_version` string first.
- For verdict-bearing payloads, read canonical fields first and use legacy fields only as fallback.
- For `reference_sources.v1`, see the dedicated schema page: `reference_sources`.
