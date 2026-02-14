# Report Generation

This module builds and renders vetting reports from light-curve inputs.

It has two responsibilities:
- Assemble structured report data (`ReportData`) from LC + candidate inputs.
- Render `ReportData` as self-contained HTML (Plotly-based).

## Public Entry Points

- `build_report(...)` in `bittr_tess_vetter.report`
- `render_html(...)` in `bittr_tess_vetter.report`

Most callers should import from `bittr_tess_vetter.report` (not private files).

## Typical Usage

### LC-only report data + HTML

```python
from bittr_tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from bittr_tess_vetter.report import build_report, render_html

lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
candidate = Candidate(
    ephemeris=Ephemeris(
        period_days=3.5,
        t0_btjd=1850.0,
        duration_hours=2.5,
    ),
    depth_ppm=500.0,
)

report = build_report(
    lc,
    candidate,
    tic_id=123456789,
    toi="TOI-1234.01",
)

html = render_html(report)
```

### End-to-end API (download + stitch + report + optional enrichment)

```python
from bittr_tess_vetter.api.generate_report import generate_report, EnrichmentConfig

result = generate_report(
    tic_id=340458804,
    period_days=194.243,
    t0_btjd=2039.7625,
    duration_hours=9.256,
    depth_ppm=811.0,
    toi="4510.01",
    include_html=True,
    include_enrichment=True,
    enrichment_config=EnrichmentConfig(network=False),
)

report = result.report
report_json = result.report_json
plot_data_json = result.plot_data_json
html = result.html
```

## Data Contract

`ReportData` is the canonical report packet. `to_json()` is JSON-safe and intended for:
- frontend rendering
- artifact persistence
- API transport

External adapter/charting seam documentation lives at `docs/report_external_charting_seam.md`.

Key top-level blocks:
- Identity/inputs: `tic_id`, `toi`, `ephemeris`, `stellar`, `input_depth_ppm`
- Summary: `lc_summary`
- Checks: `checks`, `bundle_summary`, `checks_run`
- Plot payloads: `full_lc`, `phase_folded`, `per_transit_stack`, `local_detrend`, `odd_even_phase`, `secondary_scan`, `oot_context`, `timing_series`, `alias_summary`
- Robustness: `lc_robustness`
- Optional enrichment: `enrichment` with `pixel_diagnostics`, `catalog_context`, `followup_context`

## Contract and Versioning Policy

- `schema_version` (`ReportData.version`, currently `2.0.0`) is the top-level payload contract version.
- Bump `schema_version` when JSON structure changes in a way that can break clients (add/remove/rename/move top-level or nested contract fields, or change field types).
- `payload_meta.summary_version` tracks semantic/shape changes scoped to `summary`.
- `payload_meta.plot_data_version` tracks semantic/shape changes scoped to `plot_data`.
- Bump `summary_version` or `plot_data_version` for domain-level evolution in that block when top-level contract compatibility is still maintained.
- `payload_meta.summary_hash` and `payload_meta.plot_data_hash` are deterministic content hashes; treat them as cache keys, not compatibility versions.

## Null and Missing-Key Semantics (Frontend)

- Producer validates payloads, then serializes with `exclude_none=True`; any `None` value is emitted as a missing key.
- Missing key means "unavailable/not produced" and must not be interpreted as `0`, `false`, or empty string.
- Some summary blocks are always present but may be empty objects (`{}`) when all values are unavailable (for example `summary.data_gap_summary`).
- Plot sections are additive/optional: if a plot block key is absent in `plot_data`, that panel should be hidden or rendered as unavailable.
- Keep frontend reads defensive (`key in object` checks), especially for scalar metrics that may appear/disappear by dataset quality.

## Feature Gating With `payload_meta`

- Use `payload_meta.contract_version` to gate UI logic tied to metric-contract interpretation (current value: `"1"`).
- `payload_meta.required_metrics_by_check` defines required keys per summary-relevant check (`V01`, `V02`, `V04`, `V05`, `V13`, `V15`).
- `payload_meta.missing_required_metrics_by_check` lists required keys missing from each *present* check result.
- `payload_meta.has_missing_required_metrics` is the coarse gate: if `true`, disable or degrade UI features that require strict metric completeness.
- `payload_meta.metric_keys_by_check` is the observed key set for debugging and telemetry; use it to explain degraded panels.
- For check-level gating, require both check presence (`summary.checks`/`summary.checks_run`) and no entry for that check in `payload_meta.missing_required_metrics_by_check`.

## Current HTML Sections

- Light Curve Summary
- Full Light Curve
- Phase-Folded Transit
- Per-Transit Stack
- Local Detrend Baseline Diagnostic
- Odd vs Even Transits
- Secondary Eclipse / Phase Scan
- Out-of-Transit Noise Context
- LC Robustness Summary
- Enrichment Summary (lightweight status/provenance panel)
- Check Results

## Performance and Payload Controls

Use `build_report(...)` budgets to control payload size:
- `max_lc_points`
- `max_phase_points`
- `max_transit_windows`
- `max_points_per_window`
- `max_timing_points`
- `max_lc_robustness_epochs`

Use `generate_report(..., enrichment_config=...)` for non-LC controls:
- `max_network_seconds`
- `per_request_timeout_seconds`
- `max_concurrent_requests`
- `max_pixel_points`
- `max_catalog_rows`

## Extension Guidelines

When adding new report capabilities:
- Add structured fields in `_data.py` first.
- Keep `build_report(...)` orchestration and report assembly flow in `_build_core.py`.
- Add or extend panel payload builders in `_build_panels.py`.
- Put reusable build helpers in `_build_utils.py`.
- Keep `_build.py` as legacy import path exposing `build_report` only.
- Use API-layer modules for non-LC enrichment assembly.
- Keep renderer (`_render_html.py`) presentation-only.
- Prefer additive fields and deterministic defaults.
- Keep JSON primitive-safe (`dict/list/str/int/float/bool/null`).

## File Map

- `__init__.py`: public exports
- `_data.py`: report dataclasses + serialization
- `_build_core.py`: `build_report(...)` orchestration and core assembly
- `_build_panels.py`: panel/diagnostic payload builders
- `_build_utils.py`: shared build helpers
- `_build.py`: legacy module path exposing `build_report` only
- `_render_html.py`: HTML renderer
